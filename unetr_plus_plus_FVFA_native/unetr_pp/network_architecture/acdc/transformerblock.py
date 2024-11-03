import torch.nn as nn
import torch
from unetr_pp.network_architecture.dynunet_block import UnetResBlock
from functools import partial
from timm.models.layers import DropPath
import math
import torch.nn.functional as F
import importlib.util
import numpy as np

def get_seqlen_and_mask_3d(input_resolution, window_size, device):

    D, H, W = input_resolution
    attn_map = torch.ones([1, 1, D, H, W], device=device)
    
    pad = (window_size // 2, window_size // 2, window_size // 2)
    
    # Unfold the 3D attention map
    attn_map = F.pad(attn_map, (pad[2], pad[2], pad[1], pad[1], pad[0], pad[0]))
    attn_map = attn_map.unfold(2, window_size, 1) 
    attn_map = attn_map.unfold(3, window_size, 1) 
    attn_map = attn_map.unfold(4, window_size, 1)

    attn_map = attn_map.reshape(1, 1, D, H, W, -1) 
    attn_map = attn_map.permute(0, 5, 1, 2, 3, 4)
    attn_map = attn_map.contiguous().view(1, 27, -1)
 
    attn_mask = (attn_map.squeeze(0).permute(1, 0)) == 0
    return attn_mask

def unfold_3d(x, kernel_size):
    B, C, D, H, W = x.shape
    pad = kernel_size // 2
    
    x_padded = F.pad(x, (pad, pad, pad, pad, pad, pad))

    unfolded = x_padded.unfold(2, kernel_size, 1)
    unfolded = unfolded.unfold(3, kernel_size, 1)
    unfolded = unfolded.unfold(4, kernel_size, 1)
    
    return unfolded.reshape(B, C, D*H*W, kernel_size**3)


class TransformerBlock(nn.Module):
    """
    A transformer block, based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    """

    def __init__(
            self,
            input_resolution,
            hidden_size: int,
            num_heads: int,
            dropout_rate: float = 0.0,
            sr_ratio=1,
            norm_layer=nn.LayerNorm,
            pos_embed=True,
            window_size=3
    ) -> None:
        """
        Args:
            hidden_size: dimension of hidden layer.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.
        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            print("Hidden size is ", hidden_size)
            print("Num heads is ", num_heads)
            raise ValueError("hidden_size should be divisible by num_heads.")
        
        # Note: The following line is a mistake but should not be removed
        self.pos_embed = pos_embed
        self.norm = nn.LayerNorm(hidden_size)
       
        self.gamma = nn.Parameter(1e-6 * torch.ones(hidden_size), requires_grad=True)
        self.epa_block = EPA(hidden_size=hidden_size, num_heads=num_heads,
                             channel_attn_drop=dropout_rate,spatial_attn_drop=dropout_rate, window_size=window_size, input_resolution=input_resolution, sr_ratio=sr_ratio)
        self.sr_ratio = sr_ratio
        self.window_size = window_size
        self.conv51 = UnetResBlock(3, hidden_size, hidden_size, kernel_size=3, stride=1, norm_name="batch")
        self.conv52 = UnetResBlock(3, hidden_size, hidden_size, kernel_size=3, stride=1, norm_name="batch")
        self.conv8 = nn.Sequential(nn.Dropout3d(0.1, False), nn.Conv3d(hidden_size, hidden_size, 1))

        self.drop_path = DropPath(dropout_rate) if dropout_rate > 0. else nn.Identity()

    def forward(self, x, relative_pos_index, relative_coords_table, seq_length_scale):
        B, C, D, H, W = x.shape

        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)

        if self.pos_embed is not None:
            x = x + self.pos_embed
            
        attn_mask = get_seqlen_and_mask_3d((D, H, W), self.window_size, x.device)
        attn = x + self.gamma * self.epa_block(self.norm(x), H, W, D, relative_pos_index, relative_coords_table, seq_length_scale, attn_mask)

        attn_skip = attn.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)  # (B, C, H, W, D)
        attn = self.conv51(attn_skip)
        attn = self.conv52(attn)
        x = attn_skip + self.conv8(attn)
        return x

class EPA(nn.Module):
    def __init__(self, hidden_size, num_heads=4, qkv_bias=False,
                 channel_attn_drop=0.1, spatial_attn_drop=0.1, window_size=3, sr_ratio=8, input_resolution=(16, 40, 40)):
        super().__init__()
        self.num_heads = num_heads
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.window_size = window_size

        # qkvv are 4 linear layers (query_shared, key_shared, value_spatial, value_channel)
        self.qkvv = nn.Linear(hidden_size, hidden_size * 4, bias=qkv_bias)

        self.attn_drop = nn.Dropout(channel_attn_drop)
        self.attn_drop_2 = nn.Dropout(spatial_attn_drop)
        
        self.out_proj = nn.Linear(hidden_size, hidden_size // 2)
        self.out_proj2 = nn.Linear(hidden_size, hidden_size // 2)
        
        self.dim = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.sr_ratio = sr_ratio

        # The estimated training resolution is used for bilinear interpolation of the generated relative position bias.
        self.trained_D, self.trained_H, self.trained_W = input_resolution
        self.trained_len = self.trained_H * self.trained_W * self.trained_D
        self.trained_pool_D, self.trained_pool_H, self.trained_pool_W = input_resolution[0] // self.sr_ratio, input_resolution[1] // self.sr_ratio, input_resolution[2] // self.sr_ratio
        self.trained_pool_len = self.trained_pool_H * self.trained_pool_W * self.trained_pool_D

        
        self.temperature = nn.Parameter(
            torch.log((torch.ones(num_heads, 1, 1) / 0.24).exp() - 1))

        self.sr = nn.Conv3d(hidden_size, hidden_size, kernel_size=1, stride=1, padding=0)
        self.norm = nn.LayerNorm(hidden_size)
        self.act = nn.GELU()

        # MLP to generate continuous relative position bias
        self.cpb_fc1 = nn.Linear(3, 512, bias=True)  # Changed to 3 for 3D
        self.cpb_act = nn.ReLU(inplace=True)
        self.cpb_fc2 = nn.Linear(512, num_heads, bias=True)

        self.local_len = window_size ** 3 
        # relative_bias_local:
        self.relative_pos_bias_local = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(num_heads, self.local_len), mean=0,
                                  std=0.0004))

        # dynamic_local_bias:
        self.learnable_tokens = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(num_heads, self.head_dim, self.local_len), mean=0, std=0.02))
        self.learnable_bias = nn.Parameter(torch.zeros(num_heads, 1, self.local_len))
        
        self.query_embedding = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(self.num_heads, 1, self.head_dim), mean=0, std=0.02))
        self.kv = nn.Linear(hidden_size, hidden_size * 2, bias=qkv_bias)

    def forward(self, x, H, W, D, relative_pos_index, relative_coords_table, seq_length_scale, attn_mask):
        B, N, C = x.shape

        qkvv = self.qkvv(x).reshape(B, N, 4, self.num_heads, C // self.num_heads)
        qkvv = qkvv.permute(2, 0, 3, 1, 4)
        q_shared, k_shared, v_CA, v_SA = qkvv[0], qkvv[1], qkvv[2], qkvv[3]
        
        ### Spatial Branch
        pool_H, pool_W, pool_D = H // self.sr_ratio, W // self.sr_ratio, D // self.sr_ratio
        pool_len = pool_H * pool_W * pool_D

        q_norm = F.normalize(q_shared, dim=-1)
        q_norm_scaled = (q_norm + self.query_embedding) * F.softplus(self.temperature) * seq_length_scale

        k_local = k_shared.permute(0, 1, 3, 2).reshape(B, C, N)
        v_SA = v_SA.permute(0, 1, 3, 2).reshape(B, C, N)

        k_local = unfold_3d(k_local.view(B, C, D, H, W), self.window_size).reshape(B, self.num_heads, C // self.num_heads, N, self.window_size**3).permute(0, 1, 3, 2, 4)
        v_local = unfold_3d(v_SA.view(B, C, D, H, W), self.window_size).reshape(B, self.num_heads, C // self.num_heads, N, self.window_size**3).permute(0, 1, 3, 2, 4)

        # Compute local similarity
        attn_local = ((q_norm_scaled.unsqueeze(-2) @ k_local).squeeze(-2) \
                            + self.relative_pos_bias_local.unsqueeze(1)).masked_fill(attn_mask, float('-inf'))
        
        # Pooled attn
        x_ = x.permute(0, 2, 1).reshape(B, -1, H, W, D).contiguous()
        x_ = F.adaptive_avg_pool3d(self.act(self.sr(x_)), (pool_H, pool_W, pool_D)).reshape(B, -1, pool_len).permute(0, 2, 1)
        x_ = self.norm(x_)

        kv_pool = self.kv(x_).reshape(B, pool_len, 2 * self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k_pool, v_pool = kv_pool.chunk(2, dim=1)

        pool_bias = self.cpb_fc2(self.cpb_act(self.cpb_fc1(relative_coords_table))).transpose(0, 1)[:, relative_pos_index.view(-1)].view(-1, self.trained_len, self.trained_pool_len)
        attn_pool = q_norm_scaled @ F.normalize(k_pool, dim=-1).transpose(-2, -1) + pool_bias

        

        attn = torch.cat([attn_local, attn_pool], dim=-1).softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn_local, attn_pool = torch.split(attn, [self.local_len, pool_len], dim=-1)
        attn_local = (q_norm @ self.learnable_tokens) + self.learnable_bias + attn_local


        x_local = ((attn_local).unsqueeze(
            -2) @ v_local.transpose(-2, -1)).squeeze(-2)
        
        x_pool = attn_pool @ v_pool

        x_SA = (x_local + x_pool).transpose(1, 2).reshape(B, N, C)
        
        
        
        
        
        
        ### Channel Branch
        q_shared = q_shared.transpose(-2, -1)
        k_shared = k_shared.transpose(-2, -1)
        v_CA = v_CA.transpose(-2, -1)
        
        q_shared = F.normalize(q_shared, dim=-1)
        k_shared = F.normalize(k_shared, dim=-1)
        
        attn_CA = (q_shared @ k_shared.transpose(-2, -1)) * self.temperature2
        attn_CA = attn_CA.softmax(dim=-1)
        attn_CA = self.attn_drop(attn_CA)
        x_CA = (attn_CA @ v_CA).permute(0, 3, 1, 2).reshape(B, N, C)

        # Concat fusion
        x_SA = self.out_proj(x_SA)
        x_CA = self.out_proj2(x_CA)
        x = torch.cat((x_SA, x_CA), dim=-1)

        return x
    
    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'query_embedding', 'relative_pos_bias_local', 'cpb', 'temperature', 'temperature2'}
    