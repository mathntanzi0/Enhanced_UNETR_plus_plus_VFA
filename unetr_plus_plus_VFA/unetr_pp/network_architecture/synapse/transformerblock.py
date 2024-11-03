import torch.nn as nn
import torch
from unetr_pp.network_architecture.dynunet_block import UnetResBlock
from functools import partial
from timm.models.layers import DropPath
import math
import torch.nn.functional as F
import importlib.util

CUDA_NUM_THREADS = 64

so_file_path = '/mnt/lustre/users/sntanzi/swattention_extension_3d/swattention_compiled.so'

# Load the compiled extension
spec = importlib.util.spec_from_file_location('swattention', so_file_path)
swattention = importlib.util.module_from_spec(spec)
spec.loader.exec_module(swattention)

class sw_qkrpb_cuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, rpb, height, width, depth, kernel_size):
        # Call the CUDA kernel function with the depth parameter
        attn_weight = swattention.qk_rpb_forward(query, key, rpb, height, width, depth, kernel_size, CUDA_NUM_THREADS)
        
        # Save tensors and parameters for backward pass
        ctx.save_for_backward(query, key)
        ctx.height, ctx.width, ctx.depth, ctx.kernel_size = height, width, depth, kernel_size

        return attn_weight

    @staticmethod
    def backward(ctx, d_attn_weight):
        query, key = ctx.saved_tensors
        height, width, depth, kernel_size = ctx.height, ctx.width, ctx.depth, ctx.kernel_size

        # Call the CUDA kernel function for backward pass with depth parameter
        d_query, d_key, d_rpb = swattention.qk_rpb_backward(d_attn_weight.contiguous(), query, key, height, width, depth,
                                                            kernel_size, CUDA_NUM_THREADS)

        return d_query, d_key, d_rpb, None, None, None, None


class sw_av_cuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, attn_weight, value, height, width, depth, kernel_size):
        # Call the CUDA kernel function with the depth parameter
        output = swattention.av_forward(attn_weight, value, height, width, depth, kernel_size, CUDA_NUM_THREADS)

        # Save tensors and parameters for backward pass
        ctx.save_for_backward(attn_weight, value)
        ctx.height, ctx.width, ctx.depth, ctx.kernel_size = height, width, depth, kernel_size

        return output

    @staticmethod
    def backward(ctx, d_output):
        attn_weight, value = ctx.saved_tensors
        height, width, depth, kernel_size = ctx.height, ctx.width, ctx.depth, ctx.kernel_size

        # Call the CUDA kernel function for backward pass with depth parameter
        d_attn_weight, d_value = swattention.av_backward(d_output.contiguous(), attn_weight, value, height, width, depth,
                                                         kernel_size, CUDA_NUM_THREADS)

        return d_attn_weight, d_value, None, None, None, None



class TransformerBlock(nn.Module):
    """
    A transformer block, based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_heads: int,
            kernel_size = 3,
            dropout_rate: float = 0.0,
            pos_embed=False,
    ) -> None:
        """
        Args:
            input_size: the size of the input for each stage.
            hidden_size: dimension of hidden layer.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.
            pos_embed: bool argument to determine if positional embedding is used.

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            print("Hidden size is ", hidden_size)
            print("Num heads is ", num_heads)
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.norm = nn.LayerNorm(hidden_size)
        self.gamma = nn.Parameter(1e-6 * torch.ones(hidden_size), requires_grad=True)
        self.epa_block = EPA(hidden_size=hidden_size, num_heads=num_heads, channel_attn_drop=dropout_rate,spatial_attn_drop=dropout_rate, kernel_size=kernel_size)
        self.conv51 = UnetResBlock(3, hidden_size, hidden_size, kernel_size=3, stride=1, norm_name="batch")
        self.conv8 = nn.Sequential(nn.Dropout3d(0.1, False), nn.Conv3d(hidden_size, hidden_size, 1))

        self.pos_embed = None
        if pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, input_size, hidden_size))

    def forward(self, x):
        B, C, H, W, D = x.shape

        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)

        if self.pos_embed is not None:
            x = x + self.pos_embed
        attn = x + self.gamma * self.epa_block(self.norm(x), H, W, D)

        attn_skip = attn.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)  # (B, C, H, W, D)
        attn = self.conv51(attn_skip)
        x = attn_skip + self.conv8(attn)

        return x


class EPA(nn.Module):
    def __init__(self, hidden_size, num_heads=4, qkv_bias=False,
                 channel_attn_drop=0.1, spatial_attn_drop=0.1, kernel_size=3):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.window_size = kernel_size

        # qkvv are 4 linear layers (query_shared, key_shared, value_spatial, value_channel)
        self.qkvv = nn.Linear(hidden_size, hidden_size * 4, bias=qkv_bias)

        self.attn_drop = nn.Dropout(channel_attn_drop)
        self.attn_drop_2 = nn.Dropout(spatial_attn_drop)
        self.relative_pos_bias_local = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(num_heads,  self.window_size ** 3), mean=0,
                                  std=0.0004))
        self.query_embedding = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(self.num_heads, 1, hidden_size // self.num_heads), mean=0, std=0.02))
        
        self.out_proj = nn.Linear(hidden_size, int(hidden_size // 2))
        self.out_proj2 = nn.Linear(hidden_size, int(hidden_size // 2))

    def forward(self, x, H, W, D):
        B, N, C = x.shape

        qkvv = self.qkvv(x).reshape(B, N, 4, self.num_heads, C // self.num_heads)
        qkvv = qkvv.permute(2, 0, 3, 1, 4)
        q_shared, k_shared, v_CA, v_SA = qkvv[0], qkvv[1], qkvv[2], qkvv[3]

        q_norm = F.normalize(q_shared, dim=-1) 
        q_norm_scaled = (q_norm + self.query_embedding) * F.softplus(self.temperature2)

        attn_SA = sw_qkrpb_cuda.apply(q_norm_scaled.contiguous(), F.normalize(k_shared, dim=-1).contiguous(), self.relative_pos_bias_local, H, W, D, self.window_size)
        q_shared = q_shared.transpose(-2, -1)
        k_shared = k_shared.transpose(-2, -1)
        v_CA = v_CA.transpose(-2, -1)
        
        q_shared = F.normalize(q_shared, dim=-1)
        k_shared = F.normalize(k_shared, dim=-1)
        
        attn_CA = (q_shared @ k_shared.transpose(-2, -1)) * self.temperature
        attn_CA = attn_CA.softmax(dim=-1)
        attn_CA = self.attn_drop(attn_CA)
        x_CA = (attn_CA @ v_CA).permute(0, 3, 1, 2).reshape(B, N, C)
        
        attn_SA = attn_SA.softmax(dim=-1)
        attn_SA = self.attn_drop_2(attn_SA)

        x_SA = sw_av_cuda.apply(attn_SA.type_as(v_SA), v_SA.contiguous(), H, W, D, self.window_size).reshape(B, N, C)
        
        # Concat fusion
        x_SA = self.out_proj(x_SA)
        x_CA = self.out_proj2(x_CA)
        x = torch.cat((x_SA, x_CA), dim=-1)

        return x


    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature', 'temperature2'}
