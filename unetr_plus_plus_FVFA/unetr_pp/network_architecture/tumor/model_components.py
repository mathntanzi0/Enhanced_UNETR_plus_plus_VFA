from torch import nn
import torch
from timm.models.layers import trunc_normal_
from typing import Sequence, Tuple, Union
from monai.networks.layers.utils import get_norm_layer
import torch.nn.functional as F
from monai.utils import optional_import
from unetr_pp.network_architecture.layers import LayerNorm
from unetr_pp.network_architecture.tumor.transformerblock import TransformerBlock
from unetr_pp.network_architecture.dynunet_block import get_conv_layer, UnetResBlock


einops, _ = optional_import("einops")

class UnetrPPEncoder(nn.Module):
    def __init__(self, input_size=[(32, 32, 32), (16, 16, 16), (8, 8, 8), (4, 4, 4)],dims=[32, 64, 128, 256], sr_ratios=[8, 4, 2, 1],
                 proj_size =[64,64,64,32], depths=[3, 3, 3, 3],  num_heads=4, spatial_dims=3, in_channels=4,
                 dropout=0.0, transformer_dropout_rate=0.1 ,**kwargs):
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem_layer = nn.Sequential(
            get_conv_layer(spatial_dims, in_channels, dims[0], kernel_size=(4, 4, 4), stride=(4, 4, 4),
                           dropout=dropout, conv_only=True, ),
            get_norm_layer(name=("group", {"num_groups": in_channels}), channels=dims[0]),
        )
        self.downsample_layers.append(stem_layer)
        for i in range(3):
            downsample_layer = nn.Sequential(
                get_conv_layer(spatial_dims, dims[i], dims[i + 1], kernel_size=(2, 2, 2), stride=(2, 2, 2),
                               dropout=dropout, conv_only=True, ),
                get_norm_layer(name=("group", {"num_groups": dims[i]}), channels=dims[i + 1]),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple Transformer blocks
        for i in range(4):
            H, W, D = input_size[i][2], input_size[i][1], input_size[i][0]
            sr_ratio = sr_ratios[i]
            relative_pos_index, relative_coords_table = get_relative_position_cpb(
            query_size=(H, W, D),
            key_size=(H // sr_ratio, W // sr_ratio, D // sr_ratio),
            pretrain_size=(H, W, D))

            self.register_buffer(f"relative_pos_index{i + 1}", relative_pos_index, persistent=False)
            self.register_buffer(f"relative_coords_table{i + 1}", relative_coords_table, persistent=False)
            
            stage_blocks = []
            for j in range(depths[i]):
                stage_blocks.append(TransformerBlock(input_resolution=input_size[i], hidden_size=dims[i],
                                                     num_heads=num_heads, dropout_rate=transformer_dropout_rate, sr_ratio=sr_ratios[i]))
            self.stages.append(nn.Sequential(*stage_blocks))
        self.hidden_states = []
        self.apply(self._init_weights)
        self.num_stages = 4
        self.input_size = input_size
        self.sr_ratios = sr_ratios

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {}

    def forward_features(self, x):
        hidden_states = []

        x = self.downsample_layers[0](x)

        H, W, D = self.input_size[0][2], self.input_size[0][1], self.input_size[0][0]
        sr_ratio = self.sr_ratios[0]

        relative_pos_index = getattr(self, f"relative_pos_index{0 + 1}")
        relative_coords_table = getattr(self, f"relative_coords_table{0 + 1}")

        with torch.no_grad():
            local_seq_length = get_seqlen_scale((D, H, W), 3, device=x.device)
            seq_length_scale = torch.log(local_seq_length + (D // sr_ratio) * (H // sr_ratio) * (W // sr_ratio))

        for blk in self.stages[0]:
            x = blk(x, relative_pos_index, relative_coords_table, seq_length_scale)

        hidden_states.append(x)

        for i in range(1, 4):
            x = self.downsample_layers[i](x)

            H, W, D = self.input_size[i][2], self.input_size[i][1], self.input_size[i][0]
            sr_ratio = self.sr_ratios[i]
            relative_pos_index = getattr(self, f"relative_pos_index{i + 1}")
            relative_coords_table = getattr(self, f"relative_coords_table{i + 1}")

            with torch.no_grad():
                if i != (self.num_stages - 1):
                    local_seq_length = get_seqlen_scale((D, H, W), 3, device=x.device)
                    seq_length_scale = torch.log(local_seq_length + (D // sr_ratio) * (H // sr_ratio) * (W // sr_ratio))
                else:
                    seq_length_scale = torch.log(torch.as_tensor((D // sr_ratio) * (H // sr_ratio) * (W // sr_ratio), device=x.device))
            for blk in self.stages[i]:
                x = blk(x, relative_pos_index, relative_coords_table, seq_length_scale)

            if i == 3:  # Reshape the output of the last stage
                x = einops.rearrange(x, "b c h w d -> b (h w d) c")
            hidden_states.append(x)
        return x, hidden_states

    def forward(self, x):
        x, hidden_states = self.forward_features(x)
        return x, hidden_states


class UnetrUpBlock(nn.Module):
    def     __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[Sequence[int], int],
            upsample_kernel_size: Union[Sequence[int], int],
            norm_name: Union[Tuple, str],
            proj_size: int = 64,
            num_heads: int = 4,
            out_size: int = 0,
            input_resolution=(1, 1, 1),
            depth: int = 3,
            conv_decoder: bool = False,
            sr_ratios=1,
            stage=0,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of heads inside each EPA module.
            out_size: spatial size for each decoder.
            depth: number of blocks for the current decoder stage.
        """

        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True
        )

        # 4 feature resolution stages, each consisting of multiple residual blocks
        self.decoder_block = nn.ModuleList()

        # If this is the last decoder, use ConvBlock(UnetResBlock) instead of EPA_Block
        # (see suppl. material in the paper)
        self.input_resolution = input_resolution
        self.conv_decoder = conv_decoder
        self.sr_ratios = sr_ratios
        self.stage = stage

        if conv_decoder == True:
            self.decoder_block.append(
                UnetResBlock(spatial_dims, out_channels, out_channels, kernel_size=kernel_size, stride=1,
                             norm_name=norm_name, ))
        else:
            stage_blocks = []
            for j in range(depth):
                relative_pos_index, relative_coords_table = get_relative_position_cpb(
                query_size=input_resolution,
                key_size=(input_resolution[2] // sr_ratios, input_resolution[1] // sr_ratios, input_resolution[0] // sr_ratios),
                pretrain_size=input_resolution)
                self.register_buffer(f"relative_pos_index{stage + 1}", relative_pos_index, persistent=False)
                self.register_buffer(f"relative_coords_table{stage + 1}", relative_coords_table, persistent=False)

                stage_blocks.append(TransformerBlock(input_resolution=input_resolution, hidden_size= out_channels, 
                                                     num_heads=num_heads, dropout_rate=0.1, sr_ratio=sr_ratios))
            self.decoder_block.append(nn.Sequential(*stage_blocks))

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    

    def forward(self, inp, skip):

        out = self.transp_conv(inp)
        out = out + skip
        sr_ratio =  self.sr_ratios
        H, W, D = self.input_resolution[2], self.input_resolution[1], self.input_resolution[0]
        
        if self.conv_decoder == True:
            out = self.decoder_block[0](out)
        else:
            relative_pos_index = getattr(self, f"relative_pos_index{self.stage + 1}")
            relative_coords_table = getattr(self, f"relative_coords_table{self.stage + 1}")
            with torch.no_grad():
                if self.stage != 3:
                    local_seq_length = get_seqlen_scale((D, H, W), 3, device=inp.device)
                    seq_length_scale = torch.log(local_seq_length + (D // sr_ratio) * (H // sr_ratio) * (W // sr_ratio))
                else:
                    seq_length_scale = torch.log(torch.as_tensor((D // sr_ratio) * (H // sr_ratio) * (W // sr_ratio), device=inp.device))

            for blk in self.decoder_block[0]:
                out = blk(out, relative_pos_index, relative_coords_table, seq_length_scale)

        return out


@torch.no_grad()
def get_seqlen_scale(input_resolution, window_size, device):
    """
    Calculate the sequence length scale for 3D input using average pooling.

    Args:
    - input_resolution (tuple): A tuple of 3 integers representing the depth, height, and width.
    - window_size (int): An integer representing the window size for depth, height, and width.
    - device (torch.device): The device on which to perform the computation.

    Returns:
    - torch.Tensor: A tensor containing the sequence length scales.
    """
    # Create a 3D tensor of ones with the given input resolution
    ones_tensor = torch.ones(1, 1, input_resolution[0], input_resolution[1], input_resolution[2], device=device) * (window_size ** 3)
    
    # Perform 3D average pooling
    pooled_tensor = torch.nn.functional.avg_pool3d(
        ones_tensor,
        kernel_size=(window_size, window_size, window_size),
        stride=1,
        padding=window_size // 2
    )

    # Reshape to a single-column tensor
    return pooled_tensor.reshape(-1, 1)



@torch.no_grad()
def get_relative_position_cpb(query_size, key_size, pretrain_size=None,
                              device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    pretrain_size = pretrain_size or query_size
    
    # Query and key axis for height (H)
    axis_qh = torch.arange(query_size[0], dtype=torch.float32, device=device)
    axis_kh = F.adaptive_avg_pool1d(axis_qh.unsqueeze(0), key_size[0]).squeeze(0)
    
    # Query and key axis for width (W)
    axis_qw = torch.arange(query_size[1], dtype=torch.float32, device=device)
    axis_kw = F.adaptive_avg_pool1d(axis_qw.unsqueeze(0), key_size[1]).squeeze(0)
    
    # Query and key axis for depth (D)
    axis_qd = torch.arange(query_size[2], dtype=torch.float32, device=device)
    axis_kd = F.adaptive_avg_pool1d(axis_qd.unsqueeze(0), key_size[2]).squeeze(0)
    
    # Create mesh grids
    axis_kh, axis_kw, axis_kd = torch.meshgrid(axis_kh, axis_kw, axis_kd)
    axis_qh, axis_qw, axis_qd = torch.meshgrid(axis_qh, axis_qw, axis_qd)
    
    # Reshape for calculations
    axis_kh = torch.reshape(axis_kh, [-1])
    axis_kw = torch.reshape(axis_kw, [-1])
    axis_kd = torch.reshape(axis_kd, [-1])
    axis_qh = torch.reshape(axis_qh, [-1])
    axis_qw = torch.reshape(axis_qw, [-1])
    axis_qd = torch.reshape(axis_qd, [-1])
    
    # Compute relative positions
    relative_h = (axis_qh[:, None] - axis_kh[None, :]) / (pretrain_size[0] - 1) * 8
    relative_w = (axis_qw[:, None] - axis_kw[None, :]) / (pretrain_size[1] - 1) * 8
    relative_d = (axis_qd[:, None] - axis_kd[None, :]) / (pretrain_size[2] - 1) * 8
    
    # Combine into relative positions
    relative_hw = torch.stack([relative_h, relative_w, relative_d], dim=-1).view(-1, 3)
    
    # Get unique relative coordinates and their indices
    relative_coords_table, idx_map = torch.unique(relative_hw, return_inverse=True, dim=0)
    
    # Normalize the coordinates
    relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
        torch.abs(relative_coords_table) + 1.0) / torch.log2(torch.tensor(8, dtype=torch.float32))
    
    return idx_map, relative_coords_table