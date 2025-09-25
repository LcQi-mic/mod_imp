import itertools
from typing import Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

from monai.networks.blocks import MLPBlock as Mlp
from monai.networks.blocks import PatchEmbed, UnetrBasicBlock, UnetrUpBlock
from monai.networks.layers import DropPath, trunc_normal_
from monai.utils import ensure_tuple_rep, look_up_option, optional_import

rearrange, _ = optional_import("einops", name="rearrange")


class ImputationModel(nn.Module):
    def __init__(
        self,
        img_size: Union[Sequence[int], int],
        patch_size: int = 4,
        embed_dim: int = 24,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        norm_name: Union[Tuple, str] = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        downsample="merging",
        con_dim: int = 512,
        mask_ratio_spa: float = 0.5,
    ) -> None:
        """
        Args:
            img_size: dimension of input image.
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            embed_dim: dimension of network feature size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            norm_name: feature normalization type and arguments.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            dropout_path_rate: drop path rate.
            normalize: normalize output intermediate features in each stage.
            downsample: module used for downsampling, available options are `"mergingv2"`, `"merging"` and a
                user-specified `nn.Module` following the API defined in :py:class:`monai.networks.nets.PatchMerging`.
                The default is currently `"merging"` (the original version defined in v0.9.0).
            phase: pretrain or finetune
        """

        super().__init__()        
        img_size = ensure_tuple_rep(img_size, 2)
        self.patch_size = ensure_tuple_rep(patch_size, 2)
        window_size = ensure_tuple_rep(7, 2)
        
        self.mask_ratio_spa = mask_ratio_spa

        if not (0 <= drop_rate <= 1):
            raise ValueError("dropout rate should be between 0 and 1.")

        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("attention dropout rate should be between 0 and 1.")

        if not (0 <= dropout_path_rate <= 1):
            raise ValueError("drop path rate should be between 0 and 1.")

        if embed_dim % 12 != 0:
            raise ValueError("embed_dim should be divisible by 12.")

        self.normalize = normalize
        
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_chans=1,
            embed_dim=embed_dim,
            norm_layer=nn.LayerNorm,
            spatial_dims=2
        )
        
        # SequentialEncoder
        self.SequentialEncoder = Transformer(
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads
        )
        
        # SpatialEncoder
        self.SpatialEncoder = SwinTransformer(
            embed_dim=embed_dim,
            window_size=window_size,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=nn.LayerNorm,
            downsample=look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample
        )

        # Decoder
        self.decoder5 = UnetrUpBlock(
            in_channels=16 * embed_dim,
            out_channels=8 * embed_dim,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
            spatial_dims=2
        )

        self.decoder4 = UnetrUpBlock(
            in_channels=embed_dim * 8,
            out_channels=embed_dim * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
            spatial_dims=2
        )

        self.decoder3 = UnetrUpBlock(
            in_channels=embed_dim * 4,
            out_channels=embed_dim * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
            spatial_dims=2
        )
        
        self.decoder2 = UnetrUpBlock(
            in_channels=embed_dim * 2,
            out_channels=embed_dim,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
            spatial_dims=2
        )

        self.to_mri = nn.Sequential(
            nn.ConvTranspose2d(
                    in_channels=embed_dim,
                    out_channels=embed_dim,
                    kernel_size=4,
                    stride=2,
                    padding=1
                ),
            nn.InstanceNorm2d(embed_dim),
            nn.LeakyReLU(),
            nn.Conv2d(embed_dim, 4, 3, 1, 1),
            nn.InstanceNorm2d(4)
        )
        
        # FusionBlock for f
        self.f_projector = nn.ModuleList()
        for i in range(len(depths) + 1):
            self.f_projector.append(FusionBlock(2**i * embed_dim))
        
        # Projector for f_seq and f_spa
        self.projector_seq = ContrastHead(16 * embed_dim, con_dim)
        self.projector_spa = ContrastHead(16 * embed_dim, con_dim)
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        
    def spa_masking(self, x):
        b, m, n, d = x.size()

        len_keep = int(n * (1 - self.mask_ratio_spa))
        noise = torch.rand(b, n).to(x.device)  # noise in [0, 1]
        mask = torch.ones([b, n]).to(x.device) 

        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=2, index=ids_keep.unsqueeze(-1).unsqueeze(1).repeat(1, m, 1, d))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([b, n], device=x.device)
        mask[:, :len_keep] = 0

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        mask_tokens = self.mask_token.repeat(b, m, ids_restore.shape[1] - x_masked.shape[2], 1)

        x = torch.cat([x_masked, mask_tokens], dim=2)  
        x = torch.gather(x, dim=2, index=ids_restore.unsqueeze(-1).unsqueeze(1).repeat(1, m, 1, x_masked.shape[-1]))  # unshuffle
        return x
    
    def seq_masking(self, x, mask_index):
        b, m, n, d = x.size()

        mask = torch.tensor(mask_index).unsqueeze(0).to(x.device)
        len_keep = int(sum(mask_index))

        # sort noise for each sample
        ids_shuffle = torch.argsort(mask, dim=1, descending=True)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]

        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(b, 1, n, d))
        mask_tokens = self.mask_token.repeat(b, ids_restore.shape[1] - x_masked.shape[1], n, 1)

        x = torch.cat([x_masked, mask_tokens], dim=1)  
        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).unsqueeze(-1).repeat(b, 1, n, x_masked.shape[-1]))  # unshuffle
        return x
    
    def masking(self, x, mask_index):
        n, m, c, h, w = x.size()
        x = rearrange(x, "n m c h w -> n m (h w) c")
        
        spa = self.spa_masking(x)
        seq = self.seq_masking(x, mask_index)

        spa = rearrange(spa, "n m (h w) c -> n m h w c", h=h, w=w)
        seq = rearrange(seq, "n m (h w) c -> n m h w c", h=h, w=w)
        
        spa = rearrange(spa, "n m h w c -> n m c h w")
        seq = rearrange(seq, "n m h w c -> n m c h w")
        return spa, seq
    
    def forward(self, x, mask_index=[1, 1, 0, 0]):
        x = torch.unbind(x, dim=1)
        
        tmp = []
        for i in range(len(mask_index)):
            tmp.append(self.patch_embed(x[i].unsqueeze(1)))
            
        embedding = torch.stack(tmp, dim=1)

        spa, seq = self.masking(embedding, mask_index)

        spa_hidden_states_out = self.SpatialEncoder(spa, self.normalize)
        seq_hidden_states_out = self.SequentialEncoder(seq, self.normalize)

        hidden_states_out = []
        for i, j, blk in zip(spa_hidden_states_out, seq_hidden_states_out, self.f_projector):
            hidden_states_out.append(blk(i, j))
            
        # f_5 for contrast loss: [b, con_dim] * 4
        f_5_seq = self.projector_seq(seq_hidden_states_out[-1])
        f_5_spa = self.projector_spa(spa_hidden_states_out[-1])
        
        f_bottle_neck = hidden_states_out[4]

        # [b, c, (d), h, w]
        dec4 = self.decoder5(f_bottle_neck, hidden_states_out[3])
        dec3 = self.decoder4(dec4, hidden_states_out[2])
        dec2 = self.decoder3(dec3, hidden_states_out[1])
        dec1 = self.decoder2(dec2, hidden_states_out[0])
        
        logits = self.to_mri(dec1)
        
        return logits, f_5_seq, f_5_spa

class SpaAttention(nn.Module):
    def __init__(self,
                 embed_dim) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(embed_dim, 1, 3, 1, 1)
        self.conv2 = nn.Conv2d(embed_dim, 1, 5, 1, 2)
        self.act = nn.LeakyReLU()
        
    def forward(self, x):
        att1 = self.conv1(x)
        att2 = self.conv2(x)
        att = self.act(att1 + att2)
        return F.layer_norm(att * x, (att * x).size()[1:])
    

class ChaAttention(nn.Module):
    def __init__(self, 
                 embed_dim,
                 ratio,
                 ) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.squeeze = nn.Linear(embed_dim, int(embed_dim * ratio))
        self.excite = nn.Linear(int(embed_dim * ratio), embed_dim)
        self.act = nn.LeakyReLU()
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        att = self.pool(x)
        att = att.view(x.size()[0], x.size()[1])

        att = self.squeeze(att)
        att = self.excite(att)
        att = self.act(att)

        att = att.unsqueeze(-1).unsqueeze(-1)

        return F.layer_norm(att * x, (att * x).size()[1:])
      
      
class FusionBlock(nn.Module):
    def __init__(self,
                embed_dim,
                ratio=1.0,
        ) -> None:
        super().__init__()
        
        self.spa_att = SpaAttention(
            embed_dim,
        )
        
        self.cha_att = ChaAttention(
            embed_dim,
            ratio,
        )
        
        self.conv = nn.Conv2d(embed_dim * 4, embed_dim, kernel_size=1)
        self.act = nn.ReLU()
        
    def forward(self, spa, seq):
        x = spa + seq
        x = torch.cat(torch.unbind(x, dim=1), dim=1)
        
        x = self.conv(x)
        x = self.act(x)
        x = F.layer_norm(x, x.size()[1:])
        
        att = self.cha_att(x)
        att = self.spa_att(att)
        return F.layer_norm(att, att.size()[1:])
    
    
class ContrastHead(nn.Module):
    def __init__(self, dims, con_dims) -> None:
        super().__init__()
        self.dims = dims
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(dims, con_dims)
        self.norm = nn.LayerNorm(con_dims)
        self.lin_act = nn.LeakyReLU()
        
    def forward(self, x):
        # x:[b, m, c, (d), h, w]
        b, m, c, _, _ = x.size()
        x = torch.unbind(x, dim=1)
    
        y = []
        for f in x:
            y.append(self.norm(self.lin_act(self.linear(self.pool(f).view(b, c)))))
        y = torch.stack(y, dim=1)
        return y

######  SwinTransformer
def window_partition(x, window_size):
    """window partition operation based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        x: input tensor.
        window_size: local window size.
    """
    b, h, w, c = x.shape
    x = x.view(b, h // window_size[0], window_size[0], w // window_size[1], window_size[1], c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0] * window_size[1], c)
    return windows

def window_reverse(windows, window_size, dims):
    """window reverse operation based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        windows: windows tensor.
        window_size: local window size.
        dims: dimension values.
    """
    b, h, w = dims
    x = windows.view(b, h // window_size[0], w // window_size[1], window_size[0], window_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x

def get_window_size(x_size, window_size, shift_size=None):
    """Computing window size based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        x_size: input size.
        window_size: local window size.
        shift_size: window shifting size.
    """

    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


class WindowAttention(nn.Module):
    """
    Window based multi-head self attention module with relative position bias based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int],
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            num_heads: number of attention heads.
            window_size: local window size.
            qkv_bias: add a learnable bias to query, key, value.
            attn_drop: attention dropout rate.
            proj_drop: dropout rate of output.
        """

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        mesh_args = torch.meshgrid.__kwdefaults__

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        if mesh_args is not None:
            coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        else:
            coords = torch.stack(torch.meshgrid(coords_h, coords_w))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1

        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.clone()[:n, :n].reshape(-1)
        ].reshape(n, n, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn).to(v.dtype)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int],
        shift_size: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: str = "GELU",
        norm_layer: Type[LayerNorm] = nn.LayerNorm,
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            num_heads: number of attention heads.
            window_size: local window size.
            shift_size: window shift size.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: stochastic depth rate.
            act_layer: activation layer.
            norm_layer: normalization layer.
        """

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(hidden_size=dim, mlp_dim=mlp_hidden_dim, act=act_layer, dropout_rate=drop, dropout_mode="swin")

    def forward_part1(self, x, mask_matrix):
        x = self.norm1(x)

        b, h, w, c = x.shape
        window_size, shift_size = get_window_size((h, w), self.window_size, self.shift_size)
        pad_l = pad_t = 0
        pad_b = (window_size[0] - h % window_size[0]) % window_size[0]
        pad_r = (window_size[1] - w % window_size[1]) % window_size[1]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, hp, wp, _ = x.shape
        dims = [b, hp, wp]

        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        x_windows = window_partition(shifted_x, window_size)

        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, *(window_size + (c,)))
        shifted_x = window_reverse(attn_windows, window_size, dims)
        
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :h, :w, :].contiguous()
        return x

    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x, mask_matrix):
        shortcut = x
        x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)
        x = x + self.forward_part2(x)
        return x


class PatchMergingV2(nn.Module):
    def __init__(self, dim: int, norm_layer: Type[LayerNorm] = nn.LayerNorm) -> None:
        """
        Args:
            dim: number of feature channels.
            norm_layer: normalization layer.
        """

        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        x_shape = x.size()

        b, m, h, w, c = x_shape
        pad_input = (h % 2 == 1) or (w % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))
        x = torch.cat([x[:, :, j::2, i::2, :] for i, j in itertools.product(range(2), range(2))], -1)

        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchMerging(PatchMergingV2):
    def forward(self, x):
        x_shape = x.size()
        if len(x_shape) == 5:
            return super().forward(x)
        if len(x_shape) != 6:
            raise ValueError(f"expecting 5D x, got {x.shape}.")
        b, m, d, h, w, c = x_shape
        pad_input = (h % 2 == 1) or (w % 2 == 1) or (d % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))
        x0 = x[:, :, 0::2, 0::2, 0::2, :]
        x1 = x[:, :, 1::2, 0::2, 0::2, :]
        x2 = x[:, :, 0::2, 1::2, 0::2, :]
        x3 = x[:, :, 0::2, 0::2, 1::2, :]
        x4 = x[:, :, 1::2, 0::2, 1::2, :]
        x5 = x[:, :, 0::2, 1::2, 0::2, :]
        x6 = x[:, :, 0::2, 0::2, 1::2, :]
        x7 = x[:, :, 1::2, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)
        x = self.norm(x)
        x = self.reduction(x)
        return x


MERGING_MODE = {"merging": PatchMerging, "mergingv2": PatchMergingV2}

def compute_mask(dims, window_size, shift_size, device):
    """Computing region masks based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        dims: dimension values.
        window_size: local window size.
        shift_size: shift size.
        device: device.
    """
    cnt = 0

    h, w = dims
    img_mask = torch.zeros((1, h, w, 1), device=device)
    for h in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
        for w in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
            img_mask[:, h, w, :] = cnt
            cnt += 1

    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    return attn_mask

class BasicLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: Sequence[int],
        drop_path: list,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        norm_layer: Type[LayerNorm] = nn.LayerNorm,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            depth: number of layers in each stage.
            num_heads: number of attention heads.
            window_size: local window size.
            drop_path: stochastic depth rate.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            norm_layer: normalization layer.
            downsample: an optional downsampling layer at the end of the layer.
        """

        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.no_shift = tuple(0 for i in window_size)
        self.depth = depth
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=self.window_size,
                    shift_size=self.no_shift if (i % 2 == 0) else self.shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.downsample = downsample
        if callable(self.downsample):
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)

    def merge_bm(self, x):
        x = rearrange(x, "b m c h w -> (b m) c h w")
        return x
    
    def split_bm(self, x, b, m):
        x = rearrange(x, "(b m) c h w -> b m c h w", b = b, m = m)
        return x
    
    def forward(self, x):
        x_shape = x.size()

        b, m, c, h, w = x_shape
        x = self.merge_bm(x)
        window_size, shift_size = get_window_size((h, w), self.window_size, self.shift_size)
        x = rearrange(x, "b c h w -> b h w c")
        hp = int(np.ceil(h / window_size[0])) * window_size[0]
        wp = int(np.ceil(w / window_size[1])) * window_size[1]
        attn_mask = compute_mask([hp, wp], window_size, shift_size, x.device)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = x.view(b * m, h, w, -1)
        x = self.split_bm(x, b, m)
        if self.downsample is not None:
            x = self.downsample(x)
        x = rearrange(x, "b m h w c -> b m c h w")
        return x


class SwinTransformer(nn.Module):
    """
    Swin Transformer based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        embed_dim: int,
        window_size: Sequence[int],
        depths: Sequence[int],
        num_heads: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: Type[LayerNorm] = nn.LayerNorm,
        patch_norm: bool = False,
        downsample="merging",
    ) -> None:
        """
        Args:
            in_chans: dimension of input channels.
            embed_dim: number of linear projection output channels.
            window_size: local window size.
            patch_size: patch size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            drop_path_rate: stochastic depth rate.
            norm_layer: normalization layer.
            patch_norm: add normalization after patch embedding.
            downsample: module used for downsampling, available options are `"mergingv2"`, `"merging"` and a
                user-specified `nn.Module` following the API defined in :py:class:`monai.networks.nets.PatchMerging`.
                The default is currently `"merging"` (the original version defined in v0.9.0).
            phase: pretrain or finetune
        """

        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.window_size = window_size

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers1 = nn.ModuleList()
        self.layers2 = nn.ModuleList()
        self.layers3 = nn.ModuleList()
        self.layers4 = nn.ModuleList()
        down_sample_mod = look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=self.window_size,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                downsample=down_sample_mod,
            )
            if i_layer == 0:
                self.layers1.append(layer)
            elif i_layer == 1:
                self.layers2.append(layer)
            elif i_layer == 2:
                self.layers3.append(layer)
            elif i_layer == 3:
                self.layers4.append(layer)
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

    def proj_out(self, x, normalize=False):
        if normalize:
            x_shape = x.size()
            n, m, ch, h, w = x_shape
            x = rearrange(x, "n m c h w -> n m h w c")
            x = F.layer_norm(x, [ch])
            x = rearrange(x, "n m h w c -> n m c h w")
        return x
            
    def forward(self, x, normalize=True):
        # [b m c h w]
        x0_out = self.proj_out(x, normalize)
        x1 = self.layers1[0](x.contiguous())
        x1_out = self.proj_out(x1, normalize)
        
        x2 = self.layers2[0](x1.contiguous())
        x2_out = self.proj_out(x2, normalize)
        
        x3 = self.layers3[0](x2.contiguous())
        x3_out = self.proj_out(x3, normalize)
        
        x4 = self.layers4[0](x3.contiguous())
        x4_out = self.proj_out(x4, normalize)
        return [x0_out, x1_out, x2_out, x3_out, x4_out]


######  Vanilla vision ransformer. Modified based on https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    
    
class Attention(nn.Module):
    def __init__(self, dim, 
                 num_heads = 8, 
                 dropout = 0.):
        super().__init__()
        head_dim = dim //  num_heads

        self.num_heads = num_heads
        self.scale = head_dim ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        ) 

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.num_heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, 
                 dim, 
                 num_heads, 
                 dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
                
        self.attn = Attention(dim, num_heads = num_heads, dropout = dropout)
        self.ff = FeedForward(dim, dim, dropout = dropout)

    def bhwc_2_nld(self, x):
        x_shape = x.size()
        b, h, w, m, c = x_shape
        dims = [b, h, w, m]
        return x.contiguous().view(b * h * w, m, c), dims
        
    def nld_2_bhwc(self, x, dims):
        b, h, w, m = dims
        x = x.contiguous().view(b, h, w, m, -1)
        return x
    
    def forward(self, x):
        x, dims = self.bhwc_2_nld(x)
        x = self.attn(x) + x
        x = self.norm(self.ff(x)) + x
        x = self.nld_2_bhwc(x, dims)
        
        return x


class ViTLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        attn_drop: float = 0.0,
        norm_layer: Type[LayerNorm] = nn.LayerNorm,
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            depth: number of layers in each stage.
            num_heads: number of attention heads.
            window_size: local window size.
            drop_path: stochastic depth rate.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            norm_layer: normalization layer.
            downsample: an optional downsampling layer at the end of the layer.
        """

        super().__init__()                
        self.depth = depth
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim, 
                    num_heads, 
                    attn_drop,
                )
                for i in range(depth)
            ]
        )

        self.downsample = PatchMerging(dim=dim, norm_layer=norm_layer)
    
    def forward(self, x):
        x_shape = x.size()

        b, m, c, h, w = x_shape
        x = rearrange(x, "b m c h w -> b h w m c")
        for blk in self.blocks:
            x = blk(x)
        x = x.view(b, h, w, m, -1)
        x = rearrange(x, "b h w m c -> b m h w c")
        if self.downsample is not None:
            x = self.downsample(x)
        x = rearrange(x, "b m h w c -> b m c h w")
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        depths: Sequence[int],
        num_heads: Sequence[int],

    ) -> None:
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, 4, embed_dim, 1, 1))

        self.num_layers = len(depths)
        self.embed_dim = embed_dim

        self.layers1 = nn.ModuleList()
        self.layers2 = nn.ModuleList()
        self.layers3 = nn.ModuleList()
        self.layers4 = nn.ModuleList()

        for i_layer in range(self.num_layers):
            layer = ViTLayer(
                dim=int(embed_dim * 2**i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer]
            )
            if i_layer == 0:
                self.layers1.append(layer)
            elif i_layer == 1:
                self.layers2.append(layer)
            elif i_layer == 2:
                self.layers3.append(layer)
            elif i_layer == 3:
                self.layers4.append(layer)
                
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

    def proj_out(self, x, normalize=False):
        if normalize:
            x_shape = x.size()
            n, m, ch, h, w = x_shape
            x = rearrange(x, "n m c h w -> n m h w c")
            x = F.layer_norm(x, [ch])
            x = rearrange(x, "n m h w c -> n m c h w")
        return x

    def forward(self, x, normalize=True):
        x0 = x + self.pos_embedding    

        x0_out = self.proj_out(x0, normalize)
        
        x1 = self.layers1[0](x0.contiguous())
        
        x1_out = self.proj_out(x1, normalize)
        x2 = self.layers2[0](x1.contiguous())

        x2_out = self.proj_out(x2, normalize)
        x3 = self.layers3[0](x2.contiguous())
        
        x3_out = self.proj_out(x3, normalize)
        x4 = self.layers4[0](x3.contiguous())
        
        x4_out = self.proj_out(x4, normalize)

        return [x0_out, x1_out, x2_out, x3_out, x4_out]


def test_model():
    model = ImputationModel(
        img_size=(256, 256), 
        patch_size=2
    )
    
    swinvit = SwinTransformer(
        embed_dim=24,
        window_size=[7, 7],
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
    )
    
    vit = Transformer(
        embed_dim=24,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
    )
    
    inps = torch.randn((3, 4, 256, 256))

    out, a, b= model(inps)
    
    print(out.size(), a.size(), b.size())

if __name__ == "__main__":
    test_model()
