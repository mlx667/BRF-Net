#src/models/basic_module.py
import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch import Tensor
from collections import OrderedDict
import re
import math
import torch.nn.functional as F
from typing import Type, Any, Callable, Union, List, Optional, cast, Tuple
from torch.distributions.uniform import Uniform
from models.mspDFFN import MS_DFFN_Seq


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):

        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]


        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        '''
        B,h,n1,n2 = attn.shape
        W1 = torch.mean(attn, 1, True)
        W = torch.sigmoid(W1)
        attn = attn * W
        '''
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=8,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = DualAttention(
            dim=dim,
            input_resolution=input_resolution,
            num_heads=num_heads,
            window_size=self.window_size,
            shift_size=self.shift_size,
            chsa_head_ratio=0.25,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        if self.shift_size > 0:
            self.register_buffer("attn_mask", self._build_attn_mask(), persistent=False)
        else:
            self.attn_mask = None

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = MS_DFFN_Seq(
            dim=dim,
            input_resolution=input_resolution,
            ffn_expansion_factor=mlp_ratio,
            bias=True,
            patch_size=8,
        )

    def _build_attn_mask(self):
        H, W = self.input_resolution
        ws = self.window_size
        ss = self.shift_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1

        h_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
        w_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, ws)  # (nW, ws, ws, 1)
        mask_windows = mask_windows.view(-1, ws * ws)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer) for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False,
                     dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64, dilation: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(2 * out_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat((x1, x2), dim=1)
        x = self.conv_bn_relu(x)
        return x


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64, dilation: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


def window_unpartition_4d(x, H, W, window_size):
    """
    x: (B_, heads, window_area, dim_head) or (B_, heads, window_area, C)
    return: (B, heads*dim_head, H, W)
    """
    # avoid einops dependency: use reshape+permute
    # x: (B*nH*nW, heads, ws*ws, dim_head)
    B_wn, heads, area, ch = x.shape
    ws = window_size
    nH = H // ws
    nW = W // ws
    B = B_wn // (nH * nW)

    x = x.view(B, nH, nW, heads, ws, ws, ch)          # (B, nH, nW, heads, ws, ws, ch)
    x = x.permute(0, 3, 6, 1, 4, 2, 5).contiguous()   # (B, heads, ch, nH, ws, nW, ws)
    x = x.view(B, heads * ch, H, W)                   # (B, heads*ch, H, W)
    return x


class QKVProjection2D(nn.Module):
    """
    x: (B, C, H, W) -> qkv: (B, num_head, 3*dim_head, H, W)
    """
    def __init__(self, dim, num_head, qkv_bias=True):
        super().__init__()
        assert dim % num_head == 0
        self.dim = dim
        self.num_head = num_head
        self.qkv = nn.Conv2d(dim, 3 * dim, kernel_size=1, bias=qkv_bias)

    def forward(self, x):
        qkv = self.qkv(x)  # (B, 3C, H, W)
        B, C3, H, W = qkv.shape
        qkv = qkv.view(B, self.num_head, C3 // self.num_head, H, W)  # (B, heads, 3*dim_head, H, W)
        return qkv


def get_relative_position_index_2d(win_h, win_w):
    # torch.meshgrid must specify indexing on new torch
    coords = torch.stack(torch.meshgrid(torch.arange(win_h), torch.arange(win_w), indexing='ij'))
    coords_flatten = torch.flatten(coords, 1)  # (2, win_h*win_w)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, A, A)
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (A, A, 2)
    relative_coords[:, :, 0] += win_h - 1
    relative_coords[:, :, 1] += win_w - 1
    relative_coords[:, :, 0] *= 2 * win_w - 1
    return relative_coords.sum(-1)  # (A, A)


class SpatialSelfAttentionWin(nn.Module):
    """
    Spatial window self-attention (cosine attention + relative bias) on qkv 2D.
    """
    def __init__(self, dim_head, num_head, window_size=8, shift=0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim_head = dim_head
        self.num_head = num_head
        self.window_size = window_size
        self.shift = shift
        self.window_area = window_size * window_size

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_head, 1, 1))), requires_grad=True)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_head)
        )
        self.register_buffer("relative_position_index", get_relative_position_index_2d(window_size, window_size))
        trunc_normal_(self.relative_position_bias_table, std=.02)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim_head * num_head, dim_head * num_head, kernel_size=1)
        self.proj_drop = nn.Dropout(proj_drop)

    def _get_rel_pos_bias(self):
        rpbt = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        rpbt = rpbt.view(self.window_area, self.window_area, -1)
        rpbt = rpbt.permute(2, 0, 1).contiguous()
        return rpbt.unsqueeze(0)

    def forward(self, qkv, attn_mask=None):
        # qkv: (B, heads_sp, 3*dim_head, H, W)
        B, Hh, C3, H, W = qkv.shape
        ws = self.window_size

        if self.shift > 0:
            qkv = torch.roll(qkv, shifts=(-self.shift, -self.shift), dims=(-2, -1))

        # partition windows
        qkv = qkv.view(B, Hh, C3, H // ws, ws, W // ws, ws)
        qkv = qkv.permute(0, 3, 5, 1, 4, 6, 2).contiguous()
        qkv = qkv.view(-1, Hh, ws * ws, C3)

        q, k, v = torch.chunk(qkv, 3, dim=-1)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = q @ k.transpose(-2, -1)

        logit_scale = torch.clamp(self.logit_scale, max=math.log(1. / 0.01)).exp()
        attn = attn * logit_scale.unsqueeze(0)

        attn = attn + self._get_rel_pos_bias()

        if attn_mask is not None:
            nW = attn_mask.shape[0]
            attn = attn.view(-1, nW, Hh, self.window_area, self.window_area)
            attn = attn + attn_mask.unsqueeze(0).unsqueeze(2)
            attn = attn.view(-1, Hh, self.window_area, self.window_area)

        attn = self.attn_drop(F.softmax(attn, dim=-1))
        x = attn @ v

        x = window_unpartition_4d(x, H, W, ws)
        x = self.proj_drop(self.proj(x))

        if self.shift > 0:
            x = torch.roll(x, shifts=(self.shift, self.shift), dims=(-2, -1))
        return x


class ChannelSelfAttention2D(nn.Module):
    """
    Channel self-attention on qkv 2D.
    """
    def __init__(self, dim_head, num_head, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim_head = dim_head
        self.num_head = num_head

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_head, 1, 1))), requires_grad=True)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Conv2d(dim_head * num_head, dim_head * num_head, kernel_size=1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, qkv):
        B, Hh, C3, H, W = qkv.shape
        qkv = qkv.view(B, Hh, C3, H * W)
        q, k, v = torch.chunk(qkv, 3, dim=2)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = q @ k.transpose(-2, -1)

        logit_scale = torch.clamp(self.logit_scale, max=math.log(1. / 0.01)).exp()
        attn = attn * logit_scale.unsqueeze(0)

        attn = self.attn_drop(F.softmax(attn, dim=-1))
        x = attn @ v

        x = x.view(B, Hh * self.dim_head, H, W)
        x = self.proj_drop(self.proj(x))
        return x


class CFCA2D(nn.Module):
    def __init__(self, dim, reduction=4):
        super().__init__()
        mid = max(dim // reduction, 8)
        self.sp_to_ch = nn.Sequential(
            nn.Conv2d(dim, mid, 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(mid, dim, 1, bias=True),
        )
        self.ch_to_sp = nn.Sequential(
            nn.Conv2d(dim, mid, 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(mid, dim, 1, bias=True),
        )
        self.gate_sp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, mid, 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(mid, dim, 1, bias=True),
            nn.Sigmoid()
        )
        self.gate_ch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, mid, 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(mid, dim, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, sp, ch):
        sp2ch = self.sp_to_ch(sp) * self.gate_sp(sp)
        ch2sp = self.ch_to_sp(ch) * self.gate_ch(ch)
        return sp2ch, ch2sp


class XFF2D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 2, 1, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, sp_in, ch_in):
        x = self.fuse(torch.cat([sp_in, ch_in], dim=1))
        w = self.spatial_gate(x)
        return x * w + x


class DualAttention(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        chsa_head_ratio=0.25,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.H, self.W = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size

        self.ch_heads = int(num_heads * chsa_head_ratio)
        self.sp_heads = num_heads - self.ch_heads
        dim_head = dim // num_heads

        self.qkv = QKVProjection2D(dim, num_heads, qkv_bias=qkv_bias)

        self.sp_attn = SpatialSelfAttentionWin(
            dim_head=dim_head,
            num_head=self.sp_heads,
            window_size=window_size,
            shift=shift_size,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        ) if self.sp_heads > 0 else None

        self.ch_attn = ChannelSelfAttention2D(
            dim_head=dim_head,
            num_head=self.ch_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        ) if self.ch_heads > 0 else None

        self.csp = self.sp_heads * dim_head
        self.cch = self.ch_heads * dim_head

        self.sp_proj = nn.Conv2d(self.csp, dim, kernel_size=1, bias=False) if self.sp_heads > 0 else None
        self.ch_proj = nn.Conv2d(self.cch, dim, kernel_size=1, bias=False) if self.ch_heads > 0 else None

        self.cfca = CFCA2D(dim=dim, reduction=4)
        self.xff = XFF2D(dim=dim)

        self.register_buffer("attn_mask", self._build_attn_mask() if shift_size > 0 else None, persistent=False)

    def _build_attn_mask(self):
        H, W = self.H, self.W
        ws = self.window_size
        ss = self.shift_size

        img_mask = torch.zeros((1, H, W, 1), device=self.relative_position_index_device())
        h_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
        w_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, ws)
        mask_windows = mask_windows.view(-1, ws * ws)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def relative_position_index_device(self):
        return next(self.parameters()).device

    def forward(self, x):
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W

        x_img = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        qkv = self.qkv(x_img)

        sp = None
        ch = None

        if self.sp_heads > 0:
            qkv_sp = qkv[:, :self.sp_heads]
            sp = self.sp_attn(qkv_sp, attn_mask=self.attn_mask)

        if self.ch_heads > 0:
            qkv_ch = qkv[:, self.sp_heads:]
            ch = self.ch_attn(qkv_ch)

        if sp is None:
            x_attn = self.ch_proj(ch) if self.ch_proj is not None else ch
        elif ch is None:
            x_attn = self.sp_proj(sp) if self.sp_proj is not None else sp
        else:
            sp_full = self.sp_proj(sp)
            ch_full = self.ch_proj(ch)

            sp2ch, ch2sp = self.cfca(sp_full, ch_full)
            ch_in = ch_full + sp2ch
            sp_in = sp_full + ch2sp

            x_attn = self.xff(sp_in, ch_in)

        x_out = x_attn.flatten(2).transpose(1, 2).contiguous()  # (B, L, C)
        return x_out

class SMMMEncoder4K(nn.Module):
    """
    Single-input SMMM (encoder-only), 4 kernel sizes: 1/3/5/7
    x: (B,C,H,W) -> (B,C,H,W)

    Design:
      - pre 1x1 (channel mixing)
      - 4 depthwise branches (k=1/3/5/7) + pointwise
      - softmax gating across 4 branches (spatially varying)
      - dilated conv fusion (dilation=2)
    """
    def __init__(self, channels: int, kernels=(1, 3, 5, 7), dilation: int = 2):
        super().__init__()
        self.channels = channels
        self.kernels = kernels

        # pre pointwise: 激活/混合通道（对齐 SMMM “pointwise conv activate cues” 的精神）
        self.pre = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        branches = []
        for k in kernels:
            pad = k // 2
            branches.append(nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=k, padding=pad, groups=channels, bias=False),  # DWConv
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=1, bias=False),  # PWConv
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            ))
        self.branches = nn.ModuleList(branches)
        self.gate = nn.Conv2d(channels * len(kernels), len(kernels), kernel_size=1, bias=True)
        self.fuse = nn.Sequential(
            nn.Conv2d(
                channels, channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.pre(x)  # (B,C,H,W)

        feats = [br(x) for br in self.branches]       # 4*(B,C,H,W)
        cat = torch.cat(feats, dim=1)                 # (B,4C,H,W)

        w = self.gate(cat)                            # (B,4,H,W)
        w = F.softmax(w, dim=1)

        out = 0.0
        for i, fi in enumerate(feats):
            out = out + fi * w[:, i:i+1]              # broadcast (B,1,H,W) -> (B,C,H,W)

        out = self.fuse(out)
        return out

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=[4], in_chans=3, embed_dim=96, norm_layer=nn.LayerNorm):
        super().__init__()
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

        self.smmm = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        x = self.stem(x)
        x = self.smmm(x)

        x = x.flatten(2).transpose(1, 2).contiguous()  # (B, L, C)
        if self.norm is not None:
            x = self.norm(x)
        return x

class PatchMerging(nn.Module):
    def __init__(self, dim, patch_size=[2, 4], norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.norm = norm_layer(dim)

        self.down = nn.Sequential(
            nn.Conv2d(dim, 2 * dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2 * dim),
            nn.ReLU(inplace=True),
        )

        self.smmm = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(2 * dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        B, L, C = x.shape
        x = self.norm(x)

        H = int(np.sqrt(L))
        W = H
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # (B,C,H,W)

        x = self.down(x)
        x = self.smmm(x)
        return x
