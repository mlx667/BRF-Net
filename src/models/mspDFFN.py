import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MS_DFFN(nn.Module):
    """
    Single-scale FFT-FFN (fixed patch-wise FFT modulation) — compatible replacement.
    (B, C, H, W) -> (B, C, H, W)
    """
    def __init__(self, dim: int, ffn_expansion_factor: float = 3.0, bias: bool = False, patch_size: int = 8):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)

        self.dim = dim
        self.patch_size = int(patch_size)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(
            hidden_features * 2, hidden_features * 2,
            kernel_size=3, stride=1, padding=1,
            groups=hidden_features * 2, bias=bias
        )

        P = self.patch_size
        self.fft = nn.Parameter(torch.ones((dim, 1, 1, P, P // 2 + 1)))

        # 1x1 project back
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)                      # (B, 2Hid, H, W)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)     # each (B, Hid, H, W)
        x = F.gelu(x1) * x2                         # gated
        x = self.project_out(x)                     # (B, C, H, W)

        b, c, h, w = x.shape
        P = self.patch_size

        h_pad = (P - h % P) % P
        w_pad = (P - w % P) % P
        if h_pad != 0 or w_pad != 0:
            x_pad = F.pad(x, (0, w_pad, 0, h_pad), mode="reflect")
        else:
            x_pad = x

        x_patch = rearrange(x_pad, "b c (hh p1) (ww p2) -> b c hh ww p1 p2", p1=P, p2=P)

        orig_dtype = x_patch.dtype
        x_fft = torch.fft.rfft2(x_patch.float())
        x_fft = x_fft * self.fft
        x_patch = torch.fft.irfft2(x_fft, s=(P, P)).to(orig_dtype)

        x_rec = rearrange(x_patch, "b c hh ww p1 p2 -> b c (hh p1) (ww p2)", p1=P, p2=P)

        x_rec = x_rec[:, :, :h, :w]
        return x_rec


class MS_DFFN_Seq(nn.Module):
    """
    (B, L, C) -> (B, L, C) wrapper
    """
    def __init__(self, dim: int, input_resolution, ffn_expansion_factor: float = 3.0,
                 bias: bool = False, patch_size: int = 8):
        super().__init__()
        self.H, self.W = input_resolution
        self.body = MS_DFFN(
            dim=dim,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            patch_size=patch_size,
        )

    def forward(self, x):
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, f"MS_DFFN_Seq: L({L}) != H*W({H*W})"

        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        x = self.body(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x



