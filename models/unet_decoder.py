"""轻量级 U-Net 风格解码器。

用途：
- 将 bottleneck 特征解码为 mask logits（padded 空间尺寸）。
- 可选融合来自 CNN 编码器的 skip 特征。
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


def _pick_group_count(channels: int, prefer: int = 8) -> int:
    g = min(prefer, channels)
    while g > 1 and channels % g != 0:
        g -= 1
    return g


class ConvNormAct(nn.Module):
    """Conv2d -> GroupNorm -> GELU。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels/out_channels must be > 0.")
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.norm = nn.GroupNorm(
            num_groups=_pick_group_count(out_channels),
            num_channels=out_channels,
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ConvBlock(nn.Module):
    """两个 ConvNormAct 组成的卷积块。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            ConvNormAct(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """上采样块：转置卷积 2x 上采样 -> (可选)与 skip 拼接 -> ConvBlock。"""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        if in_channels <= 0 or skip_channels < 0 or out_channels <= 0:
            raise ValueError("Invalid channels for UpBlock.")
        self.in_channels = in_channels
        self.skip_channels = skip_channels
        self.out_channels = out_channels
        self.up = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=2,
            stride=2,
            padding=0,
            bias=False,
        )
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 1) 先做转置卷积 2x 上采样
        x = self.up(x)

        # 2) 若有 skip，先对齐空间尺寸再拼接
        if skip is not None:
            if skip.ndim != 4:
                raise ValueError(f"skip must be 4D, got {tuple(skip.shape)}")
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)

        return self.conv(x)


class UNetDecoder(nn.Module):
    """四级轻量级 U-Net 解码器。"""

    def __init__(self, out_channels: int = 2, use_skips: bool = True) -> None:
        super().__init__()
        self.use_skips = use_skips

        # use_skips=True 时的通道规划：
        # x:[256] + s4:[256] -> up4 out 256
        # 256 + s3:[128] -> up3 out 128
        # 128 + s2:[64]  -> up2 out 64
        # 64  + s1:[32]  -> up1 out 32
        #
        # use_skips=False 时 skip_channels 视为 0（forward 中不传 skip）
        s4_c = 256 if use_skips else 0
        s3_c = 128 if use_skips else 0
        s2_c = 64 if use_skips else 0
        s1_c = 32 if use_skips else 0

        self.up4 = UpBlock(in_channels=256, skip_channels=s4_c, out_channels=256)
        self.up3 = UpBlock(in_channels=256, skip_channels=s3_c, out_channels=128)
        self.up2 = UpBlock(in_channels=128, skip_channels=s2_c, out_channels=64)
        self.up1 = UpBlock(in_channels=64, skip_channels=s1_c, out_channels=32)
        self.out_head = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        skips: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """前向传播。

        Args:
            x: bottleneck 特征，典型 shape `[B,256,32,16]`。
            skips: 可选 `(s1,s2,s3,s4)`，典型：
                - s1: `[B,32,512,256]`
                - s2: `[B,64,256,128]`
                - s3: `[B,128,128,64]`
                - s4: `[B,256,64,32]`

        Returns:
            torch.Tensor: `mask_logits_pad`，典型 shape `[B,2,512,256]`。
        """
        if x.ndim != 4:
            raise ValueError(f"x must be 4D [B,C,H,W], got {tuple(x.shape)}")

        if self.use_skips:
            if skips is None:
                raise ValueError("use_skips=True requires skips=(s1,s2,s3,s4).")
            if len(skips) != 4:
                raise ValueError(f"skips must have 4 tensors, got {len(skips)}")
            s1, s2, s3, s4 = skips
            y = self.up4(x, s4)
            y = self.up3(y, s3)
            y = self.up2(y, s2)
            y = self.up1(y, s1)
        else:
            y = self.up4(x, None)
            y = self.up3(y, None)
            y = self.up2(y, None)
            y = self.up1(y, None)

        return self.out_head(y)

