"""轻量级 CNN skip 编码器。

用途：
- 为 Transformer + U-Net decoder 提供多尺度 skip 特征。
- 仅处理已补零后的输入 `x_pad`，不在本文件中实现 padding。
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


def _pick_group_count(channels: int, prefer: int = 8) -> int:
    """选择可整除通道数的 GroupNorm 组数。"""
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
    """由两个 ConvNormAct 组成的卷积块。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            ConvNormAct(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """下采样块：stride=2 卷积下采样 + ConvBlock。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.down = ConvNormAct(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.conv = ConvBlock(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        x = self.conv(x)
        return x


class CNNSkipEncoder(nn.Module):
    """CNN skip 编码器。

    输入：
    - x_pad: [B, C, F_pad, T_pad]（`C = in_channels`，当前三通道输入可设为 3）

    输出：
    - s1: [B, 32, F_pad,   T_pad]
    - s2: [B, 64, F_pad/2, T_pad/2]
    - s3: [B, 128, F_pad/4, T_pad/4]
    - s4: [B, 256, F_pad/8, T_pad/8]
    """

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be > 0, got {in_channels}")
        self.in_channels = in_channels

        self.stem = ConvBlock(in_channels, 32)
        self.down1 = DownBlock(32, 64)
        self.down2 = DownBlock(64, 128)
        self.down3 = DownBlock(128, 256)

    def forward(self, x_pad: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if x_pad.ndim != 4:
            raise ValueError(f"Expected x_pad shape [B,C,F_pad,T_pad], got {tuple(x_pad.shape)}")
        if x_pad.shape[1] != self.in_channels:
            raise ValueError(f"Expected input channel={self.in_channels}, got {x_pad.shape[1]}")

        s1 = self.stem(x_pad)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        return s1, s2, s3, s4

