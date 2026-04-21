"""U-Net 分离模型。

输入/输出约定：
- 输入：`[B, 1, F, T]`（混合信号对数幅度谱）
- 输出：`[B, 2, F, T]`（两个源的 mask 概率）
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _pick_group_count(channels: int, prefer: int = 8) -> int:
    """选择可整除通道数的 GroupNorm 组数。"""
    g = min(prefer, channels)
    while g > 1 and channels % g != 0:
        g -= 1
    return g


class DoubleConv(nn.Module):
    """双卷积块：Conv -> GroupNorm -> GELU -> Conv -> GroupNorm -> GELU。

    说明：
    - 两个 3x3 卷积均使用 padding=1，因此不改变空间尺寸。
    - 使用 GroupNorm，避免 BatchNorm 对 batch size 的依赖。
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels/out_channels must be > 0.")

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=_pick_group_count(out_channels), num_channels=out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=_pick_group_count(out_channels), num_channels=out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """下采样块：MaxPool2d(2) + DoubleConv。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    """上采样块：ConvTranspose2d 上采样 + skip concat + DoubleConv。"""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # 对齐策略：若上采样后尺寸与 skip 不一致，使用双线性插值对齐到 skip 尺寸。
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetSeparator(nn.Module):
    """标准 4 层 U-Net 双源分离器。"""

    def __init__(self, in_channels: int = 1, out_masks: int = 2, base_channels: int = 16) -> None:
        super().__init__()
        if in_channels != 1:
            raise ValueError(f"Current project expects in_channels=1, got {in_channels}")
        if out_masks != 2:
            raise ValueError(f"Current project expects out_masks=2, got {out_masks}")
        if base_channels <= 0:
            raise ValueError(f"base_channels must be > 0, got {base_channels}")

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16

        self.in_channels = in_channels
        self.out_masks = out_masks

        # Encoder: 1->c1->c2->c3->c4（c1=base_channels）
        self.enc1 = DoubleConv(in_channels, c1)
        self.enc2 = DownBlock(c1, c2)
        self.enc3 = DownBlock(c2, c3)
        self.enc4 = DownBlock(c3, c4)

        # Bottleneck: c4->c5（c5=16*c1）
        self.bottleneck = DownBlock(c4, c5)

        # Decoder: (c5+c4)->c4 -> (c4+c3)->c3 -> (c3+c2)->c2 -> (c2+c1)->c1
        self.dec4 = UpBlock(c5, c4, c4)
        self.dec3 = UpBlock(c4, c3, c3)
        self.dec2 = UpBlock(c3, c2, c2)
        self.dec1 = UpBlock(c2, c1, c1)

        # Head: c1 -> 2
        self.head = nn.Conv2d(c1, out_masks, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入混合对数幅度谱，shape `[B, 1, F, T]`。

        Returns:
            torch.Tensor: 预测 mask，shape `[B, 2, F, T]`。
        """
        if x.ndim != 4:
            raise ValueError(f"Expected input [B,1,F,T], got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected channel={self.in_channels}, got channel={x.shape[1]}")

        # 编码路径
        e1 = self.enc1(x)   # [B, c1, F, T]
        e2 = self.enc2(e1)  # [B, c2, F/2, T/2]
        e3 = self.enc3(e2)  # [B, c3, F/4, T/4]
        e4 = self.enc4(e3)  # [B, c4, F/8, T/8]

        # 瓶颈
        b = self.bottleneck(e4)  # [B, c5, F/16, T/16]

        # 解码路径 + skip connection
        d4 = self.dec4(b, e4)    # [B, c4, F/8, T/8]
        d3 = self.dec3(d4, e3)   # [B, c3, F/4, T/4]
        d2 = self.dec2(d3, e2)   # [B, c2, F/2, T/2]
        d1 = self.dec1(d2, e1)   # [B, c1, F, T]

        mask_logits = self.head(d1)  # [B, 2, F, T]
        masks = torch.softmax(mask_logits, dim=1)
        return masks

