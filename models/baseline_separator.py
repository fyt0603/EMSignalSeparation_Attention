"""最小可运行分离模型（Baseline）。

用途：
- 作为训练流程联调用的“假模型”，先验证数据流、loss 与训练循环可运行。
- 后续可直接替换为正式 Transformer 分离模型。

输入/输出约定：
- 输入：`[B, 1, F, T]`（混合信号对数幅度谱）
- 输出：`[B, 2, F, T]`（两个源的 mask 概率）
"""

from __future__ import annotations

import torch
from torch import nn


class BaselineSeparator(nn.Module):
    """基于轻量卷积的双源 mask 预测器。"""

    def __init__(self, hidden_channels: int = 32) -> None:
        super().__init__()
        if hidden_channels <= 0:
            raise ValueError(f"hidden_channels must be > 0, got {hidden_channels}")

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=2,
            kernel_size=1,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入幅度谱，shape `[B, 1, F, T]`。

        Returns:
            torch.Tensor: 双源 mask，shape `[B, 2, F, T]`，在源维度做 softmax。
        """
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input [B,1,F,T], got shape={tuple(x.shape)}")
        if x.shape[1] != 1:
            raise ValueError(f"Expected channel dim=1, got x.shape[1]={x.shape[1]}")

        feat = self.encoder(x)
        logits = self.head(feat)            # [B, 2, F, T]
        masks = torch.softmax(logits, dim=1)
        return masks

