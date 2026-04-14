"""ResNet18D 分离模型。

输入/输出约定：
- 输入：`[B, 1, F, T]`（混合信号对数幅度谱）
- 输出：`[B, 2, F, T]`（两个源的 mask 概率）
"""

from __future__ import annotations

import timm
import torch
import torch.nn.functional as F
from torch import nn


class ResNet18DSeparator(nn.Module):
    """基于 timm resnet18d backbone 的双源 mask 预测器。"""

    def __init__(self, in_channels: int = 1, out_masks: int = 2) -> None:
        super().__init__()
        if in_channels != 1:
            raise ValueError(f"Current project expects in_channels=1, got {in_channels}")
        if out_masks != 2:
            raise ValueError(f"Current project expects out_masks=2, got {out_masks}")

        self.in_channels = in_channels
        self.out_masks = out_masks

        # 使用 feature extraction 方式，不使用分类头。
        self.resnet18d = timm.create_model(
            "resnet18d",
            pretrained=False,
            in_chans=in_channels,
            features_only=True,
            out_indices=(4,),
        )

        # resnet18d 最后一层特征通道为 512。
        self.head = nn.Conv2d(in_channels=512, out_channels=out_masks, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        输入必须是 `[B,1,F,T]`，输出必须是 `[B,2,F,T]`。
        """
        if x.ndim != 4:
            raise ValueError(f"Expected input [B,1,F,T], got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected channel={self.in_channels}, got channel={x.shape[1]}")

        orig_f, orig_t = int(x.shape[-2]), int(x.shape[-1])

        # resnet18d feature map，shape 约为 [B, 512, F/32, T/32]
        feat = self.resnet18d(x)[0]
        logits_low = self.head(feat)  # [B, 2, f_low, t_low]

        # 上采样回输入分辨率，得到 [B, 2, F, T]
        logits = F.interpolate(
            logits_low,
            size=(orig_f, orig_t),
            mode="bilinear",
            align_corners=False,
        )

        # 源维度 softmax，得到两个 mask 概率
        masks = torch.softmax(logits, dim=1)
        return masks
