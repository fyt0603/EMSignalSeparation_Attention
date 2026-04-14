"""LSTM 分离模型模块。

设计目标：
- 输入：混合对数幅度谱 `mix_logmag`，shape `[B, 1, F, T]`
- 内部重排：`[B, 1, F, T] -> [B, F, T] -> [B, T, F]`
- 时序建模：沿时间维使用BiLSTM
- 输出：两个源的 mask，shape `[B, 2, F, T]`，并在源维做 softmax
"""

from __future__ import annotations

import torch
from torch import nn


class LSTMSeparator(nn.Module):
    """基于BiLSTM 的双源 mask 预测器。"""

    def __init__(
        self,
        in_channels: int = 1,
        out_masks: int = 2,
        input_freq_bins: int = 512,
        hidden_size: int = 256,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if in_channels != 1:
            raise ValueError(f"Current project expects in_channels=1, got {in_channels}")
        if out_masks != 2:
            raise ValueError(f"Current project expects out_masks=2, got {out_masks}")
        if input_freq_bins <= 0 or hidden_size <= 0 or num_layers <= 0:
            raise ValueError("input_freq_bins/hidden_size/num_layers must be > 0.")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError(f"dropout must be in [0,1), got {dropout}")

        self.in_channels = in_channels
        self.out_masks = out_masks
        self.input_freq_bins = input_freq_bins
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # PyTorch LSTM 仅在 num_layers > 1 时才应用内部 dropout。
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_freq_bins, # 每个时间步的输入特征维度是F
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
            batch_first=True,  # 输入输出统一 [B, T, C]
        )

        lstm_out_dim = hidden_size * (2 if bidirectional else 1)
        # 每个时间步映射到两个源的频率 mask：2 * F
        self.head = nn.Linear(lstm_out_dim, out_masks * input_freq_bins)

    def forward(self, mix_logmag: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            mix_logmag: 输入混合对数幅度谱，shape `[B, 1, F, T]`。

        Returns:
            torch.Tensor: 预测 mask，shape `[B, 2, F, T]`。
        """
        if mix_logmag.ndim != 4:
            raise ValueError(f"Expected input [B,1,F,T], got {tuple(mix_logmag.shape)}")
        if mix_logmag.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected channel={self.in_channels}, got channel={mix_logmag.shape[1]}"
            )

        b, _, f, t = mix_logmag.shape
        if f != self.input_freq_bins:
            raise ValueError(
                f"Frequency bins mismatch: got F={f}, expected {self.input_freq_bins}. "
                "Please align input_freq_bins with STFT frequency dimension."
            )

        # [B, 1, F, T] -> [B, F, T]
        x = mix_logmag.squeeze(1)
        # [B, F, T] -> [B, T, F]，沿时间维建模
        x = x.transpose(1, 2).contiguous()

        # LSTM 输出: [B, T, H*dir]
        seq_out, _ = self.lstm(x)
        # 线性映射到 [B, T, 2*F]
        logits_bt2f = self.head(seq_out)

        # [B, T, 2*F] -> [B, T, 2, F]
        logits_bt2f = logits_bt2f.view(b, t, self.out_masks, f)
        # [B, T, 2, F] -> [B, 2, F, T]
        mask_logits = logits_bt2f.permute(0, 2, 3, 1).contiguous()

        # 源维度 softmax，得到两个 mask 概率
        masks = torch.softmax(mask_logits, dim=1)
        return masks
