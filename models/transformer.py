"""Transformer 相关模型模块。

包含：
- TransformerEncoderBlocks：Transformer Encoder 封装。
- TransformerSeparator：基于 Patch+Transformer 的双源 mask 预测器。

TransformerSeparator 维度流转：
1) 输入：`[B, 1, F, T]`
2) PatchEmbed：`[B, N, D]`（内部自动 padding）
3) + 位置编码，TransformerEncoder：`[B, N, D]`
4) token -> grid：`[B, D, grid_f, grid_t]`
5) ConvTranspose2d 解码：`[B, 2, F_pad, T_pad]`
6) 按原始尺寸裁剪：`[B, 2, F, T]`
7) 源维 softmax：`[B, 2, F, T]`
"""

from __future__ import annotations

import torch
from torch import nn

from models.patch_embed import PatchEmbed2D


class TransformerEncoderBlocks(nn.Module):
    """ Transformer Encoder 封装。"""

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        num_layers: int = 4,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if d_model <= 0 or n_heads <= 0 or num_layers <= 0 or ff_dim <= 0:
            raise ValueError("d_model/n_heads/num_layers/ff_dim must be > 0.")
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads}).")

        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.activation = activation

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation=activation,
            batch_first=True,  # 输入输出统一 [B, N, D]
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: token 序列，shape `[B, N, D]`。

        Returns:
            torch.Tensor: 编码后 token，shape `[B, N, D]`。
        """
        if x.ndim != 3:
            raise ValueError(f"Expected input shape [B,N,D], got {tuple(x.shape)}")
        if x.shape[-1] != self.d_model:
            raise ValueError(f"Expected D={self.d_model}, got D={x.shape[-1]}")
        return self.encoder(x)


class TransformerSeparator(nn.Module):
    """基于 Patch+Transformer 的双源 mask 预测器。"""

    def __init__(
        self,
        in_channels: int = 1,
        out_masks: int = 2,
        embed_dim: int = 256,
        depth: int = 4,
        num_heads: int = 8,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        patch_size: int = 16,
        max_tokens: int = 4096,
    ) -> None:
        super().__init__()
        if out_masks != 2:
            raise ValueError(f"Current project expects out_masks=2, got {out_masks}")

        self.out_masks = out_masks
        self.embed_dim = embed_dim
        self.max_tokens = max_tokens

        self.patch_embed = PatchEmbed2D(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
        )
        self.encoder = TransformerEncoderBlocks(
            d_model=embed_dim,
            n_heads=num_heads,
            num_layers=depth,
            ff_dim=ff_dim,
            dropout=dropout,
            activation="gelu",
        )

        # 简单可学习绝对位置编码：按 token 数切片使用
        self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # 反 patch 化解码头：grid -> padded 频时图
        self.decode_head = nn.ConvTranspose2d(
            in_channels=embed_dim,
            out_channels=out_masks,
            kernel_size=(self.patch_embed.patch_freq, self.patch_embed.patch_time),
            stride=(self.patch_embed.patch_freq, self.patch_embed.patch_time),
            padding=0,
        )

    def _add_pos_embed(self, tokens: torch.Tensor) -> torch.Tensor:
        """给 `[B, N, D]` token 添加位置编码。"""
        _, n, d = tokens.shape
        if d != self.embed_dim:
            raise ValueError(f"Token dim mismatch: got D={d}, expected {self.embed_dim}")
        if n > self.max_tokens:
            raise ValueError(
                f"Token length N={n} exceeds max_tokens={self.max_tokens}. "
                "Increase max_tokens in model init."
            )
        return tokens + self.pos_embed[:, :n, :]

    def forward(self, mix_logmag: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            mix_logmag: 输入混合信号对数幅度谱，shape `[B, 1, F, T]`。

        Returns:
            torch.Tensor: 预测 mask，shape `[B, 2, F, T]`。
        """
        if mix_logmag.ndim != 4:
            raise ValueError(f"Expected input [B,1,F,T], got {tuple(mix_logmag.shape)}")
        if mix_logmag.shape[1] != 1:
            raise ValueError(f"Expected channel=1, got {mix_logmag.shape[1]}")

        # 1) patch embedding + meta
        tokens, meta = self.patch_embed(mix_logmag, return_meta=True)  # [B, N, D], dict

        # 2) 位置编码 + Transformer 编码
        tokens = self._add_pos_embed(tokens)
        tokens = self.encoder(tokens)  # [B, N, D]

        # 3) token -> grid feature map
        grid_feat = self.patch_embed.tokens_to_grid(tokens, meta=meta)  # [B, D, grid_f, grid_t]

        # 4) 反 patch 化到 padded mask
        mask_logits_pad = self.decode_head(grid_feat)  # [B, 2, F_pad, T_pad]

        # 5) 裁剪回原始尺寸
        orig_f = int(meta["orig_f"])
        orig_t = int(meta["orig_t"])
        mask_logits = mask_logits_pad[:, :, :orig_f, :orig_t]  # [B, 2, F, T]

        # 6) 源维度 softmax，得到两个 mask 概率
        masks = torch.softmax(mask_logits, dim=1)
        return masks
