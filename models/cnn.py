"""CNN 相关模型模块。

包含：
- CNNEncoderBlocks：保持空间尺寸与通道数不变的 CNN 编码器堆叠。
- CNNSeparator：基于 Patch+CNN 的双源 mask 预测器。

CNNSeparator 维度流转：
1) 输入：`[B, C, F, T]`（`C = in_channels`）
2) PatchEmbed：`[B, N, D]`（内部自动 padding）
3) token -> grid：`[B, D, grid_f, grid_t]`
4) CNNEncoder：`[B, D, grid_f, grid_t]`
5) bottleneck 投影：`[B, 256, grid_f, grid_t]`
6) U-Net 解码：`[B, 2, F_pad, T_pad]`
7) 按原始尺寸裁剪：`[B, 2, F, T]`
8) 源维 softmax：`[B, 2, F, T]`
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from models.cnn_skip import CNNSkipEncoder
from models.patch_embed import PatchEmbed2D
from models.unet_decoder import UNetDecoder


def _pick_group_count(channels: int, prefer: int = 8) -> int:
    """选择可整除通道数的 GroupNorm 组数。"""
    g = min(prefer, channels)
    while g > 1 and channels % g != 0:
        g -= 1
    return g


class _ConvNormAct(nn.Module):
    """Conv2d -> GroupNorm -> GELU。"""

    def __init__(
        self,
        channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be > 0, got {channels}")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.norm = nn.GroupNorm(
            num_groups=_pick_group_count(channels),
            num_channels=channels,
        )
        self.act = nn.GELU()
        self.drop = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.norm(self.conv(x))))


class _ResidualConvBlock(nn.Module):
    """不改变 `[B,C,H,W]` 形状的残差卷积块。"""

    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv = _ConvNormAct(channels=channels, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class CNNEncoderBlocks(nn.Module):
    """CNN 编码器堆叠。

    说明：
    - 输入输出均为 `[B, C, H, W]`。
    - 空间尺寸 `H/W` 不变。
    - 通道数 `C` 不变。
    """

    def __init__(
        self,
        channels: int = 256,
        depth: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be > 0, got {channels}")
        if depth <= 0:
            raise ValueError(f"depth must be > 0, got {depth}")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.channels = channels
        self.depth = depth
        self.dropout = dropout

        self.blocks = nn.Sequential(
            *[_ResidualConvBlock(channels=channels, dropout=dropout) for _ in range(depth)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected input shape [B,C,H,W], got {tuple(x.shape)}")
        if x.shape[1] != self.channels:
            raise ValueError(f"Expected C={self.channels}, got C={x.shape[1]}")
        return self.blocks(x)


class CNNSeparator(nn.Module):
    """基于 Patch+CNN 的双源 mask 预测器。"""

    def __init__(
        self,
        in_channels: int = 3,
        out_masks: int = 2,
        embed_dim: int = 256,
        depth: int = 2,
        dropout: float = 0.1,
        patch_size: int = 16,
        decoder_type: str = "deconv",
        use_cnn_skip: bool = True,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be > 0, got {in_channels}")
        if out_masks != 2:
            raise ValueError(f"Current project expects out_masks=2, got {out_masks}")
        if decoder_type not in ("deconv", "unet"):
            raise ValueError(f"decoder_type must be 'deconv' or 'unet', got {decoder_type}")
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be > 0, got {embed_dim}")
        if depth <= 0:
            raise ValueError(f"depth must be > 0, got {depth}")
        if patch_size <= 0:
            raise ValueError(f"patch_size must be > 0, got {patch_size}")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.in_channels = in_channels
        self.out_masks = out_masks
        self.embed_dim = embed_dim
        self.depth = depth
        self.dropout = dropout
        self.decoder_type = decoder_type
        self.use_cnn_skip = use_cnn_skip

        self.patch_embed = PatchEmbed2D(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
        )
        self.encoder = CNNEncoderBlocks(
            channels=embed_dim,
            depth=depth,
            dropout=dropout,
        )

        # deconv 分支：反 patch 化解码头（grid -> padded 频时图）
        self.decode_head = nn.ConvTranspose2d(
            in_channels=embed_dim,
            out_channels=out_masks,
            kernel_size=(self.patch_embed.patch_freq, self.patch_embed.patch_time),
            stride=(self.patch_embed.patch_freq, self.patch_embed.patch_time),
            padding=0,
        )

        # unet 分支：先把 grid feature 投影到 256 通道，再经 U-Net decoder
        self.bottleneck_proj = nn.Conv2d(embed_dim, 256, kernel_size=1)
        self.unet_decoder = UNetDecoder(out_channels=out_masks, use_skips=use_cnn_skip)
        self.cnn_skip = CNNSkipEncoder(in_channels=in_channels) if use_cnn_skip else None

    def forward(self, mix_feat: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            mix_feat: 输入混合信号特征图，shape `[B, C, F, T]`，`C = in_channels`。

        Returns:
            torch.Tensor: 预测 mask，shape `[B, 2, F, T]`。
        """
        if mix_feat.ndim != 4:
            raise ValueError(f"Expected input [B,C,F,T], got {tuple(mix_feat.shape)}")
        if mix_feat.shape[1] != self.in_channels:
            raise ValueError(f"Expected channel={self.in_channels}, got {mix_feat.shape[1]}")

        # 1) patch embedding + meta
        tokens, meta = self.patch_embed(mix_feat, return_meta=True)  # [B, N, D], dict

        # 2) token -> grid feature map
        grid_feat = self.patch_embed.tokens_to_grid(tokens, meta=meta)  # [B, D, grid_f, grid_t]

        # 3) CNN 编码
        grid_feat = self.encoder(grid_feat)  # [B, D, grid_f, grid_t]

        # 4) 解码到 padded mask
        if self.decoder_type == "deconv":
            # deconv 分支下忽略 use_cnn_skip，直接由 grid feature 反 patch 化得到 mask。
            mask_logits_pad = self.decode_head(grid_feat)  # [B, 2, F_pad, T_pad]
        else:
            # 4.1) bottleneck 投影到 256 通道
            bottleneck = self.bottleneck_proj(grid_feat)  # [B, 256, grid_f, grid_t]

            # 4.2) 可选 CNN skip（x_pad 使用与 PatchEmbed 相同的补零量）
            skips = None
            if self.use_cnn_skip:
                if self.cnn_skip is None:
                    raise RuntimeError("use_cnn_skip=True but cnn_skip is not initialized.")
                pad_f = int(meta["pad_f"])
                pad_t = int(meta["pad_t"])
                if pad_f > 0 or pad_t > 0:
                    x_pad = F.pad(mix_feat, (0, pad_t, 0, pad_f), mode="constant", value=0.0)
                else:
                    x_pad = mix_feat
                skips = self.cnn_skip(x_pad)  # (s1, s2, s3, s4)

            # 4.3) U-Net 解码到 padded mask
            mask_logits_pad = self.unet_decoder(bottleneck, skips=skips)  # [B, 2, F_pad, T_pad]

        # 5) 裁剪回原始尺寸
        orig_f = int(meta["orig_f"])
        orig_t = int(meta["orig_t"])
        mask_logits = mask_logits_pad[:, :, :orig_f, :orig_t]  # [B, 2, F, T]

        # 6) 源维度 softmax，得到两个 mask 概率
        masks = torch.softmax(mask_logits, dim=1)
        return masks

