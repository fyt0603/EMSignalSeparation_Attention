"""Patch Embedding 模块。

职责：
- 将输入时频图 `[B, C, F, T]` 切分为 patch 并映射为 token。
- 自动对右侧和下侧做零填充，使 `F/T` 可被 patch size 整除。
- 提供可恢复空间布局所需的元信息（grid/orig/pad）。
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class PatchEmbed2D(nn.Module):
    """时频二维 patch 嵌入。

    维度流转：
    - 输入：`x` -> `[B, C, F, T]`（其中 `C = in_channels`）
    - padding 后：`[B, C, F_pad, T_pad]`
    - Conv2d 投影：`[B, D, grid_f, grid_t]`
    - 展平 token：`[B, N, D]`，其中 `N = grid_f * grid_t`
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 256,
        patch_size: int = 16,
        patch_freq: Optional[int] = None,
        patch_time: Optional[int] = None,
    ) -> None:
        super().__init__()
        if patch_size <= 0:
            raise ValueError(f"patch_size must be > 0, got {patch_size}")

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_freq = int(patch_freq) if patch_freq is not None else int(patch_size)
        self.patch_time = int(patch_time) if patch_time is not None else int(patch_size)
        if self.patch_freq <= 0 or self.patch_time <= 0:
            raise ValueError("patch_freq/patch_time must be > 0.")

        # patchify + linear projection
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=(self.patch_freq, self.patch_time),
            stride=(self.patch_freq, self.patch_time),
            padding=0,
        )
        self.last_meta: Dict[str, int] = {}

    def _compute_padding(self, f: int, t: int) -> Tuple[int, int]:
        """计算右侧/下侧补齐量。"""
        pad_f = (self.patch_freq - (f % self.patch_freq)) % self.patch_freq
        pad_t = (self.patch_time - (t % self.patch_time)) % self.patch_time
        return pad_f, pad_t

    def forward(
        self,
        x: torch.Tensor,
        return_meta: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, int]]:
        """将时频图转为 patch token。

        Args:
            x: 输入特征图，shape `[B, C, F, T]`（`C = in_channels`）。
            return_meta: 为 True 时额外返回元信息字典。

        Returns:
            - tokens: `[B, N, D]`
            - meta (optional):
                - `grid_f`, `grid_t`
                - `orig_f`, `orig_t`
                - `pad_f`, `pad_t`
        """
        if x.ndim != 4:
            raise ValueError(f"Expected input [B,C,F,T], got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected channel={self.in_channels}, got input channel={x.shape[1]}"
            )

        _, _, orig_f, orig_t = x.shape
        pad_f, pad_t = self._compute_padding(orig_f, orig_t)

        if pad_f > 0 or pad_t > 0:
            # F 是高度(H)，T 是宽度(W)。pad 顺序为 (W_left, W_right, H_top, H_bottom)
            x = F.pad(x, (0, pad_t, 0, pad_f), mode="constant", value=0.0)

        feat = self.proj(x)  # [B, D, grid_f, grid_t]
        b, d, grid_f, grid_t = feat.shape
        tokens = feat.flatten(2).transpose(1, 2).contiguous()  # [B, N, D]

        meta = {
            "grid_f": int(grid_f),
            "grid_t": int(grid_t),
            "orig_f": int(orig_f),
            "orig_t": int(orig_t),
            "pad_f": int(pad_f),
            "pad_t": int(pad_t),
        }
        self.last_meta = meta

        if return_meta:
            return tokens, meta
        return tokens

    def tokens_to_grid(
        self,
        tokens: torch.Tensor,
        meta: Optional[Dict[str, int]] = None,
    ) -> torch.Tensor:
        """将 token 序列还原为 patch 网格特征。

        Args:
            tokens: token 序列，shape `[B, N, D]`。
            meta: 可选网格元信息；若为 None，则使用 `self.last_meta`。

        Returns:
            torch.Tensor: 网格特征，shape `[B, D, grid_f, grid_t]`。
        """
        if tokens.ndim != 3:
            raise ValueError(f"Expected tokens [B,N,D], got {tuple(tokens.shape)}")

        if meta is None:
            meta = self.last_meta
        if not meta:
            raise ValueError("meta is required when self.last_meta is empty.")

        if "grid_f" not in meta or "grid_t" not in meta:
            raise KeyError("meta must contain 'grid_f' and 'grid_t'.")

        b, n, d = tokens.shape
        grid_f = int(meta["grid_f"])
        grid_t = int(meta["grid_t"])
        expected_n = grid_f * grid_t

        if d != self.embed_dim:
            raise ValueError(f"Token dim mismatch: got D={d}, expected embed_dim={self.embed_dim}")
        if n != expected_n:
            raise ValueError(f"Token count mismatch: got N={n}, expected {expected_n}")

        # [B, N, D] -> [B, D, N] -> [B, D, grid_f, grid_t]
        return tokens.transpose(1, 2).contiguous().view(b, d, grid_f, grid_t)
