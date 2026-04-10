"""双源分离损失函数。

公式：
- pred_srcA = pred_mask[:, 0] * mix_spec
- pred_srcB = pred_mask[:, 1] * mix_spec
- mag_loss = L1(|pred_srcA|, |srcA_spec|) + L1(|pred_srcB|, |srcB_spec|)
- mask_loss = MSE(pred_mask, target_mask)
- corr_loss = 0.5 * ((1 - corr_a) + (1 - corr_b))
- total_loss = mag_loss_weight * mag_loss + mask_loss_weight * mask_loss + corr_loss_weight * corr_loss
"""

from typing import Any, Dict

import torch
from torch import nn

from data.stft_utils import istft_reconstruct

class SeparationLoss(nn.Module):
    """用于双源分离的组合损失。"""

    def __init__(
        self,
        mag_loss_weight: float = 0.3,
        mask_loss_weight: float = 0.3,
        corr_loss_weight: float = 0.3,
    ) -> None:
        super().__init__()
        self.mag_loss_weight = mag_loss_weight
        self.mask_loss_weight = mask_loss_weight
        self.corr_loss_weight = corr_loss_weight
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.eps = 1e-8

    def _complex_corr_torch(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or y.ndim != 2:
            raise ValueError(f"x/y must be [B,N], got {tuple(x.shape)} and {tuple(y.shape)}")
        if x.shape != y.shape:
            raise ValueError(f"x/y shape mismatch: {tuple(x.shape)} vs {tuple(y.shape)}")

        numerator = torch.abs(torch.sum(x * torch.conj(y), dim=1))  # [B]
        energy_x = torch.sum(torch.abs(x) ** 2, dim=1)  # [B]
        energy_y = torch.sum(torch.abs(y) ** 2, dim=1)  # [B]
        denominator = torch.sqrt(energy_x * energy_y + self.eps)  # [B]
        return numerator / denominator

    def forward(
        self,
        pred_mask: torch.Tensor,
        target_mask: torch.Tensor,
        mix_spec: torch.Tensor,
        srcA_spec: torch.Tensor,
        srcB_spec: torch.Tensor,
        srcA_time: torch.Tensor,
        srcB_time: torch.Tensor,
        cfg: Any,
    ) -> Dict[str, torch.Tensor]:
        """计算总损失与子损失。

        Args:
            pred_mask: 预测 mask，shape [B, 2, F, T]。
            target_mask: 目标 mask，shape [B, 2, F, T]。
            mix_spec: 混合复数谱，shape [B, F, T]。
            srcA_spec: 源A真实复数谱，shape [B, F, T]。
            srcB_spec: 源B真实复数谱，shape [B, F, T]。
            srcA_time: 源A真实时域复信号，shape [B, N]。
            srcB_time: 源B真实时域复信号，shape [B, N]。
            cfg: 配置对象。

        Returns:
            Dict[str, torch.Tensor]:
                - total_loss
                - mag_loss
                - mask_loss
                - corr_loss
        """
        if pred_mask.ndim != 4 or pred_mask.shape[1] != 2:
            raise ValueError(f"pred_mask must be [B,2,F,T], got {tuple(pred_mask.shape)}")
        if target_mask.shape != pred_mask.shape:
            raise ValueError("target_mask shape must match pred_mask shape.")
        if mix_spec.ndim != 3:
            raise ValueError(f"mix_spec must be [B,F,T], got {tuple(mix_spec.shape)}")
        if srcA_spec.shape != mix_spec.shape or srcB_spec.shape != mix_spec.shape:
            raise ValueError("srcA_spec/srcB_spec shape must match mix_spec shape.")
        if not torch.is_complex(mix_spec):
            raise TypeError("mix_spec must be complex tensor.")
        if not torch.is_complex(srcA_spec) or not torch.is_complex(srcB_spec):
            raise TypeError("srcA_spec and srcB_spec must be complex tensors.")
        if srcA_time.ndim != 2 or srcB_time.ndim != 2:
            raise ValueError(
                f"srcA_time/srcB_time must be [B,N], got {tuple(srcA_time.shape)} and {tuple(srcB_time.shape)}"
            )
        if srcA_time.shape != srcB_time.shape:
            raise ValueError(
                f"srcA_time/srcB_time shape mismatch: {tuple(srcA_time.shape)} vs {tuple(srcB_time.shape)}"
            )
        if not torch.is_complex(srcA_time) or not torch.is_complex(srcB_time):
            raise TypeError("srcA_time and srcB_time must be complex tensors.")

        # 根据预测 mask 和混合复谱重建预测源谱
        pred_srcA = pred_mask[:, 0, :, :] * mix_spec  # [B, F, T], complex
        pred_srcB = pred_mask[:, 1, :, :] * mix_spec  # [B, F, T], complex

        # 幅度谱 L1 损失
        pred_srcA_mag = torch.abs(pred_srcA)
        pred_srcB_mag = torch.abs(pred_srcB)
        srcA_mag = torch.abs(srcA_spec)
        srcB_mag = torch.abs(srcB_spec)
        mag_loss = self.l1(pred_srcA_mag, srcA_mag) + self.l1(pred_srcB_mag, srcB_mag)

        # mask MSE 损失
        mask_loss = self.mse(pred_mask, target_mask)

        # 复相关损失（通过 iSTFT 恢复时域后计算）
        target_len = int(srcA_time.shape[-1])
        pred_srcA_time = istft_reconstruct(pred_srcA, cfg=cfg, length=target_len)  # [B, N], complex
        pred_srcB_time = istft_reconstruct(pred_srcB, cfg=cfg, length=target_len)  # [B, N], complex

        corr_a = torch.mean(self._complex_corr_torch(pred_srcA_time, srcA_time))
        corr_b = torch.mean(self._complex_corr_torch(pred_srcB_time, srcB_time))
        corr_loss = 0.5 * ((1.0 - corr_a) + (1.0 - corr_b))

        # 总损失
        total_loss = (
            self.mag_loss_weight * mag_loss
            + self.mask_loss_weight * mask_loss
            + self.corr_loss_weight * corr_loss
        )
        return {
            "total_loss": total_loss,
            "mag_loss": mag_loss,
            "mask_loss": mask_loss,
            "corr_loss": corr_loss,
        }
