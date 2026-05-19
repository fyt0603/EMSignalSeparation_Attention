"""复数 mask 构造、应用与理想目标计算工具。

包含函数：
- build_complex_mask: 将四通道实数 mask 组装为两个复数 mask。
- apply_complex_mask: 将复数 mask 应用于混合复数谱。
- compute_ideal_complex_mask: 基于稳定公式计算理想复数 target mask。
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def _require_tensor(name: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be torch.Tensor, got {type(value)}")


def _validate_real_4ch_mask(name: str, mask: torch.Tensor) -> None:
    """检查 mask 为实数四通道，支持 [B,4,F,T] 或 [4,F,T]。"""
    _require_tensor(name, mask)
    if torch.is_complex(mask):
        raise TypeError(f"{name} must be a real tensor, got complex dtype={mask.dtype}")
    if mask.ndim not in (3, 4):
        raise ValueError(f"{name} must be [4,F,T] or [B,4,F,T], got shape={tuple(mask.shape)}")
    ch_dim = 0 if mask.ndim == 3 else 1
    if mask.shape[ch_dim] != 4:
        raise ValueError(
            f"{name} channel dimension must be 4, got shape={tuple(mask.shape)}"
        )


def _validate_complex_spec(name: str, spec: torch.Tensor) -> None:
    """检查谱为复数，支持 [F,T] 或 [B,F,T]。"""
    _require_tensor(name, spec)
    if not torch.is_complex(spec):
        raise TypeError(f"{name} must be complex tensor, got dtype={spec.dtype}")
    if spec.ndim not in (2, 3):
        raise ValueError(f"{name} must be [F,T] or [B,F,T], got shape={tuple(spec.shape)}")


def build_complex_mask(mask_4ch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """将 4 通道实数 mask 组装为两个复数 mask。

    Args:
        mask_4ch: 实数 mask。
            - [B,4,F,T] 时：
              通道 0/1/2/3 分别为 M_A_r, M_A_i, M_B_r, M_B_i。
            - [4,F,T] 时：
              通道定义同上。

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - mask_a: 复数 mask，shape [B,F,T] 或 [F,T]
            - mask_b: 复数 mask，shape [B,F,T] 或 [F,T]
    """
    _validate_real_4ch_mask("mask_4ch", mask_4ch)

    if mask_4ch.ndim == 4:
        m_a_r = mask_4ch[:, 0, :, :]
        m_a_i = mask_4ch[:, 1, :, :]
        m_b_r = mask_4ch[:, 2, :, :]
        m_b_i = mask_4ch[:, 3, :, :]
    else:  # [4,F,T]
        m_a_r = mask_4ch[0, :, :]
        m_a_i = mask_4ch[1, :, :]
        m_b_r = mask_4ch[2, :, :]
        m_b_i = mask_4ch[3, :, :]

    mask_a = torch.complex(m_a_r, m_a_i)
    mask_b = torch.complex(m_b_r, m_b_i)
    return mask_a, mask_b


def apply_complex_mask(pred_mask: torch.Tensor, mix_spec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """将复数 mask 应用于混合复数谱。

    Args:
        pred_mask: 实数预测 mask，shape [B,4,F,T] 或 [4,F,T]。
        mix_spec: 复数混合谱，shape [B,F,T] 或 [F,T]。

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - pred_srcA: 复数预测源 A，shape 与 mix_spec 相同
            - pred_srcB: 复数预测源 B，shape 与 mix_spec 相同

    Raises:
        ValueError: 当 pred_mask 与 mix_spec 的 batch/维度形态不匹配时抛出。
    """
    _validate_real_4ch_mask("pred_mask", pred_mask)
    _validate_complex_spec("mix_spec", mix_spec)

    if pred_mask.ndim == 4 and mix_spec.ndim != 3:
        raise ValueError(
            "Shape mismatch: pred_mask is [B,4,F,T], so mix_spec must be [B,F,T], "
            f"got {tuple(mix_spec.shape)}"
        )
    if pred_mask.ndim == 3 and mix_spec.ndim != 2:
        raise ValueError(
            "Shape mismatch: pred_mask is [4,F,T], so mix_spec must be [F,T], "
            f"got {tuple(mix_spec.shape)}"
        )

    if pred_mask.ndim == 4:
        if pred_mask.shape[0] != mix_spec.shape[0] or pred_mask.shape[2:] != mix_spec.shape[1:]:
            raise ValueError(
                "Shape mismatch between pred_mask and mix_spec: "
                f"pred_mask={tuple(pred_mask.shape)} vs mix_spec={tuple(mix_spec.shape)}"
            )
    else:
        if pred_mask.shape[1:] != mix_spec.shape:
            raise ValueError(
                "Shape mismatch between pred_mask and mix_spec: "
                f"pred_mask={tuple(pred_mask.shape)} vs mix_spec={tuple(mix_spec.shape)}"
            )

    mask_a, mask_b = build_complex_mask(pred_mask)
    pred_srcA = mask_a * mix_spec
    pred_srcB = mask_b * mix_spec
    return pred_srcA, pred_srcB


def compute_ideal_complex_mask(
    mix_spec: torch.Tensor,
    srcA_spec: torch.Tensor,
    srcB_spec: torch.Tensor,
    eps: float = 1e-8,
    mask_bound: Optional[float] = 5.0,
) -> torch.Tensor:
    """计算理想复数 target mask（稳定公式版本）。

    Args:
        mix_spec: 复数混合谱，shape [B,F,T] 或 [F,T]。
        srcA_spec: 复数源 A 谱，shape 与 mix_spec 相同。
        srcB_spec: 复数源 B 谱，shape 与 mix_spec 相同。
        eps: 数值稳定项，分母加法项。
        mask_bound: 若非 None，对 target mask 执行 clamp 到 [-mask_bound, mask_bound]。

    Returns:
        torch.Tensor: 实数 target mask。
            - 输入 [B,F,T] -> 输出 [B,4,F,T]
            - 输入 [F,T]   -> 输出 [4,F,T]
    """
    _validate_complex_spec("mix_spec", mix_spec)
    _validate_complex_spec("srcA_spec", srcA_spec)
    _validate_complex_spec("srcB_spec", srcB_spec)

    if srcA_spec.shape != mix_spec.shape or srcB_spec.shape != mix_spec.shape:
        raise ValueError(
            "srcA_spec and srcB_spec must have the same shape as mix_spec, got "
            f"mix_spec={tuple(mix_spec.shape)}, "
            f"srcA_spec={tuple(srcA_spec.shape)}, srcB_spec={tuple(srcB_spec.shape)}"
        )

    x_r = mix_spec.real
    x_i = mix_spec.imag
    a_r = srcA_spec.real
    a_i = srcA_spec.imag
    b_r = srcB_spec.real
    b_i = srcB_spec.imag

    den = x_r**2 + x_i**2 + float(eps)

    m_a_r = (a_r * x_r + a_i * x_i) / den
    m_a_i = (a_i * x_r - a_r * x_i) / den
    m_b_r = (b_r * x_r + b_i * x_i) / den
    m_b_i = (b_i * x_r - b_r * x_i) / den

    if mix_spec.ndim == 3:
        target_mask = torch.stack([m_a_r, m_a_i, m_b_r, m_b_i], dim=1)  # [B,4,F,T]
    else:
        target_mask = torch.stack([m_a_r, m_a_i, m_b_r, m_b_i], dim=0)  # [4,F,T]

    if mask_bound is not None:
        target_mask = torch.clamp(target_mask, -float(mask_bound), float(mask_bound))

    return target_mask

