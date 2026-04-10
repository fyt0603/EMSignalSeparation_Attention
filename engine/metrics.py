"""分离任务评估指标。

说明：
- `pearson_corr` / `batch_pearson_corr` 面向实值信号。
  其内部会转换为 float，复数输入会隐式丢失虚部，因此不适合复基带评估。
- 对复基带时域信号，推荐使用 `complex_corr` / `batch_complex_corr`，
  基于归一化复相关系数幅值，更符合复信号一致性度量。

公式：
- r = cov(x, y) / (std(x) * std(y))
- 实现形式：
    x_c = x - mean(x), y_c = y - mean(y)
    r = sum(x_c * y_c) / sqrt(sum(x_c^2) * sum(y_c^2) + eps)
"""

from __future__ import annotations

from typing import Union

import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor]


def _to_numpy_1d(x: ArrayLike, name: str) -> np.ndarray:
    """将输入转换为一维 numpy float64。"""
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        arr = x
    else:
        raise TypeError(f"{name} must be torch.Tensor or numpy.ndarray, got {type(x)}")

    arr = np.asarray(arr)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D for pearson_corr, got shape={arr.shape}")
    return arr.astype(np.float64, copy=False)


def _to_numpy_2d(x: ArrayLike, name: str) -> np.ndarray:
    """将输入转换为二维 numpy float64（批量版本）。"""
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        arr = x
    else:
        raise TypeError(f"{name} must be torch.Tensor or numpy.ndarray, got {type(x)}")

    arr = np.asarray(arr)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if arr.ndim == 1:
        arr = arr[None, :]
    elif arr.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D for batch_pearson_corr, got shape={arr.shape}")
    return arr.astype(np.float64, copy=False)


def _to_numpy_complex_1d(x: ArrayLike, name: str) -> np.ndarray:
    """将输入转换为一维 numpy complex128。"""
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        arr = x
    else:
        raise TypeError(f"{name} must be torch.Tensor or numpy.ndarray, got {type(x)}")

    arr = np.asarray(arr)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D for complex_corr, got shape={arr.shape}")
    return arr.astype(np.complex128, copy=False)


def _to_numpy_complex_2d(x: ArrayLike, name: str) -> np.ndarray:
    """将输入转换为二维 numpy complex128（批量版本）。"""
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        arr = x
    else:
        raise TypeError(f"{name} must be torch.Tensor or numpy.ndarray, got {type(x)}")

    arr = np.asarray(arr)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if arr.ndim == 1:
        arr = arr[None, :]
    elif arr.ndim != 2:
        raise ValueError(
            f"{name} must be 1D or 2D for batch_complex_corr, got shape={arr.shape}"
        )
    return arr.astype(np.complex128, copy=False)


def pearson_corr(x: ArrayLike, y: ArrayLike, eps: float = 1e-8) -> float:
    """计算一对一维时域信号的 Pearson 相关系数。

    Args:
        x: 预测或重建信号，shape `[N]`。
        y: 参考真实信号，shape `[N]`。
        eps: 数值稳定项，避免分母为 0。

    Returns:
        float: Pearson 相关系数，范围通常在 [-1, 1]。
    """
    x_np = _to_numpy_1d(x, name="x")
    y_np = _to_numpy_1d(y, name="y")
    if x_np.shape != y_np.shape:
        raise ValueError(f"x/y shape mismatch: {x_np.shape} vs {y_np.shape}")

    x_c = x_np - x_np.mean()
    y_c = y_np - y_np.mean()
    numerator = float(np.sum(x_c * y_c))
    denominator = float(np.sqrt(np.sum(x_c * x_c) * np.sum(y_c * y_c) + eps))
    return numerator / denominator


def batch_pearson_corr(x: ArrayLike, y: ArrayLike, eps: float = 1e-8) -> float:
    """计算批量样本的平均 Pearson 相关系数。

    输入可为：
    - `[N]`（视作 batch=1）
    - `[B, N]`

    Args:
        x: 批量预测信号。
        y: 批量真实信号。
        eps: 数值稳定项。

    Returns:
        float: 批量平均相关系数。
    """
    x_np = _to_numpy_2d(x, name="x")
    y_np = _to_numpy_2d(y, name="y")
    if x_np.shape != y_np.shape:
        raise ValueError(f"x/y shape mismatch: {x_np.shape} vs {y_np.shape}")

    x_c = x_np - x_np.mean(axis=1, keepdims=True)
    y_c = y_np - y_np.mean(axis=1, keepdims=True)
    numerator = np.sum(x_c * y_c, axis=1)
    denominator = np.sqrt(np.sum(x_c * x_c, axis=1) * np.sum(y_c * y_c, axis=1) + eps)
    corr_each = numerator / denominator
    return float(np.mean(corr_each))


def complex_corr(x: ArrayLike, y: ArrayLike, eps: float = 1e-8) -> float:
    """计算一对复时域信号的归一化复相关系数幅值。

    定义：
    - rho = |sum(x * conj(y))| / sqrt(sum(|x|^2) * sum(|y|^2) + eps)

    说明：
    - 返回值范围通常在 [0, 1]，越接近 1 表示两个复信号越相似。
    - 与实值 Pearson 不同，该指标同时考虑实部与虚部关系。
    """
    x_np = _to_numpy_complex_1d(x, name="x")
    y_np = _to_numpy_complex_1d(y, name="y")
    if x_np.shape != y_np.shape:
        raise ValueError(f"x/y shape mismatch: {x_np.shape} vs {y_np.shape}")

    numerator = np.abs(np.sum(x_np * np.conj(y_np)))
    denominator = np.sqrt(
        np.sum(np.abs(x_np) ** 2) * np.sum(np.abs(y_np) ** 2) + float(eps)
    )
    return float(numerator / denominator)


def batch_complex_corr(x: ArrayLike, y: ArrayLike, eps: float = 1e-8) -> float:
    """计算批量复时域信号的平均归一化复相关系数幅值。

    输入可为：
    - `[N]`（视作 batch=1）
    - `[B, N]`

    Returns:
        float: 批量平均复相关系数幅值，范围通常在 [0, 1]。
    """
    x_np = _to_numpy_complex_2d(x, name="x")
    y_np = _to_numpy_complex_2d(y, name="y")
    if x_np.shape != y_np.shape:
        raise ValueError(f"x/y shape mismatch: {x_np.shape} vs {y_np.shape}")

    numerator = np.abs(np.sum(x_np * np.conj(y_np), axis=1))
    denominator = np.sqrt(
        np.sum(np.abs(x_np) ** 2, axis=1) * np.sum(np.abs(y_np) ** 2, axis=1) + float(eps)
    )
    rho_each = numerator / denominator
    return float(np.mean(rho_each))
