"""训练/验证单个 epoch 的基础流程。

职责边界：
- 负责单 epoch 的前向、loss 计算、反向更新（训练阶段）。
- 不负责创建数据集/模型/优化器，不负责 checkpoint 与命令行解析。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

import torch
from torch.utils.data import DataLoader


REQUIRED_BATCH_KEYS = (
    "mix_mag",
    "mix_spec",
    "srcA_spec",
    "srcB_spec",
    "mask_target",
    "srcA_time",
    "srcB_time",
)


@dataclass
class _LossAccumulator:
    total_loss: float = 0.0
    mag_loss: float = 0.0
    mask_loss: float = 0.0
    corr_loss: float = 0.0
    steps: int = 0 #累计处理了多少个batch

    def update(self, loss_dict: Mapping[str, torch.Tensor]) -> None:
        self.total_loss += float(loss_dict["total_loss"].detach().item())
        self.mag_loss += float(loss_dict["mag_loss"].detach().item())
        self.mask_loss += float(loss_dict["mask_loss"].detach().item())
        self.corr_loss += float(loss_dict["corr_loss"].detach().item())
        self.steps += 1

    def average(self) -> Dict[str, float]:
        if self.steps == 0:
            raise ValueError("Dataloader is empty: no batch was processed in this epoch.")
        denom = float(self.steps)
        return {
            "total_loss": self.total_loss / denom,
            "mag_loss": self.mag_loss / denom,
            "mask_loss": self.mask_loss / denom,
            "corr_loss": self.corr_loss / denom,
        }


def _move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    """将 batch 中 tensor 移动到 device；非 tensor 字段保持不变。"""
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _require_keys(batch: Mapping[str, Any]) -> None:
    missing = [k for k in REQUIRED_BATCH_KEYS if k not in batch]
    if missing:
        raise KeyError(f"Batch missing required keys: {missing}")


def _validate_shapes(batch: Mapping[str, Any]) -> None:
    """检查当前数据格式是否满足训练链路约定。"""
    mix_mag = batch["mix_mag"]
    mix_spec = batch["mix_spec"]
    srcA_spec = batch["srcA_spec"]
    srcB_spec = batch["srcB_spec"]
    mask_target = batch["mask_target"]
    srcA_time = batch["srcA_time"]
    srcB_time = batch["srcB_time"]

    if not isinstance(mix_mag, torch.Tensor) or mix_mag.ndim != 4 or mix_mag.shape[1] != 1:
        raise ValueError(f"mix_mag must be [B,1,F,T], got {getattr(mix_mag, 'shape', None)}")
    if not isinstance(mask_target, torch.Tensor) or mask_target.ndim != 4 or mask_target.shape[1] != 2:
        raise ValueError(
            f"mask_target must be [B,2,F,T], got {getattr(mask_target, 'shape', None)}"
        )

    for name, spec in (("mix_spec", mix_spec), ("srcA_spec", srcA_spec), ("srcB_spec", srcB_spec)):
        if not isinstance(spec, torch.Tensor) or spec.ndim != 3:
            raise ValueError(f"{name} must be [B,F,T], got {getattr(spec, 'shape', None)}")
    for name, signal in (("srcA_time", srcA_time), ("srcB_time", srcB_time)):
        if not isinstance(signal, torch.Tensor) or signal.ndim != 2:
            raise ValueError(f"{name} must be [B,N], got {getattr(signal, 'shape', None)}")

    if mix_spec.shape != srcA_spec.shape or mix_spec.shape != srcB_spec.shape:
        raise ValueError(
            "mix_spec/srcA_spec/srcB_spec shape mismatch: "
            f"{mix_spec.shape}, {srcA_spec.shape}, {srcB_spec.shape}"
        )
    # 检查batch_size维度是否一致
    if mix_mag.shape[0] != mix_spec.shape[0]:
        raise ValueError("Batch size mismatch between mix_mag and mix_spec.")
    if mask_target.shape[0] != mix_spec.shape[0]:
        raise ValueError("Batch size mismatch between mask_target and mix_spec.")
    if srcA_time.shape[0] != mix_spec.shape[0] or srcB_time.shape[0] != mix_spec.shape[0]:
        raise ValueError("Batch size mismatch among srcA_time/srcB_time and mix_spec.")


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: Any,
) -> Dict[str, float]:
    """执行一个训练 epoch，并返回平均 loss。"""
    model.train()
    acc = _LossAccumulator()

    for batch in dataloader:
        _require_keys(batch)
        batch = _move_batch_to_device(batch, device)
        _validate_shapes(batch)

        pred_mask = model(batch["mix_mag"])  # [B, 2, F, T]
        loss_dict = criterion(
            pred_mask=pred_mask,
            target_mask=batch["mask_target"],
            mix_spec=batch["mix_spec"],
            srcA_spec=batch["srcA_spec"],
            srcB_spec=batch["srcB_spec"],
            srcA_time=batch["srcA_time"],
            srcB_time=batch["srcB_time"],
            cfg=cfg,
        )

        optimizer.zero_grad(set_to_none=True)
        loss_dict["total_loss"].backward()
        optimizer.step()

        acc.update(loss_dict)

    return acc.average()


def validate_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    cfg: Any,
) -> Dict[str, float]:
    """执行一个验证 epoch（无反向传播），并返回平均 loss。"""
    model.eval()
    acc = _LossAccumulator()

    with torch.no_grad():
        for batch in dataloader:
            _require_keys(batch)
            batch = _move_batch_to_device(batch, device)
            _validate_shapes(batch)

            pred_mask = model(batch["mix_mag"])  # [B, 2, F, T]
            loss_dict = criterion(
                pred_mask=pred_mask,
                target_mask=batch["mask_target"],
                mix_spec=batch["mix_spec"],
                srcA_spec=batch["srcA_spec"],
                srcB_spec=batch["srcB_spec"],
                srcA_time=batch["srcA_time"],
                srcB_time=batch["srcB_time"],
                cfg=cfg,
            )
            acc.update(loss_dict)

    return acc.average()
