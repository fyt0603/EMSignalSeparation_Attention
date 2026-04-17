"""分离模型评估流程封装。

职责：
- 在验证/测试 dataloader 上执行推理。
- 由预测 mask 重建源谱并 iSTFT 到时域。
- 计算并汇总复相关系数指标（用于复基带时域信号）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

import torch
from torch.utils.data import DataLoader
from data.stft_utils import istft_reconstruct
from engine.metrics import complex_corr


REQUIRED_EVAL_KEYS = (
    "mix_feat",
    "mix_spec",
    "srcA_spec",
    "srcB_spec",
    "srcA_time",
    "srcB_time",
)


def _move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _require_eval_keys(batch: Mapping[str, Any]) -> None:
    missing = [k for k in REQUIRED_EVAL_KEYS if k not in batch]
    if missing:
        raise KeyError(f"Batch missing required eval keys: {missing}")


def _validate_eval_shapes(batch: Mapping[str, Any]) -> None:
    mix_feat = batch["mix_feat"]
    mix_spec = batch["mix_spec"]
    srcA_spec = batch["srcA_spec"]
    srcB_spec = batch["srcB_spec"]
    srcA_time = batch["srcA_time"]
    srcB_time = batch["srcB_time"]

    if mix_feat.ndim != 4 or mix_feat.shape[1] != 3:
        raise ValueError(f"mix_feat must be [B,3,F,T], got {tuple(mix_feat.shape)}")
    if mix_spec.ndim != 3:
        raise ValueError(f"mix_spec must be [B,F,T], got {tuple(mix_spec.shape)}")
    if srcA_spec.shape != mix_spec.shape or srcB_spec.shape != mix_spec.shape:
        raise ValueError(
            "srcA_spec/srcB_spec must match mix_spec shape, got "
            f"{tuple(srcA_spec.shape)} / {tuple(srcB_spec.shape)} vs {tuple(mix_spec.shape)}"
        )
    if srcA_time.ndim != 2 or srcB_time.ndim != 2:
        raise ValueError(
            f"srcA_time/srcB_time must be [B,N], got {tuple(srcA_time.shape)} / {tuple(srcB_time.shape)}"
        )
    if srcA_time.shape != srcB_time.shape:
        raise ValueError(
            f"srcA_time/srcB_time shape mismatch: {tuple(srcA_time.shape)} vs {tuple(srcB_time.shape)}"
        )
    if srcA_time.shape[0] != mix_feat.shape[0]:
        raise ValueError("Batch size mismatch between mix_feat and srcA_time.")


@torch.no_grad()
def evaluate_separator(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    cfg: Any,
    return_details: bool = False,
    max_detail_samples: int = 0,
) -> Dict[str, Any]:
    """评估分离模型并返回复相关系数指标。

    Args:
        model: 分离模型，输入特征图 `mix_feat`，shape `[B,3,F,T]`，输出 `[B,2,F,T]`。
        dataloader: 验证或测试 dataloader。
        device: 计算设备。
        cfg: 配置对象（供 iSTFT 参数与 window_len 使用）。
        return_details: 是否返回样本级详细结果。
        max_detail_samples: 详细结果最多保留的样本数（<=0 表示不限制）。

    Returns:
        Dict[str, Any]:
            - avg_corr_a: 源A平均复相关系数（归一化复相关幅值）
            - avg_corr_b: 源B平均复相关系数（归一化复相关幅值）
            - avg_corr_mean: (A+B)/2
            - num_samples: 样本数
            - details: 可选，样本级结果列表
    """
    model.eval()

    corr_a_values: List[float] = []
    corr_b_values: List[float] = []
    details: List[Dict[str, Any]] = []
    sample_counter = 0
    target_len = int(getattr(getattr(cfg, "data", cfg), "window_len"))

    for batch in dataloader:
        _require_eval_keys(batch)
        batch = _move_batch_to_device(batch, device)
        _validate_eval_shapes(batch)

        # 1) 模型推理预测 mask
        pred_mask = model(batch["mix_feat"])  # [B, 2, F, T]

        # 2) 按 mask 与 mix_spec 重建预测源谱
        pred_srcA_spec = pred_mask[:, 0, :, :] * batch["mix_spec"]  # [B, F, T]
        pred_srcB_spec = pred_mask[:, 1, :, :] * batch["mix_spec"]  # [B, F, T]

        # 3) iSTFT 恢复时域（复数）
        pred_srcA_time = istft_reconstruct(pred_srcA_spec, cfg=cfg, length=target_len)  # [B, N]
        pred_srcB_time = istft_reconstruct(pred_srcB_spec, cfg=cfg, length=target_len)  # [B, N]

        # 4) 与真实时域信号计算复相关系数
        bsz = pred_srcA_time.shape[0]
        for i in range(bsz):
            corr_a = complex_corr(pred_srcA_time[i], batch["srcA_time"][i])
            corr_b = complex_corr(pred_srcB_time[i], batch["srcB_time"][i])
            corr_a_values.append(corr_a)
            corr_b_values.append(corr_b)

            if return_details:
                if max_detail_samples <= 0 or len(details) < max_detail_samples:
                    details.append(
                        {
                            "sample_idx": sample_counter,
                            "corr_a": corr_a,
                            "corr_b": corr_b,
                            "corr_mean": 0.5 * (corr_a + corr_b),
                        }
                    )
            sample_counter += 1

    if sample_counter == 0:
        raise ValueError("Dataloader is empty: no sample evaluated.")

    avg_corr_a = float(sum(corr_a_values) / len(corr_a_values))
    avg_corr_b = float(sum(corr_b_values) / len(corr_b_values))
    result: Dict[str, Any] = {
        "avg_corr_a": avg_corr_a,
        "avg_corr_b": avg_corr_b,
        "avg_corr_mean": 0.5 * (avg_corr_a + avg_corr_b),
        "num_samples": int(sample_counter),
    }
    if return_details:
        result["details"] = details
    return result
