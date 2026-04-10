"""Dataset 单样本调试脚本。

用途：
- 加载 `DroneSeparationDataset`。
- 读取一条样本并打印全部字段信息（类型/shape/dtype/值摘要）。
- 快速核对后续模型输入输出维度是否正确。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _ensure_project_root_in_syspath() -> Path:
    """确保脚本可直接运行（python scripts/debug_dataset_sample.py）。"""
    project_root = Path(__file__).resolve().parents[1]
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return project_root


def _describe_value(key: str, value: Any) -> None:
    key_head = f"[{key}]"
    py_type = type(value).__name__

    if isinstance(value, torch.Tensor):
        print(
            f"{key_head:<18} type={py_type:<14} shape={tuple(value.shape)!s:<16} "
            f"dtype={str(value.dtype):<16} is_complex={torch.is_complex(value)}"
        )
        return

    if isinstance(value, np.ndarray):
        print(
            f"{key_head:<18} type={py_type:<14} shape={value.shape!s:<16} "
            f"dtype={str(value.dtype):<16} is_complex={np.iscomplexobj(value)}"
        )
        return

    if isinstance(value, (list, tuple, set)):
        print(f"{key_head:<18} type={py_type:<14} len={len(value)}")
        return

    if isinstance(value, dict):
        print(f"{key_head:<18} type={py_type:<14} len={len(value)}")
        return

    print(f"{key_head:<18} type={py_type:<14} value={value}")


def main() -> None:
    project_root = _ensure_project_root_in_syspath()

    from configs.config import get_default_config
    from data.dataset import DroneSeparationDataset

    cfg = get_default_config()
    split = "test"
    index_path = cfg.paths.outputs_dir / "indexes" / f"{split}_index.json"

    if len(cfg.file_split.drone_codes) < 2:
        raise ValueError("Need at least two drone_codes in config.file_split.drone_codes.")

    source_a_code = cfg.file_split.drone_codes[0]
    source_b_code = cfg.file_split.drone_codes[1]
    sir_db = 0.0

    print("=== Debug Dataset Sample ===")
    print(f"project_root : {project_root}")
    print(f"index_json   : {index_path}")
    print(f"split        : {split}")
    print(f"source A/B   : {source_a_code} / {source_b_code}")
    print(f"sir_db       : {sir_db}")

    dataset = DroneSeparationDataset(
        split=split,
        index_json_path=str(index_path),
        source_a_code=source_a_code,
        source_b_code=source_b_code,
        sir_db=sir_db,
        cfg=cfg,
        seed=42,
    )

    print(f"dataset_len  : {len(dataset)}")
    sample = dataset[0]

    print("\n--- Sample Fields ---")
    for k in sorted(sample.keys()):
        _describe_value(k, sample[k])

    print("\n--- Key Tensor Summary ---")
    for key in [
        "mix_mag",
        "mix_spec",
        "srcA_spec",
        "srcB_spec",
        "mask_target",
        "srcA_time",
        "srcB_time",
    ]:
        v = sample.get(key, None)
        if isinstance(v, torch.Tensor):
            print(f"{key:<12} shape={tuple(v.shape)}, dtype={v.dtype}")
        elif isinstance(v, np.ndarray):
            print(f"{key:<12} shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"{key:<12} value={v}")


if __name__ == "__main__":
    main()

# === Debug Dataset Sample ===
# project_root : D:\CODE\EMSignalSeparation_Attention
# index_json   : D:\CODE\EMSignalSeparation_Attention\outputs\indexes\train_index.json
# split        : train
# source A/B   : T0010 / T0101
# sir_db       : 0.0
# dataset_len  : 11856

# --- Sample Fields ---
# [mask_target]      type=Tensor         shape=(2, 512, 255)    dtype=torch.float32    is_complex=False
# [mix_mag]          type=Tensor         shape=(1, 512, 255)    dtype=torch.float32    is_complex=False
# [mix_spec]         type=Tensor         shape=(512, 255)       dtype=torch.complex64  is_complex=True
# [sir_db]           type=Tensor         shape=()               dtype=torch.float32    is_complex=False
# [source_a_code]    type=str            value=T0010
# [source_b_code]    type=str            value=T0101
# [srcA_spec]        type=Tensor         shape=(512, 255)       dtype=torch.complex64  is_complex=True
# [srcA_time]        type=Tensor         shape=(65536,)         dtype=torch.complex64  is_complex=True
# [srcB_spec]        type=Tensor         shape=(512, 255)       dtype=torch.complex64  is_complex=True
# [srcB_time]        type=Tensor         shape=(65536,)         dtype=torch.complex64  is_complex=True

# --- Key Tensor Summary ---
# mix_mag      shape=(1, 512, 255), dtype=torch.float32
# mix_spec     shape=(512, 255), dtype=torch.complex64
# srcA_spec    shape=(512, 255), dtype=torch.complex64
# srcB_spec    shape=(512, 255), dtype=torch.complex64
# mask_target  shape=(2, 512, 255), dtype=torch.float32
# srcA_time    shape=(65536,), dtype=torch.complex64
# srcB_time    shape=(65536,), dtype=torch.complex64


# === Debug Dataset Sample ===
# project_root : D:\CODE\EMSignalSeparation_Attention
# index_json   : D:\CODE\EMSignalSeparation_Attention\outputs\indexes\val_index.json
# split        : val
# source A/B   : T0010 / T0101
# sir_db       : 0.0
# dataset_len  : 2280

# --- Sample Fields ---
# [mask_target]      type=Tensor         shape=(2, 512, 255)    dtype=torch.float32    is_complex=False
# [mix_mag]          type=Tensor         shape=(1, 512, 255)    dtype=torch.float32    is_complex=False
# [mix_spec]         type=Tensor         shape=(512, 255)       dtype=torch.complex64  is_complex=True
# [sir_db]           type=Tensor         shape=()               dtype=torch.float32    is_complex=False
# [source_a_code]    type=str            value=T0010
# [source_b_code]    type=str            value=T0101
# [srcA_spec]        type=Tensor         shape=(512, 255)       dtype=torch.complex64  is_complex=True
# [srcA_time]        type=Tensor         shape=(65536,)         dtype=torch.complex64  is_complex=True
# [srcB_spec]        type=Tensor         shape=(512, 255)       dtype=torch.complex64  is_complex=True
# [srcB_time]        type=Tensor         shape=(65536,)         dtype=torch.complex64  is_complex=True

# --- Key Tensor Summary ---
# mix_mag      shape=(1, 512, 255), dtype=torch.float32
# mix_spec     shape=(512, 255), dtype=torch.complex64
# srcA_spec    shape=(512, 255), dtype=torch.complex64
# srcB_spec    shape=(512, 255), dtype=torch.complex64
# mask_target  shape=(2, 512, 255), dtype=torch.float32
# srcA_time    shape=(65536,), dtype=torch.complex64
# srcB_time    shape=(65536,), dtype=torch.complex64


# === Debug Dataset Sample ===
# project_root : D:\CODE\EMSignalSeparation_Attention
# index_json   : D:\CODE\EMSignalSeparation_Attention\outputs\indexes\test_index.json
# split        : test
# source A/B   : T0010 / T0101
# sir_db       : 0.0
# dataset_len  : 4560

# --- Sample Fields ---
# [mask_target]      type=Tensor         shape=(2, 512, 255)    dtype=torch.float32    is_complex=False
# [mix_mag]          type=Tensor         shape=(1, 512, 255)    dtype=torch.float32    is_complex=False
# [mix_spec]         type=Tensor         shape=(512, 255)       dtype=torch.complex64  is_complex=True
# [sir_db]           type=Tensor         shape=()               dtype=torch.float32    is_complex=False
# [source_a_code]    type=str            value=T0010
# [source_b_code]    type=str            value=T0101
# [srcA_spec]        type=Tensor         shape=(512, 255)       dtype=torch.complex64  is_complex=True
# [srcA_time]        type=Tensor         shape=(65536,)         dtype=torch.complex64  is_complex=True
# [srcB_spec]        type=Tensor         shape=(512, 255)       dtype=torch.complex64  is_complex=True
# [srcB_time]        type=Tensor         shape=(65536,)         dtype=torch.complex64  is_complex=True

# --- Key Tensor Summary ---
# mix_mag      shape=(1, 512, 255), dtype=torch.float32
# mix_spec     shape=(512, 255), dtype=torch.complex64
# srcA_spec    shape=(512, 255), dtype=torch.complex64
# srcB_spec    shape=(512, 255), dtype=torch.complex64
# mask_target  shape=(2, 512, 255), dtype=torch.float32
# srcA_time    shape=(65536,), dtype=torch.complex64
# srcB_time    shape=(65536,), dtype=torch.complex64