"""在线双源混合数据集。

职责：
- 从 build_index.py 生成的窗口索引 JSON 中读取窗口元数据。
- 按 drone_code 构建 source A/B 窗口池并进行在线混合。
- 返回训练所需时域、频域及监督目标（IRM）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from configs.config import ExperimentConfig, get_default_config
from data.io_utils import read_complex_rf0
from data.stft_utils import compute_stft, spec_to_logmag

Split = Literal["train", "val", "test"]


def _load_index_items(index_json_path: Path) -> List[Dict[str, Any]]:
    if not index_json_path.exists():
        raise FileNotFoundError(f"Index json not found: {index_json_path}")
    with index_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Index json must be a list, got: {type(data)}")
    return data


def _validate_item(item: Dict[str, Any]) -> None:
    required = ["split", "drone_code", "file_path", "start", "length"]
    missing = [k for k in required if k not in item]
    if missing:
        raise KeyError(f"Index item missing keys: {missing}")


def _normalize_complex(sig: np.ndarray, eps: float) -> np.ndarray:
    """复信号 RMS 归一化。"""
    power = float(np.mean(np.abs(sig) ** 2))
    scale = np.sqrt(power + eps)
    return (sig / scale).astype(np.complex64, copy=False)


def _sir_to_scales(sir_db: float) -> Tuple[float, float]:
    """将 SIR(dB) 转为幅度缩放系数。

    定义：
    - SIR = 10*log10(PA/PB)
    - 若 A/B 先做单位 RMS 归一化，则幅度比 alpha/beta = 10^(sir_db/20)
    """
    ratio_amp = float(10.0 ** (sir_db / 20.0))
    alpha = ratio_amp
    beta = 1.0
    return alpha, beta


def _build_irm(src_a_spec: torch.Tensor, src_b_spec: torch.Tensor, eps: float) -> torch.Tensor:
    """基于缩放后真实源频谱构建 ideal ratio mask。"""
    mag_a = torch.abs(src_a_spec).to(torch.float32)
    mag_b = torch.abs(src_b_spec).to(torch.float32)
    denom = mag_a + mag_b + float(eps)
    mask_a = mag_a / denom
    mask_b = mag_b / denom
    return torch.stack([mask_a, mask_b], dim=0)  # [2, F, T]


class DroneSeparationDataset(Dataset):
    """DroneRFa 双源分离 Dataset（在线混合）。"""

    def __init__(
        self,
        split: Split,
        index_json_path: str,
        source_a_code: Optional[str] = None,
        source_b_code: Optional[str] = None,
        sir_db: float = 0.0,
        cfg: Optional[ExperimentConfig] = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.split = split
        self.index_json_path = Path(index_json_path)
        self.cfg = cfg if cfg is not None else get_default_config()
        self.sir_db = float(sir_db)
        self.eps = float(self.cfg.numeric.eps)
        self.rng = np.random.default_rng(seed)

        raw_items = _load_index_items(self.index_json_path)
        self.items: List[Dict[str, Any]] = []
        for item in raw_items:
            _validate_item(item)
            # 兼容 split 索引文件和全量索引文件两种情况
            if str(item["split"]).lower() == split:
                self.items.append(item)

        if len(self.items) == 0:
            raise ValueError(f"No index items found for split='{split}' in {self.index_json_path}")

        codes = sorted({str(it["drone_code"]).upper() for it in self.items})
        if len(codes) < 2:
            raise ValueError(f"Need at least 2 drone_code in split='{split}', got {codes}")
        # 选择无人机类别
        self.source_a_code = source_a_code.upper() if source_a_code else codes[0]
        self.source_b_code = source_b_code.upper() if source_b_code else codes[1]
        if self.source_a_code == self.source_b_code:
            raise ValueError("source_a_code and source_b_code must be different.")
        # 构建A池和B池
        self.pool_a = self._build_pool(self.source_a_code)
        self.pool_b = self._build_pool(self.source_b_code)
        if len(self.pool_a) == 0 or len(self.pool_b) == 0:
            raise ValueError(
                f"Empty source pool: {self.source_a_code}={len(self.pool_a)}, "
                f"{self.source_b_code}={len(self.pool_b)}"
            )

        # 验证/测试固定配对：排序后按 index 对齐，长度取两池最小值
        self.fixed_pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        if self.split in ("val", "test"):
            a_sorted = sorted(self.pool_a, key=self._pair_sort_key)
            b_sorted = sorted(self.pool_b, key=self._pair_sort_key)
            n = min(len(a_sorted), len(b_sorted))
            self.fixed_pairs = [(a_sorted[i], b_sorted[i]) for i in range(n)]

    def _build_pool(self, drone_code: str) -> List[Dict[str, Any]]:
        return [it for it in self.items if str(it["drone_code"]).upper() == drone_code]

    @staticmethod
    def _pair_sort_key(item: Dict[str, Any]) -> Tuple[str, int]:
        return (str(item["file_path"]), int(item["start"]))

    def __len__(self) -> int:
        if self.split == "train":
            # 训练集不截断到最小池长度；使用较大池长度作为1个epoch中的样本数
            return max(len(self.pool_a), len(self.pool_b))
        return len(self.fixed_pairs)

    def _read_window(self, item: Dict[str, Any]) -> np.ndarray:
        return read_complex_rf0(
            mat_path=str(item["file_path"]),
            start=int(item["start"]),
            length=int(item["length"]),
        )

    def _sample_pair(self, index: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if self.split == "train":
            item_a = self.pool_a[int(self.rng.integers(0, len(self.pool_a)))]
            item_b = self.pool_b[int(self.rng.integers(0, len(self.pool_b)))]
            return item_a, item_b
        return self.fixed_pairs[index]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item_a, item_b = self._sample_pair(index)

        # 1) 读取纯净窗口
        src_a = self._read_window(item_a)  # [N], complex64
        src_b = self._read_window(item_b)  # [N], complex64

        # 2) 归一化
        src_a_n = _normalize_complex(src_a, eps=self.eps)
        src_b_n = _normalize_complex(src_b, eps=self.eps)

        # 3) 固定 sir_db 缩放并混合
        alpha, beta = _sir_to_scales(self.sir_db)
        src_a_scaled = (alpha * src_a_n).astype(np.complex64, copy=False)
        src_b_scaled = (beta * src_b_n).astype(np.complex64, copy=False)
        mix = (src_a_scaled + src_b_scaled).astype(np.complex64, copy=False)

        # 4) STFT（复数谱 [F, T]）
        mix_spec = compute_stft(mix, self.cfg)
        src_a_spec = compute_stft(src_a_scaled, self.cfg)
        src_b_spec = compute_stft(src_b_scaled, self.cfg)

        # 5) 输入特征（对数幅度谱 [1, F, T]）
        mix_mag = spec_to_logmag(mix_spec, eps=self.eps)

        # 6) 监督目标 IRM（基于缩放后真实源频谱）
        mask_target = _build_irm(src_a_spec, src_b_spec, eps=self.eps)  # [2, F, T]

        return {
            "mix_mag": mix_mag.to(torch.float32),                 # [1, F, T]
            "mix_spec": mix_spec.to(torch.complex64),             # [F, T]
            "srcA_spec": src_a_spec.to(torch.complex64),          # [F, T]
            "srcB_spec": src_b_spec.to(torch.complex64),          # [F, T]
            "mask_target": mask_target.to(torch.float32),         # [2, F, T]
            "srcA_time": torch.from_numpy(src_a_scaled),          # [N], complex64
            "srcB_time": torch.from_numpy(src_b_scaled),          # [N], complex64
            "sir_db": torch.tensor(self.sir_db, dtype=torch.float32),
            "source_a_code": self.source_a_code,
            "source_b_code": self.source_b_code,
        }
