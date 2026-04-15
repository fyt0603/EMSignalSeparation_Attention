"""项目配置定义。

说明：
- 使用 dataclass 组织路径、数据、STFT、模型、训练、损失等配置。
- train/val/test 文件列表暂为空列表，占位等待后续明确填充。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


def _default_project_root() -> Path:
    """推断项目根目录（configs/ 的上一级）。"""
    return Path(__file__).resolve().parents[1]


@dataclass
class PathConfig:
    """路径相关配置。"""

    project_root: Path = field(default_factory=_default_project_root)
    dataset_dir: Path = field(default_factory=lambda: _default_project_root() / "dataset")
    outputs_dir: Path = field(default_factory=lambda: _default_project_root() / "outputs")


@dataclass
class FileSplitConfig:
    """文件级数据划分配置（先按文件划分，再切窗）。

    设计目标：
    - 不手写完整文件名列表。
    - 通过 drone/distance/segment 规则自动生成 train/val/test 文件清单。
    - 保持对现有 `build_index.py` 的兼容（仍提供 train_files/val_files/test_files）。
    """

    drone_codes: List[str] = field(default_factory=lambda: ["T0010", "T0101"])
    distance_code: str = "D00"

    # 每个无人机代码都使用相同的 segment 划分规则。
    train_segments: List[str] = field(
        default_factory=lambda: ["S0000", "S0001", "S0010", "S0011", "S0100"]
    )
    val_segments: List[str] = field(default_factory=lambda: ["S0101"])
    test_segments: List[str] = field(default_factory=lambda: ["S0110", "S0111"])

    def _build_files(self, segments: List[str]) -> List[str]:
        files: List[str] = []
        for drone_code in self.drone_codes:
            for segment in segments:
                files.append(f"{drone_code}_{self.distance_code}_{segment}.mat")
        return files

    @property
    def train_files(self) -> List[str]:
        return self._build_files(self.train_segments)

    @property
    def val_files(self) -> List[str]:
        return self._build_files(self.val_segments)

    @property
    def test_files(self) -> List[str]:
        return self._build_files(self.test_segments)


@dataclass
class DataConfig:
    """数据处理核心参数。"""

    sample_rate: float = 100e6
    block_len: int = 10_000_000
    window_len: int = 65_536
    rf_i_key: str = "RF0_I"
    rf_q_key: str = "RF0_Q"


@dataclass
class STFTConfig:
    """STFT 参数配置。"""

    n_fft: int = 512
    win_length: int = 512
    hop_length: int = 256
    window: str = "hamming"
    center: bool = False
    onesided: bool = False


@dataclass
class ModelConfig:
    """Transformer 分离模型配置。"""

    # 输入为 [B, 1, F, T]，输出 mask 为 [B, 2, F, T]
    patch_size: int = 16
    d_model: int = 256
    n_heads: int = 8
    num_layers: int = 4
    ff_dim: int = 1024
    dropout: float = 0.1


@dataclass
class LSTMConfig:
    """LSTM 分离模型配置。"""

    hidden_size: int = 256
    num_layers: int = 2
    bidirectional: bool = True
    dropout: float = 0.1


@dataclass
class TrainConfig:
    """训练超参数配置。"""

    batch_size: int = 8
    learning_rate: float = 1e-4
    epochs: int = 50
    weight_decay: float = 1e-4
    sir_db: float = 0.0


@dataclass
class LossConfig:
    """损失配置。"""

    mag_loss_weight: float = 0.1
    mask_loss_weight: float = 2.0
    corr_loss_weight: float = 1.0


@dataclass
class NumericConfig:
    """数值稳定项配置。"""

    eps: float = 1e-8


@dataclass
class ExperimentConfig:
    """实验总配置聚合。"""

    paths: PathConfig = field(default_factory=PathConfig)
    file_split: FileSplitConfig = field(default_factory=FileSplitConfig)
    data: DataConfig = field(default_factory=DataConfig)
    stft: STFTConfig = field(default_factory=STFTConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    numeric: NumericConfig = field(default_factory=NumericConfig)


def get_default_config() -> ExperimentConfig:
    """返回默认实验配置。"""
    return ExperimentConfig() # 返回配置的实例
