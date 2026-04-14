"""LSTM 单 batch 训练链路调试脚本。

目的：
- 验证 Dataset/DataLoader、LSTM 前向、SeparationLoss、反向传播与参数更新是否打通。
- 固定同一个 batch 重复训练若干步，观察 loss 是否下降。

运行方式（项目根目录）：
- python -m scripts.debug_train_one_batch_lstm
- python scripts/debug_train_one_batch_lstm.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader


def _ensure_project_root_in_syspath() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return project_root


def _print_batch_info(batch: Dict[str, Any]) -> None:
    print("\n=== Batch Field Info ===")
    for key in sorted(batch.keys()):
        value = batch[key]
        if isinstance(value, torch.Tensor):
            print(
                f"{key:<12} shape={tuple(value.shape)!s:<18} "
                f"dtype={str(value.dtype):<16} is_complex={torch.is_complex(value)}"
            )
        elif isinstance(value, (list, tuple)):
            sample = value[0] if len(value) > 0 else None
            print(f"{key:<12} type={type(value).__name__:<10} len={len(value)} sample={sample}")
        else:
            print(f"{key:<12} type={type(value).__name__:<10} value={value}")


def _to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        out[k] = v.to(device) if isinstance(v, torch.Tensor) else v
    return out


def main() -> None:
    project_root = _ensure_project_root_in_syspath()

    from configs.config import get_default_config
    from data.dataset import DroneSeparationDataset
    from losses.separation_loss import SeparationLoss
    from models.lstm import LSTMSeparator

    cfg = get_default_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=== Debug One Batch LSTM Training ===")
    print(f"project_root: {project_root}")
    print(f"device      : {device}")

    if len(cfg.file_split.drone_codes) < 2:
        raise ValueError("Need at least two drone codes in cfg.file_split.drone_codes.")

    split = "train"
    sir_db = cfg.train.sir_db
    source_a_code = cfg.file_split.drone_codes[0]
    source_b_code = cfg.file_split.drone_codes[1]
    index_path = cfg.paths.outputs_dir / "indexes" / "train_index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")

    dataset = DroneSeparationDataset(
        split=split,
        index_json_path=str(index_path),
        source_a_code=source_a_code,
        source_b_code=source_b_code,
        sir_db=sir_db,
        cfg=cfg,
        seed=42,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    batch = next(iter(loader))
    _print_batch_info(batch)
    batch = _to_device(batch, device)

    model = LSTMSeparator(
        in_channels=1,
        out_masks=2,
        input_freq_bins=cfg.stft.n_fft,
        hidden_size=cfg.lstm.hidden_size,
        num_layers=cfg.lstm.num_layers,
        bidirectional=cfg.lstm.bidirectional,
        dropout=cfg.lstm.dropout,
    ).to(device)
    criterion = SeparationLoss(
        mag_loss_weight=cfg.loss.mag_loss_weight,
        mask_loss_weight=cfg.loss.mask_loss_weight,
        corr_loss_weight=cfg.loss.corr_loss_weight,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )

    steps = 200
    print_every = 10
    initial_loss: float = -1.0
    final_loss: float = -1.0

    model.train()
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)

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
        total_loss = loss_dict["total_loss"]
        total_loss.backward()
        optimizer.step()

        loss_val = float(total_loss.detach().item())
        if step == 1:
            initial_loss = loss_val
        if step % print_every == 0 or step == 1 or step == steps:
            print(
                f"step={step:03d}  "
                f"total={loss_val:.6f}  "
                f"mag={float(loss_dict['mag_loss'].detach().item()):.6f}  "
                f"mask={float(loss_dict['mask_loss'].detach().item()):.6f}  "
                f"corr={float(loss_dict['corr_loss'].detach().item()):.6f}"
            )
        final_loss = loss_val

    print("\n=== Loss Summary ===")
    print(f"initial_loss : {initial_loss:.6f}")
    print(f"final_loss   : {final_loss:.6f}")
    print(f"loss_decrease: {final_loss < initial_loss}")


if __name__ == "__main__":
    main()
