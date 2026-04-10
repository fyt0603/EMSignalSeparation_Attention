"""Transformer 前向链路调试脚本。

用途：
- 验证 Dataset/DataLoader 是否能提供可用 batch。
- 验证 TransformerSeparator 前向输出是否为 `[B, 2, F, T]`。
- 验证输出是否可直接送入 SeparationLoss 计算。

运行方式（项目根目录）：
- python -m scripts.debug_transformer_forward
- python scripts/debug_transformer_forward.py
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


def _move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def _print_batch_info(batch: Dict[str, Any]) -> None:
    print("\n=== Batch Fields ===")
    for key in sorted(batch.keys()):
        value = batch[key]
        if isinstance(value, torch.Tensor):
            print(
                f"{key:<12} shape={tuple(value.shape)!s:<18} "
                f"dtype={str(value.dtype):<16} is_complex={torch.is_complex(value)}"
            )
        elif isinstance(value, (list, tuple)):
            print(f"{key:<12} type={type(value).__name__:<10} len={len(value)}")
        else:
            print(f"{key:<12} type={type(value).__name__:<10} value={value}")


def main() -> None:
    project_root = _ensure_project_root_in_syspath()

    from configs.config import get_default_config
    from data.dataset import DroneSeparationDataset
    from losses.separation_loss import SeparationLoss
    from models.transformer import TransformerSeparator

    cfg = get_default_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if len(cfg.file_split.drone_codes) < 2:
        raise ValueError("cfg.file_split.drone_codes must contain at least two codes.")
    source_a_code = cfg.file_split.drone_codes[0]
    source_b_code = cfg.file_split.drone_codes[1]
    sir_db = 0.0

    index_path = cfg.paths.outputs_dir / "indexes" / "train_index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Train index not found: {index_path}")

    print("=== Debug Transformer Forward ===")
    print(f"project_root : {project_root}")
    print(f"device       : {device}")
    print(f"index_path   : {index_path}")
    print(f"source A/B   : {source_a_code} / {source_b_code}")
    print(f"sir_db       : {sir_db}")

    dataset = DroneSeparationDataset(
        split="train",
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
    batch = _move_batch_to_device(batch, device)

    model = TransformerSeparator(
        in_channels=1,
        out_masks=2,
        embed_dim=cfg.model.d_model,
        depth=cfg.model.num_layers,
        num_heads=cfg.model.n_heads,
        ff_dim=cfg.model.ff_dim,
        dropout=cfg.model.dropout,
        patch_size=cfg.model.patch_size,
    ).to(device)
    criterion = SeparationLoss(mask_loss_weight=cfg.loss.mask_loss_weight).to(device)

    model.eval()
    with torch.no_grad():
        pred_mask = model(batch["mix_mag"])

    print("\n=== Forward Output ===")
    print(
        f"pred_mask    shape={tuple(pred_mask.shape)}, dtype={pred_mask.dtype}, "
        f"is_complex={torch.is_complex(pred_mask)}"
    )

    if pred_mask.ndim != 4 or pred_mask.shape[1] != 2:
        raise ValueError(f"pred_mask shape invalid: {tuple(pred_mask.shape)}")

    # softmax 维度一致性简单检查：两个源通道在某个位置的和应接近 1
    channel_sum = pred_mask[0, :, 0, 0].sum().item()
    print(f"softmax_check pred_mask[0,:,0,0].sum() = {channel_sum:.6f} (expect ~1.0)")

    loss_dict = criterion(
        pred_mask=pred_mask,
        target_mask=batch["mask_target"],
        mix_spec=batch["mix_spec"],
        srcA_spec=batch["srcA_spec"],
        srcB_spec=batch["srcB_spec"],
    )

    total_loss = float(loss_dict["total_loss"].detach().item())
    mag_loss = float(loss_dict["mag_loss"].detach().item())
    mask_loss = float(loss_dict["mask_loss"].detach().item())

    print("\n=== Loss ===")
    print(f"total_loss   = {total_loss:.6f}")
    print(f"mag_loss     = {mag_loss:.6f}")
    print(f"mask_loss    = {mask_loss:.6f}")
    print("\nForward chain status: SUCCESS (dataset -> model -> loss).")


if __name__ == "__main__":
    main()

