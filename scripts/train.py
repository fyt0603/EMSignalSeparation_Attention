"""Transformer 正式训练入口脚本。

职责：
- 组织 train/val 数据加载、模型构建、epoch 训练与验证。
- 记录日志并保存最佳 checkpoint（按 val total_loss 最小）。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader


def _ensure_project_root_in_syspath() -> Path:
    """兼容 `python scripts/train.py` 运行方式。"""
    project_root = Path(__file__).resolve().parents[1]
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return project_root


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _dump_log(log_path: Path, history: List[Dict[str, Any]]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train separator")
    parser.add_argument("--device", type=str, default=None, help="Override device, e.g. cuda/cpu")
    parser.add_argument("--epochs", type=int, default=None, help="Override cfg.train.epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override cfg.train.batch_size")
    parser.add_argument("--sir_db", type=float, default=None, help="Override cfg.train.sir_db")
    parser.add_argument(
        "--mask_type",
        type=str,
        default=None,
        help="Mask mode override: magnitude or complex (default: use config)",
    )
    parser.add_argument(
        "--mask_bound",
        type=float,
        default=None,
        help="Complex mask bound for mask_bound * tanh(raw_output) (default: use config)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        choices=["transformer", "cnn"],
        help="Override model name",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def _resolve_model_name(args: argparse.Namespace) -> str:
    if args.model_name is not None:
        return str(args.model_name)
    return "transformer"


def _build_model(model_name: str, cfg: Any, device: torch.device) -> torch.nn.Module:
    if model_name != "transformer" and cfg.model.mask_type == "complex":
        raise ValueError("complex mask is currently only supported for transformer model.")

    if model_name == "transformer":
        from models.transformer import TransformerSeparator

        out_masks = 4 if cfg.model.mask_type == "complex" else 2
        model = TransformerSeparator(
            in_channels=3,  # mix_feat: [logmag, sin_phi, cos_phi]
            out_masks=out_masks,
            mask_type=cfg.model.mask_type,
            mask_bound=cfg.model.mask_bound,
            embed_dim=cfg.model.d_model,
            depth=cfg.model.num_layers,
            num_heads=cfg.model.n_heads,
            ff_dim=cfg.model.ff_dim,
            dropout=cfg.model.dropout,
            patch_size=cfg.model.patch_size,
        )
    elif model_name == "cnn":
        from models.cnn import CNNSeparator

        model = CNNSeparator(
            in_channels=3,
            out_masks=2,
            embed_dim=cfg.model.d_model,
            depth=2,
            dropout=cfg.model.dropout,
            patch_size=cfg.model.patch_size,
            use_cnn_skip=True,
        )
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")
    return model.to(device)


def main() -> None:
    _ensure_project_root_in_syspath()

    from configs.config import get_default_config
    from data.dataset import DroneSeparationDataset
    from engine.trainer import train_one_epoch, validate_one_epoch
    from losses.complex_mask_separation_loss import ComplexMaskSeparationLoss
    from losses.separation_loss import SeparationLoss
    args = parse_args()
    cfg = get_default_config()
    if args.mask_type is not None:
        if args.mask_type not in ("magnitude", "complex"):
            raise ValueError(f"Unsupported mask_type: {args.mask_type}")
        cfg.model.mask_type = args.mask_type
    if args.mask_bound is not None:
        cfg.model.mask_bound = float(args.mask_bound)
    _set_seed(args.seed)

    device = _build_device(args.device)
    model_name = _resolve_model_name(args)
    epochs = int(args.epochs if args.epochs is not None else cfg.train.epochs)
    batch_size = int(args.batch_size if args.batch_size is not None else cfg.train.batch_size)
    sir_db = float(args.sir_db if args.sir_db is not None else cfg.train.sir_db)

    if len(cfg.file_split.drone_codes) < 2:
        raise ValueError("cfg.file_split.drone_codes must contain at least two codes.")
    source_a_code = cfg.file_split.drone_codes[0]
    source_b_code = cfg.file_split.drone_codes[1]

    train_index_path = cfg.paths.outputs_dir / "indexes" / "train_index.json"
    val_index_path = cfg.paths.outputs_dir / "indexes" / "val_index.json"
    if not train_index_path.exists() or not val_index_path.exists():
        raise FileNotFoundError(
            "Missing train/val index json. Please run data/build_index.py first."
        )

    train_dataset = DroneSeparationDataset(
        split="train",
        index_json_path=str(train_index_path),
        source_a_code=source_a_code,
        source_b_code=source_b_code,
        sir_db=sir_db,
        cfg=cfg,
        seed=args.seed,
    )
    val_dataset = DroneSeparationDataset(
        split="val",
        index_json_path=str(val_index_path),
        source_a_code=source_a_code,
        source_b_code=source_b_code,
        sir_db=sir_db,
        cfg=cfg,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = _build_model(model_name=model_name, cfg=cfg, device=device)
    if cfg.model.mask_type == "complex":
        criterion = ComplexMaskSeparationLoss(
            mag_loss_weight=cfg.loss.mag_loss_weight,
            mask_loss_weight=cfg.loss.mask_loss_weight,
            corr_loss_weight=cfg.loss.corr_loss_weight,
        ).to(device)
    else:
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

    ckpt_dir = cfg.paths.outputs_dir / "checkpoints"
    log_dir = cfg.paths.outputs_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    best_val_total = float("inf")
    history: List[Dict[str, Any]] = []
    history_filename = f"train_history_{model_name}_{cfg.model.mask_type}.json"
    best_ckpt_filename = f"best_{model_name}_{cfg.model.mask_type}.pt"

    print("=== Train Start ===")
    print(f"model={model_name}")
    print(f"mask_type={cfg.model.mask_type}, mask_bound={cfg.model.mask_bound}")
    print(f"device={device}, epochs={epochs}, batch_size={batch_size}, sir_db={sir_db}")
    print(f"source_a={source_a_code}, source_b={source_b_code}")
    print(f"train_size={len(train_dataset)}, val_size={len(val_dataset)}")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            cfg=cfg,
        )
        val_loss = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            cfg=cfg,
        )

        record = {
            "epoch": epoch,
            "train_total_loss": train_loss["total_loss"],
            "train_mag_loss": train_loss["mag_loss"],
            "train_mask_loss": train_loss["mask_loss"],
            "train_corr_loss": train_loss["corr_loss"],
            "val_total_loss": val_loss["total_loss"],
            "val_mag_loss": val_loss["mag_loss"],
            "val_mask_loss": val_loss["mask_loss"],
            "val_corr_loss": val_loss["corr_loss"],
        }
        history.append(record)

        print(
            f"[Epoch {epoch:03d}/{epochs:03d}] "
            f"train(total={record['train_total_loss']:.6f}, "
            f"mag={record['train_mag_loss']:.6f}, mask={record['train_mask_loss']:.6f}, "
            f"corr={record['train_corr_loss']:.6f}) | "
            f"val(total={record['val_total_loss']:.6f}, "
            f"mag={record['val_mag_loss']:.6f}, mask={record['val_mask_loss']:.6f}, "
            f"corr={record['val_corr_loss']:.6f})"
        )

        if record["val_total_loss"] < best_val_total:
            best_val_total = record["val_total_loss"]
            best_ckpt_path = ckpt_dir / best_ckpt_filename
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_total_loss": best_val_total,
                    "source_a_code": source_a_code,
                    "source_b_code": source_b_code,
                    "sir_db": sir_db,
                },
                best_ckpt_path,
            )
            print(f"  -> New best checkpoint saved: {best_ckpt_path}")

        _dump_log(log_dir / history_filename, history)

    print("=== Train Done ===")
    print(f"best_val_total_loss={best_val_total:.6f}")
    print(f"log_path={log_dir / history_filename}")


if __name__ == "__main__":
    main()
