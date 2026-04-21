"""Transformer 测试入口脚本。

职责：
- 加载训练好的 checkpoint。
- 在 test split 上执行分离评估。
- 输出并可选保存评估结果。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader


def _ensure_project_root_in_syspath() -> Path:
    """兼容 `python scripts/test.py` 的导入路径。"""
    project_root = Path(__file__).resolve().parents[1]
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return project_root


def _build_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_model_name(args: argparse.Namespace) -> str:
    if args.model_name is not None:
        return str(args.model_name)
    return "transformer"


def _build_model(model_name: str, cfg: Any, device: torch.device) -> torch.nn.Module:
    if model_name == "resnet18":
        from models.resnet18 import ResNet18Separator

        model = ResNet18Separator(in_channels=1, out_masks=2)
    elif model_name == "transformer":
        from models.transformer import TransformerSeparator

        model = TransformerSeparator(
            in_channels=1,
            out_masks=2,
            embed_dim=cfg.model.d_model,
            depth=cfg.model.num_layers,
            num_heads=cfg.model.n_heads,
            ff_dim=cfg.model.ff_dim,
            dropout=cfg.model.dropout,
            patch_size=cfg.model.patch_size,
        )
    elif model_name == "lstm":
        from models.lstm import LSTMSeparator

        model = LSTMSeparator(
            in_channels=1,
            out_masks=2,
            input_freq_bins=cfg.stft.n_fft,
            hidden_size=cfg.lstm.hidden_size,
            num_layers=cfg.lstm.num_layers,
            bidirectional=cfg.lstm.bidirectional,
            dropout=cfg.lstm.dropout,
        )
    elif model_name == "unet":
        from models.unet import UNetSeparator

        model = UNetSeparator(
            in_channels=1,
            out_masks=2,
            base_channels=16,
        )
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")
    return model.to(device)


def _resolve_ckpt_path(cfg: Any, ckpt_arg: str, model_name: str) -> Path:
    if ckpt_arg:
        ckpt_path = Path(ckpt_arg)
    else:
        ckpt_path = cfg.paths.outputs_dir / "checkpoints" / f"best_{model_name}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return ckpt_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test separator")
    parser.add_argument("--ckpt", type=str, default="", help="Checkpoint path")
    parser.add_argument("--device", type=str, default=None, help="Override device, e.g. cuda/cpu")
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        choices=["resnet18", "transformer", "lstm", "unet"],
        help="Override model name",
    )
    parser.add_argument(
        "--detail_samples",
        type=int,
        default=0,
        help="If >0, keep up to this many sample-level details in output.",
    )
    return parser.parse_args()


def main() -> None:
    _ensure_project_root_in_syspath()

    from configs.config import get_default_config
    from data.dataset import DroneSeparationDataset
    from engine.evaluator import evaluate_separator

    args = parse_args()
    cfg = get_default_config()
    device = _build_device(args.device)
    model_name = _resolve_model_name(args)

    if len(cfg.file_split.drone_codes) < 2:
        raise ValueError("cfg.file_split.drone_codes must contain at least two codes.")
    source_a_code = cfg.file_split.drone_codes[0]
    source_b_code = cfg.file_split.drone_codes[1]
    sir_db = float(cfg.train.sir_db)

    test_index_path = cfg.paths.outputs_dir / "indexes" / "test_index.json"
    if not test_index_path.exists():
        raise FileNotFoundError(
            f"Test index not found: {test_index_path}. Please run build_index.py first."
        )

    test_dataset = DroneSeparationDataset(
        split="test",
        index_json_path=str(test_index_path),
        source_a_code=source_a_code,
        source_b_code=source_b_code,
        sir_db=sir_db,
        cfg=cfg,
        seed=42,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = _build_model(model_name=model_name, cfg=cfg, device=device)
    ckpt_path = _resolve_ckpt_path(cfg, args.ckpt, model_name=model_name)
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif isinstance(ckpt, dict):
        # 兼容直接保存 state_dict 的情形
        model.load_state_dict(ckpt)
    else:
        raise TypeError(f"Unsupported checkpoint format: {type(ckpt)}")

    result: Dict[str, Any] = evaluate_separator(
        model=model,
        dataloader=test_loader,
        device=device,
        cfg=cfg,
        return_details=args.detail_samples > 0,
        max_detail_samples=args.detail_samples,
    )

    print("=== Test Result ===")
    print(f"model         : {model_name}")
    print(f"checkpoint    : {ckpt_path}")
    print(f"device        : {device}")
    print(f"source_a_code : {source_a_code}")
    print(f"source_b_code : {source_b_code}")
    print(f"sir_db        : {sir_db}")
    print(f"avg_corr_a    : {result['avg_corr_a']:.6f}")
    print(f"avg_corr_b    : {result['avg_corr_b']:.6f}")
    print(f"avg_corr_mean : {result['avg_corr_mean']:.6f}")
    print(f"num_samples   : {result['num_samples']}")

    log_dir = cfg.paths.outputs_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    out_path = log_dir / f"test_result_{model_name}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"saved_json    : {out_path}")


if __name__ == "__main__":
    main()
