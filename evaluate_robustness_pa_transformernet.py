"""PA-TransformerNet 鲁棒性测试脚本（毕业论文 5.2.3）。

功能：
- 在不同混合强度（sir_db）条件下评估同一个已训练模型 checkpoint。
- 复用项目现有的数据加载、推理、频谱重建和相关系数计算逻辑。
- 输出每个 sir_db 的 corr_a / corr_b / corr_mean / corr_strong / corr_weak。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader


def _ensure_project_root_in_syspath() -> Path:
    """兼容 `python evaluate_robustness_pa_transformernet.py` 运行方式。"""
    project_root = Path(__file__).resolve().parent
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return project_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate robustness of PA-TransformerNet")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/checkpoints/s0_best_transformer.pt",
        help="Path to trained PA-TransformerNet checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/robustness_pa_transformernet.json",
        help="Output json path",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override test batch size (default: cfg.train.batch_size)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device, e.g. cuda/cpu (default: auto choose cuda if available)",
    )
    return parser.parse_args()


def build_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(cfg: Any, device: torch.device) -> torch.nn.Module:
    """构建 PA-TransformerNet（TransformerSeparator）。"""
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
    return model.to(device)


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif isinstance(ckpt, dict):
        # 兼容直接保存 state_dict 的情形
        model.load_state_dict(ckpt)
    else:
        raise TypeError(f"Unsupported checkpoint format: {type(ckpt)}")


def build_test_loader(cfg: Any, sir_db: float, batch_size: int) -> DataLoader:
    from data.dataset import DroneSeparationDataset

    if len(cfg.file_split.drone_codes) < 2:
        raise ValueError("cfg.file_split.drone_codes must contain at least two codes.")
    source_a_code = cfg.file_split.drone_codes[0]
    source_b_code = cfg.file_split.drone_codes[1]

    test_index_path = cfg.paths.outputs_dir / "indexes" / "test_index.json"
    if not test_index_path.exists():
        raise FileNotFoundError(
            f"Test index not found: {test_index_path}. Please run data/build_index.py first."
        )

    dataset = DroneSeparationDataset(
        split="test",
        index_json_path=str(test_index_path),
        source_a_code=source_a_code,
        source_b_code=source_b_code,
        sir_db=float(sir_db),
        cfg=cfg,
        seed=42,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return loader


def evaluate_one_sir(
    model: torch.nn.Module,
    cfg: Any,
    device: torch.device,
    sir_db: float,
    batch_size: int,
) -> Dict[str, Any]:
    """评估单个 sir_db 条件并返回汇总结果。"""
    from engine.evaluator import evaluate_separator

    loader = build_test_loader(cfg=cfg, sir_db=sir_db, batch_size=batch_size)
    metrics = evaluate_separator(
        model=model,
        dataloader=loader,
        device=device,
        cfg=cfg,
        return_details=False,
    )

    corr_a = float(metrics["avg_corr_a"])
    corr_b = float(metrics["avg_corr_b"])
    corr_mean = float(metrics["avg_corr_mean"])

    # 强弱源判定依据数据集定义：
    # 数据集中采用 SIR = 10*log10(PA/PB)（见 data/dataset.py 注释），
    # 因此 sir_db > 0 表示 A 更强，sir_db < 0 表示 B 更强。
    if sir_db > 0:
        corr_strong = corr_a
        corr_weak = corr_b
    elif sir_db < 0:
        corr_strong = corr_b
        corr_weak = corr_a
    else:
        corr_strong = corr_mean
        corr_weak = corr_mean

    return {
        "model": "PA-TransformerNet",
        "sir_db": float(sir_db),
        "corr_a": corr_a,
        "corr_b": corr_b,
        "corr_mean": corr_mean,
        "corr_strong": float(corr_strong),
        "corr_weak": float(corr_weak),
        "num_samples": int(metrics["num_samples"]),
    }


def main() -> None:
    _ensure_project_root_in_syspath()

    from configs.config import get_default_config

    args = parse_args()
    cfg = get_default_config()
    device = build_device(args.device)
    batch_size = int(args.batch_size) if args.batch_size is not None else int(cfg.train.batch_size)

    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)
    sir_values = [-12, -9, -6, -3, 0, 3, 6, 9, 12]

    model = build_model(cfg=cfg, device=device)
    load_checkpoint(model=model, checkpoint_path=checkpoint_path, device=device)
    model.eval()

    print("=== PA-TransformerNet Robustness Evaluation ===")
    print(f"checkpoint : {checkpoint_path}")
    print(f"device     : {device}")
    print(f"batch_size : {batch_size}")
    print(f"mask_type  : {cfg.model.mask_type}")
    print(f"mask_bound : {cfg.model.mask_bound}")

    results: List[Dict[str, Any]] = []
    for sir_db in sir_values:
        row = evaluate_one_sir(
            model=model,
            cfg=cfg,
            device=device,
            sir_db=float(sir_db),
            batch_size=batch_size,
        )
        results.append(row)
        print(
            f"sir_db={sir_db:>4} dB | "
            f"corr_a={row['corr_a']:.6f}, corr_b={row['corr_b']:.6f}, "
            f"corr_mean={row['corr_mean']:.6f}, "
            f"corr_strong={row['corr_strong']:.6f}, corr_weak={row['corr_weak']:.6f}, "
            f"num_samples={row['num_samples']}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"saved_json : {output_path}")


if __name__ == "__main__":
    main()

