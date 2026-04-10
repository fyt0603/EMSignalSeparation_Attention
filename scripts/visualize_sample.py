"""样本可视化脚本。

- 模型：`TransformerSeparator`（`--model_name` 默认 transformer）
- checkpoint：`outputs/checkpoints/best_transformer.pt`（可用 `--ckpt` 覆盖）
- 训练日志：`outputs/logs/train_history_transformer.json`（可用 `--history_json` 覆盖）

可视化内容：
1) 五联时频谱图（均为对数幅度谱）
2) train/val 合并 loss 曲线图
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch


def _ensure_project_root_in_syspath() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return project_root


def _build_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_ckpt(cfg: Any, ckpt_arg: str) -> Path:
    if ckpt_arg:
        ckpt_path = Path(ckpt_arg)
    else:
        ckpt_path = cfg.paths.outputs_dir / "checkpoints" / "best_transformer.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return ckpt_path


def _resolve_history(cfg: Any, history_arg: str) -> Path:
    if history_arg:
        return Path(history_arg)
    return cfg.paths.outputs_dir / "logs" / "train_history_transformer.json"


def _build_model(model_name: str, cfg: Any, device: torch.device) -> torch.nn.Module:
    """模型创建与切换入口"""
    if model_name == "baseline":
        from models.baseline_separator import BaselineSeparator

        model = BaselineSeparator()
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
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")
    return model.to(device)


def _load_checkpoint(model: torch.nn.Module, ckpt_path: Path, device: torch.device) -> None:
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif isinstance(ckpt, dict):
        model.load_state_dict(ckpt)
    else:
        raise TypeError(f"Unsupported checkpoint format: {type(ckpt)}")


def _extract_loss_curves(history_path: Path) -> Dict[str, List[float]]:
    if not history_path.exists():
        return {"train": [], "val": []}
    with history_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return {"train": [], "val": []}
    train_curve: List[float] = []
    val_curve: List[float] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if "train_total_loss" in item:
            train_curve.append(float(item["train_total_loss"]))
        if "val_total_loss" in item:
            val_curve.append(float(item["val_total_loss"]))
    return {"train": train_curve, "val": val_curve}


def _plot_and_save_combined_loss_curve(
    out_path: Path,
    train_curve: List[float],
    val_curve: List[float],
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_title("Train & Val loss", fontsize=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.3)

    has_any = False
    if len(train_curve) > 0:
        ax.plot(np.arange(1, len(train_curve) + 1), train_curve, linewidth=1.8, label="train")
        has_any = True
    if len(val_curve) > 0:
        ax.plot(np.arange(1, len(val_curve) + 1), val_curve, linewidth=1.8, label="val")
        has_any = True
    if has_any:
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No history found", ha="center", va="center")
        ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize one separation sample")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--sample_idx", type=int, default=5, help="Sample index in split")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="outputs/checkpoints/best_transformer.pt",
        help="Checkpoint path",
    )
    parser.add_argument(
        "--history_json",
        type=str,
        default="outputs/logs/train_history_transformer.json",
        help="Training history json",
    )
    parser.add_argument("--device", type=str, default=None, help="Override device, e.g. cuda/cpu")
    parser.add_argument(
        "--model_name",
        type=str,
        default="transformer",
        choices=["baseline", "transformer"],
        help="Model family",
    )
    return parser.parse_args()


def main() -> None:
    _ensure_project_root_in_syspath()

    from configs.config import get_default_config
    from data.dataset import DroneSeparationDataset
    from data.stft_utils import istft_reconstruct, spec_to_logmag
    from engine.metrics import complex_corr

    args = parse_args()
    cfg = get_default_config()
    device = _build_device(args.device)
    ckpt_path = _resolve_ckpt(cfg, args.ckpt)
    history_path = _resolve_history(cfg, args.history_json)

    if len(cfg.file_split.drone_codes) < 2:
        raise ValueError("cfg.file_split.drone_codes must contain at least two drone codes.")
    source_a_code = cfg.file_split.drone_codes[0]
    source_b_code = cfg.file_split.drone_codes[1]
    sir_db = float(cfg.train.sir_db)

    index_path = cfg.paths.outputs_dir / "indexes" / f"{args.split}_index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")

    dataset = DroneSeparationDataset(
        split=args.split,
        index_json_path=str(index_path),
        source_a_code=source_a_code,
        source_b_code=source_b_code,
        sir_db=sir_db,
        cfg=cfg,
        seed=42,
    )
    if args.sample_idx < 0 or args.sample_idx >= len(dataset):
        raise IndexError(f"sample_idx out of range: {args.sample_idx}, len={len(dataset)}")
    sample = dataset[args.sample_idx]

    model = _build_model(args.model_name, cfg, device)
    _load_checkpoint(model, ckpt_path, device)
    model.eval()

    # 单样本加 batch 维
    mix_mag = sample["mix_mag"].unsqueeze(0).to(device)          # [1,1,F,T]
    mix_spec = sample["mix_spec"].unsqueeze(0).to(device)        # [1,F,T]
    srcA_spec = sample["srcA_spec"].unsqueeze(0).to(device)      # [1,F,T]
    srcB_spec = sample["srcB_spec"].unsqueeze(0).to(device)      # [1,F,T]
    srcA_time_true = sample["srcA_time"].to(device)              # [N]
    srcB_time_true = sample["srcB_time"].to(device)              # [N]

    with torch.no_grad():
        pred_mask = model(mix_mag)  # [1,2,F,T]
        pred_srcA_spec = pred_mask[:, 0] * mix_spec
        pred_srcB_spec = pred_mask[:, 1] * mix_spec
        pred_srcA_time = istft_reconstruct(pred_srcA_spec, cfg=cfg, length=cfg.data.window_len)[0]
        pred_srcB_time = istft_reconstruct(pred_srcB_spec, cfg=cfg, length=cfg.data.window_len)[0]

    corr_a = complex_corr(pred_srcA_time, srcA_time_true)
    corr_b = complex_corr(pred_srcB_time, srcB_time_true)

    eps = float(cfg.numeric.eps)
    def _log_mag_ft(spec_ft: torch.Tensor) -> np.ndarray:
        """spec_ft: 复数谱 [F,T] -> 对数幅度 [F,T] numpy。"""
        return spec_to_logmag(spec_ft.detach().cpu(), eps=eps)[0].numpy()

    mix_mag_np = _log_mag_ft(mix_spec[0])
    srcA_mag_np = _log_mag_ft(srcA_spec[0])
    srcB_mag_np = _log_mag_ft(srcB_spec[0])
    predA_mag_np = _log_mag_ft(pred_srcA_spec[0])
    predB_mag_np = _log_mag_ft(pred_srcB_spec[0])

    ref_mix_from_tensor = mix_mag[0, 0].detach().cpu().numpy()
    if not np.allclose(mix_mag_np, ref_mix_from_tensor, rtol=1e-5, atol=1e-3):
        raise RuntimeError(
            "mix log-mag from mix_spec disagrees with sample['mix_mag']; check eps / STFT pipeline."
        )

    curves = _extract_loss_curves(history_path)
    train_curve = curves["train"]
    val_curve = curves["val"]

    fig = plt.figure(figsize=(23, 4.8))
    gs = fig.add_gridspec(1, 5, wspace=0.28)

    def _plot_spec(axis: Any, img: np.ndarray, title: str) -> None:
        im = axis.imshow(
            img, origin="lower", aspect="auto", interpolation="nearest", cmap="jet"
        )
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("T")
        axis.set_ylabel("F")
        fig.colorbar(im, ax=axis, fraction=0.046, pad=0.04)

    spec_axes = [fig.add_subplot(gs[0, i]) for i in range(5)]
    _plot_spec(spec_axes[0], mix_mag_np, "Mix log_magnitude")
    _plot_spec(spec_axes[1], srcA_mag_np, "SrcA real log_magnitude")
    _plot_spec(spec_axes[2], predA_mag_np, "SrcA pred log_magnitude")
    _plot_spec(spec_axes[3], srcB_mag_np, "SrcB real log_magnitude")
    _plot_spec(spec_axes[4], predB_mag_np, "SrcB pred log_magnitude")

    fig.suptitle(
        f"split={args.split} | sample_idx={args.sample_idx} | "
        f"A={source_a_code} B={source_b_code} | sir_db={sir_db} | "
        f"corr_a={corr_a:.4f} corr_b={corr_b:.4f}",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.93])

    out_dir = cfg.paths.outputs_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"viz_spec5_{args.model_name}_{args.split}_idx{args.sample_idx}_{ts}.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    both_curve_path = out_dir / f"loss_train_val_{args.model_name}_{ts}.png"
    _plot_and_save_combined_loss_curve(both_curve_path, train_curve, val_curve)

    print("=== Visualize Done ===")
    print(f"model        : {args.model_name}")
    print(f"device       : {device}")
    print(f"checkpoint   : {ckpt_path}")
    print(f"history_json : {history_path}")
    print(f"split/index  : {args.split}/{args.sample_idx}")
    print(f"corr_a/corr_b: {corr_a:.6f}/{corr_b:.6f}")
    print(f"saved_spec_figure : {out_path}")
    print(f"saved_loss_figure : {both_curve_path}")


if __name__ == "__main__":
    main()
