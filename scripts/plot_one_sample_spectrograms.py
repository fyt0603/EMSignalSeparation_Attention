"""单样本分离结果时频图导出脚本（用于论文/结构图示意图）。

从 val/test 取一个样本，用训练好的 TransformerSeparator（magnitude mask）推理，
保存混合信号与两路恢复源的时频图。

运行示例（项目根目录）：
    python scripts/plot_one_sample_spectrograms.py
    python scripts/plot_one_sample_spectrograms.py --split val --sample-index 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Tuple

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


def _default_checkpoint_path(cfg: Any) -> Path:
    """默认 magnitude Transformer 权重路径（与 train.py 保存命名一致）。"""
    return cfg.paths.outputs_dir / "checkpoints" / "best_transformer_magnitude.pt"


def _resolve_checkpoint_path(cfg: Any, checkpoint_arg: str | None) -> Path:
    if checkpoint_arg:
        ckpt_path = Path(checkpoint_arg)
    else:
        # 兼容用户文档中的 best_transformer.pt 命名
        candidates = [
            _default_checkpoint_path(cfg),
            cfg.paths.outputs_dir / "checkpoints" / "best_transformer.pt",
        ]
        ckpt_path = next((p for p in candidates if p.exists()), candidates[0])
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Please train with mask_type=magnitude or pass --checkpoint explicitly."
        )
    return ckpt_path


def _build_transformer_magnitude_model(cfg: Any, device: torch.device) -> torch.nn.Module:
    from models.transformer import TransformerSeparator

    model = TransformerSeparator(
        in_channels=3,
        out_masks=2,
        mask_type="magnitude",
        mask_bound=cfg.model.mask_bound,
        embed_dim=cfg.model.d_model,
        depth=cfg.model.num_layers,
        num_heads=cfg.model.n_heads,
        ff_dim=cfg.model.ff_dim,
        dropout=cfg.model.dropout,
        patch_size=cfg.model.patch_size,
    )
    return model.to(device)


def _load_checkpoint(model: torch.nn.Module, ckpt_path: Path, device: torch.device) -> None:
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif isinstance(ckpt, dict):
        model.load_state_dict(ckpt)
    else:
        raise TypeError(f"Unsupported checkpoint format: {type(ckpt)}")


def _linear_mag_to_log_display(mag: np.ndarray, eps: float) -> np.ndarray:
    """线性幅度谱 -> 对数域显示（与 spec_to_logmag 一致：log(mag + eps)）。"""
    return np.log(np.maximum(mag, 0.0) + float(eps))


def _recover_magnitude_masks(
    pred_mask: torch.Tensor,
    mix_spec: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """由 softmax magnitude mask 与混合谱幅度恢复两路线性幅度谱。

    pred_mask: [1, 2, F, T]，通道 0/1 分别对应 source A / source B。
    """
    if pred_mask.ndim != 4 or pred_mask.shape[0] != 1 or pred_mask.shape[1] != 2:
        raise ValueError(f"Expected pred_mask [1,2,F,T], got {tuple(pred_mask.shape)}")

    mixed_magnitude = torch.abs(mix_spec[0]).to(torch.float32)  # [F, T]
    pred_mask_a = pred_mask[0, 0]  # source A
    pred_mask_b = pred_mask[0, 1]  # source B
    pred_mag_a = pred_mask_a * mixed_magnitude
    pred_mag_b = pred_mask_b * mixed_magnitude
    return mixed_magnitude, pred_mask_a, pred_mask_b, pred_mag_a, pred_mag_b


def _apply_spec_axes(
    ax: Any,
    image: np.ndarray,
    *,
    show_axes: bool,
    show_ylabel: bool = True,
) -> None:
    """为时频图设置 extent 与坐标轴（横轴 T，纵轴 F，单位：bin 索引）。"""
    n_f, n_t = image.shape
    ax.imshow(
        image,
        origin="lower",
        aspect="equal",
        interpolation="nearest",
        cmap="jet",
        extent=(0, n_t, 0, n_f),
    )
    if not show_axes:
        ax.set_axis_off()
        return
    ax.set_xlabel("Time (T)", fontsize=9)
    if show_ylabel:
        ax.set_ylabel("Frequency (F)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_xlim(0, n_t)
    ax.set_ylim(0, n_f)


def _save_single_spec(
    image: np.ndarray,
    out_path: Path,
    *,
    show_axes: bool = True,
) -> None:
    """保存单张时频图。"""
    fig, ax = plt.subplots(figsize=(4.8, 4.6), dpi=100)
    _apply_spec_axes(ax, image, show_axes=show_axes)
    fig.tight_layout(pad=0.08)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def _save_overview(
    images: Tuple[np.ndarray, np.ndarray, np.ndarray],
    titles: Tuple[str, str, str],
    out_path: Path,
    *,
    show_axes: bool = True,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.8), dpi=100)
    for i, (ax, img, title) in enumerate(zip(axes, images, titles)):
        _apply_spec_axes(ax, img, show_axes=show_axes, show_ylabel=(i == 0))
        ax.set_title(title, fontsize=11, pad=6)
    fig.tight_layout(w_pad=0.45, h_pad=0.25)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot one-sample separation spectrograms (magnitude mask, Transformer)."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Model checkpoint path. "
            "Default: outputs/checkpoints/best_transformer_magnitude.pt "
            "(fallback: best_transformer.pt)"
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Dataset split (default: test)",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Sample index in the split dataset (default: 0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Figure output directory (default: outputs/figures/separation_sample)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override, e.g. cuda or cpu (default: auto)",
    )
    parser.add_argument(
        "--no-axes",
        action="store_true",
        help="Hide axis ticks and labels (default: show F/T axes)",
    )
    return parser.parse_args()


def main() -> None:
    _ensure_project_root_in_syspath()

    from configs.config import get_default_config
    from data.dataset import DroneSeparationDataset

    args = parse_args()
    cfg = get_default_config()
    # 本脚本仅支持 magnitude mask 示意图
    cfg.model.mask_type = "magnitude"

    device = _build_device(args.device)
    ckpt_path = _resolve_checkpoint_path(cfg, args.checkpoint)

    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else cfg.paths.outputs_dir / "figures" / "separation_sample"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if len(cfg.file_split.drone_codes) < 2:
        raise ValueError("cfg.file_split.drone_codes must contain at least two codes.")
    source_a_code = cfg.file_split.drone_codes[0]
    source_b_code = cfg.file_split.drone_codes[1]
    sir_db = float(cfg.train.sir_db)

    index_path = cfg.paths.outputs_dir / "indexes" / f"{args.split}_index.json"
    if not index_path.exists():
        raise FileNotFoundError(
            f"Index file not found: {index_path}. Please run data/build_index.py first."
        )

    dataset = DroneSeparationDataset(
        split=args.split,
        index_json_path=str(index_path),
        source_a_code=source_a_code,
        source_b_code=source_b_code,
        sir_db=sir_db,
        cfg=cfg,
        seed=42,
    )
    if args.sample_index < 0 or args.sample_index >= len(dataset):
        raise IndexError(
            f"sample-index out of range: {args.sample_index}, dataset length={len(dataset)}"
        )

    sample = dataset[args.sample_index]
    mix_feat = sample["mix_feat"].unsqueeze(0).to(device)   # [1, 3, F, T]
    mix_spec = sample["mix_spec"].unsqueeze(0).to(device)   # [1, F, T], complex

    model = _build_transformer_magnitude_model(cfg, device)
    _load_checkpoint(model, ckpt_path, device)
    model.eval()

    with torch.no_grad():
        pred_mask = model(mix_feat)  # [1, 2, F, T], softmax over source dim

    mixed_magnitude, pred_mask_a, pred_mask_b, pred_mag_a, pred_mag_b = _recover_magnitude_masks(
        pred_mask, mix_spec
    )

    eps = float(cfg.numeric.eps)
    # 混合图：优先使用 mix_feat 第 0 通道（log magnitude）
    mixed_plot = mix_feat[0, 0].detach().cpu().numpy()
    pred_a_plot = _linear_mag_to_log_display(pred_mag_a.detach().cpu().numpy(), eps)
    pred_b_plot = _linear_mag_to_log_display(pred_mag_b.detach().cpu().numpy(), eps)

    paths = {
        "mixed": out_dir / "mixed_spectrogram.png",
        "pred_a": out_dir / "pred_source_A_spectrogram.png",
        "pred_b": out_dir / "pred_source_B_spectrogram.png",
        "overview": out_dir / "separation_sample_overview.png",
    }

    show_axes = not args.no_axes
    _save_single_spec(mixed_plot, paths["mixed"], show_axes=show_axes)
    _save_single_spec(pred_a_plot, paths["pred_a"], show_axes=show_axes)
    _save_single_spec(pred_b_plot, paths["pred_b"], show_axes=show_axes)
    _save_overview(
        (mixed_plot, pred_a_plot, pred_b_plot),
        ("Mixed", "Recovered Source A", "Recovered Source B"),
        paths["overview"],
        show_axes=show_axes,
    )

    print("=== Separation Spectrogram Export Done ===")
    print(f"split         : {args.split}")
    print(f"sample_index  : {args.sample_index}")
    print(f"source_a/b    : {source_a_code} / {source_b_code}")
    print(f"device        : {device}")
    print(f"checkpoint    : {ckpt_path.resolve()}")
    print(f"pred_mask sum : {float(pred_mask_a.sum() + pred_mask_b.sum()) / pred_mask_a.numel():.4f} (per-bin mean)")
    print("Saved figures:")
    for key, path in paths.items():
        print(f"  [{key}] {path.resolve()}")


if __name__ == "__main__":
    main()
