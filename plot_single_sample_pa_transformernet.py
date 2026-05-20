"""PA-TransformerNet 单样本时频图对比脚本（毕业论文 5.2.4）。

输出两张图：
1) figures/single_sample/sample_reference.png
   - Mixture
   - Ground Truth A
   - Ground Truth B
2) figures/single_sample/sample_pa_transformernet.png
   - PA-TransformerNet Output A
   - PA-TransformerNet Output B
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _ensure_project_root_in_syspath() -> Path:
    """兼容 `python plot_single_sample_pa_transformernet.py` 运行方式。"""
    project_root = Path(__file__).resolve().parent
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return project_root


def _build_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_pa_transformernet(cfg: Any, device: torch.device) -> torch.nn.Module:
    """构建 PA-TransformerNet（TransformerSeparator）。"""
    from models.transformer import TransformerSeparator

    out_masks = 4 if cfg.model.mask_type == "complex" else 2
    model = TransformerSeparator(
        in_channels=3,
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


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise TypeError(f"Unsupported checkpoint format: {type(ckpt)}")

    if not isinstance(state_dict, dict):
        raise TypeError(f"state_dict must be dict, got {type(state_dict)}")
    model.load_state_dict(state_dict)


def _linear_mag_to_db_display(mag: np.ndarray, eps: float) -> np.ndarray:
    """线性幅度谱 -> dB 显示（20*log10）。"""
    return 20.0 * np.log10(np.maximum(mag, float(eps)))


def _recover_pred_magnitude(
    pred_mask: torch.Tensor,
    mix_spec: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """由 softmax 幅度 mask 恢复两路线性幅度谱。

    pred_mask: [1, 2, F, T]
    mix_spec:  [1, F, T] (complex)
    """
    if pred_mask.ndim != 4 or pred_mask.shape[0] != 1 or pred_mask.shape[1] != 2:
        raise ValueError(f"Expected pred_mask [1,2,F,T], got {tuple(pred_mask.shape)}")
    if mix_spec.ndim != 3 or mix_spec.shape[0] != 1:
        raise ValueError(f"Expected mix_spec [1,F,T], got {tuple(mix_spec.shape)}")

    mixed_magnitude = torch.abs(mix_spec[0]).to(torch.float32)
    pred_mag_a = pred_mask[0, 0] * mixed_magnitude
    pred_mag_b = pred_mask[0, 1] * mixed_magnitude
    return pred_mag_a, pred_mag_b


def _apply_spec_axes(
    ax: Any,
    image: np.ndarray,
    *,
    show_axes: bool,
    hop_length: int,
    sample_rate: float,
    center_freq: float,
    apply_fftshift: bool,
    show_ylabel: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
) -> Any:
    """统一时频图样式。"""
    image_plot = np.fft.fftshift(image, axes=0) if apply_fftshift else image
    n_f, n_t = image_plot.shape

    time_start_ms = 0.0
    time_end_ms = float(n_t) * float(hop_length) / float(sample_rate) * 1e3
    freq_start_mhz = (float(center_freq) - float(sample_rate) / 2.0) / 1e6
    freq_end_mhz = (float(center_freq) + float(sample_rate) / 2.0) / 1e6
    extent = [time_start_ms, time_end_ms, freq_start_mhz, freq_end_mhz]

    im = ax.imshow(
        image_plot,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="jet",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
    )
    if not show_axes:
        ax.set_axis_off()
        return im
    ax.set_xlabel("时间 (ms)", fontsize=9)
    if show_ylabel:
        ax.set_ylabel("频率 (MHz)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    return im


def _save_horizontal_figure(
    images: Sequence[np.ndarray],
    titles: Sequence[str],
    out_path: Path,
    *,
    show_axes: bool,
    hop_length: int,
    sample_rate: float,
    center_freq: float,
    apply_fftshift: bool = True,
) -> None:
    n_cols = len(images)
    if n_cols <= 0:
        raise ValueError("images must not be empty.")
    if len(titles) != n_cols:
        raise ValueError("titles length must match images length.")

    # 多列 + 每列独立 colorbar 时，tight_layout 容易使「上一列 colorbar」压住「下一列主轴左侧」。
    # constrained_layout 会给每列（含 colorbar）自动留水平间隙。
    # 总宽 ≈ 系数 × 列数；系数越大子图越宽；论文图常用 4.8~5.4 之间微调
    fig, axes = plt.subplots(
        1,
        n_cols,
        figsize=(4 * n_cols, 4.6),
        dpi=100,
        constrained_layout=True,
    )
    if n_cols == 1:
        axes = [axes]
    vmin = min(float(np.nanmin(image)) for image in images)
    vmax = max(float(np.nanmax(image)) for image in images)
    for i, (ax, image, title) in enumerate(zip(axes, images, titles)):
        im = _apply_spec_axes(
            ax,
            image,
            show_axes=show_axes,
            hop_length=hop_length,
            sample_rate=sample_rate,
            center_freq=center_freq,
            apply_fftshift=apply_fftshift,
            # 每张子图都保留纵轴标签，避免读者误以为「左侧被挡的是没画轴」
            show_ylabel=True,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(title, fontsize=11, pad=6)
        cbar = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.038, pad=0.02)
        cbar.set_label("magnitude (dB)", fontsize=9)
        cbar.ax.tick_params(labelsize=8)
    # 已使用 constrained_layout，勿再调用 tight_layout（二者易冲突）
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot single-sample reference/pred spectrograms for PA-TransformerNet."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/checkpoints/s0_best_transformer.pt",
        help="Path to PA-TransformerNet checkpoint",
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
        default=100,
        help="Sample index in split index json (default: 0)",
    )
    parser.add_argument(
        "--sir-db",
        type=float,
        default=0.0,
        help="Mixing SIR in dB (default: 0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="figures/single_sample",
        help="Output directory (default: figures/single_sample)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override, e.g. cuda/cpu (default: auto)",
    )
    parser.add_argument(
        "--no-axes",
        action="store_true",
        help="Hide axes and ticks",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=100e6,
        help="Sampling rate in Hz (default: 100e6)",
    )
    parser.add_argument(
        "--center-freq",
        type=float,
        default=2440e6,
        help="Center frequency in Hz (default: 2440e6)",
    )
    return parser.parse_args()


def main() -> None:
    _ensure_project_root_in_syspath()

    from configs.config import get_default_config
    from data.dataset import DroneSeparationDataset

    args = parse_args()
    cfg = get_default_config()
    cfg.model.mask_type = "magnitude"

    seed = 42
    device = _build_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(cfg.file_split.drone_codes) < 2:
        raise ValueError("cfg.file_split.drone_codes must contain at least two codes.")
    source_a_code = cfg.file_split.drone_codes[0]
    source_b_code = cfg.file_split.drone_codes[1]

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
        sir_db=float(args.sir_db),
        cfg=cfg,
        seed=seed,
    )
    if args.sample_index < 0 or args.sample_index >= len(dataset):
        raise IndexError(
            f"sample-index out of range: {args.sample_index}, dataset length={len(dataset)}"
        )

    sample = dataset[args.sample_index]
    mix_feat = sample["mix_feat"].unsqueeze(0).to(device)   # [1, 3, F, T]
    mix_spec = sample["mix_spec"].unsqueeze(0).to(device)   # [1, F, T], complex
    src_a_spec = sample["srcA_spec"]                        # [F, T], complex
    src_b_spec = sample["srcB_spec"]                        # [F, T], complex

    model = _build_pa_transformernet(cfg=cfg, device=device)
    _load_checkpoint(model=model, checkpoint_path=checkpoint_path, device=device)
    model.eval()

    with torch.no_grad():
        pred_mask = model(mix_feat)  # [1, 2, F, T]
    pred_mag_a, pred_mag_b = _recover_pred_magnitude(pred_mask=pred_mask, mix_spec=mix_spec)

    eps = float(cfg.numeric.eps)
    mixed_mag = torch.abs(mix_spec[0])
    mixture_plot = _linear_mag_to_db_display(mixed_mag.cpu().numpy(), eps)
    gt_a_plot = _linear_mag_to_db_display(torch.abs(src_a_spec).cpu().numpy(), eps)
    gt_b_plot = _linear_mag_to_db_display(torch.abs(src_b_spec).cpu().numpy(), eps)
    pred_a_plot = _linear_mag_to_db_display(pred_mag_a.detach().cpu().numpy(), eps)
    pred_b_plot = _linear_mag_to_db_display(pred_mag_b.detach().cpu().numpy(), eps)

    reference_path = output_dir / "sample_reference.png"
    pred_path = output_dir / "sample_pa_transformernet.png"

    show_axes = not args.no_axes
    hop_length = int(cfg.stft.hop_length)
    _save_horizontal_figure(
        images=(mixture_plot, gt_a_plot, gt_b_plot),
        titles=("Mix Signal", "Original A", "Original B"),
        out_path=reference_path,
        show_axes=show_axes,
        hop_length=hop_length,
        sample_rate=float(args.sample_rate),
        center_freq=float(args.center_freq),
        apply_fftshift=True,
    )
    _save_horizontal_figure(
        images=(pred_a_plot, pred_b_plot),
        titles=("PA-TransformerNet Output A", "PA-TransformerNet Output B"),
        out_path=pred_path,
        show_axes=show_axes,
        hop_length=hop_length,
        sample_rate=float(args.sample_rate),
        center_freq=float(args.center_freq),
        apply_fftshift=True,
    )

    sample_info = {
        "sample_index": int(args.sample_index),
        "split": str(args.split),
        "sir_db": float(args.sir_db),
        "seed": int(seed),
        "source_a_code": str(source_a_code),
        "source_b_code": str(source_b_code),
        "checkpoint": str(checkpoint_path),
        "model": "PA-TransformerNet",
    }
    sample_info_path = output_dir / "sample_info.json"
    with sample_info_path.open("w", encoding="utf-8") as f:
        json.dump(sample_info, f, ensure_ascii=False, indent=2)

    print("=== Single Sample Spectrograms Export Done ===")
    print(f"split              : {args.split}")
    print(f"sample_index       : {args.sample_index}")
    print(f"sir_db             : {args.sir_db}")
    print(f"source_a/source_b  : {source_a_code} / {source_b_code}")
    print(f"device             : {device}")
    print(f"checkpoint         : {checkpoint_path.resolve()}")
    print(f"saved_reference    : {reference_path.resolve()}")
    print(f"saved_prediction   : {pred_path.resolve()}")
    print(f"saved_sample_info  : {sample_info_path.resolve()}")


if __name__ == "__main__":
    main()
