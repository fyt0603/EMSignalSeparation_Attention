"""绘制训练/验证子损失随 epoch 变化曲线。

输入：
- 训练日志 JSON（由 scripts/train.py 输出），形如：
  outputs/logs/train_history_transformer_complex.json

输出：
- 三行一列子图（mag/mask/corr），每张包含 train/val 两条曲线。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot train/val loss curves from a training log JSON.")
    parser.add_argument("--log_path", type=str, required=True, help="Path to training history JSON.")
    parser.add_argument(
        "--out_path",
        type=str,
        default="",
        help="Output png path (default: outputs/figures/loss_curves.png).",
    )
    parser.add_argument("--title", type=str, default="", help="Figure title (optional).")
    return parser.parse_args()


def _load_history(log_path: Path) -> List[Dict[str, Any]]:
    if not log_path.exists():
        raise FileNotFoundError(f"log_path not found: {log_path}")
    if log_path.is_dir():
        raise IsADirectoryError(f"log_path is a directory, expected a JSON file: {log_path}")

    with log_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if data is None:
        raise ValueError(f"JSON is null/None: {log_path}")
    if not isinstance(data, list):
        raise TypeError(f"JSON root must be a list (epoch records), got {type(data)}: {log_path}")
    if len(data) == 0:
        raise ValueError(f"JSON is empty (no epoch records): {log_path}")

    # 基本结构校验：每项应为 dict
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise TypeError(
                f"Each epoch record must be a dict, got {type(item)} at index {i}: {log_path}"
            )
    return data


def _require_fields(history: Sequence[Dict[str, Any]], required_fields: Sequence[str], log_path: Path) -> None:
    missing_any = False
    for i, record in enumerate(history):
        missing = [k for k in required_fields if k not in record]
        if missing:
            missing_any = True
            raise KeyError(
                "Missing required fields in training log.\n"
                f"- log_path: {log_path}\n"
                f"- record_index: {i}\n"
                f"- missing_fields: {missing}\n"
                f"- available_fields: {sorted(record.keys())}"
            )
    if missing_any:
        raise KeyError(f"Missing required fields in training log: {log_path}")


def _extract_xy(history: Sequence[Dict[str, Any]], x_key: str, y_key: str) -> Tuple[List[float], List[float]]:
    xs: List[float] = []
    ys: List[float] = []
    for r in history:
        xs.append(float(r[x_key]))
        ys.append(float(r[y_key]))
    return xs, ys


def _default_out_path() -> Path:
    return _project_root() / "outputs" / "figures" / "loss_curves.png"


def main() -> None:
    args = _parse_args()

    log_path = Path(args.log_path)
    out_path = Path(args.out_path) if str(args.out_path).strip() else _default_out_path()

    history = _load_history(log_path)

    required_fields = [
        "epoch",
        "train_total_loss",
        "train_mag_loss",
        "train_mask_loss",
        "train_corr_loss",
        "val_total_loss",
        "val_mag_loss",
        "val_mask_loss",
        "val_corr_loss",
    ]
    _require_fields(history, required_fields=required_fields, log_path=log_path)

    # 延迟导入，避免在无 matplotlib 环境下读取/校验阶段就报错
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 10), sharex=True)

    plots = [
        ("mag_loss", "train_mag_loss", "val_mag_loss", "Magnitude loss"),
        ("mask_loss", "train_mask_loss", "val_mask_loss", "Mask loss"),
        ("corr_loss", "train_corr_loss", "val_corr_loss", "Correlation loss"),
    ]

    epochs, _ = _extract_xy(history, x_key="epoch", y_key="train_total_loss")

    for ax, (tag, train_key, val_key, title) in zip(axes, plots):
        _, train_y = _extract_xy(history, x_key="epoch", y_key=train_key)
        _, val_y = _extract_xy(history, x_key="epoch", y_key=val_key)

        ax.plot(epochs, train_y, label=f"train_{tag}", linewidth=2)
        ax.plot(epochs, val_y, label=f"val_{tag}", linewidth=2)
        ax.set_ylabel("loss")
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()

    axes[-1].set_xlabel("epoch")

    fig_title = str(args.title).strip()
    if fig_title:
        fig.suptitle(fig_title)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    else:
        fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"saved_png: {out_path}")


if __name__ == "__main__":
    main()

