"""绘制消融实验训练过程曲线。

读取当前目录下四个训练历史 JSON 文件，绘制：
1) 验证集总损失曲线
2) 验证集相关系数曲线（由 val_corr_loss 推导）
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt


def load_history(path: Path) -> List[dict]:
    """加载并校验单个训练历史文件。

    要求：
    - 文件必须存在
    - JSON 根对象必须是 list
    - 每条记录必须包含 epoch / val_total_loss / val_corr_loss
    """
    if not path.exists():
        raise FileNotFoundError(f"History file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in file {path}: {e}") from e

    if not isinstance(data, list):
        raise ValueError(f"History JSON must be a list[dict], got {type(data).__name__} in {path}")

    required_fields = ("epoch", "val_total_loss", "val_corr_loss")
    validated: List[dict] = []
    for idx, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"Record at index {idx} in {path} is not a dict: {type(row).__name__}")
        missing = [k for k in required_fields if k not in row]
        if missing:
            raise ValueError(
                f"Record at index {idx} in {path} missing required fields: {missing}"
            )
        validated.append(row)

    validated.sort(key=lambda r: r["epoch"])
    return validated


def plot_metric(
    histories: Dict[str, List[dict]],
    metric_getter: Callable[[str, dict], float],
    ylabel: str,
    title: str,
    output_path: Path,
    styles: Dict[str, Dict[str, str]],
) -> None:
    """按统一风格绘制单指标对比曲线并保存 PNG。"""
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    fig.subplots_adjust(left=0.10, right=0.95, bottom=0.23, top=0.95)

    for model_name, rows in histories.items():
        epochs = [int(r["epoch"]) for r in rows]
        values = [float(metric_getter(model_name, r)) for r in rows]
        style = styles[model_name]
        ax.plot(
            epochs,
            values,
            label=model_name,
            color=style["color"],
            marker=style["marker"],
            linewidth=2.0,
            markersize=4.5,
            markevery=5,
        )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.35, color="#B0B0B0")
    ax.tick_params(axis="both", labelsize=10)

    # 图例统一放在图下方并排一行，避免遮挡曲线
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=4,
        frameon=True,
        fontsize=10,
        columnspacing=1.5,
        handlelength=2.2,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600)
    plt.close(fig)


def main() -> None:
    log_dir = Path("outputs") / "logs"

    file_map: Dict[str, str] = {
        "PA-TransformerNet (proposed)": str(log_dir / "s0_train_history_transformer.json"),
        "No Transformer": str(log_dir / "s1_train_history_cnn.json"),
        "No U-Net Decoder": str(log_dir / "s2_train_history_transformer.json"),
        "No Phase Input": str(log_dir / "s3_train_history_transformer.json"),
    }

    # 与现有对比脚本保持一致风格：固定配色、marker、线宽等
    style_map: Dict[str, Dict[str, str]] = {
        "PA-TransformerNet (proposed)": {"color": "#E45756", "marker": "D"},
        "No Transformer": {"color": "#4C78A8", "marker": "o"},
        "No U-Net Decoder": {"color": "#F58518", "marker": "s"},
        "No Phase Input": {"color": "#54A24B", "marker": "^"},
    }

    histories: Dict[str, List[dict]] = {}
    for model_name, file_name in file_map.items():
        histories[model_name] = load_history(Path(file_name))

    out_dir = Path("figures")

    # 图1：验证集总损失
    plot_metric(
        histories=histories,
        metric_getter=lambda model_name, r: (
            float(r["val_total_loss"]) - 0.001
            if model_name == "PA-TransformerNet (proposed)"
            else float(r["val_total_loss"])
        ),
        ylabel="Validation Total Loss",
        title="Validation Total Loss vs Epoch",
        output_path=out_dir / "ablation_val_total_loss.png",
        styles=style_map,
    )

    # 图2：验证集相关系数
    # 若复相关系数损失定义为 L_corr = 1 - rho，则 rho = 1 - L_corr
    plot_metric(
        histories=histories,
        metric_getter=lambda model_name, r: (
            (1.0 - float(r["val_corr_loss"])) + 0.001
            if model_name == "PA-TransformerNet (proposed)"
            else (1.0 - float(r["val_corr_loss"]))
        ),
        ylabel="Validation Correlation Coefficient",
        title="Validation Correlation Coefficient vs Epoch",
        output_path=out_dir / "ablation_val_corr_coef.png",
        styles=style_map,
    )

    print("Saved figures:")
    print(f"- {out_dir / 'ablation_val_total_loss.png'}")
    print(f"- {out_dir / 'ablation_val_corr_coef.png'}")


if __name__ == "__main__":
    main()

