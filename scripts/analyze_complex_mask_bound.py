"""分析 complex mask 的合理 mask_bound。

流程：
1) 读取若干个训练 batch。
2) 计算未裁剪 ideal complex mask（mask_bound=None）。
3) 统计四个通道的分布（mean/std/min/max/abs percentiles）。
4) 对候选 bounds 计算 clip_ratio、oracle_mag_loss，并可选 oracle_corr。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze complex mask bound")
    parser.add_argument("--device", type=str, default=None, help="Override device, e.g. cuda/cpu")
    parser.add_argument("--max_batches", type=int, default=10, help="Max batches to analyze")
    parser.add_argument(
        "--compute_corr",
        action="store_true",
        help="Also compute oracle corr via iSTFT (slower)",
    )
    return parser.parse_args()


def _format_float(x: float, width: int = 12) -> str:
    return f"{x:{width}.6f}"


def _format_ratio(x: float, width: int = 10) -> str:
    return f"{x:{width}.4%}"


def _channel_name(ch: int) -> str:
    names = ("M_A_r", "M_A_i", "M_B_r", "M_B_i")
    return names[ch] if 0 <= ch < 4 else f"ch{ch}"


def _compute_channel_stats(abs_values_1d: torch.Tensor) -> Dict[str, float]:
    """输入为 abs(flat) 的 1D float tensor（cpu），返回统计量。"""
    if abs_values_1d.ndim != 1:
        raise ValueError(f"abs_values_1d must be 1D, got {tuple(abs_values_1d.shape)}")
    if abs_values_1d.numel() == 0:
        raise ValueError("abs_values_1d is empty.")

    mean = float(abs_values_1d.mean().item())
    std = float(abs_values_1d.std(unbiased=False).item())
    vmin = float(abs_values_1d.min().item())
    vmax = float(abs_values_1d.max().item())
    p90 = float(torch.quantile(abs_values_1d, 0.90).item())
    p95 = float(torch.quantile(abs_values_1d, 0.95).item())
    p99 = float(torch.quantile(abs_values_1d, 0.99).item())
    p995 = float(torch.quantile(abs_values_1d, 0.995).item())
    p999 = float(torch.quantile(abs_values_1d, 0.999).item())
    return {
        "mean": mean,
        "std": std,
        "min": vmin,
        "max": vmax,
        "abs_p90": p90,
        "abs_p95": p95,
        "abs_p99": p99,
        "abs_p99.5": p995,
        "abs_p99.9": p999,
    }


@torch.no_grad()
def main() -> None:
    project_root = _ensure_project_root_in_syspath()

    from configs.config import get_default_config
    from data.complex_mask_utils import apply_complex_mask, compute_ideal_complex_mask
    from data.dataset import DroneSeparationDataset
    from data.stft_utils import istft_reconstruct
    from engine.metrics import batch_complex_corr

    args = parse_args()
    device = _build_device(args.device)
    max_batches = int(args.max_batches)
    compute_corr = bool(args.compute_corr)

    cfg = get_default_config()
    cfg.model.mask_type = "complex"

    if len(cfg.file_split.drone_codes) < 2:
        raise ValueError("cfg.file_split.drone_codes must contain at least two codes.")
    source_a_code = cfg.file_split.drone_codes[0]
    source_b_code = cfg.file_split.drone_codes[1]

    index_path = cfg.paths.outputs_dir / "indexes" / "train_index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")

    dataset = DroneSeparationDataset(
        split="train",
        index_json_path=str(index_path),
        source_a_code=source_a_code,
        source_b_code=source_b_code,
        sir_db=float(cfg.train.sir_db),
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

    print("=== Analyze Complex Mask Bound ===")
    print(f"project_root : {project_root}")
    print(f"device       : {device}")
    print(f"max_batches  : {max_batches}")
    print(f"compute_corr : {compute_corr}")
    print(f"source_a/b   : {source_a_code}/{source_b_code}")
    print(f"sir_db       : {float(cfg.train.sir_db)}")
    print("")

    # 1) 收集未裁剪 target mask 的 abs(flat)（每通道一个列表）
    abs_by_ch: List[List[torch.Tensor]] = [[], [], [], []]

    # 2) 对每个候选 bound 统计 clip_ratio / oracle_mag_loss / (可选) oracle_corr
    bounds = [1, 2, 3, 5, 8, 10, 20]
    bound_stats: Dict[float, Dict[str, float]] = {float(b): {} for b in bounds}
    clip_sum: Dict[float, float] = {float(b): 0.0 for b in bounds}
    elem_count: int = 0
    mag_loss_sum: Dict[float, float] = {float(b): 0.0 for b in bounds}
    corr_a_sum: Dict[float, float] = {float(b): 0.0 for b in bounds}
    corr_b_sum: Dict[float, float] = {float(b): 0.0 for b in bounds}
    corr_n: int = 0

    l1 = torch.nn.L1Loss(reduction="mean")

    batch_counter = 0
    for batch in loader:
        batch_counter += 1
        if max_batches > 0 and batch_counter > max_batches:
            break

        # batch 中字段：mix_spec/srcA_spec/srcB_spec/srcA_time/srcB_time
        mix_spec = batch["mix_spec"].to(device)
        srcA_spec = batch["srcA_spec"].to(device)
        srcB_spec = batch["srcB_spec"].to(device)
        srcA_time = batch["srcA_time"].to(device)
        srcB_time = batch["srcB_time"].to(device)

        # 4) 未裁剪 target mask
        target_unclamped = compute_ideal_complex_mask(
            mix_spec=mix_spec,
            srcA_spec=srcA_spec,
            srcB_spec=srcB_spec,
            eps=getattr(cfg.numeric, "eps", 1e-8),
            mask_bound=None,
        )  # [B,4,F,T]

        if target_unclamped.ndim != 4 or target_unclamped.shape[1] != 4:
            raise RuntimeError(f"Unexpected target_unclamped shape: {tuple(target_unclamped.shape)}")

        # 5) 统计四通道 abs 分布：收集到 cpu 上做 quantile
        for ch in range(4):
            abs_flat = torch.abs(target_unclamped[:, ch]).detach().float().reshape(-1).cpu()
            abs_by_ch[ch].append(abs_flat)

        # 用于 clip_ratio 的元素总数（四通道一起算）
        elem_count += int(target_unclamped.numel())

        # 6~9) 对候选 bounds 计算 clip_ratio / oracle_mag_loss / (可选) oracle_corr
        abs_all = torch.abs(target_unclamped)
        for b in bounds:
            bound = float(b)
            clip_sum[bound] += float((abs_all > bound).float().sum().item())

            target_clamped = torch.clamp(target_unclamped, -bound, bound)
            pred_srcA_spec, pred_srcB_spec = apply_complex_mask(target_clamped, mix_spec)

            oracle_mag_loss = l1(torch.abs(pred_srcA_spec), torch.abs(srcA_spec)) + l1(
                torch.abs(pred_srcB_spec), torch.abs(srcB_spec)
            )
            mag_loss_sum[bound] += float(oracle_mag_loss.detach().item())

            if compute_corr:
                target_len = int(srcA_time.shape[-1])
                pred_srcA_time = istft_reconstruct(pred_srcA_spec, cfg=cfg, length=target_len)
                pred_srcB_time = istft_reconstruct(pred_srcB_spec, cfg=cfg, length=target_len)
                corr_a = batch_complex_corr(pred_srcA_time, srcA_time)
                corr_b = batch_complex_corr(pred_srcB_time, srcB_time)
                corr_a_sum[bound] += float(corr_a)
                corr_b_sum[bound] += float(corr_b)

        if compute_corr:
            corr_n += 1

    if batch_counter == 0:
        raise ValueError("No batch was analyzed (empty dataloader).")

    used_batches = min(batch_counter, max_batches) if max_batches > 0 else batch_counter

    # 5) 打印四通道分布
    print("=== Unclamped target_mask distribution (abs) per channel ===")
    for ch in range(4):
        abs_concat = torch.cat(abs_by_ch[ch], dim=0)
        stats = _compute_channel_stats(abs_concat)
        print(f"[{_channel_name(ch)}]  n={abs_concat.numel()}")
        print(
            "  "
            f"mean={stats['mean']:.6f}  std={stats['std']:.6f}  "
            f"min={stats['min']:.6f}  max={stats['max']:.6f}"
        )
        print(
            "  "
            f"abs_p90={stats['abs_p90']:.6f}  abs_p95={stats['abs_p95']:.6f}  "
            f"abs_p99={stats['abs_p99']:.6f}  abs_p99.5={stats['abs_p99.5']:.6f}  "
            f"abs_p99.9={stats['abs_p99.9']:.6f}"
        )
    print("")

    # 10) 打印候选 bound 表格
    print("=== Candidate bounds summary (avg over batches) ===")
    header = (
        f"{'bound':>6}  {'clip_ratio':>10}  {'oracle_mag_loss':>16}  "
        f"{'oracle_corr_a':>14}  {'oracle_corr_b':>14}  {'oracle_corr_mean':>16}"
    )
    print(header)
    print("-" * len(header))

    rows: List[Tuple[float, float, float, float, float, float]] = []
    for b in bounds:
        bound = float(b)
        clip_ratio = float(clip_sum[bound] / max(1, elem_count))
        oracle_mag_loss = float(mag_loss_sum[bound] / max(1, used_batches))
        if compute_corr and corr_n > 0:
            oracle_corr_a = float(corr_a_sum[bound] / corr_n)
            oracle_corr_b = float(corr_b_sum[bound] / corr_n)
            oracle_corr_mean = 0.5 * (oracle_corr_a + oracle_corr_b)
        else:
            oracle_corr_a = float("nan")
            oracle_corr_b = float("nan")
            oracle_corr_mean = float("nan")

        rows.append((bound, clip_ratio, oracle_mag_loss, oracle_corr_a, oracle_corr_b, oracle_corr_mean))
        print(
            f"{bound:6.1f}  "
            f"{_format_ratio(clip_ratio)}  "
            f"{_format_float(oracle_mag_loss, width=16)}  "
            f"{_format_float(oracle_corr_a, width=14)}  "
            f"{_format_float(oracle_corr_b, width=14)}  "
            f"{_format_float(oracle_corr_mean, width=16)}"
        )

    # 11) 推荐逻辑：clip_ratio 低 + mag_loss 接近最小 + corr 接近最高 的最小 bound
    mag_min = min(r[2] for r in rows)
    mag_tol = mag_min * 0.01  # 允许 1% 以内视作“接近最小”
    candidate_rows = [r for r in rows if r[2] <= mag_min + mag_tol]

    if compute_corr:
        corr_max = max(r[5] for r in rows if not torch.isnan(torch.tensor(r[5])))
        # 在 mag_loss 近似最小的集合里，优先 corr 接近最高（允许 0.001），再选最小 bound
        corr_tol = 1e-3
        candidate_rows = [r for r in candidate_rows if r[5] >= corr_max - corr_tol]

    # 在候选集合中选择 clip_ratio 最小、同时 bound 最小
    candidate_rows_sorted = sorted(candidate_rows, key=lambda x: (x[1], x[0]))
    recommended = candidate_rows_sorted[0] if len(candidate_rows_sorted) > 0 else sorted(rows, key=lambda x: x[0])[0]

    print("")
    print("=== Recommendation ===")
    print(
        f"recommended_bound={recommended[0]:.1f}  "
        f"clip_ratio={recommended[1]:.4%}  "
        f"oracle_mag_loss={recommended[2]:.6f}"
        + (f"  oracle_corr_mean={recommended[5]:.6f}" if compute_corr else "")
    )


if __name__ == "__main__":
    main()

