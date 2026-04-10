"""逻辑窗口索引构建。

职责：
- 基于 train/val/test 原始文件列表构建窗口索引（不落盘窗口数据）。
- 先按 10,000,000 点分块，再在块内按 65536 不重叠切窗，舍弃不足一窗。
- 不写死无人机类别，优先从文件名解析 `drone_code`（Txxxx）。
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple

from configs.config import ExperimentConfig, get_default_config
from data.io_utils import get_signal_length


FILENAME_RE = re.compile(r"^(T\d{4})_(D\d{2})_(S\d{4})\.mat$", re.IGNORECASE)


@dataclass
class WindowIndexItem:
    """单窗口索引项。"""

    split: str
    drone_code: str
    distance_code: str
    segment_code: str
    file_path: str
    block_id: int
    start: int
    length: int
    drone_name: Optional[str] = None


def parse_filename_tags(file_name: str) -> Dict[str, str]:
    """解析文件名标签。

    期望格式：`Txxxx_Dxx_Sxxxx.mat`
    """
    match = FILENAME_RE.match(file_name)
    if match is None:
        raise ValueError(
            f"Invalid file name format: {file_name}. Expected 'Txxxx_Dxx_Sxxxx.mat'."
        )

    drone_code, distance_code, segment_code = match.groups()
    return {
        "drone_code": drone_code.upper(),
        "distance_code": distance_code.upper(),
        "segment_code": segment_code.upper(),
    }


def build_file_windows(
    file_path: Path,
    split: str,
    block_len: int,
    window_len: int,
) -> Tuple[List[WindowIndexItem], Dict[str, str]]:
    """构建单文件窗口索引。"""
    tags = parse_filename_tags(file_path.name)
    signal_len = get_signal_length(file_path, i_key="RF0_I")
    # IQ分量等长
    items: List[WindowIndexItem] = []
    n_blocks = (signal_len + block_len - 1) // block_len

    for block_id in range(n_blocks):
        block_start = block_id * block_len
        block_end = min((block_id + 1) * block_len, signal_len)
        block_size = block_end - block_start
        n_windows = block_size // window_len

        for w in range(n_windows):
            start = block_start + w * window_len
            items.append(
                WindowIndexItem(
                    split=split,
                    drone_code=tags["drone_code"],
                    distance_code=tags["distance_code"],
                    segment_code=tags["segment_code"],
                    file_path=str(file_path.resolve()),
                    block_id=block_id,
                    start=start,
                    length=window_len,
                    drone_name=None,
                )
            )

    return items, tags


def _resolve_file_path(dataset_dir: Path, file_name_or_path: str) -> Path:
    p = Path(file_name_or_path)
    if not p.is_absolute():
        p = dataset_dir / p
    if not p.exists():
        raise FileNotFoundError(f"File listed in config not found: {p}")
    if p.suffix.lower() != ".mat":
        raise ValueError(f"Only .mat files are supported: {p}")
    return p


def build_split_index(
    split: str,
    file_list: List[str],
    dataset_dir: Path,
    block_len: int,
    window_len: int,
) -> Tuple[List[WindowIndexItem], Dict[str, Dict[str, int]]]:
    """构建某个 split 的全部索引及统计。"""
    split_items: List[WindowIndexItem] = []
    file_window_counts: Dict[str, int] = {}
    drone_window_counts: Dict[str, int] = {}

    for file_ref in file_list:
        file_path = _resolve_file_path(dataset_dir, file_ref)
        file_items, tags = build_file_windows(
            file_path=file_path,
            split=split,
            block_len=block_len,
            window_len=window_len,
        )
        split_items.extend(file_items)
        file_window_counts[file_path.name] = len(file_items)
        drone_code = tags["drone_code"]
        drone_window_counts[drone_code] = drone_window_counts.get(drone_code, 0) + len(file_items)

    stats = {
        "file_window_counts": file_window_counts,
        "split_total_windows": {split: len(split_items)},
        "drone_code_window_counts": drone_window_counts,
    }
    return split_items, stats


def save_index_json(index_items: List[WindowIndexItem], out_path: Path) -> None:
    """保存索引到 JSON。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(item) for item in index_items]
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _print_split_stats(split: str, stats: Dict[str, Dict[str, int]]) -> None:
    print(f"\n=== Split: {split} ===")
    print("Per-file window count:")
    for name, cnt in stats["file_window_counts"].items():
        print(f"  {name}: {cnt}")
    print("Total windows:")
    print(f"  {split}: {stats['split_total_windows'][split]}")
    print("Per-drone_code window count:")
    for code, cnt in stats["drone_code_window_counts"].items():
        print(f"  {code}: {cnt}")


def main() -> None:
    """构建并保存 train/val/test 索引。"""
    cfg: ExperimentConfig = get_default_config()

    dataset_dir = cfg.paths.dataset_dir
    block_len = cfg.data.block_len
    window_len = cfg.data.window_len

    split_to_files: Dict[str, List[str]] = {
        "train": cfg.file_split.train_files,
        "val": cfg.file_split.val_files,
        "test": cfg.file_split.test_files,
    }

    out_dir = cfg.paths.outputs_dir / "indexes"
    all_stats: Dict[str, Dict[str, Dict[str, int]]] = {}

    for split, file_list in split_to_files.items():
        if len(file_list) == 0:
            print(f"[Warning] split '{split}' file list is empty, skip.")
            continue

        split_items, split_stats = build_split_index(
            split=split,
            file_list=file_list,
            dataset_dir=dataset_dir,
            block_len=block_len,
            window_len=window_len,
        )
        save_index_json(split_items, out_dir / f"{split}_index.json")
        _print_split_stats(split, split_stats)
        all_stats[split] = split_stats

    if len(all_stats) > 0:
        stats_path = out_dir / "index_stats.json"
        with stats_path.open("w", encoding="utf-8") as f:
            json.dump(all_stats, f, ensure_ascii=False, indent=2)
        print(f"\nSaved stats to: {stats_path}")


if __name__ == "__main__":
    main()
