"""`.mat` 读取工具（仅 I/Q 与复基带组装）。

职责：
- 读取 `.mat` 中 `RF0_I` 与 `RF0_Q`。
- 支持按窗口片段读取，避免每次读取整文件。
- 输出复基带信号 `x[n] = I[n] + jQ[n]`，类型为 `np.complex64`。
"""

from pathlib import Path
from typing import Optional, Union

import h5py
import numpy as np

PathLike = Union[str, Path]


def _as_path(mat_path: PathLike) -> Path:
    path = Path(mat_path)
    if not path.exists():
        raise FileNotFoundError(f"MAT file not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Path is not a file: {path}")
    return path


def _require_key(h5_file: h5py.File, key: str) -> h5py.Dataset:
    if key not in h5_file:
        raise KeyError(f"Key '{key}' not found in mat file.")
    dataset = h5_file[key]
    if not isinstance(dataset, h5py.Dataset):
        raise TypeError(f"'{key}' is not an h5py.Dataset.")
    return dataset


def _dataset_signal_length(dataset: h5py.Dataset, key: str) -> int:
    """返回信号长度（当前项目约束为 (1, N)）。"""
    if dataset.ndim != 2 or dataset.shape[0] != 1:
        raise ValueError(
            f"Unsupported shape for '{key}': {dataset.shape}. Expected strictly (1, N)."
        )
    return int(dataset.shape[1])


def _validate_range(signal_len: int, start: int, length: int) -> None:
    if start < 0:
        raise ValueError(f"start must be >= 0, got {start}.")
    if length <= 0:
        raise ValueError(f"length must be > 0, got {length}.")
    if start >= signal_len:
        raise IndexError(f"start out of range: start={start}, signal_len={signal_len}.")
    if start + length > signal_len:
        raise IndexError(
            f"slice out of range: start={start}, length={length}, signal_len={signal_len}."
        )


def _read_row_slice(
    dataset: h5py.Dataset,
    start: int,
    length: int,
    key: str,
) -> np.ndarray:
    if dataset.ndim != 2 or dataset.shape[0] != 1:
        raise ValueError(
            f"Unsupported shape for '{key}': {dataset.shape}. Expected strictly (1, N)."
        )
    data = dataset[0, start : start + length]
    return np.asarray(data, dtype=np.float32)


def get_signal_length(mat_path: PathLike, i_key: str = "RF0_I") -> int:
    """读取 `.mat` 内某个 I 通道数据长度。

    Args:
        mat_path: `.mat` 文件路径。
        i_key: I 分量字段名，默认 `RF0_I`。

    Returns:
        int: 信号长度 `N`。
    """
    path = _as_path(mat_path)
    with h5py.File(path, "r") as f:
        i_ds = _require_key(f, i_key)
        return _dataset_signal_length(i_ds, i_key)


def read_complex_rf0(
    mat_path: PathLike,
    start: Optional[int] = None,
    length: Optional[int] = None,
) -> np.ndarray:
    """读取 `RF0_I/Q` 并组装复基带信号。

    Args:
        mat_path: `.mat` 文件路径。
        start: 起始下标（含），默认 `None` 表示从 0 开始。
        length: 读取长度，默认 `None` 表示读到结尾。

    Returns:
        np.ndarray: 复基带信号，shape `[L]`，dtype `np.complex64`。
    """
    path = _as_path(mat_path)

    with h5py.File(path, "r") as f:
        i_ds = _require_key(f, "RF0_I")
        q_ds = _require_key(f, "RF0_Q")

        i_len = _dataset_signal_length(i_ds, "RF0_I")
        q_len = _dataset_signal_length(q_ds, "RF0_Q")
        if i_len != q_len:
            raise ValueError(f"RF0_I/RF0_Q length mismatch: {i_len} vs {q_len}.")

        s = 0 if start is None else int(start)
        l = (i_len - s) if length is None else int(length)
        _validate_range(i_len, s, l)

        i_part = _read_row_slice(i_ds, s, l, "RF0_I")
        q_part = _read_row_slice(q_ds, s, l, "RF0_Q")

    # 明确返回 complex64
    complex_sig = i_part.astype(np.complex64) + 1j * q_part.astype(np.complex64)
    return complex_sig.astype(np.complex64, copy=False)


if __name__ == "__main__":
    # 最小可运行示例：
    # python data/io_utils.py --mat "dataset/xxx.mat" --start 0 --length 1024
    import argparse

    parser = argparse.ArgumentParser(description="Read RF0 complex signal from .mat")
    parser.add_argument("--mat", type=str, required=True, help="Path to .mat file")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--length", type=int, default=1024, help="Read length")
    args = parser.parse_args()

    total_len = get_signal_length(args.mat, i_key="RF0_I")
    sig = read_complex_rf0(args.mat, start=args.start, length=args.length)

    print(f"total_len={total_len}")
    print(f"slice_shape={sig.shape}, dtype={sig.dtype}")
    print(f"first_3_samples={sig[:3]}")

