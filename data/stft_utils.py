"""STFT 与对数幅度谱工具。

职责：
- 计算复基带信号 STFT（复数谱）。
- 将复数谱转换为对数幅度谱。
- 将时频图补齐/裁剪到统一尺寸。

维度约定：
- `compute_stft` 输出复数谱：`[F, T]`
- `spec_to_logmag` 输出对数幅度谱：`[1, F, T]`
- `pad_or_crop_spectrogram` 输入输出建议使用：`[1, F, T]`
"""

from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import torch


ArrayLike1D = Union[np.ndarray, torch.Tensor]
SpecLike = Union[np.ndarray, torch.Tensor]


@dataclass
class STFTLikeConfig:
    """最小 STFT 配置类型（用于测试示例）。"""

    n_fft: int = 512
    hop_length: int = 256
    win_length: int = 512
    window: str = "hamming"
    center: bool = False
    onesided: bool = False


def _extract_stft_cfg(cfg: Any) -> STFTLikeConfig:
    """兼容 `ExperimentConfig` 或 `STFTConfig` 风格输入。"""
    stft_obj = getattr(cfg, "stft", cfg)
    return STFTLikeConfig(
        n_fft=int(getattr(stft_obj, "n_fft")),
        hop_length=int(getattr(stft_obj, "hop_length")),
        win_length=int(getattr(stft_obj, "win_length")),
        window=str(getattr(stft_obj, "window", "hamming")),
        center=bool(getattr(stft_obj, "center")),
        onesided=bool(getattr(stft_obj, "onesided")),
    )


def _to_1d_complex_torch(x: ArrayLike1D) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        x_t = torch.from_numpy(x)
    elif isinstance(x, torch.Tensor):
        x_t = x
    else:
        raise TypeError(f"Unsupported input type: {type(x)}")

    if x_t.ndim != 1:
        raise ValueError(f"Expected 1D signal, got shape={tuple(x_t.shape)}")
    if not torch.is_complex(x_t):
        x_t = x_t.to(torch.float32).to(torch.complex64)
    else:
        x_t = x_t.to(torch.complex64)
    return x_t


def _build_window(cfg: STFTLikeConfig, device: torch.device) -> torch.Tensor:
    win_name = cfg.window.lower()
    if win_name == "hamming":
        return torch.hamming_window(cfg.win_length, periodic=True, device=device)
    if win_name == "hann":
        return torch.hann_window(cfg.win_length, periodic=True, device=device)
    raise ValueError(f"Unsupported window type: {cfg.window}")


def compute_stft(x: ArrayLike1D, cfg: Any) -> torch.Tensor:
    """计算一维复基带信号 STFT。

    Args:
        x: 一维复基带信号（numpy 或 torch），shape `[N]`。
        cfg: 配置对象，需包含 STFT 参数（可为 `cfg.stft` 或直接 STFTConfig）。

    Returns:
        torch.Tensor: 复数谱，shape `[F, T]`，dtype `complex64`。
    """
    stft_cfg = _extract_stft_cfg(cfg)
    if stft_cfg.center is not False:
        raise ValueError("Current project requires center=False.")
    if stft_cfg.onesided is not False:
        raise ValueError("Current project requires onesided=False.")

    x_t = _to_1d_complex_torch(x)
    window = _build_window(stft_cfg, x_t.device)

    spec = torch.stft(
        input=x_t,
        n_fft=stft_cfg.n_fft,
        hop_length=stft_cfg.hop_length,
        win_length=stft_cfg.win_length,
        window=window,
        center=stft_cfg.center,
        onesided=stft_cfg.onesided,
        return_complex=True,
    )
    # torch.stft 返回 [F, T]
    return spec.to(torch.complex64)


def _to_complex_spec_torch(X: SpecLike) -> torch.Tensor:
    """将谱输入转换为 complex64 torch 张量。"""
    if isinstance(X, np.ndarray):
        x_t = torch.from_numpy(X)
    elif isinstance(X, torch.Tensor):
        x_t = X
    else:
        raise TypeError(f"Unsupported spectrum type: {type(X)}")

    if not torch.is_complex(x_t):
        raise TypeError("Expected complex spectrum input.")
    return x_t.to(torch.complex64)


def istft_reconstruct(
    X: SpecLike,
    cfg: Any,
    length: Optional[int] = None,
) -> torch.Tensor:
    """将复数谱反变换为时域复信号（iSTFT）。

    Args:
        X: 复数谱，支持 `[F, T]` 或 `[B, F, T]`。
        cfg: 配置对象，需包含 STFT 参数（可为 `cfg.stft` 或直接 STFTConfig）。
        length: 期望恢复长度；若为 None，则按 iSTFT 默认长度返回。

    Returns:
        torch.Tensor: 时域复信号。
            - 输入 `[F, T]` -> 输出 `[N]`
            - 输入 `[B, F, T]` -> 输出 `[B, N]`
    """
    stft_cfg = _extract_stft_cfg(cfg)
    if stft_cfg.center is not False:
        raise ValueError("Current project requires center=False.")
    if stft_cfg.onesided is not False:
        raise ValueError("Current project requires onesided=False.")

    x_spec = _to_complex_spec_torch(X)
    squeeze_back = False
    if x_spec.ndim == 2:
        x_spec = x_spec.unsqueeze(0)  # [1, F, T]
        squeeze_back = True
    elif x_spec.ndim != 3:
        raise ValueError(f"Expected spec shape [F,T] or [B,F,T], got {tuple(x_spec.shape)}")

    window = _build_window(stft_cfg, x_spec.device)

    # 逐样本 iSTFT，确保兼容批量输入与复数输出
    waves = []
    for i in range(x_spec.shape[0]):
        wi = torch.istft(
            input=x_spec[i],# batch的第i个样本
            n_fft=stft_cfg.n_fft,
            hop_length=stft_cfg.hop_length,
            win_length=stft_cfg.win_length,
            window=window,
            center=stft_cfg.center,
            onesided=stft_cfg.onesided,
            length=length,
            return_complex=True,
        )
        waves.append(wi.to(torch.complex64))

    wav = torch.stack(waves, dim=0)  # [B, N]
    return wav.squeeze(0) if squeeze_back else wav


def spec_to_logmag(X: SpecLike, eps: float = 1e-8) -> torch.Tensor:
    """复数谱转对数幅度谱。

    Args:
        X: 复数谱，shape `[F, T]`（或可被转为该维度）。
        eps: 数值稳定项。

    Returns:
        torch.Tensor: 对数幅度谱，shape `[1, F, T]`，dtype `float32`。
    """
    if isinstance(X, np.ndarray):
        X_t = torch.from_numpy(X)
    elif isinstance(X, torch.Tensor):
        X_t = X
    else:
        raise TypeError(f"Unsupported spectrum type: {type(X)}")

    if X_t.ndim != 2:
        raise ValueError(f"Expected spec shape [F, T], got {tuple(X_t.shape)}")
    if not torch.is_complex(X_t):
        raise TypeError("spec_to_logmag expects complex spectrum input.")

    mag = torch.abs(X_t).to(torch.float32)
    logmag = torch.log(mag + float(eps))
    return logmag.unsqueeze(0)  # [1, F, T]


def pad_or_crop_spectrogram(
    spec: SpecLike,
    target_f: Optional[int] = None,
    target_t: Optional[int] = None,
) -> torch.Tensor:
    """补齐或裁剪时频图到目标尺寸。

    Args:
        spec: 时频图，支持 `[1, F, T]` 或 `[F, T]`。
        target_f: 目标频率维大小；None 表示保持不变。
        target_t: 目标时间维大小；None 表示保持不变。

    Returns:
        torch.Tensor: 调整后的时频图，若输入为 `[1,F,T]` 则输出同形态；
            若输入为 `[F,T]` 则输出 `[F,T]`。
    """
    if isinstance(spec, np.ndarray):
        x = torch.from_numpy(spec)
    elif isinstance(spec, torch.Tensor):
        x = spec
    else:
        raise TypeError(f"Unsupported spectrogram type: {type(spec)}")

    keep_2d = False
    if x.ndim == 2:
        x = x.unsqueeze(0)  # -> [1, F, T]
        keep_2d = True
    elif x.ndim != 3 or x.shape[0] != 1:
        raise ValueError(f"Expected shape [1, F, T] or [F, T], got {tuple(x.shape)}")

    _, f, t = x.shape
    out_f = f if target_f is None else int(target_f)
    out_t = t if target_t is None else int(target_t)
    if out_f <= 0 or out_t <= 0:
        raise ValueError(f"target_f/target_t must be > 0, got {out_f}, {out_t}")

    # 先裁剪
    x = x[:, : min(f, out_f), : min(t, out_t)]

    # 再右侧/下侧零填充
    pad_f = out_f - x.shape[1]
    pad_t = out_t - x.shape[2]
    if pad_f > 0 or pad_t > 0:
        out = torch.zeros((1, out_f, out_t), dtype=x.dtype, device=x.device)
        out[:, : x.shape[1], : x.shape[2]] = x
        x = out

    return x.squeeze(0) if keep_2d else x


if __name__ == "__main__":
    # 最小可运行示例
    # python data/stft_utils.py
    n = 65_536
    fs = 100e6
    t = np.arange(n, dtype=np.float32) / fs

    # 复基带测试信号：两个复指数分量叠加
    sig = np.exp(1j * 2 * np.pi * 1.0e6 * t) + 0.5 * np.exp(1j * 2 * np.pi * (-2.0e6) * t)
    sig = sig.astype(np.complex64)

    cfg = STFTLikeConfig(
        n_fft=512,
        hop_length=256,
        win_length=512,
        window="hamming",
        center=False,
        onesided=False,
    )

    X = compute_stft(sig, cfg)                  # [F, T], complex64
    logmag = spec_to_logmag(X, eps=1e-8)        # [1, F, T], float32
    logmag_fix = pad_or_crop_spectrogram(logmag, target_f=512, target_t=256)
    rec = istft_reconstruct(X, cfg, length=n)   # [N], complex64
    mae = torch.mean(torch.abs(rec - torch.from_numpy(sig)))
    max_err = torch.max(torch.abs(rec - torch.from_numpy(sig)))

    print(f"X.shape={tuple(X.shape)}, dtype={X.dtype}")
    print(f"logmag.shape={tuple(logmag.shape)}, dtype={logmag.dtype}")
    print(f"logmag_fix.shape={tuple(logmag_fix.shape)}, dtype={logmag_fix.dtype}")
    print(f"sig.shape={sig.shape}, dtype={sig.dtype}")
    print(f"rec.shape={tuple(rec.shape)}, dtype={rec.dtype}")
    print(f"recon_mae={float(mae):.6e}, recon_max_abs_err={float(max_err):.6e}")
