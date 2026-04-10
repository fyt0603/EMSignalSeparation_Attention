import os
from pathlib import Path
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

def _read_complex_channel(mat_path, i_key, q_key, n_samples) -> np.ndarray:
    with h5py.File(mat_path, "r") as f:
        i = f[i_key][0, :n_samples].astype(np.float32, copy=False)
        q = f[q_key][0, :n_samples].astype(np.float32, copy=False)
    return i + 1j * q

def stft(x, fs, n_fft, hop) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_t = torch.as_tensor(x, dtype=torch.complex64, device=device)
    win_t = torch.hamming_window(n_fft, periodic=True, dtype=torch.float32, device=device)

    X = torch.stft(
        x_t,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=win_t,
        center=False,
        onesided=False,
        return_complex=True,
    )

    frames = X.shape[1]
    eps = torch.tensor(1e-12, dtype=torch.float32, device=device)
    mag_db_t = 20.0 * torch.log10(torch.clamp(X.abs(), min=eps))

    f_t = torch.fft.fftfreq(n_fft, d=1.0 / fs, device=device)
    t_t = (torch.arange(frames, device=device, dtype=torch.float32) * hop) / fs

    mag_db_t = torch.fft.fftshift(mag_db_t, dim=0)
    f_t = torch.fft.fftshift(f_t)

    f = f_t.detach().cpu().numpy()
    t = t_t.detach().cpu().numpy()
    mag_db = mag_db_t.detach().cpu().numpy().astype(np.float32, copy=False)

    return f, t, mag_db

if __name__ == "__main__":
    dataset_dir = Path("dataset")
    mat_files = sorted(dataset_dir.glob("*.mat"))

    fs = 100e6
    seg_seconds = 0.5
    n_fft = 2048
    hop = 1024
    fc = 2440e6
    max_frames_to_plot = 8000

    out_dir = Path("outputs_stft_figure")
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, mat_path in enumerate(mat_files, start=1):
        # 根据文件内通道长度与采样率计算整段信号总时长
        with h5py.File(mat_path, "r") as f:
            n = f["RF0_I"].shape[1]
        total_seconds = n / fs
        print(f"[{idx}/{len(mat_files)}] {mat_path.name} total={total_seconds:.6f}s fs={fs}")

        n_samples = int(fs * total_seconds)
        x = _read_complex_channel(mat_path, "RF0_I", "RF0_Q", n_samples=n_samples)

        print(f"stft:samples={len(x)}, fs={fs}, N={n_fft}")
        f, t, mag_db = stft(x, fs=fs, n_fft=n_fft, hop=hop)

        if mag_db.shape[1] > max_frames_to_plot:
            step = int(np.ceil(mag_db.shape[1] / max_frames_to_plot))
            print(f"frames={mag_db.shape[1]}，每 {step} 帧取1帧用于显示")
            mag_db_plot = mag_db[:, ::step]
            t_plot = t[::step]
        else:
            mag_db_plot = mag_db
            t_plot = t

        out_png = out_dir / f"{mat_path.stem}_stft_{n_fft}_hamming.png"

        plt.figure(figsize=(12, 5), dpi=150)
        plt.pcolormesh(t_plot, (f + fc) / 1e6, mag_db_plot, shading="auto", cmap="jet")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (MHz)")
        plt.title(f"STFT (N={n_fft}, Hamming) fs={fs/1e6:.0f}MS/s fc={fc/1e6:.0f}MHz")
        plt.colorbar(label="Magnitude (dB)")
        plt.tight_layout()
        plt.savefig(out_png, bbox_inches="tight")
        plt.close()

        print(f"已保存时频图：{out_png.resolve()}")