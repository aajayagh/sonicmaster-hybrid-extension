"""Complementary STFT mixing and final LUFS / true-peak matching."""
from __future__ import annotations

import numpy as np
from scipy.signal import istft, stft

from .branch_p import match_volume_lufs
from .branch_r import soft_limit


def mix_complementary_stft(
    v: np.ndarray,
    s: np.ndarray,
    mask: np.ndarray,
    sr: int = 44100,
    n_fft: int = 2048,
    hop: int = 512,
) -> tuple[np.ndarray, dict[str, float | bool]]:
    """Compute out = ISTFT(M * STFT(V) + (1 - M) * STFT(s)), then match LUFS to s and limit -1 dBTP."""
    assert v.shape == s.shape
    n_samples, n_ch = v.shape
    channels = []
    for ch in range(n_ch):
        _, _, zv = stft(v[:, ch], fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop)
        _, _, zs = stft(s[:, ch], fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop)
        m = mask[ch]
        assert zv.shape == m.shape and zs.shape == m.shape, f"Mask shape {m.shape} vs STFT {zv.shape}"
        z_out = m * zv + (1.0 - m) * zs
        _, y_ch = istft(z_out, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop)
        channels.append(y_ch[:n_samples])

    out = np.stack(channels, axis=1).astype(np.float32)
    out, gain_db = match_volume_lufs(out, s, sr=sr)
    out, limited = soft_limit(out, ceiling_db=-1.0)
    return out, {"final_gain_db": gain_db, "final_limited": limited}
