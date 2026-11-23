"""Soft vocal mask estimation (separator-based A5 and band-only A6)."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import stft


def _smooth_mask_tf(
    mask: np.ndarray,
    freqs: np.ndarray,
    sr: int = 44100,
    hop: int = 512,
    smooth_ms: float = 50.0,
) -> np.ndarray:
    """Smooth mask over time (moving average) and frequency (1/3-octave)."""
    # Time smoothing
    n_frames = max(1, int(round((smooth_ms / 1000.0) * sr / hop)))
    if n_frames > 1:
        mask = uniform_filter1d(mask, size=n_frames, axis=-1, mode="nearest")

    # Frequency smoothing: 1/3 octave around each bin
    out = np.empty_like(mask)
    factor = 2.0 ** (1.0 / 6.0)
    for i, f in enumerate(freqs):
        if f <= 0.0:
            out[i] = mask[i]
            continue
        lo = int(np.searchsorted(freqs, f / factor, side="left"))
        hi = int(np.searchsorted(freqs, f * factor, side="right"))
        hi = max(hi, lo + 1)
        out[i] = np.mean(mask[lo:hi], axis=0)
    return out


def compute_separator_mask(
    voc: np.ndarray,
    acc: np.ndarray,
    sr: int = 44100,
    n_fft: int = 2048,
    hop: int = 512,
    smooth_ms: float = 50.0,
    clamp_min: float = 0.1,
    clamp_max: float = 0.9,
) -> np.ndarray:
    """Compute soft STFT mask M of shape (channels, freq_bins, frames) in [clamp_min, clamp_max]."""
    assert voc.shape == acc.shape
    masks = []
    for ch in range(voc.shape[1]):
        freqs, _, zv = stft(voc[:, ch], fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop)
        _, _, za = stft(acc[:, ch], fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop)
        mag_v = np.abs(zv)
        mag_a = np.abs(za)
        m = mag_v / (mag_v + mag_a + 1e-8)
        m = _smooth_mask_tf(m, freqs, sr=sr, hop=hop, smooth_ms=smooth_ms)
        m = np.clip(m, clamp_min, clamp_max)
        masks.append(m.astype(np.float32))
    return np.stack(masks, axis=0)


def compute_band_mask(
    n_samples: int,
    n_channels: int = 2,
    sr: int = 44100,
    n_fft: int = 2048,
    hop: int = 512,
    f_low: float = 1000.0,
    f_high: float = 8000.0,
    clamp_min: float = 0.1,
    clamp_max: float = 0.9,
) -> np.ndarray:
    """Model-free 1-8 kHz soft band mask with 1/2-octave cosine transitions (A6)."""
    dummy = np.zeros(n_samples, dtype=np.float32)
    freqs, times, _ = stft(dummy, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop)

    # 1/2-octave cosine transitions around f_low and f_high
    lo_start = f_low / (2.0 ** 0.5)
    hi_end = f_high * (2.0 ** 0.5)

    w = np.zeros_like(freqs, dtype=np.float32)
    for i, f in enumerate(freqs):
        if f <= lo_start or f >= hi_end:
            w[i] = 0.0
        elif f < f_low:
            t = (np.log2(f) - np.log2(lo_start)) / (np.log2(f_low) - np.log2(lo_start))
            w[i] = 0.5 * (1.0 - np.cos(np.pi * t))
        elif f <= f_high:
            w[i] = 1.0
        else:
            t = (np.log2(hi_end) - np.log2(f)) / (np.log2(hi_end) - np.log2(f_high))
            w[i] = 0.5 * (1.0 - np.cos(np.pi * t))

    m_1d = clamp_min + (clamp_max - clamp_min) * w
    m_2d = np.repeat(m_1d[:, None], len(times), axis=1).astype(np.float32)
    return np.stack([m_2d] * n_channels, axis=0)
