"""Module 3: Blind xband Complementary Spectral Crossover."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import stft

from scripts.extension.biquad_eq import fit_biquad_eq
from scripts.extension.branch_p import match_dynamics, match_stereo, match_volume_lufs
from scripts.extension.mix import mix_complementary_stft


def compute_xband_mask(
    x: np.ndarray,
    s: np.ndarray,
    sr: int = 44100,
    n_fft: int = 2048,
    hop: int = 512,
    tau_db: float = -18.0,
    delta_db: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute smooth frequency mask M_band(f) in [0, 1] from blind LTAS ratio x vs s.

    Returns (freqs, m_band) where m_band has shape (n_fft // 2 + 1,).
    """
    n = min(len(x), len(s))
    x = x[:n]
    s = s[:n]

    p_x = np.zeros(n_fft // 2 + 1, dtype=np.float64)
    p_s = np.zeros(n_fft // 2 + 1, dtype=np.float64)
    for ch in range(x.shape[1]):
        freqs, _, zx = stft(x[:, ch], fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop)
        _, _, zs = stft(s[:, ch], fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop)
        p_x += np.mean(np.abs(zx) ** 2, axis=1)
        p_s += np.mean(np.abs(zs) ** 2, axis=1)

    # Match RMS in reference mid-band (400-2000 Hz)
    mid = (freqs >= 400.0) & (freqs <= 2000.0)
    ex_mid = float(np.sum(p_x[mid]))
    es_mid = float(np.sum(p_s[mid]))
    if ex_mid > 1e-12 and es_mid > 1e-12:
        p_x = p_x * (es_mid / ex_mid)

    # Smooth power spectra across ~1/3 octave (log-frequency interpolation)
    log_f = np.log2(np.maximum(freqs, 10.0))
    grid_log = np.linspace(np.log2(20.0), np.log2(0.49 * sr), 200)
    px_log = np.interp(grid_log, log_f, p_x)
    ps_log = np.interp(grid_log, log_f, p_s)
    # 1/3 octave is ~ 200 * (0.333 / 10) ~ 6.6 bins
    px_s = gaussian_filter1d(px_log, sigma=4.0, mode="nearest")
    ps_s = gaussian_filter1d(ps_log, sigma=4.0, mode="nearest")

    r_db_grid = 10.0 * np.log10((px_s + 1e-12) / (ps_s + 1e-12))
    r_db = np.interp(log_f, grid_log, r_db_grid)

    z = np.clip((r_db - tau_db) / delta_db, -30.0, 30.0)
    m_band = 1.0 / (1.0 + np.exp(-z))
    return freqs, m_band.astype(np.float32)


def apply_xband_crossover(
    x: np.ndarray,
    s: np.ndarray,
    sr: int = 44100,
    eq_clamp_db: float = 18.0,
    eq_phase: str = "minphase",
) -> tuple[np.ndarray, dict[str, object]]:
    """Apply frequency-weighted biquad PEQ to live bands and blend s on missing bands."""
    n = min(len(x), len(s))
    x = x[:n]
    s = s[:n]

    freqs_2k, m_band_2k = compute_xband_mask(x, s, sr=sr, n_fft=2048, hop=512)
    # Interpolate weights to n_fft=4096 grid for fit_biquad_eq
    freqs_4k = np.fft.rfftfreq(4096, d=1.0 / sr)
    w_4k = np.interp(freqs_4k, freqs_2k, m_band_2k)
    w_4k[0] = 0.0

    # Match mid-band RMS before PEQ so dead bands don't skew overall RMS
    p_out, eq_info = fit_biquad_eq(
        x, s, sr=sr, eq_clamp_db=eq_clamp_db, eq_phase=eq_phase, freq_weights=w_4k
    )
    p_out, _ = match_dynamics(p_out, s, sr=sr)
    p_out, _ = match_stereo(p_out, s)
    p_out, _ = match_volume_lufs(p_out, s, sr=sr)

    # Expand m_band_2k across time frames for mix_complementary_stft (n_fft=2048, hop=512)
    _, _, zs = stft(s[:, 0], fs=sr, window="hann", nperseg=2048, noverlap=2048 - 512)
    n_frames = zs.shape[1]
    mask_tf = np.broadcast_to(m_band_2k[None, :, None], (s.shape[1], len(m_band_2k), n_frames)).astype(np.float32)
    y, mix_info = mix_complementary_stft(p_out, s, mask_tf, sr=sr)
    info = {
        "live_band_frac": float(np.mean(m_band_2k > 0.5)),
        "eq_phase": eq_phase,
        "mix": mix_info,
    }
    return y, info
