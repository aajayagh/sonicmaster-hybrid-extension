"""Branch P: Parametric DSP estimation from s vs u, applied to u (never reads clean)."""
from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve, firwin2, stft

try:
    import pyloudnorm as pyln
except ImportError:  # pragma: no cover
    pyln = None


def _band_centers_third_octave(f_min: float = 20.0, f_max: float = 20000.0, bands_per_octave: int = 3) -> np.ndarray:
    k_min = int(np.ceil(float(bands_per_octave) * np.log2(f_min / 1000.0)))
    k_max = int(np.floor(float(bands_per_octave) * np.log2(f_max / 1000.0)))
    ks = np.arange(k_min, k_max + 1)
    return 1000.0 * (2.0 ** (ks / float(bands_per_octave)))


def match_eq_fir(
    u: np.ndarray,
    s: np.ndarray,
    sr: int = 44100,
    n_fft: int = 4096,
    hop: int = 1024,
    eq_clamp_db: float = 12.0,
    numtaps: int = 4097,
    *, bands_per_octave: int = 3, smooth: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate 1/3-octave EQ curve from s/u and apply linear-phase FIR to u."""
    # Average power spectrum across channels and time
    p_u = np.zeros(n_fft // 2 + 1, dtype=np.float64)
    p_s = np.zeros(n_fft // 2 + 1, dtype=np.float64)
    for ch in range(u.shape[1]):
        freqs, _, zu = stft(u[:, ch], fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop)
        _, _, zs = stft(s[:, ch], fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop)
        p_u += np.mean(np.abs(zu) ** 2, axis=1)
        p_s += np.mean(np.abs(zs) ** 2, axis=1)

    centers = _band_centers_third_octave(20.0, min(20000.0, 0.48 * sr), bands_per_octave)
    factor = 2.0 ** (1.0 / (2 * bands_per_octave))
    band_gains_db = np.zeros(len(centers), dtype=np.float64)

    for i, fc in enumerate(centers):
        mask = (freqs >= fc / factor) & (freqs <= fc * factor)
        if not np.any(mask):
            idx = int(np.argmin(np.abs(freqs - fc)))
            eu = p_u[idx]
            es = p_s[idx]
        else:
            eu = float(np.sum(p_u[mask]))
            es = float(np.sum(p_s[mask]))
        if eu < 1e-12 and es < 1e-12:
            band_gains_db[i] = 0.0
        else:
            band_gains_db[i] = 10.0 * np.log10(max(es, 1e-12) / max(eu, 1e-12))

    # Smooth over 3 adjacent bands [0.25, 0.5, 0.25]
    if smooth and len(band_gains_db) >= 3:
        padded = np.pad(band_gains_db, 1, mode="edge")
        band_gains_db = 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]

    band_gains_db = np.clip(band_gains_db, -eq_clamp_db, eq_clamp_db)

    if np.max(np.abs(band_gains_db)) < 0.05:
        return u.astype(np.float32, copy=False), band_gains_db

    nyq = 0.5 * sr
    grid_freqs = np.concatenate(([0.0], centers, [nyq]))
    grid_db = np.concatenate(([band_gains_db[0]], band_gains_db, [band_gains_db[-1]]))
    grid_gain = 10.0 ** (grid_db / 20.0)

    taps = firwin2(numtaps, grid_freqs, grid_gain, fs=sr)
    delay = (numtaps - 1) // 2
    out = fftconvolve(u.astype(np.float64), taps[:, None], mode="full", axes=0)
    out = out[delay : delay + u.shape[0]]
    return out.astype(np.float32), band_gains_db


def _frame_rms_db(x: np.ndarray, frame: int = 2048, hop: int = 1024) -> np.ndarray:
    mono = x.mean(axis=1).astype(np.float64)
    if len(mono) < frame:
        return np.array([20.0 * np.log10(np.sqrt(np.mean(mono ** 2)) + 1e-8)])
    n_frames = 1 + (len(mono) - frame) // hop
    idx = np.arange(frame)[None, :] + (np.arange(n_frames) * hop)[:, None]
    frames = mono[idx]
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-16)
    return 20.0 * np.log10(rms + 1e-8)


def _apply_compressor(
    x: np.ndarray,
    sr: int,
    threshold_db: float,
    ratio: float,
    attack_ms: float = 10.0,
    release_ms: float = 100.0,
) -> np.ndarray:
    mono = np.max(np.abs(x), axis=1).astype(np.float64)
    alpha_a = np.exp(-1.0 / max(1.0, sr * attack_ms / 1000.0))
    alpha_r = np.exp(-1.0 / max(1.0, sr * release_ms / 1000.0))

    env = np.empty_like(mono)
    cur = 0.0
    for i, v in enumerate(mono):
        if v > cur:
            cur = alpha_a * cur + (1.0 - alpha_a) * v
        else:
            cur = alpha_r * cur + (1.0 - alpha_r) * v
        env[i] = cur

    env_db = 20.0 * np.log10(env + 1e-8)
    over = np.maximum(env_db - threshold_db, 0.0)
    gain_db = -over * (1.0 - 1.0 / ratio)
    gain = (10.0 ** (gain_db / 20.0))[:, None]
    return (x.astype(np.float64) * gain).astype(np.float32)


def match_dynamics(u: np.ndarray, s: np.ndarray, sr: int = 44100) -> tuple[np.ndarray, str]:
    """Match frame RMS std of u to s using feed-forward compressor when u is more dynamic."""
    rms_u = _frame_rms_db(u)
    rms_s = _frame_rms_db(s)
    act_u = rms_u[rms_u > -60.0]
    act_s = rms_s[rms_s > -60.0]
    if len(act_u) < 4 or len(act_s) < 4:
        return u.astype(np.float32, copy=False), "skipped_too_quiet"

    std_u = float(np.std(act_u))
    std_s = float(np.std(act_s))
    if std_u <= std_s + 0.25:
        mode = "skipped_expansion" if std_s > std_u + 0.25 else "skipped_matched"
        return u.astype(np.float32, copy=False), mode

    thresholds = np.quantile(act_u, [0.25, 0.45, 0.65, 0.80])
    ratios = [1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
    best_err = abs(std_u - std_s)
    best_out = u.astype(np.float32, copy=False)
    best_tag = "skipped_no_improvement"

    for thr in thresholds:
        for rat in ratios:
            cand = _apply_compressor(u, sr, float(thr), float(rat))
            rc = _frame_rms_db(cand)
            ac = rc[rc > -60.0]
            if len(ac) < 4:
                continue
            err = abs(float(np.std(ac)) - std_s)
            if err < best_err:
                best_err = err
                best_out = cand
                best_tag = f"compressed_thr{thr:.1f}_rat{rat:.1f}"

    return best_out, best_tag


def match_stereo(u: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, float]:
    """Match Side/Mid RMS ratio of u to s, clamped to k in [0.0, 4.0]."""
    if u.shape[1] != 2 or s.shape[1] != 2:
        return u.astype(np.float32, copy=False), 1.0

    m_u = 0.5 * (u[:, 0] + u[:, 1]).astype(np.float64)
    s_u = 0.5 * (u[:, 0] - u[:, 1]).astype(np.float64)
    m_s = 0.5 * (s[:, 0] + s[:, 1]).astype(np.float64)
    s_s = 0.5 * (s[:, 0] - s[:, 1]).astype(np.float64)

    rms_mu = float(np.sqrt(np.mean(m_u ** 2)))
    rms_su = float(np.sqrt(np.mean(s_u ** 2)))
    rms_ms = float(np.sqrt(np.mean(m_s ** 2)))
    rms_ss = float(np.sqrt(np.mean(s_s ** 2)))

    if rms_ms < 1e-7:
        return u.astype(np.float32, copy=False), 1.0

    if rms_su < 1e-4:
        # Input is mono; cannot scale zero side channel, so borrow side from s
        out = np.column_stack((m_u + s_s, m_u - s_s)).astype(np.float32)
        return out, 0.0

    k = (rms_ss / rms_su) * (rms_mu / rms_ms)
    k = float(np.clip(k, 0.0, 4.0))
    if abs(k - 1.0) < 0.01:
        return u.astype(np.float32, copy=False), 1.0

    out = np.column_stack((m_u + k * s_u, m_u - k * s_u)).astype(np.float32)
    return out, k


def match_volume_lufs(u: np.ndarray, s: np.ndarray, sr: int = 44100) -> tuple[np.ndarray, float]:
    """Match integrated LUFS of u to s (with RMS fallback), clamped to [-24, +24] dB."""
    gain_db = 0.0
    used_lufs = False
    if pyln is not None and u.shape[0] >= int(0.4 * sr):
        try:
            meter = pyln.Meter(sr)
            lufs_u = float(meter.integrated_loudness(u.astype(np.float64)))
            lufs_s = float(meter.integrated_loudness(s.astype(np.float64)))
            if np.isfinite(lufs_u) and np.isfinite(lufs_s):
                gain_db = lufs_s - lufs_u
                used_lufs = True
        except Exception:
            used_lufs = False

    if not used_lufs:
        rms_u = float(np.sqrt(np.mean(u.astype(np.float64) ** 2)))
        rms_s = float(np.sqrt(np.mean(s.astype(np.float64) ** 2)))
        if rms_u > 1e-8 and rms_s > 1e-8:
            gain_db = 20.0 * np.log10(rms_s / rms_u)

    gain_db = float(np.clip(gain_db, -24.0, 24.0))
    gain = 10.0 ** (gain_db / 20.0)
    return (u.astype(np.float64) * gain).astype(np.float32), gain_db


def apply_branch_p(
    u: np.ndarray,
    s: np.ndarray,
    sr: int = 44100,
    eq_clamp_db: float = 18.0,
) -> tuple[np.ndarray, dict[str, object]]:
    """Apply parametric DSP in order: RMS-match -> EQ -> dynamics -> stereo -> LUFS."""
    rx = float(np.sqrt(np.mean(u.astype(np.float64) ** 2)))
    rs = float(np.sqrt(np.mean(s.astype(np.float64) ** 2)))
    xv = (u.astype(np.float64) * ((rs + 1e-9) / (rx + 1e-9))).astype(np.float32)
    y, eq_curve = match_eq_fir(xv, s, sr=sr, eq_clamp_db=eq_clamp_db)
    y, dyn_mode = match_dynamics(y, s, sr=sr)
    y, stereo_k = match_stereo(y, s)
    y, vol_gain_db = match_volume_lufs(y, s, sr=sr)
    info = {
        "eq_max_abs_db": float(np.max(np.abs(eq_curve))),
        "dynamics_mode": dyn_mode,
        "stereo_k": stereo_k,
        "vol_gain_db": vol_gain_db,
    }
    return y, info

