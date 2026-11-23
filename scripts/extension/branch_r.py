"""Branch R (waveform residual) and zero-lag / phase checks."""
from __future__ import annotations

import numpy as np
from scipy.signal import correlate


class LagError(RuntimeError):
    """Raised when signals are not zero-lag aligned."""


def estimate_lag_samples(ref: np.ndarray, sig: np.ndarray, max_lag: int = 2048) -> tuple[int, float]:
    """Return (lag_samples, zero_lag_corr) on mono sum.

    Positive lag means `sig` is delayed relative to `ref`.
    """
    ref_m = ref.mean(axis=1) if ref.ndim == 2 else ref
    sig_m = sig.mean(axis=1) if sig.ndim == 2 else sig

    n = min(len(ref_m), len(sig_m))
    ref_m = ref_m[:n].astype(np.float64)
    sig_m = sig_m[:n].astype(np.float64)

    ref_norm = np.linalg.norm(ref_m)
    sig_norm = np.linalg.norm(sig_m)
    if ref_norm < 1e-12 or sig_norm < 1e-12:
        return 0, 1.0

    zero_corr = float(np.dot(ref_m, sig_m) / (ref_norm * sig_norm))

    # Use a central window up to 44100 * 10 samples for fast lag check
    win = min(n, 44100 * 10)
    start = (n - win) // 2
    a = ref_m[start : start + win]
    b = sig_m[start : start + win]
    if np.linalg.norm(a) < 1e-12 or np.linalg.norm(b) < 1e-12:
        a, b = ref_m, sig_m

    cc = correlate(b, a, mode="full", method="fft")
    mid = len(a) - 1
    lo = max(0, mid - max_lag)
    hi = min(len(cc), mid + max_lag + 1)
    best_idx = lo + int(np.argmax(cc[lo:hi]))
    lag = best_idx - mid
    return int(lag), zero_corr


def check_zero_lag(
    x: np.ndarray,
    s: np.ndarray,
    r: np.ndarray | None = None,
    max_lag: int = 2048,
    strict: bool = True,
) -> dict[str, float | int]:
    """Verify x, s (and optionally r) have identical shape and (if strict) lag == 0."""
    if x.shape != s.shape:
        raise LagError(f"Shape mismatch x={x.shape} vs s={s.shape}")
    lag_xs, corr_xs = estimate_lag_samples(x, s, max_lag=max_lag)
    if strict and lag_xs != 0:
        raise LagError(f"Non-zero lag between x and s: {lag_xs} samples (corr={corr_xs:.4f})")

    out: dict[str, float | int] = {"lag_xs": lag_xs, "phase_corr_xs": corr_xs}
    if r is not None:
        if x.shape != r.shape:
            raise LagError(f"Shape mismatch x={x.shape} vs r={r.shape}")
        lag_xr, corr_xr = estimate_lag_samples(x, r, max_lag=max_lag)
        if strict and lag_xr != 0:
            raise LagError(f"Non-zero lag between x and r: {lag_xr} samples (corr={corr_xr:.4f})")
        out["lag_xr"] = lag_xr
        out["phase_corr_xr"] = corr_xr
    return out


def soft_limit(y: np.ndarray, ceiling_db: float = -1.0) -> tuple[np.ndarray, bool]:
    """Soft-limit peaks above ceiling_db using smooth tanh knee."""
    ceiling = 10.0 ** (ceiling_db / 20.0)
    peak = float(np.max(np.abs(y)))
    if peak <= ceiling:
        return y.astype(np.float32, copy=False), False

    # Linear below 0.9 * ceiling, smooth tanh compression above
    knee = 0.9 * ceiling
    headroom = ceiling - knee
    mag = np.abs(y)
    over = np.maximum(mag - knee, 0.0)
    limited_mag = np.where(mag > knee, knee + headroom * np.tanh(over / headroom), mag)
    scale = np.where(mag > 1e-12, limited_mag / mag, 1.0)
    return (y * scale).astype(np.float32), True


def apply_branch_r(x: np.ndarray, s: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, bool]:
    """Compute V_R = x + (s - r) with true-peak soft-limiter at -1 dBTP."""
    v_r = x.astype(np.float64) + (s.astype(np.float64) - r.astype(np.float64))
    return soft_limit(v_r, ceiling_db=-1.0)
