"""Module 1: Nercessian (2020) 8-Band Differentiable Biquad PEQ + BABE-2 LTAS Initialization."""
from __future__ import annotations

import librosa
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.signal import sosfilt, sosfiltfilt, stft

_MEL_CACHE: dict[tuple[int, int, int], np.ndarray] = {}


def _get_mel_matrix(sr: int = 44100, n_fft: int = 4096, n_mels: int = 128) -> np.ndarray:
    key = (sr, n_fft, n_mels)
    if key not in _MEL_CACHE:
        _MEL_CACHE[key] = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels).astype(np.float64)
    return _MEL_CACHE[key]


def _third_octave_centers(f_min: float = 20.0, f_max: float = 20000.0) -> np.ndarray:
    k_min = int(np.ceil(3.0 * np.log2(f_min / 1000.0)))
    k_max = int(np.floor(3.0 * np.log2(f_max / 1000.0)))
    return 1000.0 * (2.0 ** (np.arange(k_min, k_max + 1) / 3.0))


def _ltas_init(
    freqs: np.ndarray,
    p_x: np.ndarray,
    p_s: np.ndarray,
    sr: int,
    eq_clamp_db: float = 18.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Initialize 8-band PEQ (HP, LP, LS, HS, PK1..PK4) via 1/3-oct Gaussian-smoothed LTAS."""
    centers = _third_octave_centers(20.0, min(20000.0, 0.46 * sr))
    factor = 2.0 ** (1.0 / 6.0)
    diff_db = np.zeros(len(centers), dtype=np.float64)
    for i, fc in enumerate(centers):
        m = (freqs >= fc / factor) & (freqs <= fc * factor)
        if not np.any(m):
            idx = int(np.argmin(np.abs(freqs - fc)))
            ex, es = p_x[idx], p_s[idx]
        else:
            ex, es = float(np.sum(p_x[m])), float(np.sum(p_s[m]))
        diff_db[i] = 10.0 * np.log10((es + 1e-12) / (ex + 1e-12))

    smooth_db = gaussian_filter1d(diff_db, sigma=1.0, mode="nearest")
    smooth_db = np.clip(smooth_db, -eq_clamp_db, eq_clamp_db)

    # Order: [LS, HS, PK1..PK6] (all exact identity when gain_db == 0)
    init_fc = np.array([120.0, 8000.0, 150.0, 350.0, 700.0, 1500.0, 3500.0, 6000.0], dtype=np.float64)
    init_g = np.zeros(8, dtype=np.float64)
    init_q = np.array([0.707, 0.707, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4], dtype=np.float64)

    low_m = centers < 150.0
    high_m = centers > 6000.0
    if np.any(low_m):
        init_g[0] = float(np.mean(smooth_db[low_m])) * 0.7
    if np.any(high_m):
        init_g[1] = float(np.mean(smooth_db[high_m])) * 0.7

    # Pick up to 6 extrema in [80 Hz, 12 kHz] spaced >= 1/2 octave apart
    mid_mask = (centers >= 80.0) & (centers <= 12000.0)
    mid_idx = np.where(mid_mask)[0]
    order = mid_idx[np.argsort(-np.abs(smooth_db[mid_idx]))]
    chosen: list[tuple[float, float]] = []
    for idx in order:
        fc = float(centers[idx])
        if all(abs(np.log2(fc / c[0])) >= 0.50 for c in chosen):
            chosen.append((fc, float(smooth_db[idx]) * 0.6))
        if len(chosen) == 6:
            break
    chosen.sort(key=lambda t: t[0])
    for k, (fc, g) in enumerate(chosen):
        init_fc[2 + k] = fc
        init_g[2 + k] = float(np.clip(g, -eq_clamp_db * 0.8, eq_clamp_db * 0.8))

    return init_fc, init_g, init_q


def biquad_coeffs_torch(
    fc: torch.Tensor,
    gain_db: torch.Tensor,
    q: torch.Tensor,
    sr: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """RBJ Audio EQ Cookbook differentiable coefficients for [LS, HS, PK1..PK6].

    Returns normalized (b0, b1, b2, a1, a2) each of shape (8,) with a0 = 1.
    """
    w0 = 2.0 * np.pi * fc / float(sr)
    cos_w0 = torch.cos(w0)
    sin_w0 = torch.sin(w0)
    alpha = sin_w0 / (2.0 * q)
    A = torch.pow(10.0, gain_db / 40.0)
    sqrt_A = torch.sqrt(torch.clamp(A, min=1e-6))

    b0 = torch.empty_like(fc)
    b1 = torch.empty_like(fc)
    b2 = torch.empty_like(fc)
    a0 = torch.empty_like(fc)
    a1 = torch.empty_like(fc)
    a2 = torch.empty_like(fc)

    # 0: Low-shelf
    t_ls = 2.0 * sqrt_A[0] * alpha[0]
    b0[0] = A[0] * ((A[0] + 1.0) - (A[0] - 1.0) * cos_w0[0] + t_ls)
    b1[0] = 2.0 * A[0] * ((A[0] - 1.0) - (A[0] + 1.0) * cos_w0[0])
    b2[0] = A[0] * ((A[0] + 1.0) - (A[0] - 1.0) * cos_w0[0] - t_ls)
    a0[0] = (A[0] + 1.0) + (A[0] - 1.0) * cos_w0[0] + t_ls
    a1[0] = -2.0 * ((A[0] - 1.0) + (A[0] + 1.0) * cos_w0[0])
    a2[0] = (A[0] + 1.0) + (A[0] - 1.0) * cos_w0[0] - t_ls

    # 1: High-shelf
    t_hs = 2.0 * sqrt_A[1] * alpha[1]
    b0[1] = A[1] * ((A[1] + 1.0) + (A[1] - 1.0) * cos_w0[1] + t_hs)
    b1[1] = -2.0 * A[1] * ((A[1] - 1.0) + (A[1] + 1.0) * cos_w0[1])
    b2[1] = A[1] * ((A[1] + 1.0) + (A[1] - 1.0) * cos_w0[1] - t_hs)
    a0[1] = (A[1] + 1.0) - (A[1] - 1.0) * cos_w0[1] + t_hs
    a1[1] = 2.0 * ((A[1] - 1.0) - (A[1] + 1.0) * cos_w0[1])
    a2[1] = (A[1] + 1.0) - (A[1] - 1.0) * cos_w0[1] - t_hs

    # 2..7: Peaking
    b0[2:] = 1.0 + alpha[2:] * A[2:]
    b1[2:] = -2.0 * cos_w0[2:]
    b2[2:] = 1.0 - alpha[2:] * A[2:]
    a0[2:] = 1.0 + alpha[2:] / A[2:]
    a1[2:] = -2.0 * cos_w0[2:]
    a2[2:] = 1.0 - alpha[2:] / A[2:]

    return b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0


def eval_biquad_mag2(
    b0: torch.Tensor,
    b1: torch.Tensor,
    b2: torch.Tensor,
    a1: torch.Tensor,
    a2: torch.Tensor,
    omega: torch.Tensor,
) -> torch.Tensor:
    """Evaluate cascaded |H(e^{jw})|^2 on angular frequencies omega (F,)."""
    cos1 = torch.cos(omega)
    cos2 = torch.cos(2.0 * omega)
    sin1 = torch.sin(omega)
    sin2 = torch.sin(2.0 * omega)

    num_re = b0[:, None] + b1[:, None] * cos1[None, :] + b2[:, None] * cos2[None, :]
    num_im = -b1[:, None] * sin1[None, :] - b2[:, None] * sin2[None, :]
    den_re = 1.0 + a1[:, None] * cos1[None, :] + a2[:, None] * cos2[None, :]
    den_im = -a1[:, None] * sin1[None, :] - a2[:, None] * sin2[None, :]

    mag2_each = (num_re ** 2 + num_im ** 2) / (den_re ** 2 + den_im ** 2 + 1e-24)
    return torch.prod(mag2_each, dim=0)


def fit_biquad_eq(
    u: np.ndarray,
    s: np.ndarray,
    sr: int = 44100,
    n_fft: int = 4096,
    hop: int = 1024,
    eq_clamp_db: float = 18.0,
    steps: int = 250,
    lr: float = 0.05,
    lambda_kl: float = 10.0,
    lambda_sp: float = 0.01,
    eq_phase: str = "minphase",
    freq_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Fit 8-band differentiable biquad PEQ from vol-matched u to s and apply to u."""
    # volFirst: RMS-match u to s before estimating EQ
    rx = float(np.sqrt(np.mean(u.astype(np.float64) ** 2)))
    rs = float(np.sqrt(np.mean(s.astype(np.float64) ** 2)))
    xv = (u.astype(np.float64) * ((rs + 1e-9) / (rx + 1e-9))).astype(np.float32)

    p_x = np.zeros(n_fft // 2 + 1, dtype=np.float64)
    p_s = np.zeros(n_fft // 2 + 1, dtype=np.float64)
    for ch in range(xv.shape[1]):
        freqs, _, zx = stft(xv[:, ch], fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop)
        _, _, zs = stft(s[:, ch], fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop)
        p_x += np.mean(np.abs(zx) ** 2, axis=1)
        p_s += np.mean(np.abs(zs) ** 2, axis=1)

    init_fc, init_g, init_q = _ltas_init(freqs, p_x, p_s, sr, eq_clamp_db=eq_clamp_db)

    f_lo, f_hi = 20.0, 0.47 * sr
    q_lo, q_hi = 0.5, 8.0

    def _inv_sig(v: np.ndarray, lo: float, hi: float) -> np.ndarray:
        p = np.clip((v - lo) / (hi - lo), 1e-4, 1.0 - 1e-4)
        return np.log(p / (1.0 - p))

    raw_fc = torch.tensor(_inv_sig(np.log(init_fc), np.log(f_lo), np.log(f_hi)), dtype=torch.float64, requires_grad=True)
    raw_g = torch.tensor(np.arctanh(np.clip(init_g / eq_clamp_db, -0.99, 0.99)), dtype=torch.float64, requires_grad=True)
    raw_q = torch.tensor(_inv_sig(init_q, q_lo, q_hi), dtype=torch.float64, requires_grad=True)

    omega = torch.tensor(2.0 * np.pi * freqs / float(sr), dtype=torch.float64)
    px_t = torch.tensor(p_x, dtype=torch.float64)
    ps_t = torch.tensor(p_s, dtype=torch.float64)
    mel_mat = torch.tensor(_get_mel_matrix(sr=sr, n_fft=n_fft, n_mels=128), dtype=torch.float64)

    if freq_weights is None:
        w_t = torch.ones_like(px_t)
        w_t[0] = 0.0  # ignore DC
    else:
        w_t = torch.tensor(freq_weights, dtype=torch.float64)
    w_t = w_t / (torch.sum(w_t) + 1e-12)

    # Target Mel distribution
    mel_s = mel_mat @ ps_t
    mel_s_db = 10.0 * torch.log10(mel_s + 1e-12)
    mel_s_db = mel_s_db - torch.max(mel_s_db)
    # Shift to positive for KL (matching logmel dB distribution shape)
    p_mel_s = torch.softmax(mel_s_db / 10.0, dim=0)

    log_ps = 10.0 * torch.log10(ps_t + 1e-12)
    power_exp = 2.0 if eq_phase == "zerophase" else 1.0

    opt = torch.optim.Adam([raw_fc, raw_g, raw_q], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        fc = torch.exp(np.log(f_lo) + (np.log(f_hi) - np.log(f_lo)) * torch.sigmoid(raw_fc))
        g_db = eq_clamp_db * torch.tanh(raw_g)
        q = q_lo + (q_hi - q_lo) * torch.sigmoid(raw_q)

        b0, b1, b2, a1, a2 = biquad_coeffs_torch(fc, g_db, q, sr)
        mag2 = eval_biquad_mag2(b0, b1, b2, a1, a2, omega)
        if power_exp == 2.0:
            mag2 = mag2 * mag2

        px_eq = mag2 * px_t
        log_px_eq = 10.0 * torch.log10(px_eq + 1e-12)
        loss_spec = torch.sum(w_t * torch.abs(log_px_eq - log_ps))

        mel_x = mel_mat @ px_eq
        mel_x_db = 10.0 * torch.log10(mel_x + 1e-12)
        mel_x_db = mel_x_db - torch.max(mel_x_db)
        p_mel_x = torch.softmax(mel_x_db / 10.0, dim=0)
        loss_kl = torch.sum(p_mel_s * torch.log((p_mel_s + 1e-10) / (p_mel_x + 1e-10)))

        loss_sp = torch.sum(torch.abs(g_db))
        loss = loss_spec + lambda_kl * loss_kl + lambda_sp * loss_sp
        loss.backward()
        opt.step()

    with torch.no_grad():
        fc = torch.exp(np.log(f_lo) + (np.log(f_hi) - np.log(f_lo)) * torch.sigmoid(raw_fc))
        g_db = eq_clamp_db * torch.tanh(raw_g)
        q = q_lo + (q_hi - q_lo) * torch.sigmoid(raw_q)
        b0, b1, b2, a1, a2 = biquad_coeffs_torch(fc, g_db, q, sr)
        sos = np.stack(
            [
                b0.cpu().numpy(),
                b1.cpu().numpy(),
                b2.cpu().numpy(),
                np.ones(8, dtype=np.float64),
                a1.cpu().numpy(),
                a2.cpu().numpy(),
            ],
            axis=1,
        )

    out = np.empty_like(xv, dtype=np.float64)
    for ch in range(xv.shape[1]):
        if eq_phase == "zerophase":
            out[:, ch] = sosfiltfilt(sos, xv[:, ch].astype(np.float64))
        else:
            out[:, ch] = sosfilt(sos, xv[:, ch].astype(np.float64))

    info = {
        "fc_hz": fc.cpu().numpy().tolist(),
        "gain_db": g_db.cpu().numpy().tolist(),
        "q": q.cpu().numpy().tolist(),
        "sos": sos,
        "eq_phase": eq_phase,
    }
    return out.astype(np.float32), info
