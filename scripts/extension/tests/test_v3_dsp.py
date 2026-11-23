"""Unit tests and sanity checks for Hybrid-v3 Differentiable DSP modules."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.signal import butter, sosfilt, sosfreqz

from scripts.extension.biquad_eq import biquad_coeffs_torch, eval_biquad_mag2, fit_biquad_eq
from scripts.extension.branch_p import _frame_rms_db
from scripts.extension.diff_drc_joint import fit_joint_drc
from scripts.extension.eval_dev import logmel, mel_ssim
from scripts.extension.xband_crossover import apply_xband_crossover, compute_xband_mask


def _pink_noise(n: int = 88200, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((n, 2))
    spec = np.fft.rfft(w, axis=0)
    f = np.fft.rfftfreq(n, d=1.0 / 44100.0)
    scale = 1.0 / np.maximum(f, 20.0) ** 0.5
    scale[0] = 0.0
    y = np.fft.irfft(spec * scale[:, None], n=n, axis=0)
    y = y / (np.max(np.abs(y)) + 1e-9) * 0.3
    return y.astype(np.float32)


def _peaking_sos(fc: float, gain_db: float, q: float, sr: int = 44100) -> np.ndarray:
    w0 = 2.0 * np.pi * fc / sr
    alpha = np.sin(w0) / (2.0 * q)
    A = 10.0 ** (gain_db / 40.0)
    b0 = 1.0 + alpha * A
    b1 = -2.0 * np.cos(w0)
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * np.cos(w0)
    a2 = 1.0 - alpha / A
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]], dtype=np.float64)


def test_biquad_freqz_exactness() -> None:
    sr = 44100
    fc = torch.tensor([120.0, 8000.0, 150.0, 350.0, 700.0, 1500.0, 3500.0, 6000.0], dtype=torch.float64)
    g = torch.tensor([3.0, -2.5, 4.5, -3.2, -6.0, 5.0, -2.5, 3.0], dtype=torch.float64)
    q = torch.tensor([0.707, 0.707, 1.0, 0.8, 1.5, 2.0, 1.2, 3.0], dtype=torch.float64)

    b0, b1, b2, a1, a2 = biquad_coeffs_torch(fc, g, q, sr)
    freqs = np.linspace(20.0, 20000.0, 512)
    omega = torch.tensor(2.0 * np.pi * freqs / sr, dtype=torch.float64)
    mag2_torch = eval_biquad_mag2(b0, b1, b2, a1, a2, omega).numpy()

    sos = np.stack([b0.numpy(), b1.numpy(), b2.numpy(), np.ones(8), a1.numpy(), a2.numpy()], axis=1)
    _, h_scipy = sosfreqz(sos, worN=2.0 * np.pi * freqs / sr)
    mag2_scipy = np.abs(h_scipy) ** 2
    assert np.max(np.abs(mag2_torch - mag2_scipy)) < 1e-5


def test_identity_all_modules() -> None:
    x = _pink_noise(88200, seed=1)
    for eq_phase in ("minphase", "zerophase"):
        y_eq, _ = fit_biquad_eq(x, x, sr=44100, eq_phase=eq_phase)
        assert np.all(np.isfinite(y_eq))
        # Compare spectral similarity (phase of HP/LP at 25Hz/19.5kHz may shift waveform slightly in minphase)
        ssim = mel_ssim(logmel(x.mean(1)), logmel(y_eq.mean(1)))
        assert ssim > 0.97

    y_drc, _ = fit_joint_drc(x, x, sr=44100, steps=40, device="cpu")
    assert np.all(np.isfinite(y_drc))
    assert mel_ssim(logmel(x.mean(1)), logmel(y_drc.mean(1))) > 0.96

    y_xb, _ = apply_xband_crossover(x, x, sr=44100)
    assert np.all(np.isfinite(y_xb))
    assert mel_ssim(logmel(x.mean(1)), logmel(y_xb.mean(1))) > 0.97


def test_synthetic_mud_recovery() -> None:
    clean = _pink_noise(132300, seed=7)
    sos_mud = _peaking_sos(350.0, 10.0, 1.5, sr=44100)
    deg = np.column_stack([sosfilt(sos_mud, clean[:, c]) for c in range(2)]).astype(np.float32)

    rec, _ = fit_biquad_eq(deg, clean, sr=44100, eq_phase="minphase", steps=300)
    ssim = mel_ssim(logmel(clean.mean(1)), logmel(rec.mean(1)))
    assert ssim > 0.98

    # Check 200-500 Hz energy ratio error
    f = np.fft.rfftfreq(len(clean), d=1.0 / 44100.0)
    band = (f >= 200.0) & (f <= 500.0)
    pc = np.abs(np.fft.rfft(clean.mean(1))) ** 2
    pr = np.abs(np.fft.rfft(rec.mean(1))) ** 2
    ratio_c = np.sum(pc[band]) / np.sum(pc)
    ratio_r = np.sum(pr[band]) / np.sum(pr)
    assert abs(ratio_r - ratio_c) < 0.02


def test_zerophase_gain_not_doubled() -> None:
    clean = _pink_noise(132300, seed=11)
    sos_mud = _peaking_sos(350.0, 8.0, 1.5, sr=44100)
    deg = np.column_stack([sosfilt(sos_mud, clean[:, c]) for c in range(2)]).astype(np.float32)
    rec, _ = fit_biquad_eq(deg, clean, sr=44100, eq_phase="zerophase", steps=300)
    ssim = mel_ssim(logmel(clean.mean(1)), logmel(rec.mean(1)))
    assert ssim > 0.98


def test_drc_compression_and_expansion() -> None:
    base = _pink_noise(132300, seed=19)
    t = np.linspace(0.0, 3.0, len(base), dtype=np.float32)[:, None]
    env = 0.25 + 0.75 * (0.5 + 0.5 * np.sin(2.0 * np.pi * 1.5 * t))
    dynamic_sig = (base * env).astype(np.float32)
    flat_sig = base.astype(np.float32)

    # 1. Needs compression: dynamic_sig -> flat_sig
    y_comp, _ = fit_joint_drc(dynamic_sig, flat_sig, sr=44100, steps=80, device="cpu")
    std_in = float(np.std(_frame_rms_db(dynamic_sig)))
    std_tgt = float(np.std(_frame_rms_db(flat_sig)))
    std_out = float(np.std(_frame_rms_db(y_comp)))
    assert abs(std_out - std_tgt) < abs(std_in - std_tgt)

    # 2. Needs expansion: flat_sig -> dynamic_sig (or compressed -> dynamic)
    y_exp, _ = fit_joint_drc(flat_sig, dynamic_sig, sr=44100, steps=100, device="cpu")
    std_in2 = float(np.std(_frame_rms_db(flat_sig)))
    std_tgt2 = float(np.std(_frame_rms_db(dynamic_sig)))
    std_out2 = float(np.std(_frame_rms_db(y_exp)))
    assert abs(std_out2 - std_tgt2) < abs(std_in2 - std_tgt2)


def test_xband_mask() -> None:
    s = _pink_noise(88200, seed=23)
    sos_lp = butter(8, 3000.0, btype="low", fs=44100, output="sos")
    x = np.column_stack([sosfilt(sos_lp, s[:, c]) for c in range(2)]).astype(np.float32)

    freqs, m_band = compute_xband_mask(x, s, sr=44100)
    assert np.mean(m_band[(freqs >= 100.0) & (freqs <= 2500.0)]) > 0.90
    assert np.mean(m_band[(freqs >= 5000.0) & (freqs <= 18000.0)]) < 0.05
