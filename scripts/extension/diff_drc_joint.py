"""Module 2: DiffVox-style Differentiable DRC (Compressor + Expander) & Joint Fitting (PEQ + DRC + M/S)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from scripts.extension.biquad_eq import _get_mel_matrix, biquad_coeffs_torch, fit_biquad_eq
from scripts.extension.branch_p import match_volume_lufs


def _biquad_complex_freqz(
    b0: torch.Tensor,
    b1: torch.Tensor,
    b2: torch.Tensor,
    a1: torch.Tensor,
    a2: torch.Tensor,
    omega: torch.Tensor,
) -> torch.Tensor:
    """Complex transfer function H(e^{jw}) for cascaded 8 biquads on omega (F,)."""
    e1 = torch.complex(torch.cos(omega), -torch.sin(omega))
    e2 = torch.complex(torch.cos(2.0 * omega), -torch.sin(2.0 * omega))
    num = b0[:, None] + b1[:, None] * e1[None, :] + b2[:, None] * e2[None, :]
    den = 1.0 + a1[:, None] * e1[None, :] + a2[:, None] * e2[None, :]
    h_each = num / (den + 1e-12)
    h_tot = torch.prod(h_each, dim=0)
    return h_tot


def _soft_knee_drc_gain_db(
    env_db: torch.Tensor,
    thr_c: torch.Tensor,
    slope_c: torch.Tensor,
    thr_e: torch.Tensor,
    slope_e: torch.Tensor,
    knee_db: float = 6.0,
) -> torch.Tensor:
    """Two-region soft-knee DRC: downward compression above thr_c, expansion around thr_e.

    slope_c >= 0 -> downward compression (gain decreases above thr_c)
    slope_e in [-0.8, 0.8] -> upward/downward expansion below thr_e
    """
    half_w = 0.5 * knee_db
    # Compression region (env_db > thr_c)
    diff_c = env_db - thr_c
    over_c = torch.where(
        diff_c < -half_w,
        torch.zeros_like(diff_c),
        torch.where(
            diff_c > half_w,
            diff_c,
            ((diff_c + half_w) ** 2) / (2.0 * knee_db),
        ),
    )
    gain_c = -slope_c * over_c

    # Expansion region (env_db < thr_e): allows restoring compressed peaks/dynamics
    diff_e = thr_e - env_db
    under_e = torch.where(
        diff_e < -half_w,
        torch.zeros_like(diff_e),
        torch.where(
            diff_e > half_w,
            diff_e,
            ((diff_e + half_w) ** 2) / (2.0 * knee_db),
        ),
    )
    gain_e = -slope_e * under_e
    return torch.clamp(gain_c + gain_e, -18.0, 18.0)


def _mss_loss(y: torch.Tensor, s: torch.Tensor, windows: dict[int, torch.Tensor]) -> torch.Tensor:
    """Multi-Scale STFT L1 loss (linear + log magnitude) on shape (2, T)."""
    loss = torch.tensor(0.0, device=y.device, dtype=y.dtype)
    for n_fft, win in windows.items():
        hop = n_fft // 4
        zy = torch.stft(y, n_fft=n_fft, hop_length=hop, window=win, return_complex=True)
        zs = torch.stft(s, n_fft=n_fft, hop_length=hop, window=win, return_complex=True)
        my = torch.abs(zy) + 1e-7
        ms = torch.abs(zs) + 1e-7
        sc = torch.norm(ms - my, p="fro") / (torch.norm(ms, p="fro") + 1e-7)
        l1_log = F.l1_loss(torch.log(my), torch.log(ms))
        loss = loss + sc + l1_log
    return loss / float(len(windows))


def _mldr_loss(y_mono: torch.Tensor, s_mono: torch.Tensor) -> torch.Tensor:
    """Microdynamics MAE loss (frame RMS dB + short-vs-long contrast)."""
    # Frame=2048, hop=1024 matching Table 2 compression metric
    fy = y_mono.unfold(0, 2048, 1024)
    fs = s_mono.unfold(0, 2048, 1024)
    ry = 20.0 * torch.log10(torch.sqrt(torch.mean(fy ** 2, dim=1) + 1e-12) + 1e-7)
    rs = 20.0 * torch.log10(torch.sqrt(torch.mean(fs ** 2, dim=1) + 1e-12) + 1e-7)
    mae_rms = F.l1_loss(ry, rs)
    std_err = torch.abs(torch.std(ry) - torch.std(rs))

    # Short-term transient contrast (frame=512, hop=256)
    sy = y_mono.unfold(0, 512, 256)
    ss = s_mono.unfold(0, 512, 256)
    dy = 20.0 * torch.log10(torch.sqrt(torch.mean(sy ** 2, dim=1) + 1e-12) + 1e-7)
    ds = 20.0 * torch.log10(torch.sqrt(torch.mean(ss ** 2, dim=1) + 1e-12) + 1e-7)
    diff_y = torch.relu(dy[1:] - dy[:-1])
    diff_s = torch.relu(ds[1:] - ds[:-1])
    onset_err = F.l1_loss(diff_y, diff_s)
    return mae_rms + std_err + 0.5 * onset_err


def fit_joint_drc(
    u: np.ndarray,
    s: np.ndarray,
    sr: int = 44100,
    eq_clamp_db: float = 18.0,
    eq_phase: str = "minphase",
    steps: int = 120,
    lr: float = 0.01,
    lr_peq: float = 0.002,
    device: str | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Warm-start PEQ from Module 1, then jointly fit PEQ + DRC (Comp/Exp) + M/S width."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    # 1. Warm-start PEQ via Module 1
    _, eq_info = fit_biquad_eq(u, s, sr=sr, eq_clamp_db=eq_clamp_db, eq_phase=eq_phase)
    init_fc = np.array(eq_info["fc_hz"], dtype=np.float32)
    init_g = np.array(eq_info["gain_db"], dtype=np.float32)
    init_q = np.array(eq_info["q"], dtype=np.float32)

    rx = float(np.sqrt(np.mean(u.astype(np.float64) ** 2)))
    rs = float(np.sqrt(np.mean(s.astype(np.float64) ** 2)))
    xv_np = (u.astype(np.float64) * ((rs + 1e-9) / (rx + 1e-9))).astype(np.float32)

    n_samples = min(len(xv_np), len(s))
    xv_t = torch.tensor(xv_np[:n_samples].T, dtype=torch.float32, device=dev)  # (2, T)
    s_t = torch.tensor(s[:n_samples].T, dtype=torch.float32, device=dev)       # (2, T)

    # Check mono input side
    s_u_np = 0.5 * (xv_np[:n_samples, 0] - xv_np[:n_samples, 1])
    rms_su = float(np.sqrt(np.mean(s_u_np ** 2)))
    mono_input = rms_su < 1e-4

    # Precompute STFT of xv_t for fast differentiable PEQ
    n_fft_eq = 4096
    hop_eq = 1024
    win_eq = torch.hann_window(n_fft_eq, device=dev, dtype=torch.float32)
    zx = torch.stft(xv_t, n_fft=n_fft_eq, hop_length=hop_eq, window=win_eq, return_complex=True)
    freqs_np = np.fft.rfftfreq(n_fft_eq, d=1.0 / sr)
    omega = torch.tensor(2.0 * np.pi * freqs_np / float(sr), dtype=torch.float32, device=dev)

    f_lo, f_hi = 20.0, 0.47 * sr
    q_lo, q_hi = 0.5, 8.0

    def _inv_sig(v: np.ndarray, lo: float, hi: float) -> np.ndarray:
        p = np.clip((v - lo) / (hi - lo), 1e-4, 1.0 - 1e-4)
        return np.log(p / (1.0 - p))

    raw_fc = torch.tensor(_inv_sig(np.log(init_fc), np.log(f_lo), np.log(f_hi)), dtype=torch.float32, device=dev, requires_grad=True)
    raw_g = torch.tensor(np.arctanh(np.clip(init_g / eq_clamp_db, -0.99, 0.99)), dtype=torch.float32, device=dev, requires_grad=True)
    raw_q = torch.tensor(_inv_sig(init_q, q_lo, q_hi), dtype=torch.float32, device=dev, requires_grad=True)

    # DRC parameters (initialized near bypass: slope_c ~ 0, slope_e ~ 0)
    raw_tc = torch.tensor(-1.0, dtype=torch.float32, device=dev, requires_grad=True)     # thr_c in [-40, -2]
    raw_sc = torch.tensor(-3.5, dtype=torch.float32, device=dev, requires_grad=True)     # slope_c in [0, 0.9]
    raw_te = torch.tensor(-0.5, dtype=torch.float32, device=dev, requires_grad=True)     # thr_e in [-45, -10]
    raw_se = torch.tensor(0.0, dtype=torch.float32, device=dev, requires_grad=True)      # slope_e in [-0.6, 0.6]
    raw_kms = torch.tensor(np.arctanh(0.0), dtype=torch.float32, device=dev, requires_grad=True)  # k_ms around 1.0
    raw_gout = torch.tensor(0.0, dtype=torch.float32, device=dev, requires_grad=True)    # dB gain

    windows = {n: torch.hann_window(n, device=dev, dtype=torch.float32) for n in (512, 1024, 2048)}
    mel_mat = torch.tensor(_get_mel_matrix(sr=sr, n_fft=2048, n_mels=128), dtype=torch.float32, device=dev)

    # Target Mel distribution
    zs_mel = torch.stft(s_t.mean(dim=0, keepdim=True), n_fft=2048, hop_length=512, window=windows[2048], return_complex=True)
    ps_mel = torch.mean(torch.abs(zs_mel[0]) ** 2, dim=1)
    mel_s_db = 10.0 * torch.log10(mel_mat @ ps_mel + 1e-12)
    mel_s_db = mel_s_db - torch.max(mel_s_db)
    p_mel_s = torch.softmax(mel_s_db / 10.0, dim=0)

    # Target M/S
    s_m = 0.5 * (s_t[0] + s_t[1])
    s_s = 0.5 * (s_t[0] - s_t[1])
    s_ms = torch.stack([s_m, s_s], dim=0)

    # Exponential smoothing kernel for frame RMS (hop=128 -> ~2.9 ms/frame; 16 frames ~ 46 ms)
    k_len = 16
    tau_frames = 5.0
    exp_k = torch.exp(-torch.arange(k_len, device=dev, dtype=torch.float32) / tau_frames)
    exp_k = (exp_k / torch.sum(exp_k)).flip(0).view(1, 1, k_len)

    opt = torch.optim.Adam(
        [
            {"params": [raw_fc, raw_g, raw_q], "lr": lr_peq},
            {"params": [raw_tc, raw_sc, raw_te, raw_se, raw_kms, raw_gout], "lr": lr},
        ]
    )

    def _forward() -> torch.Tensor:
        fc = torch.exp(np.log(f_lo) + (np.log(f_hi) - np.log(f_lo)) * torch.sigmoid(raw_fc))
        g_db = eq_clamp_db * torch.tanh(raw_g)
        q = q_lo + (q_hi - q_lo) * torch.sigmoid(raw_q)
        b0, b1, b2, a1, a2 = biquad_coeffs_torch(fc, g_db, q, sr)
        h = _biquad_complex_freqz(b0, b1, b2, a1, a2, omega)
        if eq_phase == "zerophase":
            h_eff = torch.complex(torch.abs(h) ** 2, torch.zeros_like(h.real))
        else:
            h_eff = h

        zy = zx * h_eff[None, :, None]
        y_eq = torch.istft(zy, n_fft=n_fft_eq, hop_length=hop_eq, window=win_eq, length=n_samples)

        # Differentiable DRC (frame=512, hop=128)
        rms_frames = torch.sqrt(
            F.avg_pool1d(torch.mean(y_eq ** 2, dim=0, keepdim=True).unsqueeze(0), kernel_size=512, stride=128, padding=256)
            + 1e-12
        )  # (1, 1, F)
        env = F.conv1d(F.pad(rms_frames, (k_len - 1, 0), mode="replicate"), exp_k).squeeze(0).squeeze(0)
        env_db = 20.0 * torch.log10(env + 1e-7)

        thr_c = -40.0 + 38.0 * torch.sigmoid(raw_tc)
        slope_c = 0.9 * torch.sigmoid(raw_sc)
        thr_e = -45.0 + 35.0 * torch.sigmoid(raw_te)
        slope_e = 0.6 * torch.tanh(raw_se)
        gain_db = _soft_knee_drc_gain_db(env_db, thr_c, slope_c, thr_e, slope_e)
        gain_frames = torch.pow(10.0, gain_db / 20.0).view(1, 1, -1)
        gain_samples = F.interpolate(gain_frames, size=n_samples, mode="linear", align_corners=False).squeeze(0).squeeze(0)
        y_drc = y_eq * gain_samples[None, :]

        # Stereo M/S width + output gain
        m_y = 0.5 * (y_drc[0] + y_drc[1])
        s_y = 0.5 * (y_drc[0] - y_drc[1])
        if mono_input:
            y_st = torch.stack([m_y + s_s, m_y - s_s], dim=0)
        else:
            k_ms = 2.0 * torch.sigmoid(raw_kms) * 2.0  # in (0, 4), init at 2.0? Wait: sigmoid(0)=0.5 -> *2 = 1.0!
            # Wait: 4.0 * sigmoid(raw_kms) has init at 2.0! Let's use exp(tanh(raw_kms)*log(4)) so 0 -> 1.0!
            k_ms = torch.exp(torch.tanh(raw_kms) * float(np.log(4.0)))
            y_st = torch.stack([m_y + k_ms * s_y, m_y - k_ms * s_y], dim=0)

        g_out = torch.pow(10.0, (6.0 * torch.tanh(raw_gout)) / 20.0)
        return y_st * g_out

    for _ in range(steps):
        opt.zero_grad()
        y_out = _forward()
        y_m = 0.5 * (y_out[0] + y_out[1])
        y_s = 0.5 * (y_out[0] - y_out[1])
        y_ms = torch.stack([y_m, y_s], dim=0)

        loss_lr = _mss_loss(y_out, s_t, windows)
        loss_ms = _mss_loss(y_ms, s_ms, windows)
        loss_dyn = _mldr_loss(y_m, s_m)

        zy_mel = torch.stft(y_m.unsqueeze(0), n_fft=2048, hop_length=512, window=windows[2048], return_complex=True)
        py_mel = torch.mean(torch.abs(zy_mel[0]) ** 2, dim=1)
        mel_y_db = 10.0 * torch.log10(mel_mat @ py_mel + 1e-12)
        mel_y_db = mel_y_db - torch.max(mel_y_db)
        p_mel_y = torch.softmax(mel_y_db / 10.0, dim=0)
        loss_kl = torch.sum(p_mel_s * torch.log((p_mel_s + 1e-10) / (p_mel_y + 1e-10)))

        loss = loss_lr + 0.5 * loss_ms + 0.25 * loss_dyn + 5.0 * loss_kl
        loss.backward()
        opt.step()

    with torch.no_grad():
        y_final = _forward().T.cpu().numpy().astype(np.float32)

    y_final, vol_db = match_volume_lufs(y_final, s[:n_samples], sr=sr)
    info = {
        "device": device,
        "steps": steps,
        "vol_gain_db": vol_db,
        "slope_c": float(0.9 * torch.sigmoid(raw_sc).item()),
        "slope_e": float(0.6 * torch.tanh(raw_se).item()),
    }
    return y_final, info
