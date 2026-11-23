"""Step 0 Unit Tests for Hybrid-v4 (Rev 2.2-Final)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.extension.v4_models import (  # noqa: E402
    HybridV4Net,
    compute_track_norm,
    count_parameters_breakdown,
    istft_ms,
    lr_to_ms,
    ms_to_lr,
    stft_ms,
)


@pytest.mark.parametrize("hop", [512, 1024])
def test_stft_istft_roundtrip(hop: int) -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 2, 44100 * 2) * 0.2
    z = stft_ms(x, n_fft=2048, hop=hop)
    y = istft_ms(z, length=x.shape[-1], n_fft=2048, hop=hop)
    err = float(torch.max(torch.abs(x - y)).item())
    assert err < 1e-5, f"STFT/iSTFT roundtrip err={err}"


def test_ms_lr_roundtrip() -> None:
    torch.manual_seed(1)
    x = torch.randn(2, 2, 44100)
    y = ms_to_lr(lr_to_ms(x))
    err = float(torch.max(torch.abs(x - y)).item())
    assert err < 1e-6, f"M/S roundtrip err={err}"


def test_forced_identity_v4_2() -> None:
    torch.manual_seed(2)
    x = torch.randn(1, 2, 44100 * 4) * 0.1
    s = x.clone()
    model = HybridV4Net("v4_2", hop=512)
    model.eval()
    with torch.no_grad():
        y, diag = model.forward_track(x, s, chunk_sec=4.0, force_identity=True)
    err = float(torch.max(torch.abs(y - x)).item())
    assert err < 1e-5, f"Forced identity err={err}"
    assert diag["g_mean_M"] == 0.0


def test_actual_init_near_identity() -> None:
    torch.manual_seed(3)
    x = torch.randn(1, 2, 44100 * 4) * 0.1
    s = torch.randn(1, 2, 44100 * 4) * 0.1
    for var in ("v4_1", "v4_2"):
        model = HybridV4Net(var, hop=512)
        model.eval()
        with torch.no_grad():
            y, diag = model.forward_track(x, s, chunk_sec=4.0, force_identity=False)
        assert torch.all(torch.isfinite(y))
        if var == "v4_2":
            # g = sigmoid(-5.0) ~= 0.00669 -> SNR should be > 35 dB vs x
            assert 0.005 <= diag["g_mean_M"] <= 0.009, f"Unexpected init g={diag['g_mean_M']}"
            snr = 10.0 * torch.log10(torch.mean(x**2) / (torch.mean((y - x) ** 2) + 1e-12))
            assert float(snr.item()) > 35.0, f"Actual init SNR too low: {snr}"


def test_silence_and_extreme_low_volume() -> None:
    torch.manual_seed(4)
    model = HybridV4Net("v4_2", hop=512)
    model.eval()
    # Silence
    x_sil = torch.zeros(1, 2, 44100 * 2)
    s_sil = torch.zeros(1, 2, 44100 * 2)
    with torch.no_grad():
        y_sil, _ = model.forward_track(x_sil, s_sil, chunk_sec=4.0)
    assert torch.all(torch.isfinite(y_sil))

    # -35 dB volume drop
    s_ref = torch.randn(1, 2, 44100 * 2) * 0.2
    x_low = s_ref * (10.0 ** (-35.0 / 20.0))
    x_in, s_in, scale, raw_stats = compute_track_norm(x_low, s_ref)
    assert float(raw_stats[0, 2].item()) > 1.0  # gain_db / 24 > 1
    with torch.no_grad():
        y_low, _ = model.forward_track(x_low, s_ref, chunk_sec=4.0)
    assert torch.all(torch.isfinite(y_low))


@pytest.mark.parametrize("chunk_sec,overlap", [(4.0, 0.5), (8.0, 0.5)])
def test_ola_chunk_boundaries(chunk_sec: float, overlap: float) -> None:
    torch.manual_seed(5)
    x = torch.randn(1, 2, 44100 * 12) * 0.15  # 12s -> multiple chunks
    s = x.clone()
    model = HybridV4Net("v4_2", hop=512)
    model.eval()
    with torch.no_grad():
        y, _ = model.forward_track(x, s, chunk_sec=chunk_sec, overlap_ratio=overlap, force_identity=True)
    err = float(torch.max(torch.abs(y - x)).item())
    assert err < 1e-5, f"OLA boundary err={err}"


def test_parameter_counts() -> None:
    for var in ("v4_0", "v4_1", "v4_2", "v4_3", "v4_4"):
        m = HybridV4Net(var)
        counts = count_parameters_breakdown(m)
        assert 500_000 < counts["total"] < 3_500_000, f"Unexpected param count for {var}: {counts}"


def test_v4_3_forward_and_mel_kl_loss() -> None:
    from scripts.extension.v4_losses import HybridV4Loss

    torch.manual_seed(0)
    x = torch.randn(1, 2, 44100 * 3) * 0.1
    s = torch.randn(1, 2, 44100 * 3) * 0.25
    m = HybridV4Net("v4_3", hop=512, gate_bias_init=-1.5)
    y_id, _ = m.forward_track(x, s, chunk_sec=2.0, force_identity=True)
    rms_x = torch.sqrt(torch.mean(x**2) + 1e-8)
    rms_s = torch.sqrt(torch.mean(s**2) + 1e-8)
    assert torch.allclose(y_id, x * (rms_s / rms_x), atol=1e-4)

    y_out, diag = m.forward_track(x, s, chunk_sec=2.0, force_identity=False)
    rms_y = torch.sqrt(torch.mean(y_out**2))
    assert torch.isclose(rms_y, rms_s, rtol=1e-3)
    assert "g_mean_M" in diag and "dwk_rms" in diag

    crit = HybridV4Loss()
    x_n, s_n, _, raw_stats = (
        x / 0.2,
        s / 0.2,
        0.2,
        torch.zeros(1, 3),
    )
    y_n, d_t = m.forward_normalized(x_n, s_n, raw_stats)
    loss, parts = crit(y_n, s_n, d_t, stage=2, lambda_ms=0.5, lambda_g=0.0, lambda_kl=2.0)
    assert "l_kl" in parts and parts["l_kl"] >= 0.0
    loss.backward()


def test_v4_4_additive_residual() -> None:
    from scripts.extension.v4_losses import HybridV4Loss

    torch.manual_seed(1)
    x = torch.randn(1, 2, 44100 * 3) * 0.1
    s = torch.randn(1, 2, 44100 * 3) * 0.2
    m = HybridV4Net("v4_4", hop=512, gate_bias_init=-1.5)
    y_id, _ = m.forward_track(x, s, chunk_sec=2.0, force_identity=True)
    rms_x = torch.sqrt(torch.mean(x**2) + 1e-8)
    rms_s = torch.sqrt(torch.mean(s**2) + 1e-8)
    assert torch.allclose(y_id, x * (rms_s / rms_x), atol=1e-4)

    y_out, diag = m.forward_track(x, s, chunk_sec=2.0, force_identity=False)
    assert "res_rms" in diag and diag["res_rms"] < 1e-5  # zero-init preserves identity
    assert torch.isclose(torch.sqrt(torch.mean(y_out**2)), rms_s, rtol=1e-3)

    crit = HybridV4Loss()
    y_n, d_t = m.forward_normalized(x / 0.2, s / 0.2, torch.zeros(1, 3))
    loss, _ = crit(y_n, s / 0.2, d_t, stage=2, lambda_ms=1.0, lambda_g=0.0, lambda_kl=2.5)
    loss.backward()
