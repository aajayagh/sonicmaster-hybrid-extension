"""Step 0 Unit Tests for Hybrid-v4 (Rev 2.2-Final)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import numpy as np
import soundfile as sf
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
from scripts.extension.v4_alignment import align_teacher_to_input
from scripts.extension.v4_runtime import checkpoint_inference_config, spectral_scores
from scripts.extension.eval_dev import kl_div, logmel


def test_short_v4_4_and_regularizer_gradients():
    model = HybridV4Net("v4_4", gate_bias_init=-1.5)
    with torch.no_grad():
        model.decoder.mf_head.bias.fill_(0.03)
    x = torch.randn(1, 2, 8192)
    y, diag = model.forward_normalized(x, x, torch.zeros(1, 3))
    assert y.shape == x.shape and torch.isfinite(y).all()
    gate = diag["g_mean_M"] + diag["g_mean_S"]
    weights = diag["dw0_rms"].square() + diag["dwk_rms"].square()
    assert gate.requires_grad and weights.requires_grad
    grads = torch.autograd.grad(gate, tuple(model.decoder.parameters()), allow_unused=True, retain_graph=True)
    assert any(g is not None and g.abs().max() > 0 for g in grads)
    (weight_grad,) = torch.autograd.grad(weights, (model.decoder.mf_head.bias,))
    assert weight_grad.abs().max() > 0


def test_staging_crops_all_variants_at_same_offset(tmp_path):
    from scripts.extension.v4_stage_pilot_kaggle import process_source_group

    rng = np.random.default_rng(2)
    audio = rng.normal(0, .1, (230000, 2)).astype(np.float32)
    n_short = 200000
    for key, samples in (("clean", audio), ("long", audio), ("short", audio[:n_short])):
        sf.write(tmp_path / f"{key}.wav", samples, 44100, subtype="FLOAT")
    clean_dir, clip_dir = tmp_path / "clean", tmp_path / "clips"
    clean_dir.mkdir()
    clip_dir.mkdir()
    rows = [{"sample_id": key, "path_clean": str(tmp_path / "clean.wav"),
             "path_x": str(tmp_path / f"{key}.wav"), "path_s": str(tmp_path / f"{key}.wav")}
            for key in ("long", "short")]
    assert process_source_group("1", rows, str(clean_dir), str(clip_dir)) == ("1", 2)
    clean = np.load(clean_dir / "1.npy")
    assert clean.shape == (2, n_short)
    for row in rows:
        with np.load(clip_dir / f"{row['sample_id']}.npz") as clip:
            np.testing.assert_array_equal(clip["x"], clean)
            np.testing.assert_array_equal(clip["s"], clean)


@pytest.mark.parametrize("lag", [3, -3])
def test_teacher_alignment_zeros_boundary(lag):
    x = np.zeros((2, 128), dtype=np.float32)
    x[:, 50] = 1
    s = np.roll(x, lag, axis=-1)
    s[:, :3] = .25 if lag > 0 else 0
    s[:, -3:] = .25 if lag < 0 else 0
    np.testing.assert_array_equal(align_teacher_to_input(x, s), x)


def test_checkpoint_config_and_kl_order():
    assert checkpoint_inference_config({"variant": "v4_3"}, "v4_3")["gate_bias_init"] == -1.5
    config = {"hop": 1024, "gate_bias_init": -2.0, "chunk_sec": 6.0}
    ckpt = {"variant": "v4_4", "inference_config": config}
    assert checkpoint_inference_config(ckpt, "v4_4") == config
    with pytest.raises(ValueError, match="variant"):
        checkpoint_inference_config(ckpt, "v4_3")
    rng = np.random.default_rng(1)
    audio = rng.normal(size=(2, 8192))
    reference = logmel(np.hanning(8192) * audio[0])
    _, kl = spectral_scores(reference, audio)
    assert kl == pytest.approx(1000 * kl_div(logmel(audio.mean(axis=0)), reference))


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
    for var in ("v4_0", "v4_1", "v4_2", "v4_3", "v4_4", "v4_5p"):
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
    loss, parts = crit(y_n, s / 0.2, d_t, stage=2, lambda_ms=1.5, lambda_g=0.0, lambda_kl=2.5, lambda_hf=1.5)
    assert "l_hf" in parts and parts["l_hf"] >= 0.0
    loss.backward()


def test_v4_5p_decoupled_fusion_and_modspec_loss() -> None:
    from scripts.extension.v4_losses import HybridV4Loss

    torch.manual_seed(7)
    m = HybridV4Net("v4_5p", hop=512, gate_bias_init=-1.5)

    # 1. Orthogonal phases check directly on decoder math -> zero magnitude cancellation
    bsz, n_frames, n_bins = 1, 16, 1025
    h_in = torch.randn(bsz, n_frames, 24, 64)
    z_p = torch.complex(torch.ones(bsz, 2, n_bins, n_frames), torch.zeros(bsz, 2, n_bins, n_frames))
    z_s = torch.complex(torch.zeros(bsz, 2, n_bins, n_frames), torch.ones(bsz, 2, n_bins, n_frames))  # orthogonal 90 deg
    z_hat, diag = m.decoder(h_in, z_p, z_s, force_identity=False)
    # At init: dW=0, r=0, f_w == z_p (mag=1), z_s (mag=1) -> mag_fused == 1.0 even though |0.5P+0.5S|=0.707!
    assert torch.allclose(z_hat.abs(), torch.ones_like(z_hat.abs()), atol=1e-5)
    assert "g_ph_mean_M" in diag and "g_ph_mean_S" in diag

    # 2. Forced identity & exact x==s identity & RMS match & silence finiteness
    x = torch.randn(1, 2, 44100 * 3) * 0.1
    s = torch.randn(1, 2, 44100 * 3) * 0.2
    y_id, _ = m.forward_track(x, s, chunk_sec=2.0, force_identity=True)
    rms_x = torch.sqrt(torch.mean(x**2) + 1e-8)
    rms_s = torch.sqrt(torch.mean(s**2) + 1e-8)
    assert torch.allclose(y_id, x * (rms_s / rms_x), atol=1e-4)

    y_same, _ = m.forward_track(x, x.clone(), chunk_sec=2.0, force_identity=False)
    assert torch.allclose(y_same, x, atol=1e-4)

    y_sil, _ = m.forward_track(torch.zeros_like(x), torch.zeros_like(x), chunk_sec=2.0, force_identity=False)
    assert torch.all(torch.isfinite(y_sil)) and torch.allclose(y_sil, torch.zeros_like(x), atol=1e-6)

    y_out, _ = m.forward_track(x, s, chunk_sec=2.0, force_identity=False)
    assert torch.isclose(torch.sqrt(torch.mean(y_out**2)), rms_s, rtol=1e-3)

    # 3. ModSpecLoss + re-balanced HybridV4Loss gradients flow cleanly through phase_gate_head
    crit = HybridV4Loss()
    y_n, d_t = m.forward_normalized(x / 0.2, s / 0.2, torch.zeros(1, 3))
    loss, parts = crit(
        y_n,
        s / 0.2,
        d_t,
        stage=2,
        lambda_ms=1.0,
        lambda_g=0.0,
        lambda_kl=2.0,
        w_comp=0.25,
        w_wav=0.15,
        lambda_mod=2.0,
    )
    assert "l_mod" in parts and parts["l_mod"] >= 0.0
    loss.backward()
    assert m.decoder.phase_gate_head.weight.grad is not None
    assert torch.all(torch.isfinite(m.decoder.phase_gate_head.weight.grad))
    assert float(m.decoder.phase_gate_head.weight.grad.abs().sum().item()) > 0.0
    assert m.decoder.per_bin_heads[0].weight.grad is not None
    assert torch.all(torch.isfinite(m.decoder.per_bin_heads[0].weight.grad))
