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


def test_v4_6_stage3_discriminator_and_band_ratio() -> None:
    from scripts.extension.v4_discriminators import (
        HybridV4Discriminator,
        hinge_disc_loss,
        hinge_gen_loss,
        normalized_feature_matching_loss,
    )
    from scripts.extension.v4_losses import HybridV4Loss, SymmetricBandRatio6kLoss

    torch.manual_seed(11)
    m = HybridV4Net("v4_6", hop=512, gate_bias_init=-1.5)
    assert count_parameters_breakdown(m)["total"] == 1_953_200

    ckpt_p = ROOT / "artifacts/10_extension/v4/kaggle_out_v4_5p/out/v4_5p/best.pt"
    assert ckpt_p.is_file(), f"Expected v4_5p checkpoint at {ckpt_p}"
    ck = torch.load(ckpt_p, map_location="cpu", weights_only=False)
    m.load_state_dict(ck["model"], strict=True)

    # 1. SymmetricBandRatio6kLoss: zero on identical, symmetric, nonzero finite grad
    br_loss = SymmetricBandRatio6kLoss()
    a = (torch.randn(2, 2, 44100) * 0.1).requires_grad_(True)
    b = torch.randn(2, 2, 44100) * 0.1
    assert torch.allclose(br_loss(b, b), torch.zeros(()), atol=1e-6)
    l_ab = br_loss(a, b)
    l_ba = br_loss(b, a)
    assert torch.isclose(l_ab, l_ba, rtol=1e-5, atol=1e-6)
    l_ab.backward()
    assert a.grad is not None and torch.all(torch.isfinite(a.grad)) and float(a.grad.abs().sum()) > 0.0

    # 2. HybridV4Discriminator forward & losses
    disc = HybridV4Discriminator()
    d_params = sum(p.numel() for p in disc.parameters())
    assert d_params > 0
    r_lg, r_fm = disc(b)
    f_lg, f_fm = disc(a.detach())
    assert len(r_lg) == 6 and len(r_fm) == 6
    assert all(torch.all(torch.isfinite(t)) for t in r_lg + f_lg)
    assert float(hinge_disc_loss(r_lg, f_lg)) >= 0.0
    assert float(hinge_gen_loss(f_lg)) >= 0.0
    assert float(normalized_feature_matching_loss(r_fm, f_fm)) >= 0.0

    # 3. Real 2-step D + G optimization loop verifying phase_gate_head grad & zero D mutation during G step
    crit = HybridV4Loss()
    opt_g = torch.optim.AdamW(m.parameters(), lr=1.5e-4)
    opt_d = torch.optim.AdamW(disc.parameters(), lr=2e-4, betas=(0.5, 0.9))
    x_in = torch.randn(1, 2, 44100) * 0.1
    s_in = torch.randn(1, 2, 44100) * 0.1
    c_in = torch.randn(1, 2, 44100) * 0.1
    raw_stats = torch.zeros(1, 3)

    for step_idx in range(2):
        opt_g.zero_grad(set_to_none=True)
        y_hat, diag = m.forward_normalized(x_in, s_in, raw_stats)

        for p in disc.parameters():
            p.requires_grad_(True)
        opt_d.zero_grad(set_to_none=True)
        rl_d, _ = disc(c_in)
        fl_d, _ = disc(y_hat.detach())
        l_d = hinge_disc_loss(rl_d, fl_d)
        assert torch.isfinite(l_d)
        l_d.backward()
        opt_d.step()

        d_snapshot = [p.detach().clone() for p in disc.parameters()]

        for p in disc.parameters():
            p.requires_grad_(False)
        loss_rec, parts = crit(
            y_hat,
            c_in,
            diag,
            stage=2,
            lambda_ms=1.0,
            lambda_g=0.0,
            lambda_kl=2.0,
            lambda_br=0.25,
            w_comp=0.0,
            w_wav=0.0,
            lambda_mod=2.0,
        )
        assert parts["l_comp"] == 0.0 and "l_br" in parts
        with torch.no_grad():
            _, r_fmaps_g = disc(c_in)
        f_logits_g, f_fmaps_g = disc(y_hat)
        l_adv = hinge_gen_loss(f_logits_g)
        l_fm = normalized_feature_matching_loss(r_fmaps_g, f_fmaps_g)
        loss_g = loss_rec + 0.05 * l_adv + 0.50 * l_fm
        assert torch.isfinite(loss_g)
        loss_g.backward()

        assert m.decoder.phase_gate_head.weight.grad is not None
        assert torch.all(torch.isfinite(m.decoder.phase_gate_head.weight.grad))
        assert float(m.decoder.phase_gate_head.weight.grad.abs().sum()) > 0.0
        assert m.decoder.per_bin_heads[0].weight.grad is not None
        assert torch.all(torch.isfinite(m.decoder.per_bin_heads[0].weight.grad))
        assert float(m.decoder.per_bin_heads[0].weight.grad.abs().sum()) > 0.0

        opt_g.step()
        for p in disc.parameters():
            p.requires_grad_(True)

        for p_before, p_after in zip(d_snapshot, disc.parameters()):
            assert torch.equal(p_before, p_after), f"D mutated during G step {step_idx}"


def test_v4_7_blind_reverb_routing_and_dual_bright() -> None:
    from scripts.extension.v4_losses import HybridV4Loss

    torch.manual_seed(17)
    m = HybridV4Net("v4_7", hop=512, gate_bias_init=-1.5)
    m6 = HybridV4Net("v4_6", hop=512, gate_bias_init=-1.5)

    ckpt_p = ROOT / "artifacts/10_extension/v4/kaggle_out_v4_6/out/v4_6/best.pt"
    assert ckpt_p.is_file(), f"Expected v4_6 checkpoint at {ckpt_p}"
    ck = torch.load(ckpt_p, map_location="cpu", weights_only=False)
    m6.load_state_dict(ck["model"], strict=True)

    incompat = m.load_state_dict(ck["model"], strict=False)
    assert len(incompat.unexpected_keys) == 0
    assert len(incompat.missing_keys) > 0
    assert all(k.startswith("decoder.reverb_head.") or k == "decoder.gamma_band_logits" for k in incompat.missing_keys)

    # 1. WindowedModCueExtractor: exact zero when x == s, and length-invariant on periodic broadband audio (4s vs 8s)
    period_len = 86 * 512  # exact 1.0s stride = 86 STFT frames
    base_tile = torch.randn(1, 2, period_len) * 0.1
    base_echo = 0.8 * base_tile + 0.2 * torch.roll(base_tile, 1000, dims=-1)
    sig4 = base_tile.repeat(1, 1, 4)   # 4 * 86 = 344 frames (~4s)
    echo4 = base_echo.repeat(1, 1, 4)
    sig8 = base_tile.repeat(1, 1, 8)   # 8 * 86 = 688 frames (~8s)
    echo8 = base_echo.repeat(1, 1, 8)
    zx4 = stft_ms(lr_to_ms(sig4), n_fft=2048, hop=512)
    zx8 = stft_ms(lr_to_ms(sig8), n_fft=2048, hop=512)
    cues_same = m.decoder.mod_cue_extractor(zx4, zx4)
    assert cues_same.shape == (1, 24, 6) and torch.allclose(cues_same, torch.zeros_like(cues_same), atol=1e-6)
    zs4 = stft_ms(lr_to_ms(echo4), n_fft=2048, hop=512)
    zs8 = stft_ms(lr_to_ms(echo8), n_fft=2048, hop=512)
    assert torch.allclose(m.decoder.mod_cue_extractor(zx4, zs4), m.decoder.mod_cue_extractor(zx8, zs8), atol=2e-2)

    # 2. Force p_rev -> 0 matches V4-6; p_rev -> 1 opens g_ph > 0.99, lifts g_mag, and damps R -> 0
    m.eval()
    m6.eval()
    x_t = torch.randn(1, 2, 44100 * 3) * 0.1
    s_t = torch.randn(1, 2, 44100 * 3) * 0.12
    with torch.no_grad():
        y6, _ = m6.forward_track(x_t, s_t, chunk_sec=4.0)
        m.decoder.reverb_head.fc_out.bias.fill_(-12.0)
        y7_off, d_off = m.forward_track(x_t, s_t, chunk_sec=4.0)
        assert torch.max(torch.abs(y7_off - y6)) < 1e-3

        m.decoder.reverb_head.fc_out.bias.fill_(10.0)
        y7_on, d_on = m.forward_track(x_t, s_t, chunk_sec=4.0)
        assert d_on["g_ph_mean_M"] > 0.99 and d_on["g_ph_mean_S"] > 0.99
        assert d_on["g_mean_M"] > d_off["g_mean_M"] + 0.2 and d_on["g_mean_S"] > d_off["g_mean_S"] + 0.2
        old_rs = m.decoder.residual_scale
        m.decoder.residual_scale = 0.0
        y7_noR, _ = m.forward_track(x_t, s_t, chunk_sec=4.0)
        m.decoder.residual_scale = old_rs
        assert torch.max(torch.abs(y7_on - y7_noR)) < 5e-3

    # 3. Identity, x == s, silence, and RMS match
    m_fresh = HybridV4Net("v4_7", hop=512, gate_bias_init=-1.5)
    with torch.no_grad():
        y_id, _ = m_fresh.forward_track(x_t, s_t, chunk_sec=4.0, force_identity=True)
        rms_x = torch.sqrt(torch.mean(x_t**2) + 1e-8)
        rms_s = torch.sqrt(torch.mean(s_t**2) + 1e-8)
        assert torch.allclose(y_id, x_t * (rms_s / rms_x), atol=1e-4)

        y_same, _ = m_fresh.forward_track(x_t, x_t.clone(), chunk_sec=4.0, force_identity=False)
        assert torch.allclose(y_same, x_t, atol=1e-4)

        y_sil, _ = m_fresh.forward_track(torch.zeros_like(x_t), torch.zeros_like(x_t), chunk_sec=4.0, force_identity=False)
        assert torch.all(torch.isfinite(y_sil)) and torch.allclose(y_sil, torch.zeros_like(x_t), atol=1e-6)
        assert torch.isclose(torch.sqrt(torch.mean(y7_off**2)), rms_s, rtol=1e-3)

    # 4. Backward gradients flow cleanly through reverb_head, gamma_band_logits, phase_gate_head, and per_bin_heads
    m.train()
    crit = HybridV4Loss()
    xb = torch.randn(2, 2, 44100 * 2) * 0.1
    sb = torch.randn(2, 2, 44100 * 2) * 0.1
    cb = torch.randn(2, 2, 44100 * 2) * 0.1
    y_n, d_t = m.forward_normalized(xb, sb, torch.zeros(2, 3))
    assert d_t["rev_logit"].shape == (2,) and d_t["p_hf_ratio"].shape == (2,)
    loss, parts = crit(
        y_n,
        cb,
        d_t,
        stage=2,
        lambda_ms=1.0,
        lambda_g=0.0,
        lambda_kl=2.0,
        w_comp=0.15,
        w_wav=0.10,
        lambda_mod=2.0,
        lambda_br=0.60,
        lambda_br_p=0.25,
        rev_target=torch.tensor([0.0, 1.0]),
        lambda_rev=0.5,
        rev_fp_weight=2.0,
    )
    assert "l_rev" in parts and "l_br_p" in parts and "l_br" in parts
    loss.backward()
    assert any(p.grad is not None and torch.all(torch.isfinite(p.grad)) and float(p.grad.abs().sum()) > 0.0 for p in m.decoder.reverb_head.parameters())
    assert m.decoder.gamma_band_logits.grad is not None and torch.all(torch.isfinite(m.decoder.gamma_band_logits.grad)) and float(m.decoder.gamma_band_logits.grad.abs().sum()) > 0.0
    assert m.decoder.phase_gate_head.weight.grad is not None and torch.all(torch.isfinite(m.decoder.phase_gate_head.weight.grad)) and float(m.decoder.phase_gate_head.weight.grad.abs().sum()) > 0.0
    assert m.decoder.per_bin_heads[0].weight.grad is not None and torch.all(torch.isfinite(m.decoder.per_bin_heads[0].weight.grad)) and float(m.decoder.per_bin_heads[0].weight.grad.abs().sum()) > 0.0


def test_v4_7b_bright_quad_log_loss_and_freeze_head():
    from scripts.extension.v4_losses import HybridV4Loss, SymmetricBandRatio6kLoss

    br = SymmetricBandRatio6kLoss(quad_scale=400.0, log_weight=0.15)
    sig = torch.randn(2, 2, 44100 * 2) * 0.1
    assert float(br(sig, sig).item()) < 1e-6

    # Stronger outlier gradient with quad_scale=400.0
    eh_out = torch.tensor([0.08], requires_grad=True)
    eh_base = torch.tensor([0.08], requires_grad=True)
    et_out = torch.tensor([0.03])
    br._ratio_loss(eh_out, et_out, quad_scale=400.0, log_weight=0.0).backward()
    br._ratio_loss(eh_base, et_out, quad_scale=0.0, log_weight=0.0).backward()
    assert float(eh_out.grad.abs().item()) > float(eh_base.grad.abs().item())

    # Stronger relative sensitivity at low-HF with log_weight=0.15
    eh_low1 = torch.tensor([0.0007], requires_grad=True)
    eh_low2 = torch.tensor([0.0007], requires_grad=True)
    et_low = torch.tensor([0.0020])
    br._ratio_loss(eh_low1, et_low, quad_scale=0.0, log_weight=0.15).backward()
    br._ratio_loss(eh_low2, et_low, quad_scale=0.0, log_weight=0.0).backward()
    assert float(eh_low1.grad.abs().item()) > float(eh_low2.grad.abs().item())

    # Verify frozen reverb head receives zero parameter updates while per_bin_heads update
    torch.manual_seed(11)
    m = HybridV4Net("v4_7", hop=512, gate_bias_init=-1.5)
    for n, p in m.named_parameters():
        if n.startswith("decoder.reverb_head.") or n == "decoder.gamma_band_logits":
            p.requires_grad_(False)
    head_before = [p.detach().clone() for n, p in m.named_parameters() if n.startswith("decoder.reverb_head.") or n == "decoder.gamma_band_logits"]
    perbin_before = m.decoder.per_bin_heads[0].weight.detach().clone()

    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-3)
    crit = HybridV4Loss()
    xb = torch.randn(2, 2, 44100 * 2) * 0.1
    sb = torch.randn(2, 2, 44100 * 2) * 0.1
    cb = torch.randn(2, 2, 44100 * 2) * 0.1
    y_n, d_t = m.forward_normalized(xb, sb, torch.zeros(2, 3))
    loss, _ = crit(
        y_n,
        cb,
        d_t,
        stage=2,
        lambda_ms=1.5,
        lambda_g=0.0,
        lambda_kl=2.0,
        w_comp=0.15,
        w_wav=0.10,
        lambda_mod=2.0,
        lambda_br=2.0,
        lambda_br_p=1.0,
        br_quad_scale=400.0,
        br_log_weight=0.15,
        rev_target=torch.tensor([0.0, 1.0]),
        lambda_rev=0.0,
    )
    opt.zero_grad()
    loss.backward()
    opt.step()

    head_after = [p.detach() for n, p in m.named_parameters() if n.startswith("decoder.reverb_head.") or n == "decoder.gamma_band_logits"]
    for hb, ha in zip(head_before, head_after):
        assert torch.equal(hb, ha)
    assert not torch.equal(perbin_before, m.decoder.per_bin_heads[0].weight.detach())


def test_v4_7c_hf_deficit_routing_and_frozen_upstream():
    from scripts.extension.v4_losses import HybridV4Loss

    assert checkpoint_inference_config({}, "v4_7c")["gate_bias_init"] == -1.5
    torch.manual_seed(23)
    m7 = HybridV4Net("v4_7", hop=512, gate_bias_init=-1.5)
    m7c = HybridV4Net("v4_7c", hop=512, gate_bias_init=-1.5)
    p7 = count_parameters_breakdown(m7)
    p7c = count_parameters_breakdown(m7c)
    assert p7c["total"] == p7["total"] + 2 == 1966351

    incompat = m7c.load_state_dict(m7.state_dict(), strict=False)
    assert incompat.unexpected_keys == []
    assert sorted(incompat.missing_keys) == ["decoder.hf_def_alpha_logit", "decoder.hf_def_tau"]

    # Freeze encoder, blocks, reverb_head, and gamma_band_logits
    frozen_before = []
    for n, p in m7c.named_parameters():
        if n.startswith("encoder.") or n.startswith("blocks.") or n.startswith("decoder.reverb_head.") or n == "decoder.gamma_band_logits":
            p.requires_grad_(False)
            frozen_before.append((n, p.detach().clone()))

    # High E_s_hf / E_x_hf opens g_m_eff selectively at >=6kHz without changing g_ph_eff
    m7.load_state_dict(m7.state_dict())
    zx = torch.randn(1, 2, 1025, 172, dtype=torch.complex64) * 0.1
    zs = zx.clone()
    hf_mask = m7c.decoder.hf_band_mask
    zx[:, :, hf_mask, :] *= 0.1
    zs[:, :, hf_mask, :] *= 2.0
    m7c.eval()
    m7.eval()
    # Sync shared weights into m7 to verify selective HF mag lift vs phase invariance
    m7.load_state_dict({k: v for k, v in m7c.state_dict().items() if k in m7.state_dict()})
    with torch.no_grad():
        h_in = torch.randn(1, 172, 24, 64)
        _, d7 = m7.decoder(h_in, zx, zs)
        _, d7c = m7c.decoder(h_in, zx, zs)
    assert torch.allclose(d7["rev_logit"], d7c["rev_logit"], atol=1e-6)
    assert float(d7c["g_mean_M"]) > float(d7["g_mean_M"])
    assert torch.allclose(d7c["g_ph_mean_M"], d7["g_ph_mean_M"], atol=1e-6)
    assert torch.allclose(d7c["g_ph_mean_S"], d7["g_ph_mean_S"], atol=1e-6)

    # One optimizer step keeps frozen upstream + reverb_head bitwise identical while updating hf_def & per_bin
    m7c.train()
    opt = torch.optim.AdamW([p for p in m7c.parameters() if p.requires_grad], lr=5e-4)
    crit = HybridV4Loss()
    t = torch.linspace(0, 1, 44100 * 2)
    xb = (0.1 * torch.sin(2 * 3.14159 * 440 * t)).view(1, 1, -1).repeat(2, 2, 1)
    sb = xb + (0.08 * torch.sin(2 * 3.14159 * 10000 * t)).view(1, 1, -1).repeat(2, 2, 1)
    cb = sb.clone()
    a0_before = m7c.decoder.hf_def_alpha_logit.detach().clone()
    tau_before = m7c.decoder.hf_def_tau.detach().clone()
    perbin_before = m7c.decoder.per_bin_heads[20].weight.detach().clone()
    with torch.no_grad():
        _, d_before = m7c.forward_normalized(xb, sb, torch.zeros(2, 3))
    y_n, d_t = m7c.forward_normalized(xb, sb, torch.zeros(2, 3))
    loss, _ = crit(y_n, cb, d_t, stage=2, lambda_ms=1.3, lambda_g=0.0, lambda_kl=2.0, w_comp=0.15, w_wav=0.10, lambda_mod=2.0, lambda_br=0.80, lambda_br_p=0.30)
    opt.zero_grad()
    loss.backward()
    assert m7c.decoder.hf_def_alpha_logit.grad is not None and torch.isfinite(m7c.decoder.hf_def_alpha_logit.grad)
    assert m7c.decoder.hf_def_tau.grad is not None and torch.isfinite(m7c.decoder.hf_def_tau.grad)
    opt.step()

    with torch.no_grad():
        _, d_after = m7c.forward_normalized(xb, sb, torch.zeros(2, 3))
    assert torch.equal(d_before["rev_logit"], d_after["rev_logit"])
    for _, fb in frozen_before:
        # Compare all frozen parameters directly from named_parameters
        pass
    frozen_after = dict(m7c.named_parameters())
    for fn, fb in frozen_before:
        assert torch.equal(fb, frozen_after[fn].detach())
    assert not torch.equal(a0_before, m7c.decoder.hf_def_alpha_logit.detach())
    assert not torch.equal(tau_before, m7c.decoder.hf_def_tau.detach())
    assert not torch.equal(perbin_before, m7c.decoder.per_bin_heads[20].weight.detach())
