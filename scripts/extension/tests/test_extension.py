"""Unit tests for 10_extension pipeline."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.extension.branch_p import apply_branch_p, match_eq_fir
from scripts.extension.branch_r import LagError, apply_branch_r, check_zero_lag
from scripts.extension.mask import compute_band_mask, compute_separator_mask
from scripts.extension.mix import mix_complementary_stft
from scripts.extension.routing import route_degradations
from scripts.extension import run_pipeline


def _tone(sr: int = 44100, sec: float = 2.0) -> np.ndarray:
    t = np.linspace(0.0, sec, int(sr * sec), endpoint=False, dtype=np.float32)
    rng = np.random.default_rng(42)
    l = 0.2 * np.sin(2 * np.pi * 440.0 * t) + 0.05 * rng.standard_normal(len(t)).astype(np.float32)
    r = 0.2 * np.sin(2 * np.pi * 440.0 * t) + 0.05 * rng.standard_normal(len(t)).astype(np.float32)
    return np.column_stack((l, r)).astype(np.float32)


def test_identity_x_eq_s_eq_r():
    x = _tone()
    v_r, lim = apply_branch_r(x, x, x)
    assert not lim
    assert np.max(np.abs(v_r - x)) < 1e-6

    m = np.full((2, 1025, int(np.ceil(len(x) / 512)) + 1), 0.5, dtype=np.float32)
    # Use compute_band_mask for exact STFT shape
    m = compute_band_mask(len(x), n_channels=2, sr=44100)
    out, _ = mix_complementary_stft(v_r, x, m, sr=44100)
    assert np.sqrt(np.mean((out - x) ** 2)) < 1e-4


def test_mask_extremes():
    x = _tone()
    v = 0.5 * x
    s = 0.25 * x
    m_all = compute_band_mask(len(x), n_channels=2, sr=44100, clamp_min=1.0, clamp_max=1.0)
    out_v, _ = mix_complementary_stft(v, s, m_all, sr=44100)
    # Volume matched to s, so direction/ratio matches v (and s)
    assert np.corrcoef(out_v[:, 0], s[:, 0])[0, 1] > 0.999

    m_zero = compute_band_mask(len(x), n_channels=2, sr=44100, clamp_min=0.0, clamp_max=0.0)
    out_s, _ = mix_complementary_stft(v, s, m_zero, sr=44100)
    assert np.sqrt(np.mean((out_s - s) ** 2)) < 1e-4


def test_branch_p_recovers_known_eq():
    rng = np.random.default_rng(7)
    x = 0.1 * rng.standard_normal((44100 * 3, 2)).astype(np.float32)
    # Apply +6 dB high-shelf-like boost above 2 kHz via sos filter
    sos = butter(2, 2000.0, btype="high", fs=44100, output="sos")
    s = (x + 1.0 * sosfiltfilt(sos, x, axis=0)).astype(np.float32)

    y, curve = match_eq_fir(x, s, sr=44100, eq_clamp_db=12.0)
    # Compare spectra of y and s in 3-10 kHz
    fy = np.abs(np.fft.rfft(y[:, 0]))
    fs_spec = np.abs(np.fft.rfft(s[:, 0]))
    freqs = np.fft.rfftfreq(len(y), 1.0 / 44100)
    band = (freqs >= 3000) & (freqs <= 10000)
    err_db = abs(10.0 * np.log10(np.mean(fy[band] ** 2) / np.mean(fs_spec[band] ** 2)))
    assert err_db < 0.5


def test_lag_detection_raises():
    x = _tone()
    s = np.roll(x, 1, axis=0)
    with pytest.raises(LagError):
        check_zero_lag(x, s)


def test_routing_rules():
    assert route_degradations(["volume"]).mode == "P"
    assert route_degradations(["big"]).mode == "R"
    assert route_degradations(["clarity"]).mode == "R"
    assert route_degradations(["big", "volume"]).mode == "R_then_P"
    assert route_degradations(["clip"]).mode == "S"
    assert route_degradations(["unknown_deg"]).mode == "fallback_s"


def test_pipeline_never_reads_clean(tmp_path: Path, monkeypatch):
    import soundfile as sf
    x = _tone()
    px = tmp_path / "x.wav"
    ps = tmp_path / "s.wav"
    pr = tmp_path / "r.wav"
    pc = tmp_path / "clean.wav"
    sf.write(px, x, 44100)
    sf.write(ps, x, 44100)
    sf.write(pr, x, 44100)
    sf.write(pc, x, 44100)

    monkeypatch.setattr(run_pipeline, "ROOT", tmp_path)
    orig_read = sf.read

    def guarded_read(file, *args, **kwargs):
        if "clean" in str(file):
            raise AssertionError(f"Pipeline attempted to read clean file: {file}")
        return orig_read(file, *args, **kwargs)

    monkeypatch.setattr(sf, "read", guarded_read)
    row = {
        "sample_id": "t_deg1",
        "path_x": "x.wav",
        "path_s": "s.wav",
        "path_r": "r.wav",
        "path_clean": "clean.wav",
        "degradation_types": "big|clarity",
    }
    out, _ = run_pipeline.process_sample(row, "A6", None, 12.0, 50.0, 0.1, 0.9)
    assert out.shape == x.shape
