"""Phase E: Blind / Zero-Label Routing (E1 Residual Fit Gate & E2 TF Confidence Mask)."""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.ndimage import uniform_filter1d
from scipy.signal import stft

ROOT = Path(r"d:\SonicMaster_Reproducing")
GDRIVE = Path(r"G:\My Drive\SonicMaster_Reproducing")
sys.path.insert(0, str(ROOT))

from scripts.extension.biquad_eq import fit_biquad_eq  # noqa: E402
from scripts.extension.branch_p import match_dynamics, match_stereo, match_volume_lufs  # noqa: E402
from scripts.extension.eval_dev import kl_div, logmel, mel_ssim  # noqa: E402
from scripts.extension.mix import mix_complementary_stft  # noqa: E402
from scripts.extension.xband_crossover import apply_xband_crossover, compute_xband_mask  # noqa: E402


def res(p: str | Path) -> Path:
    dp = ROOT / p
    return dp if dp.is_file() else GDRIVE / p


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    for attempt in range(3):
        try:
            y, sr = sf.read(io.BytesIO(path.read_bytes()), dtype="float32", always_2d=True)
            if y.shape[1] == 1:
                y = np.repeat(y, 2, axis=1)
            return y, sr
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1.0)
    raise RuntimeError(f"Failed to load {path}")


def _mod_spec_dist(y_m: np.ndarray, s_m: np.ndarray, sr: int = 44100) -> float:
    """Normalized modulation spectrum L2 distance (sensitive to reverb tails)."""
    my = np.abs(librosa.stft(y_m, n_fft=2048, hop_length=512))
    ms = np.abs(librosa.stft(s_m, n_fft=2048, hop_length=512))
    # Modulation spectrum along time axis of mel/stft bands
    fy = np.mean(np.abs(np.fft.rfft(my, axis=1))[:, 1:32], axis=0)
    fs = np.mean(np.abs(np.fft.rfft(ms, axis=1))[:, 1:32], axis=0)
    fy = fy / (np.linalg.norm(fy) + 1e-9)
    fs = fs / (np.linalg.norm(fs) + 1e-9)
    return float(np.linalg.norm(fy - fs))


def _flatness_diff(y_m: np.ndarray, s_m: np.ndarray) -> float:
    fy = float(np.mean(librosa.feature.spectral_flatness(y=y_m, n_fft=2048, hop_length=512)))
    fs = float(np.mean(librosa.feature.spectral_flatness(y=s_m, n_fft=2048, hop_length=512)))
    return abs(fy - fs) * 1000.0


def compute_blind_candidate(x: np.ndarray, s: np.ndarray, sr: int = 44100) -> tuple[np.ndarray, dict[str, float]]:
    """Zero-label candidate generation + blind residual diagnostics vs s."""
    n = min(len(x), len(s))
    x, s = x[:n], s[:n]
    rx = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    rs = float(np.sqrt(np.mean(s.astype(np.float64) ** 2)))
    gain_db = 20.0 * np.log10((rs + 1e-9) / (rx + 1e-9))
    if gain_db > 24.0:
        return s.copy(), {"guard_s": 1.0, "dead_frac": 0.0, "d_kl": 0.0, "d_mod": 0.0, "d_flat": 0.0}

    freqs, m_band = compute_xband_mask(x, s, sr=sr)
    dead_frac = float(np.mean(m_band[freqs <= 16000.0] < 0.25))
    if dead_frac > 0.12:
        y_cand, _ = apply_xband_crossover(x, s, sr=sr, eq_clamp_db=18.0, eq_phase="minphase")
    else:
        y_cand, _ = fit_biquad_eq(x, s, sr=sr, eq_clamp_db=18.0, eq_phase="minphase")
        y_cand, _ = match_dynamics(y_cand, s, sr=sr)
        y_cand, _ = match_stereo(y_cand, s)
        y_cand, _ = match_volume_lufs(y_cand, s, sr=sr)

    ym = y_cand.mean(axis=1)
    sm = s.mean(axis=1)
    d_kl = kl_div(logmel(sm), logmel(ym)) * 1000.0
    d_mod = _mod_spec_dist(ym, sm, sr=sr)
    d_flat = _flatness_diff(ym, sm)
    return y_cand, {"guard_s": 0.0, "dead_frac": dead_frac, "d_kl": d_kl, "d_mod": d_mod, "d_flat": d_flat}


def apply_e2_tf_mask(
    y_cand: np.ndarray,
    s: np.ndarray,
    sr: int = 44100,
    tau_log: float = 0.8,
    delta_log: float = 0.25,
    m_min: float = 0.0,
    m_max: float = 1.0,
) -> np.ndarray:
    """E2: Blind Time-Frequency Confidence Mask M(t, f) blending y_cand and s."""
    n = min(len(y_cand), len(s))
    y_cand, s = y_cand[:n], s[:n]
    n_fft, hop = 2048, 512
    n_frames_t = max(1, int(round(0.050 * sr / hop)))
    masks = []
    for ch in range(y_cand.shape[1]):
        freqs, _, zy = stft(y_cand[:, ch], fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop)
        _, _, zs = stft(s[:, ch], fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop)
        diff = np.abs(np.log(np.abs(zy) + 1e-6) - np.log(np.abs(zs) + 1e-6))
        diff = uniform_filter1d(diff, size=n_frames_t, axis=1, mode="nearest")
        # 1/3-octave frequency smoothing
        diff_f = np.empty_like(diff)
        factor = 2.0 ** (1.0 / 6.0)
        for i, f in enumerate(freqs):
            if f < 20.0:
                diff_f[i] = diff[i]
            else:
                lo, hi = np.searchsorted(freqs, f / factor), np.searchsorted(freqs, f * factor, side="right")
                diff_f[i] = np.mean(diff[max(0, lo) : max(lo + 1, hi)], axis=0)
        z = np.clip((tau_log - diff_f) / delta_log, -20.0, 20.0)
        m = np.clip(1.0 / (1.0 + np.exp(-z)), m_min, m_max)
        masks.append(m.astype(np.float32))
    mask_tf = np.stack(masks, axis=0)
    y_out, _ = mix_complementary_stft(y_cand, s, mask_tf, sr=sr, n_fft=n_fft, hop=hop)
    return y_out


def _worker_dev(r: dict[str, str], cand_dir_str: str, e2_dir_str: str) -> dict[str, object]:
    sid = r["sample_id"]
    x, sr = load_audio(res(r["path_x"]))
    s, _ = load_audio(res(r["path_s"]))
    c, _ = load_audio(res(r["path_clean"]))
    n = min(len(x), len(s), len(c))
    x, s, c = x[:n], s[:n], c[:n]

    y_cand, diag = compute_blind_candidate(x, s, sr=sr)
    sf.write(str(Path(cand_dir_str) / f"{sid}.wav"), np.clip(y_cand, -1.0, 1.0), sr, subtype="PCM_16")

    if diag["guard_s"] > 0.5:
        y_e2 = s
    else:
        y_e2 = apply_e2_tf_mask(y_cand, s, sr=sr)
    sf.write(str(Path(e2_dir_str) / f"{sid}.wav"), np.clip(y_e2, -1.0, 1.0), sr, subtype="PCM_16")

    mc = logmel(c.mean(1))
    ms = logmel(s.mean(1))
    my = logmel(y_cand.mean(1))
    me2 = logmel(y_e2.mean(1))
    return {
        "sample_id": sid,
        "degradation_types": r["degradation_types"],
        "n_degradations": int(r["n_degradations"]),
        **diag,
        "ssim_a0": mel_ssim(mc, ms),
        "kl_a0": kl_div(mc, ms) * 1000.0,
        "ssim_cand": mel_ssim(mc, my),
        "kl_cand": kl_div(mc, my) * 1000.0,
        "ssim_e2": mel_ssim(mc, me2),
        "kl_e2": kl_div(mc, me2) * 1000.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    splits = json.loads(res("artifacts/10_extension/splits.json").read_text(encoding="utf-8"))
    dev_sources = set(splits["dev_sources"])
    rows = [
        r
        for r in csv.DictReader(res("artifacts/10_extension/inputs_manifest.csv").open("r", encoding="utf-8"))
        if r["source_id"] in dev_sources
    ]

    base = ROOT / "artifacts/10_extension/dev/hybrid_v3_blind"
    cand_dir = base / "cand_wav"
    e2_dir = base / "e2_wav"
    cand_dir.mkdir(parents=True, exist_ok=True)
    e2_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    out_rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_worker_dev, r, str(cand_dir), str(e2_dir)) for r in rows]
        for i, fut in enumerate(as_completed(futs), 1):
            out_rows.append(fut.result())
            if i % 50 == 0 or i == len(rows):
                print(f"[{i}/{len(rows)}] elapsed={time.time()-t0:.1f}s", flush=True)

    csv_p = base / "blind_dev_diagnostics.csv"
    with csv_p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print("Saved:", csv_p)


if __name__ == "__main__":
    main()
