"""Priority 1 Verification on Dev (280 samples):
1. Zero-Training Blind V3.5 detectors (clip flat-top + reverb kurtosis/mod) vs E1-tight.
2. Data-Consistent De-clipping prototype on the 48 Dev 'clip' samples.
Read-only on locked Hybrid-v3 artifacts; writes only to artifacts/10_extension/dev/exp_v35/.
"""
from __future__ import annotations

import csv
import io
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.ndimage import binary_dilation, uniform_filter1d
from scipy.signal import correlate, stft
from scipy.stats import kurtosis

ROOT = Path(r"d:\SonicMaster_Reproducing")
GDRIVE = Path(r"G:\My Drive\SonicMaster_Reproducing")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "SonicMaster" / "evaluation"))

from scripts.extension.eval_dev import kl_div, logmel, mel_ssim  # noqa: E402
from scripts.extension.mix import mix_complementary_stft  # noqa: E402
import evaluate_control_multiple_degs_mass as attr_eval  # noqa: E402


def res(p: str | Path) -> Path:
    dp = ROOT / p
    return dp if dp.is_file() else GDRIVE / p


def load_stereo(path: Path, frames: int = 1320960) -> np.ndarray:
    for attempt in range(3):
        try:
            y, _ = sf.read(io.BytesIO(path.read_bytes()), dtype="float32", always_2d=True)
            if y.shape[1] == 1:
                y = np.repeat(y, 2, axis=1)
            return y[:frames]
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1.0)
    raise RuntimeError(f"Failed to load {path}")


def clip_features(x: np.ndarray, p: np.ndarray, s: np.ndarray) -> dict[str, float]:
    """Blind clipping & saturation detectors from x, p, s."""
    xm = np.abs(x)
    peak = np.max(xm, axis=0, keepdims=True) + 1e-9
    norm_x = xm / peak
    peak_sat_99 = float(np.mean(norm_x >= 0.99))
    peak_sat_98 = float(np.mean(norm_x >= 0.98))
    dx = np.abs(np.diff(x, axis=0))
    flat_high = (dx < 1e-4) & (norm_x[:-1] >= 0.85)
    flat_top_ratio = float(np.mean(flat_high))
    cf_x = 20.0 * np.log10((np.max(np.abs(x)) + 1e-9) / (np.sqrt(np.mean(x**2)) + 1e-9))
    cf_p = 20.0 * np.log10((np.max(np.abs(p)) + 1e-9) / (np.sqrt(np.mean(p**2)) + 1e-9))
    cf_s = 20.0 * np.log10((np.max(np.abs(s)) + 1e-9) / (np.sqrt(np.mean(s**2)) + 1e-9))
    sp_stft = np.abs(librosa.stft(p.mean(axis=1), n_fft=2048, hop_length=512))
    ss_stft = np.abs(librosa.stft(s.mean(axis=1), n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=44100, n_fft=2048)
    hf = freqs >= 4000.0
    gp = np.exp(np.mean(np.log(sp_stft[hf] ** 2 + 1e-12), axis=0)) / (np.mean(sp_stft[hf] ** 2, axis=0) + 1e-12)
    gs = np.exp(np.mean(np.log(ss_stft[hf] ** 2 + 1e-12), axis=0)) / (np.mean(ss_stft[hf] ** 2, axis=0) + 1e-12)
    hf_flat_diff = float(np.mean(gp) - np.mean(gs)) * 1000.0
    return {
        "peak_sat_99": peak_sat_99,
        "peak_sat_98": peak_sat_98,
        "flat_top_ratio": flat_top_ratio,
        "cf_diff_s_minus_p": float(cf_s - cf_p),
        "cf_diff_s_minus_x": float(cf_s - cf_x),
        "hf_flat_diff": hf_flat_diff,
    }


def reverb_features(p: np.ndarray, s: np.ndarray, sr: int = 44100) -> dict[str, float]:
    """Blind reverb/smearing detectors comparing p vs s."""
    pm = p.mean(axis=1)
    sm = s.mean(axis=1)
    sp_stft = np.abs(librosa.stft(pm, n_fft=2048, hop_length=512))
    ss_stft = np.abs(librosa.stft(sm, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    mid = (freqs >= 300.0) & (freqs <= 6000.0)
    kp = float(np.nanmedian(kurtosis(sp_stft[mid], axis=1, fisher=True, nan_policy="omit")))
    ks = float(np.nanmedian(kurtosis(ss_stft[mid], axis=1, fisher=True, nan_policy="omit")))
    fp = np.mean(np.abs(np.fft.rfft(sp_stft[mid], axis=1))[:, 1:32], axis=0)
    fs = np.mean(np.abs(np.fft.rfft(ss_stft[mid], axis=1))[:, 1:32], axis=0)
    fp = fp / (np.linalg.norm(fp) + 1e-9)
    fs = fs / (np.linalg.norm(fs) + 1e-9)
    d_mod_mid = float(np.linalg.norm(fp - fs))
    op = float(np.mean(librosa.onset.onset_strength(y=pm, sr=sr)))
    os_ = float(np.mean(librosa.onset.onset_strength(y=sm, sr=sr)))
    onset_ratio = float((os_ - op) / (os_ + 1e-9))
    return {
        "kurt_diff_s_minus_p": float(ks - kp),
        "d_mod_mid": d_mod_mid,
        "onset_ratio_s_minus_p": onset_ratio,
    }


def analyze_one(r: dict[str, str], cand_dir_str: str) -> dict[str, object]:
    sid = r["sample_id"]
    x = load_stereo(res(r["path_x"]))
    s = load_stereo(res(r["path_s"]))
    p = load_stereo(Path(cand_dir_str) / f"{sid}.wav")
    n = min(len(x), len(s), len(p))
    x, s, p = x[:n], s[:n], p[:n]
    feats: dict[str, object] = {"sample_id": sid}
    feats.update(clip_features(x, p, s))
    feats.update(reverb_features(p, s))
    return feats


def test_declip_one(r: dict[str, str], cand_dir_str: str) -> dict[str, object]:
    """Compare A0 vs time-domain declip vs STFT HF crossover declip on a Dev 'clip' sample."""
    sid = r["sample_id"]
    c = load_stereo(res(r["path_clean"]))
    x = load_stereo(res(r["path_x"]))
    s = load_stereo(res(r["path_s"]))
    p = load_stereo(Path(cand_dir_str) / f"{sid}.wav")
    n = min(len(c), len(x), len(s), len(p))
    c, x, s, p = c[:n], x[:n], s[:n], p[:n]
    degs = [d.strip() for d in r["degradation_types"].split("|") if d.strip()]

    # Method 1: Time-domain peak replacement (aligned s replaces clipped peaks of p)
    xm = np.max(np.abs(x), axis=1)
    peak = float(np.max(xm)) + 1e-9
    clip_mask = (xm >= 0.96 * peak).astype(np.float32)
    clip_mask = binary_dilation(clip_mask > 0.5, iterations=64).astype(np.float32)
    w_t = uniform_filter1d(clip_mask, size=128)[:, None]

    pm, sm = p.mean(axis=1), s.mean(axis=1)
    seg = min(len(pm), 44100 * 5)
    cc = correlate(pm[:seg], sm[:seg], mode="full")
    mid = seg - 1
    win = cc[mid - 256 : mid + 257]
    idx_max = int(np.argmax(np.abs(win)))
    lag = idx_max - 256
    pol = 1.0 if win[idx_max] >= 0 else -1.0
    s_al = np.roll(s, lag, axis=0) * pol
    y_td = (1.0 - w_t) * p + w_t * s_al

    # Method 2: STFT HF-only declip (keep p below 3.5 kHz, blend to s above 3.5 kHz)
    n_fft = 2048
    hop = 512
    freqs = np.linspace(0, 22050, n_fft // 2 + 1)
    _, _, zp = stft(p[:, 0], fs=44100, window="hann", nperseg=n_fft, noverlap=n_fft - hop)
    n_frames = zp.shape[1]
    m_f = (1.0 / (1.0 + np.exp((freqs - 3500.0) / 500.0))).astype(np.float32)
    mask_3d = np.broadcast_to(m_f[:, None], (n_fft // 2 + 1, n_frames))
    mask_3d = np.broadcast_to(mask_3d[None, :, :], (p.shape[1], n_fft // 2 + 1, n_frames)).copy()
    y_hf, _ = mix_complementary_stft(p, s, mask_3d, sr=44100)
    nh = min(len(c), len(y_hf))

    cm = c[:nh].mean(axis=1)
    xm_m = x[:nh].mean(axis=1)
    mc = logmel(cm)

    out: dict[str, object] = {"sample_id": sid, "n_deg": int(r["n_degradations"]), "clip_frac": float(np.mean(clip_mask))}
    for name, sig in [("A0", s[:nh]), ("P", p[:nh]), ("TD_declip", y_td[:nh]), ("HF_declip", y_hf[:nh])]:
        sig_m = sig.mean(axis=1)
        ms = logmel(sig_m)
        out[f"{name}_ssim"] = mel_ssim(mc, ms)
        out[f"{name}_kl"] = kl_div(mc, ms) * 1000.0
        ev = attr_eval.evaluate_sample(cm, xm_m, sig_m, degs, c[:nh], x[:nh], sig)
        if "clip" in ev:
            out[f"{name}_clip_err"] = float(ev["clip"].get("abs_error_after", np.nan))
    return out


def main() -> None:
    out_dir = ROOT / "artifacts" / "10_extension" / "dev" / "exp_v35"
    out_dir.mkdir(parents=True, exist_ok=True)
    cand_dir = ROOT / "artifacts" / "10_extension" / "dev" / "hybrid_v3_blind" / "cand_wav"

    with res("artifacts/10_extension/inputs_manifest.csv").open("r", encoding="utf-8") as f:
        dev_rows = [r for r in csv.DictReader(f) if r["split_ext"] == "dev"]

    print(f"1) Extracting blind clip/reverb features on {len(dev_rows)} Dev samples...", flush=True)
    t0 = time.time()
    feat_rows = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(analyze_one, r, str(cand_dir)) for r in dev_rows]
        for fut in as_completed(futs):
            feat_rows.append(fut.result())
    fdf = pd.DataFrame(feat_rows)
    base_df = pd.read_csv(ROOT / "artifacts/10_extension/dev/hybrid_v3_blind/blind_dev_diagnostics.csv")
    df = base_df.merge(fdf, on="sample_id")
    df.to_csv(out_dir / "dev_v35_features.csv", index=False)
    print(f"Features extracted in {time.time()-t0:.1f}s", flush=True)

    s_fb = {"real", "clip", "small", "big", "mix"}
    df["is_nonlin"] = df["degradation_types"].apply(lambda s: bool({d.strip() for d in str(s).split("|") if d.strip()} & s_fb))
    df["has_clip"] = df["degradation_types"].apply(lambda s: "clip" in {d.strip() for d in str(s).split("|") if d.strip()})
    df["has_reverb"] = df["degradation_types"].apply(lambda s: bool({d.strip() for d in str(s).split("|") if d.strip()} & {"real", "small", "big", "mix"}))

    print("\n--- Feature medians: Linear vs Clip vs Reverb (among guard_s==0) ---")
    sub = df[df["guard_s"] < 0.5]
    for col in ["flat_top_ratio", "peak_sat_99", "cf_diff_s_minus_p", "hf_flat_diff", "kurt_diff_s_minus_p", "d_mod_mid", "onset_ratio_s_minus_p", "d_kl", "d_mod"]:
        m_lin = sub[~sub["is_nonlin"]][col].median()
        m_clp = sub[sub["has_clip"]][col].median()
        m_rev = sub[sub["has_reverb"]][col].median()
        print(f"  {col:22s}: linear={m_lin:+.5f}  clip={m_clp:+.5f}  reverb={m_rev:+.5f}")

    print("\n--- Zero-Training Blind V3.5 Rule Sweep (Dev 280) ---")
    rules = {
        "E1_tight (prev)": (df["guard_s"] < 0.5) & (df["d_kl"] <= 1.0) & (df["d_mod"] <= 0.08),
        "V3.5_A (+flat_top<1e-4 & d_mod_mid<=0.08)": (df["guard_s"] < 0.5) & (df["d_kl"] <= 1.2) & (df["d_mod"] <= 0.08) & (df["flat_top_ratio"] < 1e-4) & (df["d_mod_mid"] <= 0.08),
        "V3.5_B (+flat_top<5e-5 & kurt_diff<=1.5 & cf_diff<=1.5)": (df["guard_s"] < 0.5) & (df["d_kl"] <= 1.2) & (df["d_mod_mid"] <= 0.07) & (df["flat_top_ratio"] < 5e-5) & (df["kurt_diff_s_minus_p"] <= 1.5) & (df["cf_diff_s_minus_p"] <= 1.5),
        "V3.5_C (strict clip+rev)": (df["guard_s"] < 0.5) & (df["d_kl"] <= 1.0) & (df["d_mod_mid"] <= 0.06) & (df["flat_top_ratio"] < 1e-5) & (df["peak_sat_99"] < 2e-4) & (df["kurt_diff_s_minus_p"] <= 1.0),
        "V3.5_D (+flat_top<1e-4 & cf<=1.0 & onset<=0.0 & kurt<=2.5)": (df["guard_s"] < 0.5) & (df["d_kl"] <= 1.2) & (df["d_mod"] <= 0.08) & (df["flat_top_ratio"] < 1e-4) & (df["cf_diff_s_minus_p"] <= 1.0) & (df["onset_ratio_s_minus_p"] <= 0.0) & (df["kurt_diff_s_minus_p"] <= 2.5),
    }
    for name, mask in rules.items():
        ssim = float(np.where(mask, df["ssim_cand"], df["ssim_a0"]).mean())
        kl = float(np.where(mask, df["kl_cand"], df["kl_a0"]).mean())
        n_k = int(mask.sum())
        n_nl = int((mask & df["is_nonlin"]).sum())
        n_clp = int((mask & df["has_clip"]).sum())
        n_rev = int((mask & df["has_reverb"]).sum())
        n_lin = int((mask & ~df["is_nonlin"]).sum())
        print(f"  {name:55s}: kept={n_k:3d} (lin={n_lin:3d}, nonlin={n_nl:2d} [clip={n_clp:2d}, rev={n_rev:2d}]) SSIM={ssim:.4f} KL={kl:.4f}")

    clip_rows = [r for r in dev_rows if "clip" in {d.strip() for d in r["degradation_types"].split("|") if d.strip()}]
    print(f"\n2) Testing Data-Consistent De-clipping on {len(clip_rows)} Dev 'clip' samples...", flush=True)
    t1 = time.time()
    dc_rows = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(test_declip_one, r, str(cand_dir)) for r in clip_rows]
        for fut in as_completed(futs):
            dc_rows.append(fut.result())
    dcdf = pd.DataFrame(dc_rows)
    dcdf.to_csv(out_dir / "dev_declip_pilot.csv", index=False)
    print(f"De-clipping pilot done in {time.time()-t1:.1f}s")
    for grp_name, gdf in [("All clip (n=48)", dcdf), ("Single clip", dcdf[dcdf["n_deg"] == 1])]:
        print(f"  [{grp_name}, n={len(gdf)}]")
        for m in ["A0", "P", "TD_declip", "HF_declip"]:
            print(f"    {m:10s}: SSIM={gdf[f'{m}_ssim'].mean():.4f}  KL={gdf[f'{m}_kl'].mean():.4f}  clip_err={gdf[f'{m}_clip_err'].mean():.6f}")


if __name__ == "__main__":
    main()
