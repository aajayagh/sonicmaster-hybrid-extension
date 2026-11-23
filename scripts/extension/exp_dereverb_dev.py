"""Step 1 Verification: Guided Dereverberation (WPE, Guided_Wiener, WPE_Guided) on Dev reverb samples."""
from __future__ import annotations

import csv
import io
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from audiobox_aesthetics.infer import initialize_predictor
from scipy.ndimage import gaussian_filter, uniform_filter1d
from scipy.signal import istft, stft

ROOT = Path(r"d:\SonicMaster_Reproducing")
GDRIVE = Path(r"G:\My Drive\SonicMaster_Reproducing")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "SonicMaster" / "evaluation"))

from scripts.extension.branch_p import match_volume_lufs  # noqa: E402
from scripts.extension.branch_r import soft_limit  # noqa: E402
from scripts.extension.eval_dev import kl_div, logmel, mel_ssim  # noqa: E402
import evaluate_control_multiple_degs_mass as attr_eval  # noqa: E402

REVERB_SET = {"small", "big", "mix", "real"}


def res(p: str | Path) -> Path:
    dp = ROOT / p
    return dp if dp.is_file() else GDRIVE / p


def _sf_load(filepath, *args, **kwargs):
    p = Path(filepath)
    for attempt in range(3):
        try:
            y, sr = sf.read(io.BytesIO(p.read_bytes()), dtype="float32", always_2d=True)
            return torch.from_numpy(y.T.copy()), sr
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1.0)
    raise RuntimeError(f"Failed to load {filepath}")


torchaudio.load = _sf_load


def load_stereo(path: Path, frames: int = 1320960) -> tuple[np.ndarray, int]:
    for attempt in range(3):
        try:
            y, sr = sf.read(io.BytesIO(path.read_bytes()), dtype="float32", always_2d=True)
            if y.shape[1] == 1:
                y = np.repeat(y, 2, axis=1)
            return y[:frames], sr
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1.0)
    raise RuntimeError(f"Failed to load {path}")


def _wpe_channel(y_ch: np.ndarray, sr: int = 44100, n_fft: int = 1024, hop: int = 256, taps: int = 10, delay: int = 3, iterations: int = 3) -> np.ndarray:
    """Vectorized batched pure-NumPy single-channel WPE dereverberation."""
    n_samples = len(y_ch)
    _, _, z = stft(y_ch, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop)
    z = z.astype(np.complex128)
    n_freqs, n_frames = z.shape

    z_tilde = np.zeros((n_freqs, n_frames, taps), dtype=np.complex128)
    for k in range(taps):
        shift = delay + k
        if shift < n_frames:
            z_tilde[:, shift:, k] = z[:, : n_frames - shift]

    z_out = z.copy()
    eye = np.eye(taps, dtype=np.complex128)[None, :, :]
    for _ in range(iterations):
        psd = uniform_filter1d(np.abs(z_out) ** 2, size=3, axis=1)
        floor = np.maximum(1e-6 * np.mean(psd, axis=1, keepdims=True), 1e-12)
        inv_lam = 1.0 / np.maximum(psd, floor)

        # Weighted covariance R: (F, K, K) and rhs r: (F, K)
        wz = z_tilde * inv_lam[:, :, None]
        r_mat = np.einsum("ftk,ftm->fkm", z_tilde.conj(), wz)
        r_vec = np.einsum("ftk,ft->fk", wz.conj(), z)

        diag_mean = np.mean(np.real(np.diagonal(r_mat, axis1=1, axis2=2)), axis=1)
        reg = np.maximum(1e-4 * diag_mean, 1e-10)[:, None, None] * eye
        g = np.linalg.solve(r_mat + reg, r_vec[..., None])[..., 0]
        z_out = z - np.einsum("fk,ftk->ft", g, z_tilde)

    _, y_out = istft(z_out, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop)
    return y_out[:n_samples].astype(np.float32)


def apply_wpe(y: np.ndarray, sr: int = 44100) -> np.ndarray:
    chs = [_wpe_channel(y[:, ch], sr=sr) for ch in range(y.shape[1])]
    return np.stack(chs, axis=1)


def apply_guided_wiener(y: np.ndarray, s: np.ndarray, sr: int = 44100, n_fft: int = 2048, hop: int = 512) -> np.ndarray:
    """Apply smooth envelope Wiener gain from s onto y while keeping phase of y."""
    n_samples = min(len(y), len(s))
    y, s = y[:n_samples], s[:n_samples]
    chs = []
    for ch in range(y.shape[1]):
        _, _, zy = stft(y[:, ch], fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop)
        _, _, zs = stft(s[:, ch], fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop)
        nf = min(zy.shape[1], zs.shape[1])
        zy_c, zs_c = zy[:, :nf], zs[:, :nf]
        py = gaussian_filter(np.abs(zy_c) ** 2, sigma=(1.0, 2.0))
        ps = gaussian_filter(np.abs(zs_c) ** 2, sigma=(1.0, 2.0))
        gain = np.clip(np.sqrt((ps + 1e-8) / (py + 1e-8)), 0.15, 1.5)
        _, y_ch = istft(gain * zy_c, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop)
        chs.append(y_ch[:n_samples])
    return np.stack(chs, axis=1).astype(np.float32)


def _finalize(y: np.ndarray, s: np.ndarray, sr: int = 44100) -> np.ndarray:
    n = min(len(y), len(s))
    y, _ = match_volume_lufs(y[:n], s[:n], sr=sr)
    y, _ = soft_limit(y, ceiling_db=-1.0)
    return np.clip(y, -1.0, 1.0).astype(np.float32)


def process_and_eval_one(r: dict[str, str], cand_dir_str: str, out_root_str: str) -> dict[str, object]:
    sid = r["sample_id"]
    c, sr = load_stereo(res(r["path_clean"]))
    x, _ = load_stereo(res(r["path_x"]))
    s, _ = load_stereo(res(r["path_s"]))
    p, _ = load_stereo(Path(cand_dir_str) / f"{sid}.wav")
    n = min(len(c), len(x), len(s), len(p))
    c, x, s, p = c[:n], x[:n], s[:n], p[:n]
    degs = [d.strip() for d in r["degradation_types"].split("|") if d.strip()]

    y_wpe_raw = apply_wpe(p, sr=sr)
    y_wpe = _finalize(y_wpe_raw, s, sr=sr)
    y_gw = _finalize(apply_guided_wiener(p, s, sr=sr), s, sr=sr)
    y_wpe_gw = _finalize(apply_guided_wiener(y_wpe_raw, s, sr=sr), s, sr=sr)

    out_root = Path(out_root_str)
    for m_name, sig in [("A0", s), ("WPE", y_wpe), ("Guided_Wiener", y_gw), ("WPE_Guided", y_wpe_gw)]:
        mdir = out_root / m_name
        mdir.mkdir(parents=True, exist_ok=True)
        sf.write(str(mdir / f"{sid}.wav"), np.clip(sig, -1.0, 1.0), sr, subtype="PCM_16")

    cm = c.mean(axis=1)
    xm = x.mean(axis=1)
    mc = logmel(cm)

    rec: dict[str, object] = {
        "sample_id": sid,
        "source_id": r["source_id"],
        "n_deg": int(r["n_degradations"]),
        "degradation_types": r["degradation_types"],
    }
    for m_name, sig in [("A0", s), ("P", p), ("WPE", y_wpe), ("Guided_Wiener", y_gw), ("WPE_Guided", y_wpe_gw)]:
        ns = min(len(c), len(sig))
        sm = sig[:ns].mean(axis=1)
        ms = logmel(sm)
        rec[f"{m_name}_ssim"] = mel_ssim(mc, ms)
        rec[f"{m_name}_kl"] = kl_div(mc, ms) * 1000.0
        ev = attr_eval.evaluate_sample(cm[:ns], xm[:ns], sm, degs, c[:ns], x[:ns], sig[:ns])
        for deg in REVERB_SET:
            if deg in ev:
                val = ev[deg].get("abs_error_after", ev[deg].get("cosine_abs_error_after"))
                if val is not None and np.isfinite(val):
                    rec[f"{m_name}_err_{deg}"] = float(val)
                    vb = ev[deg].get("abs_error_before", ev[deg].get("cosine_abs_error_before"))
                    if vb is not None and np.isfinite(vb):
                        rec[f"inp_err_{deg}"] = float(vb)
    return rec


def main() -> None:
    out_root = ROOT / "artifacts" / "10_extension" / "dev" / "exp_dereverb"
    out_root.mkdir(parents=True, exist_ok=True)
    cand_dir = ROOT / "artifacts" / "10_extension" / "dev" / "hybrid_v3_blind" / "cand_wav"

    with res("artifacts/10_extension/inputs_manifest.csv").open("r", encoding="utf-8") as f:
        rev_rows = [
            r
            for r in csv.DictReader(f)
            if r["split_ext"] == "dev"
            and r["path_x"]
            and ({d.strip() for d in r["degradation_types"].split("|") if d.strip()} & REVERB_SET)
        ]

    print(f"Testing Guided Dereverberation on {len(rev_rows)} Dev reverb samples...", flush=True)
    t0 = time.time()
    recs = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(process_and_eval_one, r, str(cand_dir), str(out_root)) for r in rev_rows]
        for i, fut in enumerate(as_completed(futs), 1):
            recs.append(fut.result())
            if i % 20 == 0 or i == len(rev_rows):
                print(f"  DSP+Metrics [{i}/{len(rev_rows)}] elapsed={time.time()-t0:.1f}s", flush=True)

    df = pd.DataFrame(recs).sort_values("sample_id").reset_index(drop=True)

    # Check finite WAVs
    for m_name in ["A0", "WPE", "Guided_Wiener", "WPE_Guided"]:
        bad = [p.name for p in (out_root / m_name).glob("*.wav") if not np.all(np.isfinite(sf.read(p)[0]))]
        assert not bad, f"Non-finite WAVs in {m_name}: {bad}"

    # Local Audiobox PQ
    print("Computing Audiobox Aesthetics PQ...", flush=True)
    predictor = initialize_predictor()
    sids = list(df["sample_id"])
    for m_name in ["A0", "WPE", "Guided_Wiener", "WPE_Guided"]:
        paths = [str(out_root / m_name / f"{sid}.wav") for sid in sids]
        pqs = []
        for i in range(0, len(paths), 8):
            res_pq = predictor.forward([{"path": p} for p in paths[i : i + 8]])
            pqs.extend(float(x["PQ"]) for x in res_pq)
        df[f"{m_name}_pq"] = pqs

    df.to_csv(out_root / "dev_dereverb_pilot.csv", index=False)

    summary: dict[str, object] = {"n_total": len(df), "n_single": int((df["n_deg"] == 1).sum())}
    print("\n=== Dev Reverb Comparison ===")
    for grp_name, gdf in [("All reverb", df), ("Single reverb", df[df["n_deg"] == 1])]:
        print(f"\n[{grp_name}, n={len(gdf)}]")
        grp_sum = {}
        for m in ["A0", "P", "WPE", "Guided_Wiener", "WPE_Guided"]:
            ss = float(gdf[f"{m}_ssim"].mean())
            kl = float(gdf[f"{m}_kl"].mean())
            pq = float(gdf[f"{m}_pq"].mean()) if f"{m}_pq" in gdf.columns else float("nan")
            dpq = pq - float(gdf["A0_pq"].mean()) if f"{m}_pq" in gdf.columns else float("nan")
            grp_sum[m] = {"ssim": ss, "kl_x1000": kl, "pq": pq, "delta_pq_vs_A0": dpq}
            pq_s = f"{pq:.4f} (dPQ={dpq:+.4f})" if np.isfinite(pq) else "   -   "
            print(f"  {m:14s}: SSIM={ss:.4f}  KL={kl:.4f}  PQ={pq_s}")
        summary[grp_name] = grp_sum

    print("\n=== Reverb Attribute Guards (All Dev Reverb) ===")
    attr_sum = {}
    for deg in sorted(REVERB_SET):
        col_i = f"inp_err_{deg}"
        col_0 = f"A0_err_{deg}"
        if col_0 not in df.columns:
            continue
        sub = df[df[col_0].notna()]
        mi = float(sub[col_i].mean())
        m0 = float(sub[col_0].mean())
        deg_dict = {"n": len(sub), "inp": mi, "A0": m0}
        line = f"  {deg:6s} (n={len(sub):2d}): inp={mi:.4f} A0={m0:.4f}"
        for m in ["P", "WPE", "Guided_Wiener", "WPE_Guided"]:
            mv = float(sub[f"{m}_err_{deg}"].mean())
            det = (mv - m0) / mi if mi > 1e-12 else 0.0
            deg_dict[m] = {"err": mv, "norm_det": det}
            line += f" | {m}={mv:.4f}({det:+.3f})"
        print(line)
        attr_sum[deg] = deg_dict
    summary["reverb_attributes"] = attr_sum

    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved results to {out_root}")


if __name__ == "__main__":
    main()
