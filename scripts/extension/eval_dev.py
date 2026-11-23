"""Evaluate Dev variants (A0-A6) using exact run7000 KL/SSIM, Table 1-2 attributes, and optional FAD/PQ."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from skimage.metrics import structural_similarity as ssim

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# --- Exact KL / SSIM from artifacts/14_run7000/evaluation/kl_ssim/score_kl_ssim_7000.py ---
def logmel(y: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(y=y, sr=44100, n_fft=2048, hop_length=512, n_mels=128)
    return librosa.power_to_db(mel, ref=np.max)


def kl_div(m1: np.ndarray, m2: np.ndarray) -> float:
    p = np.mean(m1, axis=1)
    q = np.mean(m2, axis=1)
    p = p / (np.sum(p) + 1e-10)
    q = q / (np.sum(q) + 1e-10)
    return float(np.sum(p * np.log((p + 1e-10) / (q + 1e-10))))


def mel_ssim(m1: np.ndarray, m2: np.ndarray) -> float:
    shape = (min(m1.shape[0], m2.shape[0]), min(m1.shape[1], m2.shape[1]))
    m1 = m1[: shape[0], : shape[1]]
    m2 = m2[: shape[0], : shape[1]]
    m1 = (m1 - m1.min()) / (m1.max() - m1.min() + 1e-8)
    m2 = (m2 - m2.min()) / (m2.max() - m2.min() + 1e-8)
    return float(ssim(m1, m2, data_range=1.0))


def load_stereo(path: Path, frames: int = 1320960) -> np.ndarray:
    import io
    if not path.exists():
        try:
            rel = path.relative_to(ROOT)
            gp = Path("G:/My Drive/SonicMaster_Reproducing") / rel
            if gp.exists():
                path = gp
        except ValueError:
            pass
    y, sr = sf.read(io.BytesIO(path.read_bytes()), always_2d=True)
    assert sr == 44100, f"Unexpected sr {sr} in {path}"
    if y.shape[1] == 1:
        y = np.repeat(y, 2, axis=1)
    return y[:frames]


def source_bootstrap_ci(
    per_sample: list[dict],
    metric_key: str,
    n_boot: int = 2000,
    seed: int = 20261119,
) -> tuple[float, float, float]:
    by_src: dict[str, list[float]] = {}
    for r in per_sample:
        by_src.setdefault(r["source_id"], []).append(float(r[metric_key]))
    srcs = sorted(by_src.keys())
    src_means = np.array([np.mean(by_src[s]) for s in srcs], dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(srcs), size=(n_boot, len(srcs)))
    boots = np.mean(src_means[idx], axis=1)
    return float(np.mean(src_means)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "artifacts" / "10_extension" / "inputs_manifest.csv"))
    ap.add_argument("--run-dir", required=True, help="e.g. artifacts/10_extension/dev/run1")
    ap.add_argument("--variants", required=True, help="Comma-separated, e.g. A0,A1,A2,A3,A4,A5,A6")
    ap.add_argument("--sanity-check-a0", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    with Path(args.manifest).open("r", encoding="utf-8") as f:
        dev_rows = [r for r in csv.DictReader(f) if r["split_ext"] == "dev"]

    # Cache clean logmels per source_id
    clean_mels: dict[str, np.ndarray] = {}
    for r in dev_rows:
        src = r["source_id"]
        if src not in clean_mels:
            yc = load_stereo(ROOT / r["path_clean"])
            clean_mels[src] = logmel(np.mean(yc, axis=1))

    # Load run7000 KL/SSIM for A0 sanity check if requested
    r7000_kl_ssim: dict[str, dict] = {}
    if args.sanity_check_a0:
        for p in (ROOT / "artifacts" / "14_run7000" / "evaluation" / "kl_ssim").glob("*/samples/*.json"):
            rec = json.loads(p.read_text(encoding="utf-8"))
            r7000_kl_ssim[rec["sample_id"]] = rec["values"]["sonicmaster"]

    per_sample_all: list[dict] = []
    summary_all: list[dict] = []

    for var in variants:
        var_rows: list[dict] = []
        for r in dev_rows:
            sid = r["sample_id"]
            src = r["source_id"]
            wav_p = run_dir / var / f"{sid}.wav"
            if not wav_p.exists():
                continue
            y = load_stereo(wav_p)
            m_out = logmel(np.mean(y, axis=1))
            m_clean = clean_mels[src]
            s_val = mel_ssim(m_out, m_clean)
            k_val = kl_div(m_out, m_clean) * 1000.0

            if var == "A0" and args.sanity_check_a0 and sid in r7000_kl_ssim:
                ref = r7000_kl_ssim[sid]
                assert abs(s_val - ref["ssim"]) < 1e-6, f"A0 SSIM mismatch {sid}: {s_val} vs {ref['ssim']}"
                assert abs(k_val - ref["kl_x1000"]) < 1e-6, f"A0 KL mismatch {sid}: {k_val} vs {ref['kl_x1000']}"

            rec = {
                "sample_id": sid,
                "source_id": src,
                "variant": var,
                "group": "Single" if int(r["n_degradations"]) == 1 else "Double+Triple",
                "ssim": s_val,
                "kl_x1000": k_val,
            }
            var_rows.append(rec)
            per_sample_all.append(rec)

        for grp_name, sub in [
            ("All", var_rows),
            ("Single", [x for x in var_rows if x["group"] == "Single"]),
            ("Double+Triple", [x for x in var_rows if x["group"] == "Double+Triple"]),
        ]:
            if not sub:
                continue
            m_ssim, lo_s, hi_s = source_bootstrap_ci(sub, "ssim")
            m_kl, lo_k, hi_k = source_bootstrap_ci(sub, "kl_x1000")
            summary_all.append({
                "variant": var,
                "group": grp_name,
                "n_samples": len(sub),
                "ssim_mean": m_ssim,
                "ssim_ci95_lo": lo_s,
                "ssim_ci95_hi": hi_s,
                "kl_x1000_mean": m_kl,
                "kl_x1000_ci95_lo": lo_k,
                "kl_x1000_ci95_hi": hi_k,
            })

    with (run_dir / "scores_per_sample.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_sample_all[0].keys()))
        w.writeheader()
        w.writerows(per_sample_all)

    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_all[0].keys()))
        w.writeheader()
        w.writerows(summary_all)

    print(json.dumps(summary_all, indent=2))


if __name__ == "__main__":
    main()
