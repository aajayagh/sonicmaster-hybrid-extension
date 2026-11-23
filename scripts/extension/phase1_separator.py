"""Phase 1: Benchmark vocal separators on Dev vocal sources (clean, x, s)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from audio_separator.separator import Separator

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "tmp" / "separator_models"
OUT_DIR = ROOT / "artifacts" / "10_extension" / "phase1_separator"

CANDIDATES = [
    ("htdemucs_ft", "htdemucs_ft.yaml"),
    ("bs_roformer_1297", "model_bs_roformer_ep_317_sdr_12.9755.ckpt"),
    ("mel_band_roformer_kim", "vocals_mel_band_roformer.ckpt"),
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def si_sdr(est: np.ndarray, ref: np.ndarray) -> float:
    est_m = est.mean(axis=1).astype(np.float64)
    ref_m = ref.mean(axis=1).astype(np.float64)
    n = min(len(est_m), len(ref_m))
    est_m, ref_m = est_m[:n], ref_m[:n]
    est_m = est_m - np.mean(est_m)
    ref_m = ref_m - np.mean(ref_m)
    alpha = np.dot(est_m, ref_m) / (np.dot(ref_m, ref_m) + 1e-12)
    target = alpha * ref_m
    noise = est_m - target
    return float(10.0 * np.log10((np.sum(target ** 2) + 1e-12) / (np.sum(noise ** 2) + 1e-12)))


def leakage_db(voc_hat: np.ndarray, acc_ref: np.ndarray) -> float:
    n = min(len(voc_hat), len(acc_ref))
    v = voc_hat[:n].astype(np.float64)
    a = acc_ref[:n].astype(np.float64)
    num = float(np.sum(v * a))
    den_a = float(np.sum(a * a)) + 1e-12
    proj = (num / den_a) * a
    return float(10.0 * np.log10((np.sum(proj * proj) + 1e-12) / (np.sum(v * v) + 1e-12)))


def load_stem(path: Path, n_samples: int) -> np.ndarray:
    y, _ = sf.read(path, dtype="float32", always_2d=True)
    if y.shape[1] == 1:
        y = np.repeat(y, 2, axis=1)
    if len(y) > n_samples:
        y = y[:n_samples]
    elif len(y) < n_samples:
        y = np.pad(y, ((0, n_samples - len(y)), (0, 0)))
    return y


def separate_clip(sep: Separator, in_wav: Path, work_dir: Path, n_samples: int) -> tuple[np.ndarray, np.ndarray, float]:
    work_dir.mkdir(parents=True, exist_ok=True)
    sep.output_dir = str(work_dir)
    t0 = time.perf_counter()
    files = sep.separate(str(in_wav))
    dt = time.perf_counter() - t0

    voc_path = None
    acc_paths = []
    for f in files:
        p = work_dir / Path(f).name
        if not p.exists():
            p = Path(f)
        low = p.name.lower()
        if "(vocals)" in low:
            voc_path = p
        else:
            acc_paths.append(p)

    if voc_path is None:
        raise RuntimeError(f"No vocals stem found in {files}")

    voc = load_stem(voc_path, n_samples)
    if acc_paths:
        acc = sum(load_stem(p, n_samples) for p in acc_paths)
    else:
        mix = load_stem(in_wav, n_samples)
        acc = mix - voc

    # Cleanup temp stems
    for f in files:
        p = work_dir / Path(f).name
        if p.exists():
            p.unlink()
    return voc, acc, dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "artifacts" / "10_extension" / "inputs_manifest.csv"))
    ap.add_argument("--max-sources", type=int, default=15, help="Number of vocal Dev sources to benchmark")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    work_dir = OUT_DIR / "tmp_work"

    with Path(args.manifest).open("r", encoding="utf-8") as f:
        dev_vocal = [
            r for r in csv.DictReader(f)
            if r["split_ext"] == "dev" and r["vocalinstrumental"] == "vocal"
        ]

    # Pick 1 sample per source, cycling deg1..deg7 across sources
    by_src: dict[str, list[dict[str, str]]] = {}
    for r in dev_vocal:
        by_src.setdefault(r["source_id"], []).append(r)
    chosen = []
    for i, (src_id, items) in enumerate(sorted(by_src.items())[: args.max_sources]):
        items_sorted = sorted(items, key=lambda x: x["variant"])
        chosen.append(items_sorted[i % len(items_sorted)])

    print(f"Phase 1: benchmarking {len(chosen)} Dev vocal clips across {len(CANDIDATES)} models + ensemble")

    # Separate clean, x, s for each model (cached to stems_cache for resume)
    cache_dir = OUT_DIR / "stems_cache"
    stems: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray, float]] = {}
    peak_vram: dict[str, float] = {}

    for m_key, m_file in CANDIDATES:
        m_cache = cache_dir / m_key
        m_cache.mkdir(parents=True, exist_ok=True)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        sep = None

        def get_or_run(key_id: str, br: str, in_wav: Path, n_samp: int) -> tuple[np.ndarray, np.ndarray, float]:
            nonlocal sep
            vp = m_cache / f"{key_id}_{br}_voc.wav"
            ap = m_cache / f"{key_id}_{br}_acc.wav"
            tp = m_cache / f"{key_id}_{br}_dt.txt"
            if vp.exists() and ap.exists() and tp.exists():
                return load_stem(vp, n_samp), load_stem(ap, n_samp), float(tp.read_text().strip())
            if sep is None:
                sep = Separator(
                    model_file_dir=str(MODEL_DIR),
                    output_dir=str(work_dir),
                    output_format="WAV",
                    use_autocast=torch.cuda.is_available(),
                )
                sep.load_model(model_filename=m_file)
            voc, acc, dt = separate_clip(sep, in_wav, work_dir, n_samp)
            sf.write(vp, voc, 44100, format="WAV", subtype="PCM_16")
            sf.write(ap, acc, 44100, format="WAV", subtype="PCM_16")
            tp.write_text(f"{dt:.4f}")
            return voc, acc, dt

        for row in chosen:
            sid = row["sample_id"]
            src_id = row["source_id"]
            s_arr, _ = sf.read(ROOT / row["path_s"], dtype="float32", always_2d=True)
            n_samp = len(s_arr)
            # clean (once per source_id)
            if (m_key, src_id, "clean") not in stems:
                c_tmp = work_dir / f"{src_id}_clean_1320960.wav"
                c_tmp.parent.mkdir(parents=True, exist_ok=True)
                c_arr, sr = sf.read(ROOT / row["path_clean"], dtype="float32", always_2d=True)
                sf.write(c_tmp, c_arr[:n_samp], sr)
                stems[(m_key, src_id, "clean")] = get_or_run(src_id, "clean", c_tmp, n_samp)
                c_tmp.unlink(missing_ok=True)
            # x
            x_tmp = work_dir / f"{sid}_x_1320960.wav"
            x_tmp.parent.mkdir(parents=True, exist_ok=True)
            x_arr, sr = sf.read(ROOT / row["path_x"], dtype="float32", always_2d=True)
            sf.write(x_tmp, x_arr[:n_samp], sr)
            stems[(m_key, sid, "x")] = get_or_run(sid, "x", x_tmp, n_samp)
            x_tmp.unlink(missing_ok=True)
            # s
            stems[(m_key, sid, "s")] = get_or_run(sid, "s", ROOT / row["path_s"], n_samp)
            print(f"[{m_key}] {sid} done", flush=True)

        peak_vram[m_key] = float(torch.cuda.max_memory_allocated() / 1024**2) if torch.cuda.is_available() else 0.0
        if sep is not None:
            del sep
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)

    # Compute ensemble (bs_roformer_1297 + mel_band_roformer_kim)
    for row in chosen:
        sid = row["sample_id"]
        src_id = row["source_id"]
        for br, key_id in [("clean", src_id), ("x", sid), ("s", sid)]:
            v1, a1, t1 = stems[("bs_roformer_1297", key_id, br)]
            v2, a2, t2 = stems[("mel_band_roformer_kim", key_id, br)]
            stems[("ensemble_bs_mel", key_id, br)] = (0.5 * (v1 + v2), 0.5 * (a1 + a2), t1 + t2)
    peak_vram["ensemble_bs_mel"] = max(peak_vram["bs_roformer_1297"], peak_vram["mel_band_roformer_kim"])

    all_models = [c[0] for c in CANDIDATES] + ["ensemble_bs_mel"]
    per_clip_rows = []
    summary_rows = []

    for m_key in all_models:
        leak_s_list, sdr_s_list, dt_list = [], [], []
        leak_x_list, sdr_x_list = [], []
        for row in chosen:
            sid = row["sample_id"]
            src_id = row["source_id"]
            voc_ref, acc_ref, _ = stems[("ensemble_bs_mel", src_id, "clean")]

            vx, _, dtx = stems[(m_key, sid, "x")]
            vs, _, dts = stems[(m_key, sid, "s")]

            lx = leakage_db(vx, acc_ref)
            ls = leakage_db(vs, acc_ref)
            sx = si_sdr(vx, voc_ref)
            ss = si_sdr(vs, voc_ref)

            leak_x_list.append(lx)
            leak_s_list.append(ls)
            sdr_x_list.append(sx)
            sdr_s_list.append(ss)
            dt_list.append(dts)

            per_clip_rows.append({
                "model": m_key,
                "sample_id": sid,
                "source_id": src_id,
                "leakage_x_db": lx,
                "leakage_s_db": ls,
                "si_sdr_x_db": sx,
                "si_sdr_s_db": ss,
                "seconds_s": dts,
            })

        summary_rows.append({
            "model": m_key,
            "n_clips": len(chosen),
            "median_leakage_s_db": float(np.median(leak_s_list)),
            "median_leakage_x_db": float(np.median(leak_x_list)),
            "median_si_sdr_s_db": float(np.median(sdr_s_list)),
            "median_si_sdr_x_db": float(np.median(sdr_x_list)),
            "mean_seconds_per_clip": float(np.mean(dt_list)),
            "peak_vram_mb": peak_vram[m_key],
        })

    # Write per_clip.csv & summary.csv
    with (OUT_DIR / "per_clip.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_clip_rows[0].keys()))
        w.writeheader()
        w.writerows(per_clip_rows)

    with (OUT_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    # Select winner: lowest median_leakage_s_db; tie (< 0.5 dB) -> fastest
    best_leak = min(r["median_leakage_s_db"] for r in summary_rows)
    near_best = [r for r in summary_rows if r["median_leakage_s_db"] <= best_leak + 0.5]
    winner = min(near_best, key=lambda r: r["mean_seconds_per_clip"])

    # Lock model hashes
    model_hashes = {}
    for p in sorted(MODEL_DIR.glob("*")):
        if p.is_file() and p.suffix in {".ckpt", ".yaml", ".th", ".onnx", ".safetensors"}:
            model_hashes[p.name] = sha256_file(p)

    lock_ext = {
        "winner": winner["model"],
        "selection_rule": "lowest median_leakage_s_db; tie (<0.5 dB) -> fastest mean_seconds_per_clip",
        "candidates": dict(CANDIDATES),
        "model_files_sha256": model_hashes,
        "summary": summary_rows,
    }
    (ROOT / "artifacts" / "10_extension" / "source-lock-extension.json").write_text(
        json.dumps(lock_ext, indent=2), encoding="utf-8"
    )
    print("Phase 1 winner:", winner["model"])
    print(json.dumps(summary_rows, indent=2))


if __name__ == "__main__":
    main()
