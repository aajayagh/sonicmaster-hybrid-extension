"""Stage Dev audio as a private Kaggle dataset and run Phase 1 + Dev separator stems on Kaggle GPU."""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = Path(__file__).resolve().parents[2]
EXT_DIR = ROOT / "artifacts" / "10_extension"


def get_api(line_idx: int = 0) -> tuple[KaggleApi, str]:
    lines = [l.strip() for l in Path("C:/Users/ADMIN/.kaggle/access_token.txt").read_text().splitlines() if l.strip()]
    os.environ["KAGGLE_API_TOKEN"] = lines[line_idx]
    api = KaggleApi()
    api.authenticate()
    return api, str(api.config_values["username"])


def stage_dev_dataset(stage_dir: Path) -> None:
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    audio_dir = stage_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    with (EXT_DIR / "inputs_manifest.csv").open("r", encoding="utf-8") as f:
        dev_rows = [r for r in csv.DictReader(f) if r["split_ext"] == "dev"]

    # Copy all 280 Dev s files
    for r in dev_rows:
        shutil.copy2(ROOT / r["path_s"], audio_dir / f"{r['sample_id']}_s.wav")

    # Select 10 vocal Dev sources for Phase 1
    by_src: dict[str, list[dict[str, str]]] = {}
    for r in dev_rows:
        if r["vocalinstrumental"] == "vocal":
            by_src.setdefault(r["source_id"], []).append(r)

    p1_rows = []
    for i, (src_id, items) in enumerate(sorted(by_src.items())[:10]):
        items_sorted = sorted(items, key=lambda x: x["variant"])
        chosen = items_sorted[i % len(items_sorted)]
        p1_rows.append({"sample_id": chosen["sample_id"], "source_id": src_id})
        shutil.copy2(ROOT / chosen["path_x"], audio_dir / f"{chosen['sample_id']}_x.wav")
        shutil.copy2(ROOT / chosen["path_clean"], audio_dir / f"{src_id}_clean.wav")

    (stage_dir / "dev_ids.json").write_text(
        json.dumps({"dev_ids": [r["sample_id"] for r in dev_rows], "phase1_clips": p1_rows}, indent=2),
        encoding="utf-8",
    )
    print(f"Staged {len(dev_rows)} Dev s files + {len(p1_rows)} Phase 1 (x, clean) pairs in {stage_dir}")


def write_kaggle_kernel(kernel_dir: Path, owner: str, dataset_slug: str, kernel_slug: str) -> None:
    kernel_dir.mkdir(parents=True, exist_ok=True)
    script = '''"""Kaggle GPU: Phase 1 Separator Benchmark + Dev Stems Extraction."""
import csv, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "audio-separator[gpu]"])
from audio_separator.separator import Separator

INPUT_ROOT = Path("/kaggle/input")
DATA_DIR = None
for _ in range(10):
    found = list(INPUT_ROOT.rglob("dev_ids.json"))
    if found:
        DATA_DIR = found[0].parent
        break
    time.sleep(5)
if DATA_DIR is None:
    raise RuntimeError(f"dev_ids.json not found in /kaggle/input: {list(INPUT_ROOT.rglob('*'))[:20]}")

AUDIO_DIR = DATA_DIR / "audio"
meta = json.loads((DATA_DIR / "dev_ids.json").read_text())
dev_ids = meta["dev_ids"]
p1_clips = meta["phase1_clips"]

OUT = Path("/kaggle/working/out")
OUT.mkdir(parents=True, exist_ok=True)
MODEL_DIR = Path("/kaggle/working/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
WORK_DIR = Path("/kaggle/working/tmp_work")

CANDIDATES = [
    ("htdemucs_ft", "htdemucs_ft.yaml"),
    ("bs_roformer_1297", "model_bs_roformer_ep_317_sdr_12.9755.ckpt"),
    ("mel_band_roformer_kim", "vocals_mel_band_roformer.ckpt"),
]

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while b := f.read(1024 * 1024):
            h.update(b)
    return h.hexdigest()

def load_stem(p: Path, n_samples: int) -> np.ndarray:
    y, _ = sf.read(p, dtype="float32", always_2d=True)
    if y.shape[1] == 1:
        y = np.repeat(y, 2, axis=1)
    if len(y) > n_samples:
        y = y[:n_samples]
    elif len(y) < n_samples:
        y = np.pad(y, ((0, n_samples - len(y)), (0, 0)))
    return y

def si_sdr(est: np.ndarray, ref: np.ndarray) -> float:
    e = est.mean(axis=1).astype(np.float64)
    r = ref.mean(axis=1).astype(np.float64)
    e -= e.mean(); r -= r.mean()
    a = np.dot(e, r) / (np.dot(r, r) + 1e-12)
    t = a * r
    return float(10.0 * np.log10((np.sum(t**2) + 1e-12) / (np.sum((e - t)**2) + 1e-12)))

def leakage_db(voc_hat: np.ndarray, acc_ref: np.ndarray) -> float:
    v = voc_hat.astype(np.float64); a = acc_ref.astype(np.float64)
    proj = (np.sum(v * a) / (np.sum(a * a) + 1e-12)) * a
    return float(10.0 * np.log10((np.sum(proj**2) + 1e-12) / (np.sum(v**2) + 1e-12)))

def separate_clip(sep: Separator, in_wav: Path, n_samples: int):
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    sep.output_dir = str(WORK_DIR)
    t0 = time.perf_counter()
    files = sep.separate(str(in_wav))
    dt = time.perf_counter() - t0
    voc_p, acc_ps = None, []
    for f in files:
        p = WORK_DIR / Path(f).name
        if not p.exists(): p = Path(f)
        if "(vocals)" in p.name.lower(): voc_p = p
        else: acc_ps.append(p)
    voc = load_stem(voc_p, n_samples)
    acc = sum(load_stem(p, n_samples) for p in acc_ps) if acc_ps else (load_stem(in_wav, n_samples) - voc)
    for f in files:
        p = WORK_DIR / Path(f).name
        if p.exists(): p.unlink()
    return voc, acc, dt

# 1. Phase 1 Benchmark
stems = {}
peak_vram = {}
N_SAMP = 1320960
for m_key, m_file in CANDIDATES:
    torch.cuda.reset_peak_memory_stats()
    sep = Separator(model_file_dir=str(MODEL_DIR), output_dir=str(WORK_DIR), output_format="WAV", use_autocast=True)
    sep.load_model(model_filename=m_file)
    for item in p1_clips:
        sid, src_id = item["sample_id"], item["source_id"]
        if (m_key, src_id, "clean") not in stems:
            c_arr, sr = sf.read(AUDIO_DIR / f"{src_id}_clean.wav", dtype="float32", always_2d=True)
            c_tmp = WORK_DIR / f"{src_id}_c.wav"; WORK_DIR.mkdir(parents=True, exist_ok=True)
            sf.write(c_tmp, c_arr[:N_SAMP], sr)
            stems[(m_key, src_id, "clean")] = separate_clip(sep, c_tmp, N_SAMP)
            c_tmp.unlink(missing_ok=True)
        x_arr, sr = sf.read(AUDIO_DIR / f"{sid}_x.wav", dtype="float32", always_2d=True)
        x_tmp = WORK_DIR / f"{sid}_x.wav"
        sf.write(x_tmp, x_arr[:N_SAMP], sr)
        stems[(m_key, sid, "x")] = separate_clip(sep, x_tmp, N_SAMP)
        x_tmp.unlink(missing_ok=True)
        stems[(m_key, sid, "s")] = separate_clip(sep, AUDIO_DIR / f"{sid}_s.wav", N_SAMP)
    peak_vram[m_key] = float(torch.cuda.max_memory_allocated() / 1024**2)
    del sep; torch.cuda.empty_cache()

for item in p1_clips:
    sid, src_id = item["sample_id"], item["source_id"]
    for br, kid in [("clean", src_id), ("x", sid), ("s", sid)]:
        v1, a1, t1 = stems[("bs_roformer_1297", kid, br)]
        v2, a2, t2 = stems[("mel_band_roformer_kim", kid, br)]
        stems[("ensemble_bs_mel", kid, br)] = (0.5*(v1+v2), 0.5*(a1+a2), t1+t2)
peak_vram["ensemble_bs_mel"] = max(peak_vram["bs_roformer_1297"], peak_vram["mel_band_roformer_kim"])

summary_rows = []
per_clip_rows = []
for m_key in [c[0] for c in CANDIDATES] + ["ensemble_bs_mel"]:
    ls_l, lx_l, ss_l, sx_l, dt_l = [], [], [], [], []
    for item in p1_clips:
        sid, src_id = item["sample_id"], item["source_id"]
        voc_ref, acc_ref, _ = stems[("ensemble_bs_mel", src_id, "clean")]
        vx, _, _ = stems[(m_key, sid, "x")]
        vs, _, dts = stems[(m_key, sid, "s")]
        lx, ls = leakage_db(vx, acc_ref), leakage_db(vs, acc_ref)
        sx, ss = si_sdr(vx, voc_ref), si_sdr(vs, voc_ref)
        lx_l.append(lx); ls_l.append(ls); sx_l.append(sx); ss_l.append(ss); dt_l.append(dts)
        per_clip_rows.append({"model": m_key, "sample_id": sid, "leakage_x_db": lx, "leakage_s_db": ls, "si_sdr_x_db": sx, "si_sdr_s_db": ss, "seconds_s": dts})
    summary_rows.append({"model": m_key, "n_clips": len(p1_clips), "median_leakage_s_db": float(np.median(ls_l)), "median_leakage_x_db": float(np.median(lx_l)), "median_si_sdr_s_db": float(np.median(ss_l)), "median_si_sdr_x_db": float(np.median(sx_l)), "mean_seconds_per_clip": float(np.mean(dt_l)), "peak_vram_mb": peak_vram[m_key]})

with (OUT / "phase1_summary.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys())); w.writeheader(); w.writerows(summary_rows)
with (OUT / "phase1_per_clip.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(per_clip_rows[0].keys())); w.writeheader(); w.writerows(per_clip_rows)

best_leak = min(r["median_leakage_s_db"] for r in summary_rows)
winner = min([r for r in summary_rows if r["median_leakage_s_db"] <= best_leak + 0.5], key=lambda r: r["mean_seconds_per_clip"])["model"]
model_hashes = {p.name: sha256_file(p) for p in sorted(MODEL_DIR.glob("*")) if p.is_file() and p.suffix in {".ckpt", ".yaml", ".th"}}
(OUT / "source-lock-extension.json").write_text(json.dumps({"winner": winner, "model_files_sha256": model_hashes, "summary": summary_rows}, indent=2))
print("WINNER:", winner)

# 2. Separate all 280 Dev s files with winning model (save as FLAC to save space)
stems_dir = OUT / "dev_stems_flac"
stems_dir.mkdir(parents=True, exist_ok=True)
if winner == "ensemble_bs_mel":
    models_to_run = [("bs_roformer_1297", "model_bs_roformer_ep_317_sdr_12.9755.ckpt"), ("mel_band_roformer_kim", "vocals_mel_band_roformer.ckpt")]
else:
    models_to_run = [(winner, dict(CANDIDATES)[winner])]

# Free phase1 stems
stems.clear()
for i, sid in enumerate(dev_ids):
    if (stems_dir / f"{sid}_voc.flac").exists() and (stems_dir / f"{sid}_acc.flac").exists():
        continue
    v_sum, a_sum = None, None
    for _, mf in models_to_run:
        # Load lazily once per model if single, or reuse
        pass
'''
    # Write cleaner loop for step 2
    script = script.split("# Free phase1 stems")[0] + '''stems.clear()
accum = {sid: [None, None] for sid in dev_ids}
for _, mf in models_to_run:
    sep = Separator(model_file_dir=str(MODEL_DIR), output_dir=str(WORK_DIR), output_format="WAV", use_autocast=True)
    sep.load_model(model_filename=mf)
    for i, sid in enumerate(dev_ids):
        v, a, _ = separate_clip(sep, AUDIO_DIR / f"{sid}_s.wav", N_SAMP)
        if accum[sid][0] is None:
            accum[sid] = [v / len(models_to_run), a / len(models_to_run)]
        else:
            accum[sid][0] += v / len(models_to_run)
            accum[sid][1] += a / len(models_to_run)
        if (i + 1) % 25 == 0:
            print(f"[{mf}] separated {i+1}/{len(dev_ids)}", flush=True)
    del sep; torch.cuda.empty_cache()

for sid, (v, a) in accum.items():
    sf.write(stems_dir / f"{sid}_voc.flac", v, 44100, format="FLAC")
    sf.write(stems_dir / f"{sid}_acc.flac", a, 44100, format="FLAC")
print("ALL 280 DEV STEMS DONE!")
'''
    (kernel_dir / "run_phase1_dev.py").write_text(script, encoding="utf-8")
    meta_k = {
        "id": f"{owner}/{kernel_slug}",
        "title": kernel_slug,
        "code_file": "run_phase1_dev.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [f"{owner}/{dataset_slug}"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (kernel_dir / "kernel-metadata.json").write_text(json.dumps(meta_k, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--account-line", type=int, default=0)
    ap.add_argument("--stage-only", action="store_true")
    args = ap.parse_args()

    api, owner = get_api(args.account_line)
    ds_slug = "sonicmaster-ext-dev-audio"
    k_slug = "sonicmaster-ext-phase1-dev-stems"

    stage_dir = EXT_DIR / "kaggle_dev_stage"
    stage_dev_dataset(stage_dir)
    ds_meta = {
        "title": ds_slug,
        "id": f"{owner}/{ds_slug}",
        "licenses": [{"name": "other"}],
    }
    (stage_dir / "dataset-metadata.json").write_text(json.dumps(ds_meta, indent=2), encoding="utf-8")

    if args.stage_only:
        return

    print(f"Uploading dataset {owner}/{ds_slug}...")
    try:
        api.dataset_create_new(str(stage_dir), dir_mode="zip", quiet=False)
    except Exception as e:
        print(f"dataset_create_new note: {e}; trying dataset_create_version...")
        api.dataset_create_version(str(stage_dir), version_notes="update", dir_mode="zip", quiet=False)

    import time
    time.sleep(20)
    k_dir = EXT_DIR / "kaggle_kernel_p1"
    write_kaggle_kernel(k_dir, owner, ds_slug, k_slug)
    print(f"Pushing GPU kernel {owner}/{k_slug}...")
    for attempt in range(5):
        try:
            api.kernels_push(str(k_dir))
            break
        except Exception as e:
            print(f"kernels_push retry {attempt+1}: {e}")
            time.sleep(15)
    print(f"Pushed! Monitor: {owner}/{k_slug}")


if __name__ == "__main__":
    main()
