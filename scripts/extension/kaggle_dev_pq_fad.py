"""Run PQ (Audiobox Aesthetics) and FAD-CLAP for Dev A0 vs A5_final on Kaggle T4 GPU."""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = Path(__file__).resolve().parents[2]
GDRIVE = Path("G:/My Drive/SonicMaster_Reproducing")
EXT_DIR = ROOT / "artifacts" / "10_extension"


def res(p: str | Path) -> Path:
    dp = ROOT / p
    return dp if dp.exists() else GDRIVE / p


def get_api(line_idx: int = 5) -> tuple[KaggleApi, str]:
    lines = [l.strip() for l in Path("C:/Users/ADMIN/.kaggle/access_token.txt").read_text().splitlines() if l.strip()]
    os.environ["KAGGLE_API_TOKEN"] = lines[line_idx]
    api = KaggleApi()
    api.authenticate()
    return api, str(api.config_values["username"])


def stage_pq_fad_dataset(stage_dir: Path, cand_dir: Path) -> None:
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    (stage_dir / "clean").mkdir(parents=True, exist_ok=True)
    (stage_dir / "A0").mkdir(parents=True, exist_ok=True)
    (stage_dir / "A5").mkdir(parents=True, exist_ok=True)

    with res("artifacts/10_extension/inputs_manifest.csv").open("r", encoding="utf-8") as f:
        dev_rows = [r for r in csv.DictReader(f) if r["split_ext"] == "dev"]

    seen_src = set()
    meta_rows = []
    for r in dev_rows:
        sid = r["sample_id"]
        src = r["source_id"]
        if src not in seen_src:
            shutil.copy2(res(r["path_clean"]), stage_dir / "clean" / f"{src}.wav")
            seen_src.add(src)
        shutil.copy2(res(r["path_s"]), stage_dir / "A0" / f"{sid}.wav")
        shutil.copy2(cand_dir / f"{sid}.wav", stage_dir / "A5" / f"{sid}.wav")
        meta_rows.append({"sample_id": sid, "source_id": src, "n_degradations": int(r["n_degradations"])})

    (stage_dir / "dev_meta.json").write_text(json.dumps(meta_rows), encoding="utf-8")
    print(f"Staged {len(meta_rows)} Dev A0/A5 + {len(seen_src)} clean in {stage_dir}")


def write_kernel(k_dir: Path, owner: str, ds_slug: str, k_slug: str) -> None:
    k_dir.mkdir(parents=True, exist_ok=True)
    script = '''"""Kaggle GPU: Compute Dev PQ (Audiobox Aesthetics) and FAD-CLAP for A0 vs A5."""
import json, subprocess, sys, time
from pathlib import Path
import numpy as np
from scipy.linalg import sqrtm

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torchlibrosa", "ftfy", "braceexpand", "webdataset", "wget", "progressbar"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "laion-clap", "audiobox-aesthetics"])
from audiobox_aesthetics.infer import initialize_predictor
from laion_clap import CLAP_Module

INPUT_ROOT = Path("/kaggle/input")
DATA_DIR = None
for _ in range(10):
    found = list(INPUT_ROOT.rglob("dev_meta.json"))
    if found:
        DATA_DIR = found[0].parent
        break
    time.sleep(5)
if DATA_DIR is None:
    raise RuntimeError("dev_meta.json not found")

rows = json.loads((DATA_DIR / "dev_meta.json").read_text())
import hashlib
_sample_wav = DATA_DIR / "A5" / "1077462_deg5.wav"
print("KAGGLE_SAMPLE_SHA256:", hashlib.sha256(_sample_wav.read_bytes()).hexdigest(), flush=True)
OUT = Path("/kaggle/working/out")
OUT.mkdir(parents=True, exist_ok=True)

def calculate_fad(emb1: np.ndarray, emb2: np.ndarray) -> float:
    mu1, sigma1 = np.mean(emb1, axis=0), np.cov(emb1, rowvar=False)
    mu2, sigma2 = np.mean(emb2, axis=0), np.cov(emb2, rowvar=False)
    diff = mu1 - mu2
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean))

clap = CLAP_Module(enable_fusion=False)
clap.load_ckpt()
clean_srcs = sorted({r["source_id"] for r in rows})
clean_paths = [str(DATA_DIR / "clean" / f"{s}.wav") for s in clean_srcs]
clean_embs = {}
for i in range(0, len(clean_paths), 32):
    b = clean_paths[i:i+32]
    em = clap.get_audio_embedding_from_filelist(b, use_tensor=False)
    for s, e in zip(clean_srcs[i:i+32], em):
        clean_embs[s] = e

c_embs = np.stack([clean_embs[r["source_id"]] for r in rows])
embs_by_name = {}
for name in ["A0", "A5"]:
    paths = [str(DATA_DIR / name / f"{r['sample_id']}.wav") for r in rows]
    o_list = []
    for i in range(0, len(paths), 32):
        o_list.append(clap.get_audio_embedding_from_filelist(paths[i:i+32], use_tensor=False))
    embs_by_name[name] = np.concatenate(o_list, axis=0)

import torch
del clap
torch.cuda.empty_cache()

predictor = initialize_predictor()
clean_pq = []
for i in range(0, len(clean_paths), 4):
    clean_pq.extend(float(x["PQ"]) for x in predictor.forward([{"path": p} for p in clean_paths[i:i+4]]))

results = {"Clean_PQ": float(np.mean(clean_pq))}
per_sample_pq = {}

for name in ["A0", "A5"]:
    paths = [str(DATA_DIR / name / f"{r['sample_id']}.wav") for r in rows]
    pq_vals = []
    for i in range(0, len(paths), 4):
        pq_vals.extend(float(x["PQ"]) for x in predictor.forward([{"path": p} for p in paths[i:i+4]]))
    o_embs = embs_by_name[name]
    per_sample_pq[name] = pq_vals

    idx_all = list(range(len(rows)))
    idx_s = [i for i, r in enumerate(rows) if r["n_degradations"] == 1]
    idx_m = [i for i, r in enumerate(rows) if r["n_degradations"] > 1]
    results[name] = {}
    for grp, idxs in [("All", idx_all), ("Single", idx_s), ("Double+Triple", idx_m)]:
        results[name][grp] = {
            "n": len(idxs),
            "FAD": calculate_fad(o_embs[idxs], c_embs[idxs]),
            "PQ": float(np.mean([pq_vals[i] for i in idxs])),
        }

# Source-level bootstrap CI95 for delta_PQ (A5 - A0)
by_src = {}
for i, r in enumerate(rows):
    by_src.setdefault(r["source_id"], []).append(per_sample_pq["A5"][i] - per_sample_pq["A0"][i])
srcs = sorted(by_src.keys())
s_means = np.array([np.mean(by_src[s]) for s in srcs])
rng = np.random.default_rng(20261119)
boots = np.mean(s_means[rng.integers(0, len(srcs), size=(2000, len(srcs)))], axis=1)
results["delta_PQ_A5_minus_A0"] = {
    "mean": float(np.mean(s_means)),
    "ci95_lo": float(np.percentile(boots, 2.5)),
    "ci95_hi": float(np.percentile(boots, 97.5)),
}

(OUT / "dev_pq_fad.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
'''
    (k_dir / "run_pq_fad.py").write_text(script, encoding="utf-8")
    meta = {
        "id": f"{owner}/{k_slug}",
        "title": k_slug,
        "code_file": "run_pq_fad.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [f"{owner}/{ds_slug}"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (k_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--account-line", type=int, default=5)
    ap.add_argument("--push-only", action="store_true")
    ap.add_argument("--cand-dir", default=str(EXT_DIR / "dev" / "hybrid_v2" / "out_wav"))
    ap.add_argument("--ds-slug", default="sonicmaster-ext-dev-pq-fad-v2")
    args = ap.parse_args()

    api, owner = get_api(args.account_line)
    ds_slug = args.ds_slug
    k_slug = "sonicmaster-ext-dev-pq-fad-run"

    if not args.push_only:
        stage_dir = EXT_DIR / "kaggle_pq_fad_stage"
        stage_pq_fad_dataset(stage_dir, Path(args.cand_dir))
        (stage_dir / "dataset-metadata.json").write_text(
            json.dumps({"title": ds_slug, "id": f"{owner}/{ds_slug}", "licenses": [{"name": "other"}]}, indent=2),
            encoding="utf-8",
        )

        try:
            api.dataset_create_new(str(stage_dir), dir_mode="zip", quiet=False)
        except Exception as e:
            print(f"dataset_create_new note: {e}; trying version...")
            api.dataset_create_version(str(stage_dir), version_notes="update", dir_mode="zip", quiet=False)

        shutil.rmtree(stage_dir, ignore_errors=True)
    for _ in range(12):
        try:
            st = api.dataset_status(f"{owner}/{ds_slug}")
            print("dataset status:", st, flush=True)
            if str(st).lower() == "ready":
                break
        except Exception as e:
            print("dataset_status note:", e, flush=True)
        time.sleep(5)
    k_dir = EXT_DIR / "kaggle_kernel_pq_fad"
    write_kernel(k_dir, owner, ds_slug, k_slug)
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
