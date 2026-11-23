"""Upload the ~1,888 changed Confirm WAVs to Kaggle T4 GPU to compute CLAP embeddings & Audiobox PQ."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = Path(__file__).resolve().parents[2]
GDRIVE = Path("G:/My Drive/SonicMaster_Reproducing")
EXT_DIR = ROOT / "artifacts" / "10_extension"
S_FALLBACK = {"real", "clip", "xband", "small", "big", "mix", "mud"}


def res(p: str | Path) -> Path:
    dp = ROOT / p
    return dp if dp.is_file() else GDRIVE / p


def get_api(line_idx: int = 5) -> tuple[KaggleApi, str]:
    lines = [l.strip() for l in Path("C:/Users/ADMIN/.kaggle/access_token.txt").read_text().splitlines() if l.strip()]
    os.environ["KAGGLE_API_TOKEN"] = lines[line_idx]
    api = KaggleApi()
    api.authenticate()
    return api, str(api.config_values["username"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(EXT_DIR / "confirm" / "hybrid_v2" / "out_wav"))
    parser.add_argument("--ds-slug", default="sonicmaster-ext-confirm-pq-fad-v2")
    parser.add_argument("--k-slug", default="sonicmaster-ext-confirm-pq-fad-run")
    parser.add_argument("--s-fallback", default="real,clip,xband,small,big,mix,mud")
    parser.add_argument("--all-out-wav", action="store_true")
    parser.add_argument("--push-only", action="store_true")
    args = parser.parse_args()

    s_fallback = {s.strip() for s in args.s_fallback.split(",") if s.strip()}

    splits = json.loads(res("artifacts/10_extension/splits.json").read_text(encoding="utf-8"))
    confirm_sources = set(splits["confirm_sources"])
    out_dir = Path(args.out_dir)

    changed_sids = []
    with res("artifacts/10_extension/inputs_manifest.csv").open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["source_id"] not in confirm_sources:
                continue
            if args.all_out_wav:
                if (out_dir / f"{r['sample_id']}.wav").is_file() and r["path_x"]:
                    changed_sids.append(r["sample_id"])
            else:
                degs = {d.strip() for d in r["degradation_types"].split("|") if d.strip()}
                if not (degs & s_fallback) and r["path_x"]:
                    changed_sids.append(r["sample_id"])

    changed_sids.sort()
    first_sha = hashlib.sha256((out_dir / f"{changed_sids[0]}.wav").read_bytes()).hexdigest()
    print(f"Staging {len(changed_sids)} candidate changed Confirm WAVs... LOCAL_SHA256[{changed_sids[0]}]={first_sha}", flush=True)

    api, owner = get_api(5)
    ds_slug = args.ds_slug
    k_slug = args.k_slug

    stage_dir = EXT_DIR / "kaggle_confirm_pq_fad_stage"
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    if not args.push_only:
        # Split into 2 zips so each is ~2.5GB
        mid = len(changed_sids) // 2
        for zname, chunk in [("wavs_part1.zip", changed_sids[:mid]), ("wavs_part2.zip", changed_sids[mid:])]:
            with zipfile.ZipFile(stage_dir / zname, "w", compression=zipfile.ZIP_STORED) as zf:
                for sid in chunk:
                    p = out_dir / f"{sid}.wav"
                    zf.write(p, arcname=f"{sid}.wav")
        (stage_dir / "sids.json").write_text(json.dumps(changed_sids), encoding="utf-8")
        (stage_dir / "dataset-metadata.json").write_text(
            json.dumps({"title": ds_slug, "id": f"{owner}/{ds_slug}", "licenses": [{"name": "other"}]}, indent=2),
            encoding="utf-8",
        )
        Path("C:/Users/ADMIN/AppData/Local/Temp/.kaggle-uploads").mkdir(parents=True, exist_ok=True)
        try:
            api.dataset_create_new(str(stage_dir), dir_mode="zip", quiet=False)
        except Exception as e:
            print(f"dataset_create_new note: {e}; trying version...")
            api.dataset_create_version(str(stage_dir), version_notes="update", dir_mode="zip", quiet=False)

        shutil.rmtree(stage_dir, ignore_errors=True)
        for _ in range(18):
            try:
                st = api.dataset_status(f"{owner}/{ds_slug}")
                print("dataset status:", st, flush=True)
                if str(st).lower() == "ready":
                    break
            except Exception as e:
                print(f"dataset_status note: {e}", flush=True)
            time.sleep(5)
    else:
        shutil.rmtree(stage_dir, ignore_errors=True)

    k_dir = EXT_DIR / "kaggle_kernel_confirm_pq_fad"
    k_dir.mkdir(parents=True, exist_ok=True)
    script = '''"""Kaggle T4 GPU: Compute CLAP embeddings & Audiobox PQ for changed Confirm WAVs."""
import hashlib, json, subprocess, sys, time, zipfile
from pathlib import Path

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "laion-clap", "audiobox_aesthetics"])
import numpy as np
import laion_clap
from audiobox_aesthetics.infer import initialize_predictor

INPUT_ROOT = Path("/kaggle/input")
DATA_DIR = None
for _ in range(10):
    found = list(INPUT_ROOT.rglob("sids.json"))
    if found:
        DATA_DIR = found[0].parent
        break
    time.sleep(5)
if DATA_DIR is None:
    raise RuntimeError("sids.json not found")

sids = json.loads((DATA_DIR / "sids.json").read_text())
WORK = Path("/kaggle/working/in_wav")
WORK.mkdir(parents=True, exist_ok=True)
for zp in DATA_DIR.rglob("*.zip"):
    with zipfile.ZipFile(zp, "r") as zf:
        zf.extractall(WORK)
wav_map = {p.stem: p for p in DATA_DIR.rglob("*.wav")}
wav_map.update({p.stem: p for p in WORK.rglob("*.wav")})
print(f"Found {len(wav_map)} WAVs for {len(sids)} sids", flush=True)

paths = [str(wav_map[sid]) for sid in sids]
first_sha = hashlib.sha256(Path(paths[0]).read_bytes()).hexdigest()
print(f"KAGGLE_SAMPLE_SHA256[{sids[0]}]={first_sha}", flush=True)

clap = laion_clap.CLAP_Module(enable_fusion=False)
clap.load_ckpt()
t0 = time.time()
embs = []
for i in range(0, len(paths), 32):
    e = clap.get_audio_embedding_from_filelist(x=paths[i:i+32], use_tensor=False)
    embs.append(e.astype(np.float32))
    if (i // 32 + 1) % 10 == 0:
        print(f"CLAP [{min(i+32, len(paths))}/{len(paths)}] {time.time()-t0:.1f}s", flush=True)
emb_arr = np.concatenate(embs, axis=0)

predictor = initialize_predictor()
t1 = time.time()
pqs = []
for i in range(0, len(paths), 8):
    res = predictor.forward([{"path": p} for p in paths[i:i+8]])
    pqs.extend([float(x["PQ"]) for x in res])
    if (i // 8 + 1) % 40 == 0:
        print(f"PQ [{min(i+8, len(paths))}/{len(paths)}] {time.time()-t1:.1f}s", flush=True)
pq_arr = np.array(pqs, dtype=np.float32)

OUT = Path("/kaggle/working/out")
OUT.mkdir(parents=True, exist_ok=True)
np.savez(OUT / "changed_pq_fad.npz", sample_ids=np.array(sids), emb=emb_arr, pq=pq_arr)
print("DONE:", emb_arr.shape, pq_arr.shape, float(np.mean(pq_arr)))
'''
    (k_dir / "run_confirm_pq_fad.py").write_text(script, encoding="utf-8")
    meta = {
        "id": f"{owner}/{k_slug}",
        "title": k_slug,
        "code_file": "run_confirm_pq_fad.py",
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

    for attempt in range(5):
        try:
            api.kernels_push(str(k_dir))
            break
        except Exception as e:
            print(f"kernels_push retry {attempt+1}: {e}")
            time.sleep(15)
    print(f"Pushed! {owner}/{k_slug}")


if __name__ == "__main__":
    main()
