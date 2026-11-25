"""Stage 5.0s float16 Pilot dataset (400 train / 80 val sources) with full-track RMS metadata and upload to Kaggle."""
from __future__ import annotations

import csv
import io
import json
import os
import random
import shutil
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(r"d:\SonicMaster_Reproducing")
GDRIVE = Path(r"G:\My Drive\SonicMaster_Reproducing")
sys.path.insert(0, str(ROOT))

CHUNK_SAMPLES = 44100 * 5  # 5.0s = 220,500 samples


def res(p: str | Path) -> Path:
    dp = ROOT / p
    return dp if dp.is_file() else GDRIVE / p


def load_stereo_np(path: Path, frames: int = 1320960) -> np.ndarray:
    for attempt in range(3):
        try:
            y, _ = sf.read(io.BytesIO(path.read_bytes()), dtype="float32", always_2d=True)
            if y.shape[1] == 1:
                y = np.repeat(y, 2, axis=1)
            return y[:frames].T.copy()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1.0)
    raise RuntimeError(f"Failed to load {path}")


def process_source_group(
    src: str,
    srows: list[dict[str, str]],
    clean_out_dir_str: str,
    clips_out_dir_str: str,
) -> tuple[str, int]:
    from scripts.extension.v4_alignment import align_teacher_to_input
    c_path = res(srows[0]["path_clean"])
    c_full = load_stereo_np(c_path)
    # Choose the crop on the intersection of all valid extents for this source.
    n_c = min(c_full.shape[-1], *(sf.info(res(r[key])).frames
              for r in srows for key in ("path_x", "path_s")))

    # Pick an active 5.0s window for this source song
    rng = random.Random(int(src) if src.isdigit() else hash(src))
    if n_c > CHUNK_SAMPLES:
        best_st = 0
        best_rms = -1.0
        for _ in range(8):
            st = rng.randint(0, n_c - CHUNK_SAMPLES)
            r_val = float(np.sqrt(np.mean(c_full[:, st : st + CHUNK_SAMPLES] ** 2)))
            if r_val >= 0.04:
                best_st = st
                break
            if r_val > best_rms:
                best_rms = r_val
                best_st = st
        en = best_st + CHUNK_SAMPLES
    else:
        best_st, en = 0, n_c

    c_crop = c_full[:, best_st:en].astype(np.float16)
    np.save(Path(clean_out_dir_str) / f"{src}.npy", c_crop)

    saved = 0
    for r in srows:
        sid = r["sample_id"]
        x_full = load_stereo_np(res(r["path_x"]))
        s_full = load_stereo_np(res(r["path_s"]))
        n = min(x_full.shape[-1], s_full.shape[-1], n_c)
        x_full, s_full = x_full[:, :n], s_full[:, :n]

        s_full = align_teacher_to_input(x_full, s_full)

        rms_x_full = float(np.sqrt(np.mean(x_full.astype(np.float64) ** 2)) + 1e-8)
        rms_s_full = float(np.sqrt(np.mean(s_full.astype(np.float64) ** 2)) + 1e-8)

        np.savez(
            Path(clips_out_dir_str) / f"{sid}.npz",
            x=x_full[:, best_st:en].astype(np.float16),
            s=s_full[:, best_st:en].astype(np.float16),
            rms_x=np.float32(rms_x_full),
            rms_s=np.float32(rms_s_full),
        )
        saved += 1
    return src, saved


def get_api(line_idx: int = 0):
    from kaggle.api.kaggle_api_extended import KaggleApi
    lines = [l.strip() for l in Path("C:/Users/ADMIN/.kaggle/access_token.txt").read_text().splitlines() if l.strip()]
    os.environ["KAGGLE_API_TOKEN"] = lines[line_idx]
    api = KaggleApi()
    api.authenticate()
    return api, str(api.config_values["username"])


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--account-line", type=int, default=0)
    ap.add_argument("--ds-slug", default="sonicmaster-v4-pilot-5s")
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--extract-only", action="store_true")
    args = ap.parse_args()

    splits = json.loads((ROOT / "artifacts/10_extension/v4/pilot_splits.json").read_text(encoding="utf-8"))["pilot"]
    tr_srcs = set(splits["train_sources"])
    va_srcs = set(splits["val_sources"])
    target_srcs = tr_srcs | va_srcs

    all_rows = [
        r for r in csv.DictReader(res("artifacts/10_extension/inputs_manifest.csv").open("r", encoding="utf-8"))
        if r["source_id"] in target_srcs and r["path_x"] and r["path_s"] and r["path_clean"]
    ]
    by_src: dict[str, list[dict[str, str]]] = {}
    for r in all_rows:
        by_src.setdefault(r["source_id"], []).append(r)

    stage_dir = ROOT / "artifacts" / "10_extension" / "v4" / "pilot_kaggle_stage"
    clean_dir = stage_dir / "clean"
    clips_dir = stage_dir / "clips"

    if not args.skip_extract:
        if stage_dir.exists():
            shutil.rmtree(stage_dir)
        clean_dir.mkdir(parents=True, exist_ok=True)
        clips_dir.mkdir(parents=True, exist_ok=True)

        print(f"Extracting 5.0s float16 crops for {len(by_src)} sources ({len(all_rows)} clips)...", flush=True)
        t0 = time.time()
        total_clips = 0
        with ProcessPoolExecutor(max_workers=8) as ex:
            futs = [ex.submit(process_source_group, src, srows, str(clean_dir), str(clips_dir)) for src, srows in by_src.items()]
            for i, fut in enumerate(as_completed(futs), 1):
                _, n_s = fut.result()
                total_clips += n_s
                if i % 40 == 0 or i == len(by_src):
                    print(f"  [{i}/{len(by_src)}] clips={total_clips} elapsed={time.time()-t0:.1f}s", flush=True)

        meta = {
            "train_rows": [
                {"sample_id": r["sample_id"], "source_id": r["source_id"], "degradation_types": r["degradation_types"], "n_degradations": int(r["n_degradations"])}
                for r in all_rows if r["source_id"] in tr_srcs
            ],
            "val_rows": [
                {"sample_id": r["sample_id"], "source_id": r["source_id"], "degradation_types": r["degradation_types"], "n_degradations": int(r["n_degradations"])}
                for r in all_rows if r["source_id"] in va_srcs
            ],
        }
        (stage_dir / "pilot_meta.json").write_text(json.dumps(meta), encoding="utf-8")

    if args.extract_only:
        print(f"Extraction finished in {stage_dir}")
        return

    api, owner = get_api(args.account_line)
    (stage_dir / "dataset-metadata.json").write_text(
        json.dumps({"title": args.ds_slug, "id": f"{owner}/{args.ds_slug}", "licenses": [{"name": "other"}]}, indent=2),
        encoding="utf-8",
    )
    print(f"Uploading {stage_dir} to {owner}/{args.ds_slug}...", flush=True)
    Path("C:/Users/ADMIN/AppData/Local/Temp/.kaggle-uploads").mkdir(parents=True, exist_ok=True)
    try:
        api.dataset_create_new(str(stage_dir), dir_mode="zip", quiet=False)
    except Exception as e:
        print(f"dataset_create_new note: {e}; trying version...")
        api.dataset_create_version(str(stage_dir), version_notes="update", dir_mode="zip", quiet=False)

    shutil.rmtree(stage_dir, ignore_errors=True)
    print(f"Done! Dataset {owner}/{args.ds_slug} uploaded and local stage cleaned.")


if __name__ == "__main__":
    main()
