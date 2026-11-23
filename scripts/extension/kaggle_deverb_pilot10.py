"""Upload the 4 reverb Dev clips (21 MB) to Kaggle T4 GPU and run deverb_bs_roformer_8_384dim_10depth.ckpt."""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = Path(__file__).resolve().parents[2]
EXT_DIR = ROOT / "artifacts" / "10_extension"
KEY_PATH = EXT_DIR / "dev" / "listening_check" / "key_DO_NOT_OPEN_BEFORE_LISTENING.json"


def get_api(line_idx: int = 0) -> tuple[KaggleApi, str]:
    lines = [l.strip() for l in Path("C:/Users/ADMIN/.kaggle/access_token.txt").read_text().splitlines() if l.strip()]
    os.environ["KAGGLE_API_TOKEN"] = lines[line_idx]
    api = KaggleApi()
    api.authenticate()
    return api, str(api.config_values["username"])


def main() -> None:
    api, owner = get_api(0)
    ds_slug = "sonicmaster-ext-deverb-pilot4"
    k_slug = "sonicmaster-ext-deverb-pilot4-run"

    keys = json.loads(KEY_PATH.read_text(encoding="utf-8"))
    reverb_clips = [k for k in keys if k["clip_num"] in {6, 7, 9, 10}]

    import csv
    manifest = {r["sample_id"]: r for r in csv.DictReader((EXT_DIR / "inputs_manifest.csv").open(encoding="utf-8"))}

    stage_dir = EXT_DIR / "kaggle_deverb4_stage"
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    audio_dir = stage_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    sids = []
    for k in reverb_clips:
        sid = k["sample_id"]
        sids.append(sid)
        shutil.copy2(ROOT / manifest[sid]["path_x"], audio_dir / f"{sid}_x.wav")

    (stage_dir / "sids.json").write_text(json.dumps(sids), encoding="utf-8")
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
    # Wait until dataset is ready
    for _ in range(20):
        st = api.dataset_status(f"{owner}/{ds_slug}")
        print("dataset status:", st, flush=True)
        if str(st).lower() == "ready":
            break
        time.sleep(5)

    k_dir = EXT_DIR / "kaggle_kernel_deverb4"
    k_dir.mkdir(parents=True, exist_ok=True)
    script = '''"""Kaggle T4 GPU: Run deverb_bs_roformer_8_384dim_10depth.ckpt on 4 reverb Dev clips."""
import json, subprocess, sys, time
from pathlib import Path
import numpy as np
import soundfile as sf

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "audio-separator[gpu]"])
from audio_separator.separator import Separator

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
OUT = Path("/kaggle/working/out")
OUT.mkdir(parents=True, exist_ok=True)
MODEL_DIR = Path("/kaggle/working/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
WORK_DIR = Path("/kaggle/working/tmp_work")
WORK_DIR.mkdir(parents=True, exist_ok=True)

sep = Separator(model_file_dir=str(MODEL_DIR), output_dir=str(WORK_DIR), output_format="WAV", use_autocast=True)
sep.load_model("deverb_bs_roformer_8_384dim_10depth.ckpt")

for sid in sids:
    x_arr, sr = sf.read(DATA_DIR / "audio" / f"{sid}_x.wav", dtype="float32", always_2d=True)
    x_tmp = WORK_DIR / f"{sid}_x.wav"
    sf.write(x_tmp, x_arr[:1320960], sr)
    files = sep.separate(str(x_tmp))
    print(sid, "outputs:", files, flush=True)
    norev = None
    for f in files:
        p = WORK_DIR / Path(f).name
        if not p.exists():
            p = Path(f)
        if "noreverb" in p.name.lower() or "dry" in p.name.lower():
            norev = p
    if norev is None:
        norev = WORK_DIR / Path(files[0]).name
    y, _ = sf.read(norev, dtype="float32", always_2d=True)
    sf.write(OUT / f"{sid}_deverb.wav", y[:1320960], sr, subtype="PCM_16")
    for f in files:
        p = WORK_DIR / Path(f).name
        if p.exists():
            p.unlink()
    x_tmp.unlink(missing_ok=True)

print("ALL 4 DEVERB DONE!")
'''
    (k_dir / "run_deverb4.py").write_text(script, encoding="utf-8")
    meta = {
        "id": f"{owner}/{k_slug}",
        "title": k_slug,
        "code_file": "run_deverb4.py",
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
    api.kernels_push(str(k_dir))
    print(f"Pushed! {owner}/{k_slug}")


if __name__ == "__main__":
    main()
