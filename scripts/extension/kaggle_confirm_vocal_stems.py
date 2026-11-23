"""Upload Confirm vocal clips (excluding S-fallback) to Kaggle T4 GPU and run bs_roformer_1297."""
from __future__ import annotations

import csv
import io
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


def get_api(line_idx: int = 0) -> tuple[KaggleApi, str]:
    lines = [l.strip() for l in Path("C:/Users/ADMIN/.kaggle/access_token.txt").read_text().splitlines() if l.strip()]
    os.environ["KAGGLE_API_TOKEN"] = lines[line_idx]
    api = KaggleApi()
    api.authenticate()
    return api, str(api.config_values["username"])


def main() -> None:
    splits = json.loads(res("artifacts/10_extension/splits.json").read_text(encoding="utf-8"))
    confirm_sources = set(splits["confirm_sources"])

    rows = []
    with res("artifacts/10_extension/inputs_manifest.csv").open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["source_id"] not in confirm_sources:
                continue
            degs = {d.strip() for d in r["degradation_types"].split("|") if d.strip()}
            if "vocal" in degs and not (degs & S_FALLBACK) and r["path_x"]:
                rows.append(r)

    print(f"Found {len(rows)} Confirm vocal clips needing bs_roformer_1297", flush=True)

    api, owner = get_api(0)
    ds_slug = "sonicmaster-ext-confirm-vocal"
    k_slug = "sonicmaster-ext-confirm-vocal-run"

    stage_dir = EXT_DIR / "kaggle_confirm_vocal_stage"
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    if "--push-only" not in sys.argv:
        zip_path = stage_dir / "audio.zip"
        sids = []
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
            for r in rows:
                sid = r["sample_id"]
                sids.append(sid)
                x_path = res(r["path_x"])
                zf.writestr(f"{sid}.wav", x_path.read_bytes())

        (stage_dir / "sids.json").write_text(json.dumps(sids), encoding="utf-8")
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
        for _ in range(12):
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

    k_dir = EXT_DIR / "kaggle_kernel_confirm_vocal"
    k_dir.mkdir(parents=True, exist_ok=True)
    script = '''"""Kaggle T4 GPU: Run model_bs_roformer_ep_317_sdr_12.9755.ckpt on Confirm vocal clips."""
import json, subprocess, sys, time, zipfile
from pathlib import Path
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
wav_map = {p.stem: p for p in DATA_DIR.rglob("*.wav")}
if not wav_map and (DATA_DIR / "audio.zip").exists():
    WORK = Path("/kaggle/working/in_wav")
    WORK.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(DATA_DIR / "audio.zip", "r") as zf:
        zf.extractall(WORK)
    wav_map = {p.stem: p for p in WORK.rglob("*.wav")}

OUT = Path("/kaggle/working/out")
OUT.mkdir(parents=True, exist_ok=True)
TMP = Path("/kaggle/working/tmp_sep")
TMP.mkdir(parents=True, exist_ok=True)

sep = Separator(output_dir=str(TMP), output_format="FLAC", sample_rate=44100)
sep.load_model(model_filename="model_bs_roformer_ep_317_sdr_12.9755.ckpt")

t0 = time.time()
for i, sid in enumerate(sids):
    inp = wav_map[sid]
    out_files = sep.separate(str(inp))
    for f in out_files:
        fp = TMP / Path(f).name
        if not fp.exists():
            fp = Path(f)
        low = fp.name.lower()
        if "(vocals)" in low:
            fp.rename(OUT / f"{sid}_voc.flac")
        elif "(instrumental)" in low or "(no vocals)" in low:
            fp.rename(OUT / f"{sid}_acc.flac")
    if (i + 1) % 20 == 0 or (i + 1) == len(sids):
        print(f"[{i+1}/{len(sids)}] elapsed={time.time()-t0:.1f}s", flush=True)
'''
    (k_dir / "run_confirm_vocal.py").write_text(script, encoding="utf-8")
    meta = {
        "id": f"{owner}/{k_slug}",
        "title": k_slug,
        "code_file": "run_confirm_vocal.py",
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
