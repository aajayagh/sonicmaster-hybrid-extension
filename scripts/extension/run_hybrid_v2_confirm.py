"""Run Hybrid-v2 PA1 on Confirm (6,720 samples) using ProcessPoolExecutor."""
from __future__ import annotations

import csv
import io
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(r"d:\SonicMaster_Reproducing")
GDRIVE_ROOT = Path(r"G:\My Drive\SonicMaster_Reproducing")
sys.path.insert(0, str(ROOT))

from scripts.extension.branch_p import apply_branch_p  # noqa: E402
from scripts.extension.mask import compute_separator_mask  # noqa: E402
from scripts.extension.mix import mix_complementary_stft  # noqa: E402

S_FALLBACK = {"real", "clip", "xband", "small", "big", "mix", "mud"}


def resolve_path(rel_or_abs: str | Path) -> Path:
    p = Path(rel_or_abs)
    if p.is_absolute():
        if p.is_file():
            return p
        try:
            rel = p.relative_to(ROOT)
            gp = GDRIVE_ROOT / rel
            if gp.is_file():
                return gp
        except ValueError:
            pass
        return p
    dp = ROOT / p
    if dp.is_file():
        return dp
    gp = GDRIVE_ROOT / p
    if gp.is_file():
        return gp
    return dp


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    for attempt in range(3):
        try:
            data = path.read_bytes()
            audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
            return audio, sr
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1.0)
    raise RuntimeError(f"Failed to load {path}")


def process_one(r: dict[str, str], stems_dir_str: str, out_dir_str: str) -> str:
    sid = r["sample_id"]
    out_path = Path(out_dir_str) / f"{sid}.wav"
    if out_path.exists():
        return "SKIP"

    degs = {d.strip() for d in r["degradation_types"].split("|") if d.strip()}
    s_path = resolve_path(r["path_s"])
    s, sr = load_audio(s_path)

    if (degs & S_FALLBACK) or not r["path_x"]:
        route = "S"
        y = s
    else:
        x_path = resolve_path(r["path_x"])
        x, _ = load_audio(x_path)
        n_xs = min(len(x), len(s))
        rx = float(np.sqrt(np.mean(x[:n_xs].astype(np.float64) ** 2)))
        rs = float(np.sqrt(np.mean(s[:n_xs].astype(np.float64) ** 2)))
        gain_db = 20.0 * np.log10((rs + 1e-9) / (rx + 1e-9))
        if gain_db > 24.0:
            route = "S"
            y = s
        elif "vocal" in degs:
            stems_dir = Path(stems_dir_str)
            voc_p = stems_dir / f"{sid}_voc.flac"
            acc_p = stems_dir / f"{sid}_acc.flac"
            if not voc_p.exists() or not acc_p.exists():
                return "WAIT_STEMS"
            route = "vocal_mask_P"
            p_out, _ = apply_branch_p(x[:n_xs], s[:n_xs], sr=sr)
            v_stem, _ = load_audio(voc_p)
            a_stem, _ = load_audio(acc_p)
            n = min(len(p_out), len(s), len(v_stem), len(a_stem))
            mask = compute_separator_mask(v_stem[:n], a_stem[:n], sr=sr, clamp_min=0.2, clamp_max=0.8)
            y, _ = mix_complementary_stft(p_out[:n], s[:n], mask, sr=sr)
        else:
            route = "full_mix_P"
            y, _ = apply_branch_p(x[:n_xs], s[:n_xs], sr=sr)

    y = np.clip(y, -1.0, 1.0)
    sf.write(str(out_path), y, sr, subtype="PCM_16")
    return route


def main() -> None:
    splits = json.loads(resolve_path("artifacts/10_extension/splits.json").read_text(encoding="utf-8"))
    confirm_sources = set(splits["confirm_sources"])

    rows = []
    with resolve_path("artifacts/10_extension/inputs_manifest.csv").open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["source_id"] in confirm_sources:
                rows.append(r)

    stems_dir = ROOT / "artifacts/10_extension/confirm/hybrid_v2/vocal_stems_flac"
    out_dir = ROOT / "artifacts/10_extension/confirm/hybrid_v2/out_wav"
    out_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(process_one, r, str(stems_dir), str(out_dir)) for r in rows]
        for i, fut in enumerate(as_completed(futs), 1):
            rt = fut.result()
            counts[rt] = counts.get(rt, 0) + 1
            if i % 500 == 0 or i == len(rows):
                print(f"[{i}/{len(rows)}] elapsed={time.time()-t0:.1f}s counts={counts}", flush=True)

    print("DONE:", counts)


if __name__ == "__main__":
    main()
