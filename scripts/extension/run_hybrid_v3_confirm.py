"""Run Hybrid-v3 (locked V3-best) on Confirm (6,720 samples) using ProcessPoolExecutor."""
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
GDRIVE = Path(r"G:\My Drive\SonicMaster_Reproducing")
sys.path.insert(0, str(ROOT))

from scripts.extension.biquad_eq import fit_biquad_eq  # noqa: E402
from scripts.extension.branch_p import match_dynamics, match_stereo, match_volume_lufs  # noqa: E402
from scripts.extension.mask import compute_separator_mask  # noqa: E402
from scripts.extension.mix import mix_complementary_stft  # noqa: E402
from scripts.extension.xband_crossover import apply_xband_crossover  # noqa: E402

S_FALLBACK = {"real", "clip", "small", "big", "mix"}


def res(p: str | Path) -> Path:
    dp = ROOT / p
    return dp if dp.is_file() else GDRIVE / p


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    for attempt in range(3):
        try:
            y, sr = sf.read(io.BytesIO(path.read_bytes()), dtype="float32", always_2d=True)
            if y.shape[1] == 1:
                y = np.repeat(y, 2, axis=1)
            return y, sr
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1.0)
    raise RuntimeError(f"Failed to load {path}")


def _apply_peq_branch(x: np.ndarray, s: np.ndarray, sr: int) -> np.ndarray:
    y, _ = fit_biquad_eq(x, s, sr=sr, eq_clamp_db=18.0, eq_phase="minphase")
    y, _ = match_dynamics(y, s, sr=sr)
    y, _ = match_stereo(y, s)
    y, _ = match_volume_lufs(y, s, sr=sr)
    return y


def process_one(r: dict[str, str], stems_dir_str: str, out_dir_str: str) -> str:
    sid = r["sample_id"]
    out_path = Path(out_dir_str) / f"{sid}.wav"
    if out_path.exists():
        return "SKIP"

    degs = {d.strip() for d in r["degradation_types"].split("|") if d.strip()}
    s, sr = load_audio(res(r["path_s"]))

    if (degs & S_FALLBACK) or not r["path_x"]:
        sf.write(str(out_path), np.clip(s, -1.0, 1.0), sr, subtype="PCM_16")
        return "S"

    x_path = res(r["path_x"])
    if not x_path.is_file():
        sf.write(str(out_path), np.clip(s, -1.0, 1.0), sr, subtype="PCM_16")
        return "S_missing_x"

    x, _ = load_audio(x_path)
    n_xs = min(len(x), len(s))
    x, s = x[:n_xs], s[:n_xs]
    rx = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    rs = float(np.sqrt(np.mean(s.astype(np.float64) ** 2)))
    if 20.0 * np.log10((rs + 1e-9) / (rx + 1e-9)) > 24.0:
        sf.write(str(out_path), np.clip(s, -1.0, 1.0), sr, subtype="PCM_16")
        return "S_guard"

    if "xband" in degs:
        y, _ = apply_xband_crossover(x, s, sr=sr, eq_clamp_db=18.0, eq_phase="minphase")
        sf.write(str(out_path), np.clip(y, -1.0, 1.0), sr, subtype="PCM_16")
        return "xband_crossover"

    if "vocal" in degs:
        stems_dir = Path(stems_dir_str)
        voc_p = stems_dir / f"{sid}_voc.flac"
        acc_p = stems_dir / f"{sid}_acc.flac"
        if not voc_p.is_file() or not acc_p.is_file():
            sf.write(str(out_path), np.clip(s, -1.0, 1.0), sr, subtype="PCM_16")
            return "S_missing_stems"
        v_stem, _ = load_audio(voc_p)
        a_stem, _ = load_audio(acc_p)
        p_out = _apply_peq_branch(x, s, sr)
        n = min(len(p_out), len(s), len(v_stem), len(a_stem))
        mask = compute_separator_mask(v_stem[:n], a_stem[:n], sr=sr, clamp_min=0.2, clamp_max=0.8)
        y, _ = mix_complementary_stft(p_out[:n], s[:n], mask, sr=sr)
        sf.write(str(out_path), np.clip(y, -1.0, 1.0), sr, subtype="PCM_16")
        return "vocal_mask_P"

    y = _apply_peq_branch(x, s, sr)
    sf.write(str(out_path), np.clip(y, -1.0, 1.0), sr, subtype="PCM_16")
    return "peq_P"


def main() -> None:
    splits = json.loads(res("artifacts/10_extension/splits.json").read_text(encoding="utf-8"))
    confirm_sources = set(splits["confirm_sources"])
    rows = [
        r
        for r in csv.DictReader(res("artifacts/10_extension/inputs_manifest.csv").open("r", encoding="utf-8"))
        if r["source_id"] in confirm_sources
    ]

    stems_dir = ROOT / "artifacts/10_extension/confirm/hybrid_v2/vocal_stems_flac"
    out_dir = ROOT / "artifacts/10_extension/confirm/hybrid_v3/out_wav"
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
