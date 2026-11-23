"""Run Phase E (V3-Blind-E1-tight: guard_s<0.5 & d_kl<=1.0 & d_mod<=0.08) on 6,720 Confirm samples."""
from __future__ import annotations

import csv
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

from scripts.extension.blind_router import compute_blind_candidate, load_audio, res  # noqa: E402


def process_one(r: dict[str, str], out_dir_str: str) -> dict[str, object]:
    sid = r["sample_id"]
    out_path = Path(out_dir_str) / f"{sid}.wav"
    if not r["path_x"]:
        return {
            "sample_id": sid,
            "source_id": r["source_id"],
            "n_degradations": int(r["n_degradations"]),
            "degradation_types": r["degradation_types"],
            "kept": 0,
            "guard_s": 1.0,
            "dead_frac": 0.0,
            "d_kl": 0.0,
            "d_mod": 0.0,
            "d_flat": 0.0,
        }

    x_path = res(r["path_x"])
    s_path = res(r["path_s"])
    if not x_path.is_file() or not s_path.is_file():
        return {
            "sample_id": sid,
            "source_id": r["source_id"],
            "n_degradations": int(r["n_degradations"]),
            "degradation_types": r["degradation_types"],
            "kept": 0,
            "guard_s": 1.0,
            "dead_frac": 0.0,
            "d_kl": 0.0,
            "d_mod": 0.0,
            "d_flat": 0.0,
        }

    x, sr = load_audio(x_path)
    s, _ = load_audio(s_path)
    y_cand, diag = compute_blind_candidate(x, s, sr=sr)
    kept = 1 if (diag["guard_s"] < 0.5 and diag["d_kl"] <= 1.0 and diag["d_mod"] <= 0.08) else 0
    if kept:
        sf.write(str(out_path), np.clip(y_cand, -1.0, 1.0), sr, subtype="PCM_16")
    return {
        "sample_id": sid,
        "source_id": r["source_id"],
        "n_degradations": int(r["n_degradations"]),
        "degradation_types": r["degradation_types"],
        "kept": kept,
        **diag,
    }


def main() -> None:
    splits = json.loads(res("artifacts/10_extension/splits.json").read_text(encoding="utf-8"))
    confirm_sources = set(splits["confirm_sources"])
    rows = [
        r
        for r in csv.DictReader(res("artifacts/10_extension/inputs_manifest.csv").open("r", encoding="utf-8"))
        if r["source_id"] in confirm_sources
    ]

    base_dir = ROOT / "artifacts" / "10_extension" / "confirm" / "hybrid_v3_blind"
    out_dir = base_dir / "out_wav"
    out_dir.mkdir(parents=True, exist_ok=True)
    diag_csv = base_dir / "blind_confirm_diagnostics.csv"

    print(f"Running V3-Blind-E1-tight on {len(rows)} Confirm samples...", flush=True)
    t0 = time.time()
    results = []
    kept_total = 0
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(process_one, r, str(out_dir)) for r in rows]
        for i, fut in enumerate(as_completed(futs), 1):
            it = fut.result()
            results.append(it)
            kept_total += int(it["kept"])
            if i % 500 == 0 or i == len(rows):
                print(f"[{i}/{len(rows)}] elapsed={time.time()-t0:.1f}s kept={kept_total}", flush=True)

    results.sort(key=lambda d: str(d["sample_id"]))
    fieldnames = ["sample_id", "source_id", "n_degradations", "degradation_types", "kept", "guard_s", "dead_frac", "d_kl", "d_mod", "d_flat"]
    with diag_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)
    print(f"Saved {diag_csv} (kept {kept_total}/{len(rows)}) in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
