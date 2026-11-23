"""Lock Phase 3 parameters, script hashes, and 20 Confirm Listen sources into lock.json."""
from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXT_DIR = ROOT / "artifacts" / "10_extension"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while b := f.read(1024 * 1024):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    splits_path = EXT_DIR / "splits.json"
    splits = json.loads(splits_path.read_text(encoding="utf-8"))

    # Select 20 vocal sources from Confirm (seed=20261119) for Phase 5 blind listening
    with (EXT_DIR / "inputs_manifest.csv").open("r", encoding="utf-8") as f:
        confirm_vocal_srcs = sorted({
            r["source_id"] for r in csv.DictReader(f)
            if r["split_ext"] == "confirm" and r["vocalinstrumental"] == "vocal"
        })

    listen_sources = sorted(random.Random(20261119).sample(confirm_vocal_srcs, 20))
    splits["listen_sources"] = listen_sources
    splits["listen_status"] = "locked_for_phase5_unopened"
    splits_path.write_text(json.dumps(splits, indent=2), encoding="utf-8")

    script_files = [
        "scripts/extension/routing.py",
        "scripts/extension/branch_r.py",
        "scripts/extension/branch_p.py",
        "scripts/extension/mask.py",
        "scripts/extension/mix.py",
        "scripts/extension/run_pipeline.py",
    ]
    script_hashes = {p: sha256_file(ROOT / p) for p in script_files}

    lock = {
        "locked_at": "2026-11-20",
        "selected_variant": "A5",
        "separator": {
            "name": "bs_roformer_1297",
            "model_filename": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
            "sha256": "5b84f37e8d444c8cb30c79d77f613a41c05868ff9c9ac6c7049c00aefae115aa",
        },
        "params": {
            "clamp_min": 0.2,
            "clamp_max": 0.8,
            "smooth_ms": 50.0,
            "eq_clamp_db": 12.0,
            "stft_n_fft": 2048,
            "stft_hop": 512,
        },
        "routing": {
            "P": ["comp", "punch", "stereo", "volume"],
            "R": ["airy", "big", "boom", "bright", "clarity", "dark", "mic", "mix", "mud", "noise", "real", "small", "vocal", "warm"],
            "S": ["clip", "xband"],
        },
        "script_sha256": script_hashes,
        "splits_sha256": sha256_file(splits_path),
        "inputs_manifest_sha256": sha256_file(EXT_DIR / "inputs_manifest.csv"),
        "source_lock_extension_sha256": sha256_file(EXT_DIR / "source-lock-extension.json"),
        "dev_gate_passed": True,
        "confirm_thresholds": {
            "delta_ssim_min": 0.01,
            "delta_pq_ci95_lo_min": -0.05,
            "fad_ratio_max": 1.10,
            "kl_ratio_max": 1.10,
            "attr_norm_det_max": 0.05,
            "listening_win_rate_min": 0.65,
            "listening_p_max": 0.05,
        },
    }
    lock_path = EXT_DIR / "lock.json"
    lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    print("Locked Phase 3 ->", lock_path)
    print("Listen sources (20):", listen_sources)


if __name__ == "__main__":
    main()
