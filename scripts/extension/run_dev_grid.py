"""Phase 3 Dev grid search on A5 (clamp and smooth_ms) with automatic disk cleanup."""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXT_PY = str(ROOT / "tools" / "extension-env" / "Scripts" / "python.exe")
EVAL_PY = str(ROOT / "tools" / "eval-env" / "Scripts" / "python.exe")
STEMS_DIR = ROOT / "artifacts" / "10_extension" / "kaggle_kernel_p1_out" / "out" / "dev_stems_flac"
GRID_DIR = ROOT / "artifacts" / "10_extension" / "dev" / "grid_runs"

GRID = [
    (0.1, 0.9, 20.0),
    (0.1, 0.9, 50.0),
    (0.1, 0.9, 100.0),
    (0.2, 0.8, 20.0),
    (0.2, 0.8, 50.0),
    (0.2, 0.8, 100.0),
    (0.0, 1.0, 50.0),
]


def main() -> None:
    GRID_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    # Load A0 baseline
    a0_sum = list(csv.DictReader((ROOT / "artifacts" / "10_extension" / "dev" / "pilot_p" / "summary.csv").open()))
    a0_all = next(r for r in a0_sum if r["variant"] == "A0" and r["group"] == "All")
    a0_ssim = float(a0_all["ssim_mean"])
    a0_kl = float(a0_all["kl_x1000_mean"])

    for cmin, cmax, sm in GRID:
        tag = f"c{int(cmin*10)}_{int(cmax*10)}_s{int(sm)}"
        out_dir = GRID_DIR / tag
        print(f"=== Grid {tag} ===", flush=True)

        # Reuse pilot_p_v4/A5 if cmin=0.1, cmax=0.9, sm=50
        if (cmin, cmax, sm) == (0.1, 0.9, 50.0) and (ROOT / "artifacts" / "10_extension" / "dev" / "pilot_p_v4" / "attributes_A5.json").exists():
            sum_csv = ROOT / "artifacts" / "10_extension" / "dev" / "pilot_p_v4" / "summary.csv"
            attr_json = ROOT / "artifacts" / "10_extension" / "dev" / "pilot_p_v4" / "attributes_A5.json"
        else:
            subprocess.check_call([
                EXT_PY, str(ROOT / "scripts" / "extension" / "run_pipeline.py"),
                "--split", "dev", "--variants", "A5",
                "--separator-dir", str(STEMS_DIR),
                "--clamp-min", str(cmin), "--clamp-max", str(cmax), "--smooth-ms", str(sm),
                "--out", str(out_dir),
            ])
            subprocess.check_call([
                EXT_PY, str(ROOT / "scripts" / "extension" / "eval_dev.py"),
                "--run-dir", str(out_dir), "--variants", "A5",
            ])
            attr_json = out_dir / "attributes_A5.json"
            subprocess.check_call([
                EVAL_PY, str(ROOT / "scripts" / "extension" / "eval_tables123.py"),
                "--cand-dir", str(out_dir / "A5"), "--cand-name", f"A5_{tag}",
                "--out-json", str(attr_json),
            ])
            sum_csv = out_dir / "summary.csv"
            # Clean up WAVs to preserve disk space
            shutil.rmtree(out_dir / "A5", ignore_errors=True)

        s_rows = list(csv.DictReader(sum_csv.open()))
        s_all = next(r for r in s_rows if r["variant"] == "A5" and r["group"] == "All")
        ssim_v = float(s_all["ssim_mean"])
        kl_v = float(s_all["kl_x1000_mean"])
        attr = json.loads(attr_json.read_text())

        rows.append({
            "tag": tag,
            "clamp_min": cmin,
            "clamp_max": cmax,
            "smooth_ms": sm,
            "ssim": ssim_v,
            "delta_ssim": ssim_v - a0_ssim,
            "kl_x1000": kl_v,
            "kl_ratio_vs_a0": kl_v / a0_kl,
            "attr_violations": "|".join(attr["violations"]),
            "n_violations": len(attr["violations"]),
        })

    grid_csv = ROOT / "artifacts" / "10_extension" / "dev" / "grid.csv"
    with grid_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
