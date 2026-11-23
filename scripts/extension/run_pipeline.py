"""Run Codec-Bypass Hybrid pipeline (A0-A6) for 10_extension.

IMPORTANT: This script NEVER reads clean ground-truth audio.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.extension.branch_p import apply_branch_p
from scripts.extension.branch_r import LagError, apply_branch_r, check_zero_lag
from scripts.extension.mask import compute_band_mask, compute_separator_mask
from scripts.extension.mix import mix_complementary_stft
from scripts.extension.routing import route_degradations


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def write_wav_atomic(path: Path, y: np.ndarray, sr: int = 44100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part.wav")
    sf.write(tmp, np.clip(y, -1.0, 1.0).astype(np.float32), sr, format="WAV", subtype="PCM_16")
    os.replace(tmp, path)


def load_stereo(path: Path) -> tuple[np.ndarray, int]:
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    if y.shape[1] == 1:
        y = np.repeat(y, 2, axis=1)
    return y, sr


def process_sample(
    row: dict[str, str],
    variant: str,
    separator_dir: Path | None,
    eq_clamp_db: float,
    smooth_ms: float,
    clamp_min: float,
    clamp_max: float,
) -> tuple[np.ndarray, dict[str, object]]:
    """Compute output audio for one (sample_id, variant). Never accesses clean."""
    s_path = ROOT / row["path_s"]
    s, sr = load_stereo(s_path)
    meta: dict[str, object] = {
        "sample_id": row["sample_id"],
        "variant": variant,
        "branch_mode": variant,
        "lag_xs": 0,
        "lag_xr": 0,
        "phase_corr_xs": 1.0,
        "eq_max_abs_db": 0.0,
        "dynamics_mode": "none",
        "stereo_k": 1.0,
        "final_gain_db": 0.0,
        "final_limited": False,
    }

    if variant == "A0":
        return s, meta

    x_path = ROOT / row["path_x"]
    x, sr_x = load_stereo(x_path)
    if sr_x != sr:
        raise ValueError(f"Sample rate mismatch: {sr_x} vs {sr}")
    if x.shape[0] > s.shape[0] and s.shape[0] == (x.shape[0] // 2048) * 2048:
        x = x[: s.shape[0]]

    if variant == "A1":
        lag_info = check_zero_lag(x, s, strict=False)
        meta.update(lag_info)
        out = (0.9 * s + 0.1 * x).astype(np.float32)
        return out, meta

    r_path_str = row.get("path_r", "")
    r: np.ndarray | None = None
    if variant in {"A2", "A4", "A5", "A6"}:
        if not r_path_str:
            raise FileNotFoundError(f"Missing path_r for {row['sample_id']}")
        r, sr_r = load_stereo(ROOT / r_path_str)
        if sr_r != sr:
            raise ValueError(f"Sample rate mismatch r: {sr_r} vs {sr}")
        lag_info = check_zero_lag(x, s, r, strict=False)
    else:
        lag_info = check_zero_lag(x, s, strict=False)
    meta.update(lag_info)

    if float(lag_info["phase_corr_xs"]) < 0.0:
        meta["branch_mode"] = "fallback_s_neg_corr"
        return s, meta

    if variant == "A2":
        assert r is not None
        v_r, lim = apply_branch_r(x, s, r)
        meta["final_limited"] = lim
        return v_r, meta

    if variant == "A3":
        v_p, p_info = apply_branch_p(x, s, sr=sr, eq_clamp_db=eq_clamp_db)
        meta.update(p_info)
        return v_p, meta

    # A4, A5, A6 use degradation routing
    deg_types = [t for t in row["degradation_types"].split("|") if t]
    route = route_degradations(deg_types)
    meta["branch_mode"] = route.mode
    if route.mode == "fallback_s":
        raise ValueError(f"Unknown degradation types: {route.unknown_types}")
    if route.mode == "S":
        return s, meta

    if route.mode == "R":
        assert r is not None
        v, lim = apply_branch_r(x, s, r)
        meta["final_limited"] = lim
    elif route.mode == "P":
        v, p_info = apply_branch_p(x, s, sr=sr, eq_clamp_db=eq_clamp_db)
        meta.update(p_info)
    elif route.mode == "R_then_P":
        assert r is not None
        v_r, lim = apply_branch_r(x, s, r)
        v, p_info = apply_branch_p(v_r, s, sr=sr, eq_clamp_db=eq_clamp_db)
        meta.update(p_info)
        meta["final_limited"] = lim
    else:
        raise ValueError(f"Unexpected route mode: {route.mode}")

    if variant == "A4":
        return v, meta

    if variant == "A5":
        if separator_dir is None:
            raise ValueError("--separator-dir is required for A5")
        sid = row["sample_id"]
        voc_path = separator_dir / f"{sid}_voc.wav"
        acc_path = separator_dir / f"{sid}_acc.wav"
        if not voc_path.exists():
            voc_path = separator_dir / f"{sid}_voc.flac"
            acc_path = separator_dir / f"{sid}_acc.flac"
        if not voc_path.exists() or not acc_path.exists():
            raise FileNotFoundError(f"Missing separator stems for {sid} in {separator_dir}")
        voc, _ = load_stereo(voc_path)
        acc, _ = load_stereo(acc_path)
        if voc.shape != s.shape:
            voc = voc[: s.shape[0]]
            acc = acc[: s.shape[0]]
        mask = compute_separator_mask(
            voc, acc, sr=sr, smooth_ms=smooth_ms, clamp_min=clamp_min, clamp_max=clamp_max
        )
        out, mix_info = mix_complementary_stft(v, s, mask, sr=sr)
        meta.update(mix_info)
        return out, meta

    if variant == "A6":
        mask = compute_band_mask(
            s.shape[0], n_channels=s.shape[1], sr=sr, clamp_min=clamp_min, clamp_max=clamp_max
        )
        out, mix_info = mix_complementary_stft(v, s, mask, sr=sr)
        meta.update(mix_info)
        return out, meta

    raise ValueError(f"Unknown variant: {variant}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["dev", "confirm"])
    ap.add_argument("--variants", required=True, help="Comma-separated, e.g. A0,A1,A5")
    ap.add_argument("--manifest", default=str(ROOT / "artifacts" / "10_extension" / "inputs_manifest.csv"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--separator-dir", default=None)
    ap.add_argument("--eq-clamp-db", type=float, default=12.0)
    ap.add_argument("--smooth-ms", type=float, default=50.0)
    ap.add_argument("--clamp-min", type=float, default=0.1)
    ap.add_argument("--clamp-max", type=float, default=0.9)
    ap.add_argument("--limit", type=int, default=0, help="Optional max samples for pilot")
    args = ap.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    sep_dir = Path(args.separator_dir) if args.separator_dir else None

    # Save environment lock
    try:
        pip_freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        (out_dir / "environment-lock.txt").write_text(pip_freeze, encoding="utf-8")
    except Exception:
        pass

    with Path(args.manifest).open("r", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split_ext"] == args.split]

    if args.limit > 0:
        rows = rows[: args.limit]

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    out_manifest_path = out_dir / "outputs_manifest.csv"
    failures_path = out_dir / "failures.csv"

    existing: set[tuple[str, str]] = set()
    if out_manifest_path.exists():
        with out_manifest_path.open("r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                p = ROOT / r["path"]
                if p.exists():
                    existing.add((r["sample_id"], r["variant"]))

    mf_fields = [
        "sample_id", "variant", "path", "sha256", "n_samples", "branch_mode",
        "lag_xs", "lag_xr", "phase_corr_xs", "eq_max_abs_db", "dynamics_mode",
        "stereo_k", "final_gain_db", "final_limited",
    ]
    fail_fields = ["sample_id", "variant", "error_type", "error_msg"]

    mf_new = not out_manifest_path.exists()
    fail_new = not failures_path.exists()

    with out_manifest_path.open("a", newline="", encoding="utf-8") as mf, \
         failures_path.open("a", newline="", encoding="utf-8") as ff:
        mw = csv.DictWriter(mf, fieldnames=mf_fields)
        fw = csv.DictWriter(ff, fieldnames=fail_fields)
        if mf_new:
            mw.writeheader()
        if fail_new:
            fw.writeheader()

        for var in variants:
            for row in rows:
                sid = row["sample_id"]
                if (sid, var) in existing:
                    continue
                wav_path = out_dir / var / f"{sid}.wav"
                try:
                    y, meta = process_sample(
                        row, var, sep_dir, args.eq_clamp_db, args.smooth_ms, args.clamp_min, args.clamp_max
                    )
                    write_wav_atomic(wav_path, y, sr=44100)
                    mw.writerow({
                        "sample_id": sid,
                        "variant": var,
                        "path": str(wav_path.relative_to(ROOT)).replace("\\", "/"),
                        "sha256": sha256_file(wav_path),
                        "n_samples": y.shape[0],
                        "branch_mode": meta["branch_mode"],
                        "lag_xs": meta["lag_xs"],
                        "lag_xr": meta["lag_xr"],
                        "phase_corr_xs": meta["phase_corr_xs"],
                        "eq_max_abs_db": meta["eq_max_abs_db"],
                        "dynamics_mode": meta["dynamics_mode"],
                        "stereo_k": meta["stereo_k"],
                        "final_gain_db": meta["final_gain_db"],
                        "final_limited": meta["final_limited"],
                    })
                    mf.flush()
                except Exception as exc:
                    fw.writerow({
                        "sample_id": sid,
                        "variant": var,
                        "error_type": type(exc).__name__,
                        "error_msg": str(exc),
                    })
                    ff.flush()

    print(f"Done: split={args.split}, variants={variants}, out={out_dir}")


if __name__ == "__main__":
    main()
