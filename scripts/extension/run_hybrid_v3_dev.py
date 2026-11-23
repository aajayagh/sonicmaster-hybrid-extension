"""Run Hybrid-v3 ablation (V3-A, V3-B, V3-C) on 280 Dev samples."""
from __future__ import annotations

import argparse
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
from scripts.extension.diff_drc_joint import fit_joint_drc  # noqa: E402
from scripts.extension.mask import compute_separator_mask  # noqa: E402
from scripts.extension.mix import mix_complementary_stft  # noqa: E402
from scripts.extension.xband_crossover import apply_xband_crossover  # noqa: E402


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


def _apply_peq_branch(x: np.ndarray, s: np.ndarray, sr: int, eq_phase: str) -> np.ndarray:
    y, _ = fit_biquad_eq(x, s, sr=sr, eq_clamp_db=18.0, eq_phase=eq_phase)
    y, _ = match_dynamics(y, s, sr=sr)
    y, _ = match_stereo(y, s)
    y, _ = match_volume_lufs(y, s, sr=sr)
    return y


def _process_cpu_one(
    r: dict[str, str],
    mode: str,
    eq_phase: str,
    stems_dir_str: str,
    out_dir_str: str,
    use_joint: bool = True,
) -> str:
    sid = r["sample_id"]
    out_path = Path(out_dir_str) / f"{sid}.wav"
    if out_path.exists():
        return "SKIP"

    degs = {d.strip() for d in r["degradation_types"].split("|") if d.strip()}
    n_degs = int(r["n_degradations"])
    s_fallback = {"real", "clip", "small", "big", "mix"}
    if mode in ("v3a", "v3b"):
        s_fallback = s_fallback | {"xband"}

    s, sr = load_audio(res(r["path_s"]))
    if degs & s_fallback:
        sf.write(str(out_path), np.clip(s, -1.0, 1.0), sr, subtype="PCM_16")
        return "S"

    x, _ = load_audio(res(r["path_x"]))
    n_xs = min(len(x), len(s))
    x, s = x[:n_xs], s[:n_xs]
    rx = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    rs = float(np.sqrt(np.mean(s.astype(np.float64) ** 2)))
    if 20.0 * np.log10((rs + 1e-9) / (rx + 1e-9)) > 24.0:
        sf.write(str(out_path), np.clip(s, -1.0, 1.0), sr, subtype="PCM_16")
        return "S_guard"

    if mode == "v3c" and "xband" in degs:
        y, _ = apply_xband_crossover(x, s, sr=sr, eq_clamp_db=18.0, eq_phase=eq_phase)
        sf.write(str(out_path), np.clip(y, -1.0, 1.0), sr, subtype="PCM_16")
        return "xband_crossover"

    if "vocal" in degs:
        stems_dir = Path(stems_dir_str)
        v_stem, _ = load_audio(stems_dir / f"{sid}_voc.flac")
        a_stem, _ = load_audio(stems_dir / f"{sid}_acc.flac")
        p_out = _apply_peq_branch(x, s, sr, eq_phase)
        n = min(len(p_out), len(s), len(v_stem), len(a_stem))
        mask = compute_separator_mask(v_stem[:n], a_stem[:n], sr=sr, clamp_min=0.2, clamp_max=0.8)
        y, _ = mix_complementary_stft(p_out[:n], s[:n], mask, sr=sr)
        sf.write(str(out_path), np.clip(y, -1.0, 1.0), sr, subtype="PCM_16")
        return "vocal_mask_P"

    # Defer joint DRC samples to GPU pass if mode in (v3b, v3c) and use_joint
    if use_joint and mode in ("v3b", "v3c") and ((degs & {"comp", "punch", "stereo"}) or n_degs >= 2):
        return "DEFER_GPU"

    y = _apply_peq_branch(x, s, sr, eq_phase)
    sf.write(str(out_path), np.clip(y, -1.0, 1.0), sr, subtype="PCM_16")
    return "peq_P"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["v3a", "v3b", "v3c"], required=True)
    ap.add_argument("--eq-phase", choices=["minphase", "zerophase"], default="minphase")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-joint", action="store_true")
    args = ap.parse_args()

    splits = json.loads(res("artifacts/10_extension/splits.json").read_text(encoding="utf-8"))
    dev_sources = set(splits["dev_sources"])
    rows = []
    with res("artifacts/10_extension/inputs_manifest.csv").open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["source_id"] in dev_sources:
                rows.append(r)

    stems_dir = GDRIVE / "artifacts/10_extension/kaggle_kernel_p1_out/out/dev_stems_flac"
    suffix = "_nojoint" if args.no_joint else ""
    run_dir = ROOT / f"artifacts/10_extension/dev/hybrid_v3_{args.mode}_{args.eq_phase}{suffix}"
    out_dir = run_dir / "out_wav"
    out_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    deferred: list[dict[str, str]] = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(_process_cpu_one, r, args.mode, args.eq_phase, str(stems_dir), str(out_dir), not args.no_joint): r
            for r in rows
        }
        for fut in as_completed(futs):
            r = futs[fut]
            rt = fut.result()
            if rt == "DEFER_GPU":
                deferred.append(r)
            else:
                counts[rt] = counts.get(rt, 0) + 1

    print(f"CPU pass done in {time.time()-t0:.1f}s: counts={counts}, deferred_gpu={len(deferred)}", flush=True)

    if deferred:
        t1 = time.time()
        for i, r in enumerate(deferred, 1):
            sid = r["sample_id"]
            x, sr = load_audio(res(r["path_x"]))
            s, _ = load_audio(res(r["path_s"]))
            n_xs = min(len(x), len(s))
            y, _ = fit_joint_drc(x[:n_xs], s[:n_xs], sr=sr, eq_clamp_db=18.0, eq_phase=args.eq_phase)
            sf.write(str(out_dir / f"{sid}.wav"), np.clip(y, -1.0, 1.0), sr, subtype="PCM_16")
            counts["joint_drc_P"] = counts.get("joint_drc_P", 0) + 1
            if i % 10 == 0 or i == len(deferred):
                print(f"GPU joint [{i}/{len(deferred)}] elapsed={time.time()-t1:.1f}s", flush=True)

    print(f"DONE ({args.mode}, {args.eq_phase}) total={time.time()-t0:.1f}s: {counts}")


if __name__ == "__main__":
    main()
