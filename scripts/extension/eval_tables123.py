"""Evaluate Table 1-2 attributes (and optional PQ/FAD) on Dev runs using official SonicMaster code."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "SonicMaster" / "evaluation"))
import evaluate_control_multiple_degs_mass as attr_eval  # noqa: E402


def load_pair(path: Path, frames: int = 1320960) -> tuple[np.ndarray, np.ndarray]:
    import io, time
    if not path.exists():
        try:
            rel = path.relative_to(ROOT)
            gp = Path("G:/My Drive/SonicMaster_Reproducing") / rel
            if gp.exists():
                path = gp
        except ValueError:
            pass
    for attempt in range(3):
        try:
            st, sr = sf.read(io.BytesIO(path.read_bytes()), dtype="float32", always_2d=True)
            break
        except OSError:
            if attempt == 2:
                raise
            time.sleep(1.0)
    assert sr == 44100
    if st.shape[1] == 1:
        st = np.repeat(st, 2, axis=1)
    st = st[:frames]
    mono = np.mean(st, axis=1)
    return mono, st


def eval_attributes_for_variant(
    dev_rows: list[dict[str, str]],
    wav_map: dict[str, Path],
    clean_cache: dict[str, tuple[np.ndarray, np.ndarray]],
    x_cache: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, list[float]]:
    attr_eval.metrics_by_degradation.clear()
    by_attr: dict[str, list[float]] = defaultdict(list)

    for r in dev_rows:
        sid = r["sample_id"]
        src = r["source_id"]
        p = wav_map.get(sid)
        if p is None or not p.exists():
            continue
        degs = [d for d in r["degradation_types"].split("|") if d]
        c_m, c_s = clean_cache[src]
        x_m, x_s = x_cache[sid]
        o_m, o_s = load_pair(p)

        res = attr_eval.evaluate_sample(c_m, x_m, o_m, degs, c_s, x_s, o_s)
        for deg, metrics in res.items():
            v = metrics.get("abs_error_after", metrics.get("cosine_abs_error_after"))
            if v is not None and np.isfinite(v):
                by_attr[deg].append(float(v))
    return by_attr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "artifacts" / "10_extension" / "inputs_manifest.csv"))
    ap.add_argument("--a0-dir", default=str(ROOT / "artifacts" / "10_extension" / "dev" / "pilot_p" / "A0"))
    ap.add_argument("--cand-dir", required=True, help="Directory containing <sample_id>.wav")
    ap.add_argument("--cand-name", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    with Path(args.manifest).open("r", encoding="utf-8") as f:
        dev_rows = [r for r in csv.DictReader(f) if r["split_ext"] == "dev"]

    clean_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    x_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for r in dev_rows:
        src = r["source_id"]
        if src not in clean_cache:
            clean_cache[src] = load_pair(ROOT / r["path_clean"])
        x_cache[r["sample_id"]] = load_pair(ROOT / r["path_x"])

    a0_dir = Path(args.a0_dir)
    if not a0_dir.exists():
        gp = Path("G:/My Drive/SonicMaster_Reproducing/artifacts/10_extension/dev/pilot_p/A0")
        if gp.exists():
            a0_dir = gp
    inp_map = {r["sample_id"]: ROOT / r["path_x"] for r in dev_rows}
    a0_map = {r["sample_id"]: a0_dir / f"{r['sample_id']}.wav" for r in dev_rows}
    cand_map = {r["sample_id"]: Path(args.cand_dir) / f"{r['sample_id']}.wav" for r in dev_rows}

    err_inp = eval_attributes_for_variant(dev_rows, inp_map, clean_cache, x_cache)
    err_a0 = eval_attributes_for_variant(dev_rows, a0_map, clean_cache, x_cache)
    err_cand = eval_attributes_for_variant(dev_rows, cand_map, clean_cache, x_cache)

    rows = []
    violations = []
    for deg in sorted(err_a0.keys()):
        n = len(err_a0[deg])
        mi = float(np.mean(err_inp[deg])) if err_inp[deg] else float("nan")
        m0 = float(np.mean(err_a0[deg]))
        mc = float(np.mean(err_cand[deg]))
        norm_det = (mc - m0) / mi if (mi and np.isfinite(mi) and mi > 1e-12) else 0.0
        guard_ok = True if n < 10 else (norm_det <= 0.05)
        if not guard_ok:
            violations.append(deg)
        rows.append({
            "attribute": deg,
            "n": n,
            "input_err": mi,
            "A0_err": m0,
            f"{args.cand_name}_err": mc,
            "norm_deterioration": norm_det,
            "guard_eligible": n >= 10,
            "guard_ok": guard_ok,
        })

    assert len(rows) > 0, "No attributes evaluated! Check a0_dir and cand_dir."
    out = {"cand_name": args.cand_name, "violations": violations, "attributes": rows}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Violations (n>=10, norm_det>0.05): {violations}")
    for r in rows:
        flag = "FAIL" if not r["guard_ok"] else ("low-n" if not r["guard_eligible"] else "ok")
        print(f"  {r['attribute']:10s} n={r['n']:3d} inp={r['input_err']:.4f} A0={r['A0_err']:.4f} {args.cand_name}={r[args.cand_name+'_err']:.4f} det={r['norm_deterioration']:+.3f} [{flag}]")


if __name__ == "__main__":
    main()
