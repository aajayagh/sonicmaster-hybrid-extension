"""Evaluate Hybrid-v2 PA1 on Confirm (6,720 samples): SSIM, KLx1000, and 15 Attribute Guards using ProcessPoolExecutor."""
from __future__ import annotations

import csv
import io
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(r"d:\SonicMaster_Reproducing")
GDRIVE = Path(r"G:\My Drive\SonicMaster_Reproducing")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "SonicMaster" / "evaluation"))

from scripts.extension.eval_dev import kl_div, logmel, mel_ssim  # noqa: E402
import evaluate_control_multiple_degs_mass as attr_eval  # noqa: E402


def res(p: str | Path) -> Path:
    dp = ROOT / p
    return dp if dp.is_file() else GDRIVE / p


def load_stereo(path: Path, frames: int = 1320960) -> np.ndarray:
    for attempt in range(3):
        try:
            y, sr = sf.read(io.BytesIO(path.read_bytes()), dtype="float32", always_2d=True)
            if y.shape[1] == 1:
                y = np.repeat(y, 2, axis=1)
            return y[:frames]
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1.0)
    raise RuntimeError(f"Failed to load {path}")


def eval_one(r: dict[str, str], out_dir_str: str) -> dict[str, object]:
    sid = r["sample_id"]
    out_p = Path(out_dir_str) / f"{sid}.wav"
    if not out_p.exists() or not r["path_clean"] or not r["path_s"]:
        return {"sample_id": sid, "missing": True}

    c_path = res(r["path_clean"])
    s_path = res(r["path_s"])
    if not c_path.is_file() or not s_path.is_file():
        return {"sample_id": sid, "missing": True}

    c_st = load_stereo(c_path)
    s_st = load_stereo(s_path)
    o_st = load_stereo(out_p)
    n = min(len(c_st), len(s_st), len(o_st))
    c_st, s_st, o_st = c_st[:n], s_st[:n], o_st[:n]

    c_m = np.mean(c_st, axis=1)
    s_m = np.mean(s_st, axis=1)
    o_m = np.mean(o_st, axis=1)

    mc = logmel(c_m)
    ms = logmel(s_m)
    mo = logmel(o_m)

    ssim_a0 = mel_ssim(mc, ms)
    kl_a0 = kl_div(mc, ms) * 1000.0
    ssim_v2 = mel_ssim(mc, mo)
    kl_v2 = kl_div(mc, mo) * 1000.0

    degs = [d.strip() for d in r["degradation_types"].split("|") if d.strip()]

    attr_a0 = {}
    attr_v2 = {}
    attr_inp = {}
    if r["path_x"]:
        x_st = load_stereo(res(r["path_x"]))[:n]
        x_m = np.mean(x_st, axis=1)
        ra0 = attr_eval.evaluate_sample(c_m, x_m, s_m, degs, c_st, x_st, s_st)
        rv2 = attr_eval.evaluate_sample(c_m, x_m, o_m, degs, c_st, x_st, o_st)
        for deg, m0 in ra0.items():
            v0 = m0.get("abs_error_after", m0.get("cosine_abs_error_after"))
            vi = m0.get("abs_error_before", m0.get("cosine_abs_error_before"))
            if v0 is not None and np.isfinite(v0):
                attr_a0[deg] = float(v0)
            if vi is not None and np.isfinite(vi):
                attr_inp[deg] = float(vi)
        for deg, mv in rv2.items():
            vv = mv.get("abs_error_after", mv.get("cosine_abs_error_after"))
            if vv is not None and np.isfinite(vv):
                attr_v2[deg] = float(vv)

    return {
        "sample_id": sid,
        "source_id": r["source_id"],
        "n_degradations": int(r["n_degradations"]),
        "missing": False,
        "ssim_a0": ssim_a0,
        "kl_a0": kl_a0,
        "ssim_v2": ssim_v2,
        "kl_v2": kl_v2,
        "attr_inp": attr_inp,
        "attr_a0": attr_a0,
        "attr_v2": attr_v2,
    }


def main() -> None:
    splits = json.loads(res("artifacts/10_extension/splits.json").read_text(encoding="utf-8"))
    confirm_sources = set(splits["confirm_sources"])
    rows = []
    with res("artifacts/10_extension/inputs_manifest.csv").open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["source_id"] in confirm_sources:
                rows.append(r)

    out_dir = ROOT / "artifacts/10_extension/confirm/hybrid_v2/out_wav"
    eval_dir = ROOT / "artifacts/10_extension/confirm/hybrid_v2/eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    results = []
    missing = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(eval_one, r, str(out_dir)) for r in rows]
        for i, fut in enumerate(as_completed(futs), 1):
            res_item = fut.result()
            if res_item.get("missing"):
                missing += 1
            else:
                results.append(res_item)
            if i % 500 == 0 or i == len(rows):
                print(f"[{i}/{len(rows)}] elapsed={time.time()-t0:.1f}s missing={missing}", flush=True)

    # Save per-sample CSV
    csv_path = eval_dir / "confirm_kl_ssim_per_sample.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "source_id", "n_degradations", "ssim_a0", "kl_a0", "ssim_v2", "kl_v2"])
        w.writeheader()
        for it in results:
            w.writerow({k: it[k] for k in ["sample_id", "source_id", "n_degradations", "ssim_a0", "kl_a0", "ssim_v2", "kl_v2"]})

    # Summary KL/SSIM
    summary = {}
    for grp, cond in [
        ("All", lambda it: True),
        ("Single", lambda it: it["n_degradations"] == 1),
        ("Double+Triple", lambda it: it["n_degradations"] > 1),
    ]:
        sub = [it for it in results if cond(it)]
        summary[grp] = {
            "n": len(sub),
            "A0_SSIM": float(np.mean([it["ssim_a0"] for it in sub])),
            "V2_SSIM": float(np.mean([it["ssim_v2"] for it in sub])),
            "A0_KLx1000": float(np.mean([it["kl_a0"] for it in sub])),
            "V2_KLx1000": float(np.mean([it["kl_v2"] for it in sub])),
        }

    # Attribute guards
    err_inp: dict[str, list[float]] = defaultdict(list)
    err_a0: dict[str, list[float]] = defaultdict(list)
    err_v2: dict[str, list[float]] = defaultdict(list)
    for it in results:
        for deg, v in it["attr_inp"].items():
            err_inp[deg].append(v)
        for deg, v in it["attr_a0"].items():
            err_a0[deg].append(v)
        for deg, v in it["attr_v2"].items():
            err_v2[deg].append(v)

    attr_rows = []
    violations = []
    for deg in sorted(err_a0.keys()):
        n = len(err_a0[deg])
        mi = float(np.mean(err_inp[deg])) if err_inp[deg] else float("nan")
        m0 = float(np.mean(err_a0[deg]))
        mv = float(np.mean(err_v2[deg]))
        norm_det = (mv - m0) / mi if (mi and np.isfinite(mi) and mi > 1e-12) else 0.0
        guard_ok = True if n < 10 else (norm_det <= 0.05)
        if not guard_ok:
            violations.append(deg)
        attr_rows.append({
            "attribute": deg,
            "n": n,
            "input_err": mi,
            "A0_err": m0,
            "Hybrid_v2_PA1_err": mv,
            "norm_deterioration": norm_det,
            "guard_ok": guard_ok,
        })

    out_json = {
        "n_evaluated": len(results),
        "n_missing": missing,
        "summary": summary,
        "violations": violations,
        "attributes": attr_rows,
    }
    (eval_dir / "confirm_eval_summary.json").write_text(json.dumps(out_json, indent=2), encoding="utf-8")
    print(json.dumps({"n_evaluated": len(results), "n_missing": missing, "summary": summary, "violations": violations}, indent=2))


if __name__ == "__main__":
    main()
