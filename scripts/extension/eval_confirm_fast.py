"""Fast Confirm evaluation: reuse 14_run7000 A0/input scores for S-fallback (4,832) and evaluate only changed samples (~1,888)."""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(r"d:\SonicMaster_Reproducing")
GDRIVE = Path(r"G:\My Drive\SonicMaster_Reproducing")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "SonicMaster" / "evaluation"))

from scripts.extension.eval_dev import kl_div, logmel, mel_ssim  # noqa: E402
import evaluate_control_multiple_degs_mass as attr_eval  # noqa: E402

S_FALLBACK = {"real", "clip", "xband", "small", "big", "mix", "mud"}


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


def eval_changed_one(r: dict[str, str], out_dir_str: str) -> dict[str, object]:
    sid = r["sample_id"]
    out_p = Path(out_dir_str) / f"{sid}.wav"
    if not out_p.exists() or not r["path_clean"] or not r["path_x"]:
        return {"sample_id": sid, "skip": True}

    c_path = res(r["path_clean"])
    x_path = res(r["path_x"])
    if not c_path.is_file() or not x_path.is_file():
        return {"sample_id": sid, "skip": True}

    c_st = load_stereo(c_path)
    x_st = load_stereo(x_path)
    o_st = load_stereo(out_p)
    n = min(len(c_st), len(x_st), len(o_st))
    c_st, x_st, o_st = c_st[:n], x_st[:n], o_st[:n]

    c_m = np.mean(c_st, axis=1)
    x_m = np.mean(x_st, axis=1)
    o_m = np.mean(o_st, axis=1)

    mc = logmel(c_m)
    mo = logmel(o_m)

    ssim_v2 = mel_ssim(mc, mo)
    kl_v2 = kl_div(mc, mo) * 1000.0

    degs = [d.strip() for d in r["degradation_types"].split("|") if d.strip()]
    rv2 = attr_eval.evaluate_sample(c_m, x_m, o_m, degs, c_st, x_st, o_st)
    attr_v2 = {}
    for deg, mv in rv2.items():
        vv = mv.get("abs_error_after", mv.get("cosine_abs_error_after"))
        if vv is not None and np.isfinite(vv):
            attr_v2[deg] = float(vv)

    return {
        "sample_id": sid,
        "skip": False,
        "ssim_v2": ssim_v2,
        "kl_v2": kl_v2,
        "attr_v2": attr_v2,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(ROOT / "artifacts/10_extension/confirm/hybrid_v2/out_wav"))
    parser.add_argument("--eval-dir", default=str(ROOT / "artifacts/10_extension/confirm/hybrid_v2/eval"))
    parser.add_argument("--s-fallback", default="real,clip,xband,small,big,mix,mud")
    parser.add_argument("--all-out-wav", action="store_true", help="Evaluate every sample whose WAV exists in --out-dir")
    args = parser.parse_args()

    s_fallback = {s.strip() for s in args.s_fallback.split(",") if s.strip()}

    splits = json.loads(res("artifacts/10_extension/splits.json").read_text(encoding="utf-8"))
    confirm_sources = set(splits["confirm_sources"])
    rows = []
    with res("artifacts/10_extension/inputs_manifest.csv").open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["source_id"] in confirm_sources:
                rows.append(r)

    # Load 14_run7000 A0 baselines
    kl_df = pd.read_csv(GDRIVE / "artifacts/14_run7000/evaluation/kl_ssim/kaggle/per_sample_7000.csv")
    kl_a0_df = kl_df[kl_df["branch"] == "sonicmaster"]
    a0_ssim = dict(zip(kl_a0_df[kl_a0_df["metric"] == "ssim"]["sample_id"], kl_a0_df[kl_a0_df["metric"] == "ssim"]["value"]))
    a0_kl = dict(zip(kl_a0_df[kl_a0_df["metric"] == "kl_x1000"]["sample_id"], kl_a0_df[kl_a0_df["metric"] == "kl_x1000"]["value"]))

    attr_df = pd.read_csv(GDRIVE / "artifacts/14_run7000/evaluation/attributes/per_sample_attributes_7000.csv")
    a0_attr: dict[str, dict[str, float]] = defaultdict(dict)
    inp_attr: dict[str, dict[str, float]] = defaultdict(dict)
    for sid, br, met, val in zip(attr_df["sample_id"], attr_df["branch"], attr_df["metric"], attr_df["value"]):
        if br == "sonicmaster" and np.isfinite(val):
            a0_attr[sid][met] = float(val)
        elif br == "input" and np.isfinite(val):
            inp_attr[sid][met] = float(val)

    out_dir = Path(args.out_dir)
    eval_dir = Path(args.eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Identify changed samples
    changed_rows = []
    for r in rows:
        if args.all_out_wav:
            if (out_dir / f"{r['sample_id']}.wav").is_file() and r["path_x"]:
                changed_rows.append(r)
        else:
            degs = {d.strip() for d in r["degradation_types"].split("|") if d.strip()}
            if not (degs & s_fallback) and r["path_x"]:
                changed_rows.append(r)

    print(f"Confirm total={len(rows)}, changed={len(changed_rows)}. Evaluating changed samples...", flush=True)
    chg_res: dict[str, dict] = {}
    t0 = time.time()
    skipped = 0
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(eval_changed_one, r, str(out_dir)) for r in changed_rows]
        for i, fut in enumerate(as_completed(futs), 1):
            it = fut.result()
            if it.get("skip"):
                skipped += 1
            else:
                chg_res[it["sample_id"]] = it
            if i % 250 == 0 or i == len(changed_rows):
                print(f"[{i}/{len(changed_rows)}] elapsed={time.time()-t0:.1f}s skipped={skipped}", flush=True)

    # Combine all 6,720
    combined = []
    for r in rows:
        sid = r["sample_id"]
        if sid not in a0_ssim or sid not in a0_kl:
            continue
        sa0 = float(a0_ssim[sid])
        ka0 = float(a0_kl[sid])
        if sid in chg_res:
            sv2 = float(chg_res[sid]["ssim_v2"])
            kv2 = float(chg_res[sid]["kl_v2"])
            av2 = chg_res[sid]["attr_v2"]
        else:
            sv2 = sa0
            kv2 = ka0
            av2 = a0_attr.get(sid, {})
        combined.append({
            "sample_id": sid,
            "source_id": r["source_id"],
            "n_degradations": int(r["n_degradations"]),
            "ssim_a0": sa0,
            "kl_a0": ka0,
            "ssim_v2": sv2,
            "kl_v2": kv2,
            "attr_inp": inp_attr.get(sid, {}),
            "attr_a0": a0_attr.get(sid, {}),
            "attr_v2": av2,
        })

    summary = {}
    for grp, cond in [
        ("All", lambda it: True),
        ("Single", lambda it: it["n_degradations"] == 1),
        ("Double+Triple", lambda it: it["n_degradations"] > 1),
    ]:
        sub = [it for it in combined if cond(it)]
        summary[grp] = {
            "n": len(sub),
            "A0_SSIM": float(np.mean([it["ssim_a0"] for it in sub])),
            "V2_SSIM": float(np.mean([it["ssim_v2"] for it in sub])),
            "A0_KLx1000": float(np.mean([it["kl_a0"] for it in sub])),
            "V2_KLx1000": float(np.mean([it["kl_v2"] for it in sub])),
        }

    err_inp: dict[str, list[float]] = defaultdict(list)
    err_a0: dict[str, list[float]] = defaultdict(list)
    err_v2: dict[str, list[float]] = defaultdict(list)
    for it in combined:
        for deg, v in it["attr_a0"].items():
            if deg in it["attr_inp"] and deg in it["attr_v2"]:
                err_inp[deg].append(it["attr_inp"][deg])
                err_a0[deg].append(v)
                err_v2[deg].append(it["attr_v2"][deg])

    attr_rows = []
    violations = []
    for deg in sorted(err_a0.keys()):
        n = len(err_a0[deg])
        mi = float(np.mean(err_inp[deg]))
        m0 = float(np.mean(err_a0[deg]))
        mv = float(np.mean(err_v2[deg]))
        norm_det = (mv - m0) / mi if mi > 1e-12 else 0.0
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
        "n_evaluated": len(combined),
        "n_changed_evaluated": len(chg_res),
        "n_changed_skipped": skipped,
        "summary": summary,
        "violations": violations,
        "attributes": attr_rows,
    }
    (eval_dir / "confirm_eval_summary.json").write_text(json.dumps(out_json, indent=2), encoding="utf-8")
    print(json.dumps({"n_evaluated": len(combined), "n_changed_evaluated": len(chg_res), "n_changed_skipped": skipped, "summary": summary, "violations": violations}, indent=2))


if __name__ == "__main__":
    main()
