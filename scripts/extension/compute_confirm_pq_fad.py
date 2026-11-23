"""Compute Confirm FAD-CLAP and Audiobox PQ summary from Kaggle changed_pq_fad.npz + 14_run7000 baselines."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import sqrtm

ROOT = Path(__file__).resolve().parents[2]
GDRIVE = Path(r"G:\My Drive\SonicMaster_Reproducing")


def res(p: str | Path) -> Path:
    dp = ROOT / p
    return dp if dp.is_file() else GDRIVE / p


def calculate_fad(emb1: np.ndarray, emb2: np.ndarray) -> float:
    mu1, sigma1 = np.mean(emb1, axis=0), np.cov(emb1, rowvar=False)
    mu2, sigma2 = np.mean(emb2, axis=0), np.cov(emb2, rowvar=False)
    diff = mu1 - mu2
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--changed-npz", required=True, help="Path to changed_pq_fad.npz from Kaggle")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--cand-name", default="V3")
    args = ap.parse_args()

    splits = json.loads(res("artifacts/10_extension/splits.json").read_text(encoding="utf-8"))
    confirm_sources = set(splits["confirm_sources"])
    rows = [r for r in csv.DictReader(res("artifacts/10_extension/inputs_manifest.csv").open("r", encoding="utf-8")) if r["source_id"] in confirm_sources]

    # Load changed NPZ
    chg_z = np.load(args.changed_npz)
    chg_sids = [str(s) for s in chg_z["sample_ids"]]
    chg_emb = {sid: chg_z["emb"][i] for i, sid in enumerate(chg_sids)}
    chg_pq = {sid: float(chg_z["pq"][i]) for i, sid in enumerate(chg_sids)}

    # Load 14_run7000 CLAP embeddings (s01..s16)
    fad_base = GDRIVE / "artifacts/14_run7000/evaluation/fad_pq"
    base_embs: dict[str, np.ndarray] = {}
    for sdir in sorted(fad_base.glob("s[0-9][0-9]")):
        npz_p = sdir / "clap_embeddings.npz"
        if npz_p.is_file():
            z = np.load(npz_p)
            for k in z.keys():
                base_embs[k] = z[k]

    # Load 14_run7000 PQ
    pq_df = pd.read_csv(fad_base / "per_sample_pq_7000.csv")
    pq_clean_src = dict(zip(pq_df[pq_df["branch"] == "clean"]["source_id"].astype(str), pq_df[pq_df["branch"] == "clean"]["PQ"]))
    pq_a0_sid = dict(zip(pq_df[pq_df["branch"] == "sonicmaster"]["sample_id"], pq_df[pq_df["branch"] == "sonicmaster"]["PQ"]))

    # Paired evaluation on samples with base CLAP embeddings (5,306)
    paired = []
    missing = 0
    for r in rows:
        sid = r["sample_id"]
        src = r["source_id"]
        kc = f"{src}::clean"
        ka0 = f"{sid}::sonicmaster"
        if kc not in base_embs or ka0 not in base_embs or sid not in pq_a0_sid:
            missing += 1
            continue
        ec = base_embs[kc]
        ea0 = base_embs[ka0]
        ev = chg_emb.get(sid, ea0)
        pa0 = float(pq_a0_sid[sid])
        pv = chg_pq.get(sid, pa0)
        paired.append({
            "sample_id": sid,
            "source_id": src,
            "n_deg": int(r["n_degradations"]),
            "ec": ec,
            "ea0": ea0,
            "ev": ev,
            "pa0": pa0,
            "pv": pv,
            "pclean": float(pq_clean_src[src]),
        })

    out = {
        "n_evaluated": len(paired),
        "n_missing": missing,
        "n_changed_in_npz": len(chg_sids),
        "Clean_PQ": float(np.mean([it["pclean"] for it in paired])),
    }
    for grp, cond in [
        ("All", lambda it: True),
        ("Single", lambda it: it["n_deg"] == 1),
        ("Double+Triple", lambda it: it["n_deg"] > 1),
    ]:
        sub = [it for it in paired if cond(it)]
        ec = np.stack([it["ec"] for it in sub])
        ea0 = np.stack([it["ea0"] for it in sub])
        ev = np.stack([it["ev"] for it in sub])
        out[grp] = {
            "n": len(sub),
            "A0_FAD": calculate_fad(ec, ea0),
            f"{args.cand_name}_FAD": calculate_fad(ec, ev),
            "A0_PQ": float(np.mean([it["pa0"] for it in sub])),
            f"{args.cand_name}_PQ": float(np.mean([it["pv"] for it in sub])),
        }

    # Bootstrap 95% CI on delta_PQ = V - A0 (paired over all 5,306 and all 6,720)
    diffs_5306 = np.array([it["pv"] - it["pa0"] for it in paired], dtype=np.float64)
    rng = np.random.default_rng(42)
    boot = [float(np.mean(rng.choice(diffs_5306, size=len(diffs_5306), replace=True))) for _ in range(2000)]
    out[f"delta_PQ_{args.cand_name}_minus_A0"] = {
        "mean": float(np.mean(diffs_5306)),
        "ci95_lo": float(np.percentile(boot, 2.5)),
        "ci95_hi": float(np.percentile(boot, 97.5)),
    }

    # Also full 6,720 PQ
    diffs_all = np.array([chg_pq.get(r["sample_id"], float(pq_a0_sid[r["sample_id"]])) - float(pq_a0_sid[r["sample_id"]]) for r in rows], dtype=np.float64)
    boot_all = [float(np.mean(rng.choice(diffs_all, size=len(diffs_all), replace=True))) for _ in range(2000)]
    out["full_6720_PQ"] = {
        "n": len(rows),
        "A0_PQ": float(np.mean([float(pq_a0_sid[r["sample_id"]]) for r in rows])),
        f"{args.cand_name}_PQ": float(np.mean([chg_pq.get(r["sample_id"], float(pq_a0_sid[r["sample_id"]])) for r in rows])),
        "delta_mean": float(np.mean(diffs_all)),
        "ci95_lo": float(np.percentile(boot_all, 2.5)),
        "ci95_hi": float(np.percentile(boot_all, 97.5)),
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
