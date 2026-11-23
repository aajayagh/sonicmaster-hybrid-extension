"""Evaluate PQ (Audiobox Aesthetics) and FAD-CLAP on Dev variants using official checkpoints."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from audiobox_aesthetics.infer import initialize_predictor
from laion_clap import CLAP_Module
from scipy.linalg import sqrtm


def _sf_load(filepath, *args, **kwargs):
    y, sr = sf.read(filepath, dtype="float32", always_2d=True)
    return torch.from_numpy(y.T.copy()), sr


torchaudio.load = _sf_load

ROOT = Path(__file__).resolve().parents[2]


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
    ap.add_argument("--manifest", default=str(ROOT / "artifacts" / "10_extension" / "inputs_manifest.csv"))
    ap.add_argument("--a0-dir", default=str(ROOT / "artifacts" / "10_extension" / "dev" / "pilot_p" / "A0"))
    ap.add_argument("--cand-dir", required=True)
    ap.add_argument("--cand-name", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    with Path(args.manifest).open("r", encoding="utf-8") as f:
        dev_rows = [r for r in csv.DictReader(f) if r["split_ext"] == "dev"]

    clean_paths = sorted({str(ROOT / r["path_clean"]) for r in dev_rows})
    clean_by_src = {r["source_id"]: str(ROOT / r["path_clean"]) for r in dev_rows}

    # 1. CLAP embeddings (batched by 16 to stay within RAM)
    clap = CLAP_Module(enable_fusion=False)
    clap.load_ckpt()
    clean_emb_map = {}
    for i in range(0, len(clean_paths), 16):
        batch = clean_paths[i : i + 16]
        embs_c = clap.get_audio_embedding_from_filelist(batch, use_tensor=False)
        for p, e in zip(batch, embs_c):
            clean_emb_map[p] = e
    print(f"CLAP clean done ({len(clean_emb_map)})", flush=True)

    # 2. Audiobox Aesthetics PQ (batched by 16)
    predictor = initialize_predictor()
    print("Audiobox predictor initialized", flush=True)

    results = {}
    for name, folder in [("A0", Path(args.a0_dir)), (args.cand_name, Path(args.cand_dir))]:
        paths = [str(folder / f"{r['sample_id']}.wav") for r in dev_rows]
        c_embs = np.stack([clean_emb_map[clean_by_src[r["source_id"]]] for r in dev_rows])
        o_list = []
        pq_vals = []
        for i in range(0, len(paths), 16):
            batch = paths[i : i + 16]
            o_list.append(clap.get_audio_embedding_from_filelist(batch, use_tensor=False))
            pq_rows = predictor.forward([{"path": p} for p in batch])
            pq_vals.extend(float(x["PQ"]) for x in pq_rows)
            print(f"[{name}] {min(i+16, len(paths))}/{len(paths)}", flush=True)
        o_embs = np.concatenate(o_list, axis=0)

        # Group indices
        idx_all = list(range(len(dev_rows)))
        idx_single = [i for i, r in enumerate(dev_rows) if int(r["n_degradations"]) == 1]
        idx_multi = [i for i, r in enumerate(dev_rows) if int(r["n_degradations"]) > 1]

        results[name] = {}
        for grp, idxs in [("All", idx_all), ("Single", idx_single), ("Double+Triple", idx_multi)]:
            results[name][grp] = {
                "n": len(idxs),
                "FAD": calculate_fad(o_embs[idxs], c_embs[idxs]),
                "PQ": float(np.mean([pq_vals[i] for i in idxs])),
            }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
