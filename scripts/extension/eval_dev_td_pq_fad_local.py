"""Compute local Dev (280) FAD-CLAP and Audiobox PQ for A0 vs V3_TD_declip, plus 36-sample TD_declip breakdown."""
from __future__ import annotations

import csv
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from audiobox_aesthetics.infer import initialize_predictor
from laion_clap import CLAP_Module
from scipy.linalg import sqrtm

ROOT = Path(r"d:\SonicMaster_Reproducing")
GDRIVE = Path(r"G:\My Drive\SonicMaster_Reproducing")


def res(p: str | Path) -> Path:
    dp = ROOT / p
    return dp if dp.is_file() else GDRIVE / p


def _sf_load(filepath, *args, **kwargs):
    p = Path(filepath)
    for attempt in range(3):
        try:
            y, sr = sf.read(io.BytesIO(p.read_bytes()), dtype="float32", always_2d=True)
            return torch.from_numpy(y.T.copy()), sr
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1.0)
    raise RuntimeError(f"Failed to load {filepath}")


torchaudio.load = _sf_load


def calculate_fad(emb1: np.ndarray, emb2: np.ndarray) -> float:
    mu1, sigma1 = np.mean(emb1, axis=0), np.cov(emb1, rowvar=False)
    mu2, sigma2 = np.mean(emb2, axis=0), np.cov(emb2, rowvar=False)
    diff = mu1 - mu2
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean))


def main() -> None:
    with res("artifacts/10_extension/inputs_manifest.csv").open("r", encoding="utf-8") as f:
        dev_rows = [r for r in csv.DictReader(f) if r["split_ext"] == "dev"]

    cand_dir = ROOT / "artifacts" / "10_extension" / "dev" / "exp_v35" / "td_declip_v3_wav"
    out_json = ROOT / "artifacts" / "10_extension" / "dev" / "exp_v35" / "dev_pq_fad_td_declip_local.json"

    clap = CLAP_Module(enable_fusion=False)
    clap.load_ckpt()

    # Load waveforms into CLAP directly or filelist (patch librosa/soundfile works via file paths since local+G:)
    clean_srcs = sorted({r["source_id"] for r in dev_rows})
    clean_by_src = {r["source_id"]: str(res(r["path_clean"])) for r in dev_rows}
    clean_paths = [clean_by_src[s] for s in clean_srcs]

    clean_emb_map = {}
    for i in range(0, len(clean_paths), 8):
        b = clean_paths[i : i + 8]
        e = clap.get_audio_embedding_from_filelist(b, use_tensor=False)
        for s, vec in zip(clean_srcs[i : i + 8], e):
            clean_emb_map[s] = vec.astype(np.float32)
    c_embs = np.stack([clean_emb_map[r["source_id"]] for r in dev_rows])
    print(f"CLAP clean done ({len(clean_emb_map)})", flush=True)

    predictor = initialize_predictor()
    print("Audiobox predictor initialized", flush=True)

    # Clean PQ
    clean_pq_map = {}
    for i in range(0, len(clean_paths), 8):
        pq_r = predictor.forward([{"path": p} for p in clean_paths[i : i + 8]])
        for s, item in zip(clean_srcs[i : i + 8], pq_r):
            clean_pq_map[s] = float(item["PQ"])

    results: dict[str, object] = {"Clean_PQ": float(np.mean([clean_pq_map[r["source_id"]] for r in dev_rows]))}
    embs_by_name = {}
    pq_by_name = {}

    for name, paths in [
        ("A0", [str(res(r["path_s"])) for r in dev_rows]),
        ("V3_TD_declip", [str(cand_dir / f"{r['sample_id']}.wav") for r in dev_rows]),
    ]:
        embs = []
        pqs = []
        for i in range(0, len(paths), 8):
            b = paths[i : i + 8]
            embs.append(clap.get_audio_embedding_from_filelist(b, use_tensor=False).astype(np.float32))
            pq_r = predictor.forward([{"path": p} for p in b])
            pqs.extend(float(x["PQ"]) for x in pq_r)
            if (i // 8 + 1) % 5 == 0 or i + 8 >= len(paths):
                print(f"[{name}] {min(i+8, len(paths))}/{len(paths)}", flush=True)
        e_arr = np.concatenate(embs, axis=0)
        embs_by_name[name] = e_arr
        pq_by_name[name] = pqs

        idx_all = list(range(len(dev_rows)))
        idx_single = [i for i, r in enumerate(dev_rows) if int(r["n_degradations"]) == 1]
        idx_multi = [i for i, r in enumerate(dev_rows) if int(r["n_degradations"]) > 1]

        results[name] = {}
        for grp, idxs in [("All", idx_all), ("Single", idx_single), ("Double+Triple", idx_multi)]:
            results[name][grp] = {
                "n": len(idxs),
                "FAD": calculate_fad(e_arr[idxs], c_embs[idxs]),
                "PQ": float(np.mean([pqs[i] for i in idxs])),
            }

    # Source-level bootstrap CI95 for delta PQ
    by_src: dict[str, list[float]] = {}
    for i, r in enumerate(dev_rows):
        by_src.setdefault(r["source_id"], []).append(pq_by_name["V3_TD_declip"][i] - pq_by_name["A0"][i])
    srcs = sorted(by_src.keys())
    s_means = np.array([np.mean(by_src[s]) for s in srcs], dtype=np.float64)
    rng = np.random.default_rng(20261119)
    boots = np.mean(s_means[rng.integers(0, len(srcs), size=(2000, len(srcs)))], axis=1)
    results["delta_PQ_V3_TD_minus_A0"] = {
        "mean": float(np.mean(s_means)),
        "ci95_lo": float(np.percentile(boots, 2.5)),
        "ci95_hi": float(np.percentile(boots, 97.5)),
    }

    # 36 TD_declip subset breakdown
    reverb_set = {"real", "small", "big", "mix"}
    idx_td = [
        i
        for i, r in enumerate(dev_rows)
        if ("clip" in {d.strip() for d in r["degradation_types"].split("|") if d.strip()})
        and not ({d.strip() for d in r["degradation_types"].split("|") if d.strip()} & reverb_set)
        and r["path_x"]
    ]
    results["td_declip_36_subset"] = {
        "n": len(idx_td),
        "A0_FAD": calculate_fad(embs_by_name["A0"][idx_td], c_embs[idx_td]),
        "V3_TD_declip_FAD": calculate_fad(embs_by_name["V3_TD_declip"][idx_td], c_embs[idx_td]),
        "A0_PQ": float(np.mean([pq_by_name["A0"][i] for i in idx_td])),
        "V3_TD_declip_PQ": float(np.mean([pq_by_name["V3_TD_declip"][i] for i in idx_td])),
        "delta_PQ": float(np.mean([pq_by_name["V3_TD_declip"][i] - pq_by_name["A0"][i] for i in idx_td])),
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
