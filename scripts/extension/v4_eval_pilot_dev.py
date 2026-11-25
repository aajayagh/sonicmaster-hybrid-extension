"""Evaluate Pilot V4-0, V4-1, V4-2 checkpoints on independent Dev (280 samples):
- Full-track 30s inference (chunk 8s, overlap 50%, hop=512)
- Internal diagnostics: mean(g_M), mean(g_S) per degradation family, dw0_rms, dwk_rms, clip_rate, stereo corr
- SSIM, KLx1000, 15 Attribute Guards, and local FAD-CLAP + Audiobox PQ.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
from collections import defaultdict
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "SonicMaster" / "evaluation"))

from scripts.extension.v4_alignment import align_teacher_to_input  # noqa: E402
from scripts.extension.eval_dev import logmel, source_bootstrap_ci  # noqa: E402
from scripts.extension.v4_clap import matched_embeddings  # noqa: E402
from scripts.extension.v4_models import HybridV4Net  # noqa: E402
from scripts.extension.v4_runtime import checkpoint_inference_config, spectral_scores  # noqa: E402


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


def load_stereo_np(path: Path, frames: int = 1320960) -> tuple[np.ndarray, int]:
    for attempt in range(3):
        try:
            y, sr = sf.read(io.BytesIO(path.read_bytes()), dtype="float32", always_2d=True)
            if y.shape[1] == 1:
                y = np.repeat(y, 2, axis=1)
            return y[:frames].T.copy(), sr
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1.0)
    raise RuntimeError(f"Failed to load {path}")


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
    ap.add_argument("--ckpt-root", default=str(ROOT / "artifacts/10_extension/v4/pilot_ckpts"))
    ap.add_argument("--variants", default="v4_0,v4_1,v4_2")
    ap.add_argument("--chunk-sec", type=float)
    ap.add_argument("--gate-bias-init", type=float)
    ap.add_argument("--from-infer-json", action="store_true")
    ap.add_argument("--out-json", default=str(ROOT / "artifacts" / "10_extension" / "v4" / "pilot_dev_eval" / "pilot_summary.json"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with res("artifacts/10_extension/inputs_manifest.csv").open("r", encoding="utf-8") as f:
        dev_rows = [r for r in csv.DictReader(f) if r["split_ext"] == "dev"]

    # Cache Clean, Input x, A0 s in memory (280 Dev clips)
    clean_cache: dict[str, np.ndarray] = {}
    x_cache: dict[str, np.ndarray] = {}
    s_cache: dict[str, np.ndarray] = {}
    for r in dev_rows:
        sid = r["sample_id"]
        src = r["source_id"]
        if src not in clean_cache:
            c_np, _ = load_stereo_np(res(r["path_clean"]))
            clean_cache[src] = c_np
        x_np, _ = load_stereo_np(res(r["path_x"]))
        s_np, _ = load_stereo_np(res(r["path_s"]))
        x_cache[sid] = x_np
        s_cache[sid] = s_np

    clap = CLAP_Module(enable_fusion=False)
    clap.load_ckpt()
    predictor = initialize_predictor()

    frame_counts = [min(clean_cache[r["source_id"]].shape[-1], x_cache[r["sample_id"]].shape[-1],
                        s_cache[r["sample_id"]].shape[-1]) for r in dev_rows]
    clean_paths = [str(res(r["path_clean"])) for r in dev_rows]
    c_embs = matched_embeddings(clap, clean_paths, frame_counts)
    a0_paths = [str(res(r["path_s"])) for r in dev_rows]
    a0_embs = matched_embeddings(clap, a0_paths, frame_counts)
    a0_pqs = []
    for i in range(0, len(a0_paths), 8):
        b = a0_paths[i : i + 8]
        a0_pqs.extend(float(x["PQ"]) for x in predictor.forward([{"path": p} for p in b]))

    summary_all: dict[str, object] = {
        "A0": {
            "All_FAD": calculate_fad(a0_embs, c_embs),
            "All_PQ": float(np.mean(a0_pqs)),
        }
    }

    reverb_set = {"small", "big", "mix", "real"}
    eq_set = {"mud", "boom", "warm", "bright", "dark", "clarity", "airy", "mic"}
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    for var in variants:
        out_dir = ROOT / "artifacts" / "10_extension" / "v4" / "pilot_dev_eval" / var
        out_wav_dir = out_dir / "out_wav"
        if args.from_infer_json:
            meta = json.loads((out_dir / "infer_meta.json").read_text(encoding="utf-8"))
            by_id = {item["sample_id"]: item for item in meta["per_sample"]}
            manifest_ids = {r["sample_id"] for r in dev_rows}
            if len(by_id) != len(meta["per_sample"]) or set(by_id) != manifest_ids:
                raise ValueError("Inference metadata and Dev manifest contain different sample IDs")
            per_sample = [by_id[r["sample_id"]].copy() for r in dev_rows]
            ckpt = {"epoch": meta["epoch"]}
            g_by_fam = {k: [v] for k, v in meta["g_by_family"].items()}
            t0 = time.time() - float(meta["elapsed_sec"])
        else:
            ckpt_p = Path(args.ckpt_root) / var / "best.pt"
            if not ckpt_p.is_file():
                print(f"Skipping {var}: {ckpt_p} not found", flush=True)
                continue
            ckpt = torch.load(ckpt_p, map_location="cpu", weights_only=False)
            config = checkpoint_inference_config(ckpt, var, args.gate_bias_init, args.chunk_sec)
            model = HybridV4Net(var, hop=config["hop"], gate_bias_init=config["gate_bias_init"]).to(device)
            model.load_state_dict(ckpt["model"])
            model.eval()

            out_wav_dir.mkdir(parents=True, exist_ok=True)

            per_sample = []
            g_by_fam: dict[str, list[float]] = defaultdict(list)
            t0 = time.time()
            for r in dev_rows:
                sid = r["sample_id"]
                src = r["source_id"]
                x_np = x_cache[sid]
                s_np = s_cache[sid]
                n = min(x_np.shape[-1], s_np.shape[-1])
                x_np, s_np = x_np[:, :n], s_np[:, :n]

                if model.has_teacher:
                    s_al = align_teacher_to_input(x_np, s_np)
                    s_t = torch.from_numpy(s_al).unsqueeze(0).to(device)
                else:
                    s_t = None
                x_t = torch.from_numpy(x_np).unsqueeze(0).to(device)

                with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    y_t, diag = model.forward_track(x_t, s_t, chunk_sec=config["chunk_sec"], overlap_ratio=0.5)

                y_np = y_t.squeeze(0).cpu().numpy()
                clip_rate = float(np.mean(np.abs(y_np) >= 0.999))
                y_save = np.clip(y_np.T, -1.0, 1.0).astype(np.float32)
                sf.write(str(out_wav_dir / f"{sid}.wav"), y_save, 44100, subtype="PCM_16")

                stereo_corr = float(np.corrcoef(y_np[0], y_np[1])[0, 1]) if np.std(y_np[0]) > 1e-8 and np.std(y_np[1]) > 1e-8 else 1.0

                degs = {d.strip() for d in r["degradation_types"].split("|") if d.strip()}
                if "g_mean_M" in diag:
                    g_val = 0.5 * (diag["g_mean_M"] + diag["g_mean_S"])
                    g_by_fam["all"].append(g_val)
                    if not (degs & (reverb_set | {"clip", "xband", "vocal"})):
                        g_by_fam["linear_eq_dyn"].append(g_val)
                    if degs & reverb_set:
                        g_by_fam["reverb"].append(g_val)
                    if "clip" in degs:
                        g_by_fam["clip"].append(g_val)
                    if "xband" in degs:
                        g_by_fam["xband"].append(g_val)

                per_sample.append({
                    "sample_id": sid,
                    "source_id": src,
                    "n_deg": int(r["n_degradations"]),
                    "degs": r["degradation_types"],
                    "clip_rate": clip_rate,
                    "stereo_corr": stereo_corr,
                    **diag,
                })

        # FAD & PQ
        paths = [str(out_wav_dir / f"{r['sample_id']}.wav") for r in dev_rows]
        for row, path, frames, item in zip(dev_rows, paths, frame_counts, per_sample, strict=True):
            audio, _ = load_stereo_np(Path(path), frames=frames)
            if audio.shape[-1] != frames:
                raise ValueError(f"Exported WAV is shorter than the common evaluation span: {path}")
            reference = logmel(clean_cache[row["source_id"]][:, :frames].mean(axis=0))
            item["ssim"], item["kl"] = spectral_scores(reference, audio)
        v_embs = matched_embeddings(clap, paths, frame_counts)
        pqs = []
        for i in range(0, len(paths), 8):
            b = paths[i : i + 8]
            pqs.extend(float(x["PQ"]) for x in predictor.forward([{"path": p} for p in b]))

        for i, it in enumerate(per_sample):
            it["pq"] = pqs[i]
            it["dpq_vs_a0"] = pqs[i] - a0_pqs[i]

        idx_all = list(range(len(dev_rows)))
        idx_s = [i for i, r in enumerate(dev_rows) if int(r["n_degradations"]) == 1]
        idx_dt = [i for i, r in enumerate(dev_rows) if int(r["n_degradations"]) > 1]
        idx_rev = [i for i, r in enumerate(dev_rows) if ({d.strip() for d in r["degradation_types"].split("|") if d.strip()} & reverb_set)]
        idx_clp = [i for i, r in enumerate(dev_rows) if ("clip" in {d.strip() for d in r["degradation_types"].split("|") if d.strip()})]

        _, dpq_lo, dpq_hi = source_bootstrap_ci(per_sample, "dpq_vs_a0")

        var_res = {
            "epoch": int(ckpt.get("epoch", -1)),
            "elapsed_sec": round(time.time() - t0, 1),
            "All": {
                "ssim": float(np.mean([it["ssim"] for it in per_sample])),
                "kl": float(np.mean([it["kl"] for it in per_sample])),
                "fad": calculate_fad(v_embs, c_embs),
                "pq": float(np.mean(pqs)),
                "dpq_mean": float(np.mean([it["dpq_vs_a0"] for it in per_sample])),
                "dpq_ci95_lo": dpq_lo,
                "dpq_ci95_hi": dpq_hi,
            },
            "Single": {
                "ssim": float(np.mean([per_sample[i]["ssim"] for i in idx_s])),
                "kl": float(np.mean([per_sample[i]["kl"] for i in idx_s])),
                "fad": calculate_fad(v_embs[idx_s], c_embs[idx_s]),
                "pq": float(np.mean([pqs[i] for i in idx_s])),
            },
            "Double+Triple": {
                "ssim": float(np.mean([per_sample[i]["ssim"] for i in idx_dt])),
                "kl": float(np.mean([per_sample[i]["kl"] for i in idx_dt])),
                "fad": calculate_fad(v_embs[idx_dt], c_embs[idx_dt]),
                "pq": float(np.mean([pqs[i] for i in idx_dt])),
            },
            "Reverb_106": {
                "ssim": float(np.mean([per_sample[i]["ssim"] for i in idx_rev])),
                "kl": float(np.mean([per_sample[i]["kl"] for i in idx_rev])),
                "pq": float(np.mean([pqs[i] for i in idx_rev])),
                "dpq_vs_a0": float(np.mean([per_sample[i]["dpq_vs_a0"] for i in idx_rev])),
            },
            "Clip_48": {
                "ssim": float(np.mean([per_sample[i]["ssim"] for i in idx_clp])),
                "kl": float(np.mean([per_sample[i]["kl"] for i in idx_clp])),
                "pq": float(np.mean([pqs[i] for i in idx_clp])),
                "dpq_vs_a0": float(np.mean([per_sample[i]["dpq_vs_a0"] for i in idx_clp])),
            },
            "internal_diagnostics": {
                "mean_clip_rate": float(np.mean([it["clip_rate"] for it in per_sample])),
                "mean_stereo_corr": float(np.mean([it["stereo_corr"] for it in per_sample])),
                "g_by_family": {k: float(np.mean(v)) for k, v in g_by_fam.items()},
                "dw0_rms": float(np.mean([it["dw0_rms"] for it in per_sample])) if "dw0_rms" in per_sample[0] else None,
                "dwk_rms": float(np.mean([it["dwk_rms"] for it in per_sample])) if "dwk_rms" in per_sample[0] else None,
            },
        }
        summary_all[var] = var_res
        print(f"\n=== {var} (Ep {var_res['epoch']}) ===\n" + json.dumps(var_res, indent=2), flush=True)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary_all, indent=2), encoding="utf-8")
    print(f"\nSaved {out_json}")


if __name__ == "__main__":
    main()
