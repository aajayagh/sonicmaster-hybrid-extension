"""Fast GPU inference for Hybrid-v4 checkpoints on Dev (280 samples) using extension-env (CUDA)."""
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

ROOT = Path(r"d:\SonicMaster_Reproducing")
GDRIVE = Path(r"G:\My Drive\SonicMaster_Reproducing")
sys.path.insert(0, str(ROOT))

from scripts.extension.v4_alignment import align_teacher_to_input  # noqa: E402
from scripts.extension.eval_dev import logmel  # noqa: E402
from scripts.extension.v4_models import HybridV4Net  # noqa: E402
from scripts.extension.v4_runtime import checkpoint_inference_config, spectral_scores  # noqa: E402


def res(p: str | Path) -> Path:
    dp = ROOT / p
    return dp if dp.is_file() else GDRIVE / p


def load_stereo_np(path: Path, frames: int = 1320960) -> tuple[np.ndarray, int]:
    y, sr = sf.read(io.BytesIO(path.read_bytes()), dtype="float32", always_2d=True)
    if y.shape[1] == 1:
        y = np.repeat(y, 2, axis=1)
    return y[:frames].T.copy(), sr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--chunk-sec", type=float)
    ap.add_argument("--gate-bias-init", type=float)
    ap.add_argument("--output-dir", type=Path)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    with res("artifacts/10_extension/inputs_manifest.csv").open("r", encoding="utf-8") as f:
        dev_rows = [r for r in csv.DictReader(f) if r["split_ext"] == "dev"]

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    config = checkpoint_inference_config(ckpt, args.variant, args.gate_bias_init, args.chunk_sec)
    model = HybridV4Net(args.variant, hop=config["hop"], gate_bias_init=config["gate_bias_init"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    out_dir = args.output_dir or ROOT / "artifacts" / "10_extension" / "v4" / "pilot_dev_eval" / args.variant
    out_wav_dir = out_dir / "out_wav"
    out_wav_dir.mkdir(parents=True, exist_ok=True)

    clean_cache: dict[str, np.ndarray] = {}
    reverb_set = {"small", "big", "mix", "real"}
    g_by_fam: dict[str, list[float]] = defaultdict(list)
    per_sample: list[dict] = []

    t0 = time.time()
    for idx, r in enumerate(dev_rows):
        sid = r["sample_id"]
        src = r["source_id"]
        if src not in clean_cache:
            c_np, _ = load_stereo_np(res(r["path_clean"]))
            clean_cache[src] = c_np
        c_np = clean_cache[src]
        x_np, _ = load_stereo_np(res(r["path_x"]))
        s_np, _ = load_stereo_np(res(r["path_s"]))
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
        wav = out_wav_dir / f"{sid}.wav"
        sf.write(str(wav), y_save, 44100, subtype="PCM_16")
        exported, _ = load_stereo_np(wav)
        score_frames = min(c_np.shape[-1], exported.shape[-1])
        mc = logmel(c_np[:, :score_frames].mean(axis=0))
        s_val, k_val = spectral_scores(mc, exported[:, :score_frames])
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
            "ssim": s_val,
            "kl": k_val,
            "clip_rate": clip_rate,
            "stereo_corr": stereo_corr,
            **diag,
        })
        if (idx + 1) % 50 == 0 or (idx + 1) == len(dev_rows):
            print(f"[{args.variant}] {idx+1}/{len(dev_rows)} in {time.time()-t0:.1f}s", flush=True)

    meta = {
        "variant": args.variant,
        "epoch": int(ckpt.get("epoch", -1)),
        "inference_config": config,
        "elapsed_sec": round(time.time() - t0, 1),
        "g_by_family": {k: float(np.mean(v)) for k, v in g_by_fam.items()},
        "per_sample": per_sample,
    }
    (out_dir / "infer_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved {out_dir / 'infer_meta.json'} | SSIM={np.mean([x['ssim'] for x in per_sample]):.4f} | KL={np.mean([x['kl'] for x in per_sample]):.4f}")


if __name__ == "__main__":
    main()
