"""Evaluate a single V4 arm against the cached matched_dev_recheck common Dev protocol."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "SonicMaster" / "evaluation"))

from scripts.extension.eval_dev import logmel, source_bootstrap_ci  # noqa: E402
from scripts.extension.v4_clap import matched_embeddings  # noqa: E402
from scripts.extension.v4_rescore_dev import fad  # noqa: E402
from scripts.extension.v4_rescore_perceptual import load_clap  # noqa: E402
from scripts.extension.v4_runtime import spectral_scores  # noqa: E402
from scripts.neural_codec_bypass.fallback_attributes import evaluator  # noqa: E402
import evaluate_control_multiple_degs_mass as attr_eval  # noqa: E402

RECHECK = ROOT / "artifacts/10_extension/v4/matched_dev_recheck"
CHECKPOINTS = ROOT / "tmp/rescore_failure_segments"
REVERB_SET = {"small", "big", "mix", "real"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-name", required=True)
    ap.add_argument("--wav-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    t0 = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((RECHECK / "manifest.json").read_text(encoding="utf-8"))
    ids = [r["sample_id"] for r in manifest]

    cached_spec = json.loads((RECHECK / "spectral.json").read_text(encoding="utf-8"))
    spec_map = {(r["sample_id"], r["arm"]): r for r in cached_spec}
    cached_pq = json.loads((RECHECK / "pq.json").read_text(encoding="utf-8"))
    assert cached_pq["ids"] == ids
    cached_clap = np.load(RECHECK / "clap_embeddings.npz", allow_pickle=False)
    assert list(cached_clap["ids"]) == ids

    evaluate_fn = evaluator()

    @lru_cache(maxsize=48)
    def clean_ref(rel_path: str, frames: int):
        audio, rate = sf.read(ROOT / rel_path, frames=frames, always_2d=True)
        assert rate == 44100 and len(audio) == frames
        if audio.shape[1] == 1:
            audio = np.repeat(audio, 2, axis=1)
        mono = audio.mean(axis=1)
        return audio, mono, logmel(mono)

    def read_wav(path: Path, frames: int) -> np.ndarray:
        audio, rate = sf.read(path, frames=frames, always_2d=True)
        assert rate == 44100 and len(audio) == frames, f"Extent mismatch for {path}"
        return np.repeat(audio, 2, axis=1) if audio.shape[1] == 1 else audio

    def score_clip(rec: dict) -> dict:
        sid = rec["sample_id"]
        frames = int(rec["frames"])
        c, cm, ref_mel = clean_ref(rec["paths"]["clean"], frames)
        x = read_wav(ROOT / rec["paths"]["input"], frames)
        xm = x.mean(axis=1)
        y = read_wav(args.wav_dir / f"{sid}.wav", frames)
        ym = y.mean(axis=1)

        ssim, kl = spectral_scores(ref_mel, y.T)
        active = [d for d in rec["degradations"] if d not in REVERB_SET]
        details = evaluate_fn(cm, xm, ym, active, c, x, y)
        errors = {}
        for deg, fields in details.items():
            val = fields.get("abs_error_after", fields.get("cosine_abs_error_after"))
            if val is not None and np.isfinite(val):
                errors[deg] = float(val)

        msd_cand = None
        msd_a0 = None
        msd_v44 = None
        if set(rec["degradations"]) & REVERB_SET:
            msd_cand = float(attr_eval.modulation_spectrum_distance(cm, ym, fs=44100))
            msd_a0 = float(attr_eval.modulation_spectrum_distance(cm, read_wav(ROOT / rec["paths"]["A0"], frames).mean(axis=1), fs=44100))
            msd_v44 = float(attr_eval.modulation_spectrum_distance(cm, read_wav(ROOT / rec["paths"]["v4_4"], frames).mean(axis=1), fs=44100))

        return {
            "sample_id": sid,
            "source_id": rec["source_id"],
            "arm": args.arm_name,
            "ssim": float(ssim),
            "kl1000": float(kl),
            "errors": errors,
            "msd": msd_cand,
            "msd_A0": msd_a0,
            "msd_v4_4": msd_v44,
        }

    print(f"Scoring spectral, 15 guards, and Reverb MSD for {args.arm_name} ({len(manifest)} clips)...", flush=True)
    cand_spec: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, res_item in enumerate(pool.map(score_clip, manifest), 1):
            cand_spec.append(res_item)
            if i % 70 == 0 or i == len(manifest):
                print(f"  spectral+guards {i}/{len(manifest)} in {time.time()-t0:.1f}s", flush=True)

    # CLAP embeddings (only candidate arm!)
    print(f"Scoring CLAP embeddings for {args.arm_name}...", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clap_model = load_clap(torch, device)
    wav_paths = [args.wav_dir / f"{r['sample_id']}.wav" for r in manifest]
    frame_counts = [int(r["frames"]) for r in manifest]
    with torch.inference_mode():
        cand_clap = matched_embeddings(clap_model, wav_paths, frame_counts, batch_size=8).astype(np.float64)
    del clap_model

    # Audiobox PQ (only candidate arm!)
    print(f"Scoring Audiobox PQ for {args.arm_name}...", flush=True)
    from audiobox_aesthetics.infer import initialize_predictor

    predictor = initialize_predictor(str(CHECKPOINTS / "audiobox_checkpoint.pt"))
    cand_pq: list[float] = []
    for i in range(0, len(manifest), 4):
        batch = manifest[i : i + 4]
        inputs = [{"path": torch.from_numpy(read_wav(args.wav_dir / f"{r['sample_id']}.wav", int(r["frames"])).T.astype(np.float32)), "sample_rate": 44100} for r in batch]
        cand_pq.extend(float(x["PQ"]) for x in predictor.forward(inputs))
        if (i + 4) % 40 == 0 or (i + 4) >= len(manifest):
            print(f"  PQ {min(i+4, len(manifest))}/{len(manifest)} in {time.time()-t0:.1f}s", flush=True)

    for i, item in enumerate(cand_spec):
        item["pq"] = cand_pq[i]

    # Attach cached PQ & spec baselines
    by_sid_cand = {r["sample_id"]: r for r in cand_spec}
    pq_base = {arm: dict(zip(ids, cached_pq["scores"][arm], strict=True)) for arm in ("A0", "v3_blind", "v3_label", "v4_3", "v4_4")}

    families: dict[str, list[str]] = {"all": ids, "single": [], "multiple": [], "reverb": [], "nonreverb": [], "clip": []}
    for r in manifest:
        sid = r["sample_id"]
        degs = set(r["degradations"])
        families["single" if len(degs) == 1 else "multiple"].append(sid)
        families["reverb" if (degs & REVERB_SET) else "nonreverb"].append(sid)
        if "clip" in degs:
            families["clip"].append(sid)

    summary_groups = {}
    for gname, subset in families.items():
        idx = [i for i, sid in enumerate(ids) if sid in set(subset)]
        c_arr = [by_sid_cand[sid] for sid in subset]
        item: dict = {
            "n_clips": len(subset),
            "ssim": float(np.mean([x["ssim"] for x in c_arr])),
            "kl1000": float(np.mean([x["kl1000"] for x in c_arr])),
            "pq": float(np.mean([x["pq"] for x in c_arr])),
            "fad_clap": fad(cached_clap["clean"][idx], cand_clap[idx]),
            "versus": {},
        }
        if gname == "reverb":
            item["msd"] = float(np.mean([x["msd"] for x in c_arr]))
            item["msd_A0"] = float(np.mean([x["msd_A0"] for x in c_arr]))
            item["msd_v4_4"] = float(np.mean([x["msd_v4_4"] for x in c_arr]))

        for base in ("A0", "v4_4", "v4_3", "v3_blind", "v3_label"):
            item["versus"][base] = {}
            for m_key in ("ssim", "kl1000", "pq"):
                pairs = []
                for sid in subset:
                    cand_v = by_sid_cand[sid][m_key]
                    base_v = pq_base[base][sid] if m_key == "pq" else spec_map[sid, base][m_key]
                    pairs.append({"source_id": by_sid_cand[sid]["source_id"], "delta": cand_v - base_v})
                mean_d, lo_d, hi_d = source_bootstrap_ci(pairs, "delta")
                item["versus"][base][m_key] = {"source_mean": mean_d, "ci95": [lo_d, hi_d]}
        summary_groups[gname] = item

    # Matched 15 Attribute Guards (exact matched_dev_recheck denominator & errors)
    guards = {}
    violations = []
    all_degs = sorted({k for r in cand_spec for k in r["errors"]})
    for deg in all_degs:
        matched = [sid for sid in ids if deg in by_sid_cand[sid]["errors"] and deg in spec_map[sid, "A0"]["errors"] and deg in spec_map[sid, "input"]["errors"]]
        if not matched:
            continue
        inp_e = float(np.mean([spec_map[sid, "input"]["errors"][deg] for sid in matched]))
        a0_e = float(np.mean([spec_map[sid, "A0"]["errors"][deg] for sid in matched]))
        v44_e = float(np.mean([spec_map[sid, "v4_4"]["errors"][deg] for sid in matched]))
        cand_e = float(np.mean([by_sid_cand[sid]["errors"][deg] for sid in matched]))
        eligible = len(matched) >= 10 and not deg.endswith("_msd") and inp_e > 1e-12
        norm_cand = (cand_e - a0_e) / inp_e if inp_e > 1e-12 else 0.0
        norm_v44 = (v44_e - a0_e) / inp_e if inp_e > 1e-12 else 0.0
        ok = (not eligible) or (norm_cand <= 0.05)
        if eligible and norm_cand > 0.05:
            violations.append(deg)
        guards[deg] = {
            "n": len(matched),
            "eligible": eligible,
            "input_err": inp_e,
            "A0_err": a0_e,
            "v4_4_err": v44_e,
            "cand_err": cand_e,
            "norm": norm_cand,
            "v4_4_norm": norm_v44,
            "ok": ok,
        }

    out_summary = {
        "arm": args.arm_name,
        "elapsed_sec": round(time.time() - t0, 1),
        "groups": summary_groups,
        "guards": guards,
        "violations": violations,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(out_summary, indent=2), encoding="utf-8")

    # Save per_sample.csv
    with (args.out_dir / "per_sample.csv").open("w", newline="", encoding="utf-8") as f:
        fn = ["sample_id", "source_id", "ssim", "kl1000", "pq", "pq_A0", "pq_v4_4", "msd", "msd_A0"]
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        for sid in ids:
            r = by_sid_cand[sid]
            w.writerow({
                "sample_id": sid, "source_id": r["source_id"],
                "ssim": r["ssim"], "kl1000": r["kl1000"], "pq": r["pq"],
                "pq_A0": pq_base["A0"][sid], "pq_v4_4": pq_base["v4_4"][sid],
                "msd": r["msd"], "msd_A0": r["msd_A0"],
            })

    # Write detailed Markdown results table
    g_all = summary_groups["all"]
    g_rev = summary_groups["reverb"]
    g_nonrev = summary_groups["nonreverb"]
    md = [
        f"# Kết Quả Chấm Chung `{args.arm_name}` Trên Dev Độc Lập (`N = 280`, Common Protocol)",
        "",
        f"- **Checkpoint / Exports:** `{args.wav_dir}`",
        f"- **Thời gian chấm:** `{out_summary['elapsed_sec']}s` (tận dụng cache `matched_dev_recheck` cho `clean`, `input`, `A0`, `v4_3`, `v4_4`)",
        f"- **Guard Violations (`n >= 10, norm > 0.05`):** `{violations}`",
        "",
        "## 1. Chỉ Số Toàn Tập & Phân Nhóm",
        "",
        "| Nhóm | `n` | Mel-SSIM ↑ | Mel-KL×1000 ↓ | FAD-CLAP ↓ | Audiobox PQ ↑ | ΔPQ vs `A0` | ΔPQ vs `V4-4` | Official MSD ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| **All** | {g_all['n_clips']} | `{g_all['ssim']:.4f}` | `{g_all['kl1000']:.4f}` | `{g_all['fad_clap']:.4f}` | `{g_all['pq']:.4f}` | `{g_all['versus']['A0']['pq']['source_mean']:+.4f}` | `{g_all['versus']['v4_4']['pq']['source_mean']:+.4f}` | — |",
        f"| **Single** | {summary_groups['single']['n_clips']} | `{summary_groups['single']['ssim']:.4f}` | `{summary_groups['single']['kl1000']:.4f}` | `{summary_groups['single']['fad_clap']:.4f}` | `{summary_groups['single']['pq']:.4f}` | `{summary_groups['single']['versus']['A0']['pq']['source_mean']:+.4f}` | `{summary_groups['single']['versus']['v4_4']['pq']['source_mean']:+.4f}` | — |",
        f"| **Multiple** | {summary_groups['multiple']['n_clips']} | `{summary_groups['multiple']['ssim']:.4f}` | `{summary_groups['multiple']['kl1000']:.4f}` | `{summary_groups['multiple']['fad_clap']:.4f}` | `{summary_groups['multiple']['pq']:.4f}` | `{summary_groups['multiple']['versus']['A0']['pq']['source_mean']:+.4f}` | `{summary_groups['multiple']['versus']['v4_4']['pq']['source_mean']:+.4f}` | — |",
        f"| **Reverb** | {g_rev['n_clips']} | `{g_rev['ssim']:.4f}` | `{g_rev['kl1000']:.4f}` | `{g_rev['fad_clap']:.4f}` | `{g_rev['pq']:.4f}` | `{g_rev['versus']['A0']['pq']['source_mean']:+.4f}` | `{g_rev['versus']['v4_4']['pq']['source_mean']:+.4f}` | `{g_rev['msd']:.4f}` (A0=`{g_rev['msd_A0']:.4f}`) |",
        f"| **Non-Reverb** | {g_nonrev['n_clips']} | `{g_nonrev['ssim']:.4f}` | `{g_nonrev['kl1000']:.4f}` | `{g_nonrev['fad_clap']:.4f}` | `{g_nonrev['pq']:.4f}` | `{g_nonrev['versus']['A0']['pq']['source_mean']:+.4f}` | `{g_nonrev['versus']['v4_4']['pq']['source_mean']:+.4f}` | — |",
        f"| **Clip** | {summary_groups['clip']['n_clips']} | `{summary_groups['clip']['ssim']:.4f}` | `{summary_groups['clip']['kl1000']:.4f}` | `{summary_groups['clip']['fad_clap']:.4f}` | `{summary_groups['clip']['pq']:.4f}` | `{summary_groups['clip']['versus']['A0']['pq']['source_mean']:+.4f}` | `{summary_groups['clip']['versus']['v4_4']['pq']['source_mean']:+.4f}` | — |",
        "",
        "## 2. Đầy Đủ 15 Attribute Guards (Matched Common Protocol)",
        "",
        "| Attribute | `n` | `input_err` | `A0_err` | `V4-4_err` | `V4-5p_err` | `V4-4_norm` | `V4-5p_norm` | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for deg, gd in guards.items():
        st = "low-n" if not gd["eligible"] else ("PASS" if gd["ok"] else "FAIL")
        md.append(
            f"| `{deg}` | {gd['n']} | `{gd['input_err']:.4f}` | `{gd['A0_err']:.4f}` | `{gd['v4_4_err']:.4f}` | `{gd['cand_err']:.4f}` | `{gd['v4_4_norm']:+.4f}` | **`{gd['norm']:+.4f}`** | **{st}** |"
        )
    md_path = args.out_dir / f"{args.arm_name}_results_table.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"ALL DONE -> {args.out_dir / 'summary.json'} | violations={violations}", flush=True)


if __name__ == "__main__":
    main()
