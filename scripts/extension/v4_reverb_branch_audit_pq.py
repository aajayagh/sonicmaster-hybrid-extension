"""Step 2 of Reverb Branch Audit: Official Reverb MSD, Audiobox PQ, listening bundles, MD + PNG tables."""
from __future__ import annotations

import csv
import io
import json
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from audiobox_aesthetics.infer import initialize_predictor
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(r"d:\SonicMaster_Reproducing")
GDRIVE = Path(r"G:\My Drive\SonicMaster_Reproducing")
BRAIN_ART = Path(r"C:\Users\ADMIN\.gemini\antigravity\brain\bd54de32-517b-4715-954e-8096d7c41b00")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "SonicMaster" / "evaluation"))

from scripts.extension.eval_dev import source_bootstrap_ci  # noqa: E402
import evaluate_control_multiple_degs_mass as attr_eval  # noqa: E402

REVERB_TYPES = ("small", "big", "mix", "real")
BRANCHES = ("Input", "P", "Pg", "Full", "A0")


def res(p: str | Path) -> Path:
    dp = ROOT / p
    return dp if dp.is_file() else GDRIVE / p


def _sf_load(filepath, *args, **kwargs):
    p = Path(filepath)
    y, sr = sf.read(io.BytesIO(p.read_bytes()), dtype="float32", always_2d=True)
    return torch.from_numpy(y.T.copy()), sr


torchaudio.load = _sf_load


def load_stereo_np(path: Path, frames: int) -> np.ndarray:
    y, _ = sf.read(io.BytesIO(path.read_bytes()), dtype="float32", always_2d=True)
    if y.shape[1] == 1:
        y = np.repeat(y, 2, axis=1)
    return y[:frames].T.copy()


def _fnt(size: int, bold: bool = False):
    path = r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf"
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


def render_audit_png(summary: dict, out_ws: Path, out_art: Path) -> None:
    overall = summary["overall"]
    sub = summary["by_subtype"]
    headers = ["Metric (Dev Reverb, N = 106)", "Input", "P (F_W only)", "Pg (no R)", "Full (V4-4)", "A0 (SonicMaster)"]
    rows = [
        ("Official Reverb MSD ↓ (n=106)",) + tuple(f"{overall[b]['msd']:.4f}" for b in BRANCHES),
        ("  - small (n=" + str(sub["small"]["n"]) + ") MSD ↓",) + tuple(f"{sub['small'][b]['msd']:.4f}" for b in BRANCHES),
        ("  - big (n=" + str(sub["big"]["n"]) + ") MSD ↓",) + tuple(f"{sub['big'][b]['msd']:.4f}" for b in BRANCHES),
        ("  - mix (n=" + str(sub["mix"]["n"]) + ") MSD ↓",) + tuple(f"{sub['mix'][b]['msd']:.4f}" for b in BRANCHES),
        ("  - real (n=" + str(sub["real"]["n"]) + ") MSD ↓",) + tuple(f"{sub['real'][b]['msd']:.4f}" for b in BRANCHES),
        ("Audiobox PQ ↑ (n=106)",) + tuple(f"{overall[b]['pq']:.4f}" for b in BRANCHES),
        ("  dPQ vs A0 ↑",) + tuple(f"{overall[b]['pq'] - overall['A0']['pq']:+.4f}" for b in BRANCHES),
        ("Mel-SSIM ↑ (n=106)",) + tuple(f"{overall[b]['ssim']:.4f}" for b in BRANCHES),
        ("Mel-KL x 1000 ↓ (n=106)",) + tuple(f"{overall[b]['kl1000']:.4f}" for b in BRANCHES),
        ("Delta MSD vs Input (%)",) + tuple(
            f"{(overall[b]['msd'] - overall['Input']['msd']) / overall['Input']['msd'] * 100:+.1f}%" for b in BRANCHES
        ),
    ]

    col_w = [360, 175, 195, 185, 185, 210]
    pad_x, pad_y, row_h, hdr_h, title_h, note_h = 36, 32, 52, 58, 70, 56
    W = pad_x * 2 + sum(col_w)
    H = pad_y * 2 + title_h + hdr_h + len(rows) * row_h + note_h
    img = Image.new("RGB", (W, H), "#FFFFFF")
    d = ImageDraw.Draw(img)

    f_title, f_hdr, f_cell, f_bold, f_note = _fnt(26, True), _fnt(19, True), _fnt(18, False), _fnt(18, True), _fnt(15, False)
    d.text((W // 2, pad_y + 26), "Hybrid-v4 (V4-4) Reverb Branch Audit on Dev Reverb (N = 106)", fill="#0F172A", font=f_title, anchor="mm")

    y = pad_y + title_h
    x = pad_x
    d.rectangle([pad_x, y, W - pad_x, y + hdr_h], fill="#1E293B")
    for j, (h, cw) in enumerate(zip(headers, col_w)):
        d.text((x + 16 if j == 0 else x + cw // 2, y + hdr_h // 2), h, fill="#FFFFFF", font=f_hdr, anchor="lm" if j == 0 else "mm")
        x += cw

    # Best col per row across P, Pg, Full, A0 (cols 2..5)
    y += hdr_h
    for i, r in enumerate(rows):
        bg = "#F8FAFC" if i % 2 == 1 else "#FFFFFF"
        d.rectangle([pad_x, y, W - pad_x, y + row_h], fill=bg)
        d.line([(pad_x, y + row_h), (W - pad_x, y + row_h)], fill="#E2E8F0", width=1)
        x = pad_x
        for j, (val, cw) in enumerate(zip(r, col_w)):
            align = "lm" if j == 0 else "mm"
            tx = x + 16 if j == 0 else x + cw // 2
            font_use = f_bold if j == 0 and not val.startswith(" ") else f_cell
            d.text((tx, y + row_h // 2), val, fill="#0F172A", font=font_use, anchor=align)
            x += cw
        y += row_h

    d.rectangle([pad_x, pad_y + title_h, W - pad_x, y], outline="#94A3B8", width=2)
    diag = summary["phase_gate_diag"]
    d.text(
        (pad_x, y + 26),
        f"Band mean(g): 0-1k={diag['mean_g_0-1kHz']:.3f}, 1-4k={diag['mean_g_1-4kHz']:.3f}, 4-12k={diag['mean_g_4-12kHz']:.3f}, 12-22k={diag['mean_g_12-22kHz']:.3f} | Rel RMS vs |Zx|: |P-Zx|={diag['rel_dP']:.3f}, |g(S-P)|={diag['rel_gSP']:.3f}, |R|={diag['rel_R']:.3f}",
        fill="#475569",
        font=f_note,
        anchor="lm",
    )
    out_ws.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_ws)
    img.save(out_art)


def main() -> None:
    t0 = time.time()
    audit_dir = ROOT / "artifacts/10_extension/v4/reverb_branch_audit"
    spec_data = json.loads((audit_dir / "spectral_reverb.json").read_text(encoding="utf-8"))
    records = spec_data["records"]
    manifest_map = {it["sample_id"]: it for it in json.loads((ROOT / "artifacts/10_extension/v4/matched_dev_recheck/manifest.json").read_text(encoding="utf-8"))}

    pq_matched = json.loads((ROOT / "artifacts/10_extension/v4/matched_dev_recheck/pq.json").read_text(encoding="utf-8"))
    idx_matched = {sid: i for i, sid in enumerate(pq_matched["ids"])}
    pq_sweep = json.loads((ROOT / "artifacts/10_extension/v4/residual_sweep_dev/pq.json").read_text(encoding="utf-8"))
    idx_sweep = {sid: i for i, sid in enumerate(pq_sweep["ids"])}

    print(f"Computing official Modulation Spectrum Distance for {len(records)} Reverb clips across 5 branches...", flush=True)
    for rec in records:
        sid = rec["sample_id"]
        frames = int(rec["frames"])
        it = manifest_map[sid]
        c_mono = load_stereo_np(res(it["paths"]["clean"]), frames).mean(axis=0)
        x_mono = load_stereo_np(res(it["paths"]["input"]), frames).mean(axis=0)
        a0_mono = load_stereo_np(res(it["paths"]["A0"]), frames).mean(axis=0)
        p_mono = load_stereo_np(audit_dir / "wav/P" / f"{sid}.wav", frames).mean(axis=0)
        pg_mono = load_stereo_np(audit_dir / "wav/Pg" / f"{sid}.wav", frames).mean(axis=0)
        full_mono = load_stereo_np(audit_dir / "wav/Full" / f"{sid}.wav", frames).mean(axis=0)

        rec["msd"] = {
            "Input": float(attr_eval.modulation_spectrum_distance(c_mono, x_mono, fs=44100)),
            "P": float(attr_eval.modulation_spectrum_distance(c_mono, p_mono, fs=44100)),
            "Pg": float(attr_eval.modulation_spectrum_distance(c_mono, pg_mono, fs=44100)),
            "Full": float(attr_eval.modulation_spectrum_distance(c_mono, full_mono, fs=44100)),
            "A0": float(attr_eval.modulation_spectrum_distance(c_mono, a0_mono, fs=44100)),
        }

    print("Loading Audiobox Aesthetics predictor...", flush=True)
    predictor = initialize_predictor()

    def score_pq_paths(paths: list[Path]) -> list[float]:
        scores: list[float] = []
        for i in range(0, len(paths), 8):
            batch = [{"path": str(p)} for p in paths[i : i + 8]]
            scores.extend(float(x["PQ"]) for x in predictor.forward(batch))
            if (i + 8) % 40 == 0 or (i + 8) >= len(paths):
                print(f"  PQ scored {min(i + 8, len(paths))}/{len(paths)}", flush=True)
        return scores

    # Score P directly, and export frame-exact temp WAVs for Input to guarantee identical truncation
    p_paths = [audit_dir / "wav/P" / f"{r['sample_id']}.wav" for r in records]
    print("Scoring P PQ (106 clips)...", flush=True)
    pq_p = score_pq_paths(p_paths)

    with tempfile.TemporaryDirectory() as tmpd:
        inp_paths: list[Path] = []
        for r in records:
            sid = r["sample_id"]
            frames = int(r["frames"])
            arr = load_stereo_np(res(manifest_map[sid]["paths"]["input"]), frames)
            tp = Path(tmpd) / f"{sid}.wav"
            sf.write(str(tp), np.clip(arr.T, -1.0, 1.0).astype(np.float32), 44100, subtype="PCM_16")
            inp_paths.append(tp)
        print("Scoring Input PQ (106 clips)...", flush=True)
        pq_inp = score_pq_paths(inp_paths)

    # Spot-check 5 clips of Pg and Full vs cached r0 and r1 PQ
    spot_ids = [r["sample_id"] for r in records[:5]]
    spot_pg_fresh = score_pq_paths([audit_dir / "wav/Pg" / f"{sid}.wav" for sid in spot_ids])
    spot_full_fresh = score_pq_paths([audit_dir / "wav/Full" / f"{sid}.wav" for sid in spot_ids])
    max_diff_pg = max(abs(f - float(pq_sweep["scores"]["r0"][idx_sweep[sid]])) for sid, f in zip(spot_ids, spot_pg_fresh))
    max_diff_full = max(abs(f - float(pq_sweep["scores"]["r1"][idx_sweep[sid]])) for sid, f in zip(spot_ids, spot_full_fresh))
    print(f"Spot-check vs cached sweep PQ: max_diff_Pg={max_diff_pg:.4f}, max_diff_Full={max_diff_full:.4f}", flush=True)

    if max(max_diff_pg, max_diff_full) > 0.02:
        print("Spot-check > 0.02 -> scoring all 106 Pg and Full directly!", flush=True)
        pq_pg = score_pq_paths([audit_dir / "wav/Pg" / f"{r['sample_id']}.wav" for r in records])
        pq_full = score_pq_paths([audit_dir / "wav/Full" / f"{r['sample_id']}.wav" for r in records])
    else:
        pq_pg = [float(pq_sweep["scores"]["r0"][idx_sweep[r["sample_id"]]]) for r in records]
        pq_full = [float(pq_sweep["scores"]["r1"][idx_sweep[r["sample_id"]]]) for r in records]

    pq_a0 = [float(pq_matched["scores"]["A0"][idx_matched[r["sample_id"]]]) for r in records]

    for i, rec in enumerate(records):
        rec["pq"] = {
            "Input": pq_inp[i],
            "P": pq_p[i],
            "Pg": pq_pg[i],
            "Full": pq_full[i],
            "A0": pq_a0[i],
        }

    # Aggregate overall, by subtype, and single vs double+triple
    def agg(sub_recs: list[dict]) -> dict:
        out: dict = {"n": len(sub_recs)}
        for b in BRANCHES:
            out[b] = {
                "msd": float(np.mean([r["msd"][b] for r in sub_recs])),
                "pq": float(np.mean([r["pq"][b] for r in sub_recs])),
                "ssim": float(np.mean([r["ssim"][b] for r in sub_recs])),
                "kl1000": float(np.mean([r["kl1000"][b] for r in sub_recs])),
            }
        return out

    overall = agg(records)
    by_subtype = {rt: agg([r for r in records if rt in r["reverb_types"]]) for rt in REVERB_TYPES}
    by_complexity = {
        "single": agg([r for r in records if r["n_deg"] == 1]),
        "double_triple": agg([r for r in records if r["n_deg"] > 1]),
    }

    # Paired bootstrap CIs (source-clustered)
    ci_rows = []
    for r in records:
        ci_rows.append({
            "source_id": r["source_id"],
            "d_pq_P_vs_Inp": r["pq"]["P"] - r["pq"]["Input"],
            "d_pq_Pg_vs_P": r["pq"]["Pg"] - r["pq"]["P"],
            "d_pq_Full_vs_Pg": r["pq"]["Full"] - r["pq"]["Pg"],
            "d_pq_Full_vs_A0": r["pq"]["Full"] - r["pq"]["A0"],
            "d_pq_P_vs_A0": r["pq"]["P"] - r["pq"]["A0"],
            "d_msd_P_vs_Inp": r["msd"]["P"] - r["msd"]["Input"],
            "d_msd_Pg_vs_P": r["msd"]["Pg"] - r["msd"]["P"],
            "d_msd_Full_vs_Pg": r["msd"]["Full"] - r["msd"]["Pg"],
            "d_msd_Full_vs_A0": r["msd"]["Full"] - r["msd"]["A0"],
        })
    paired_cis = {}
    for key in [k for k in ci_rows[0] if k != "source_id"]:
        m, lo, hi = source_bootstrap_ci(ci_rows, key)
        paired_cis[key] = {"mean": m, "ci95": [lo, hi]}

    diag_keys = list(records[0]["diag"].keys())
    phase_gate_diag = {k: float(np.mean([r["diag"][k] for r in records])) for k in diag_keys}

    summary = {
        "n_reverb": len(records),
        "elapsed_sec": round(time.time() - t0, 1),
        "overall": overall,
        "by_subtype": by_subtype,
        "by_complexity": by_complexity,
        "paired_cis": paired_cis,
        "phase_gate_diag": phase_gate_diag,
        "spot_check_pq_max_diff": {"Pg_vs_r0": max_diff_pg, "Full_vs_r1": max_diff_full},
    }
    (audit_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Save per_sample.csv
    csv_path = audit_dir / "per_sample.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fn = [
            "sample_id", "source_id", "degradations", "reverb_types", "n_deg",
            *[f"msd_{b}" for b in BRANCHES],
            *[f"pq_{b}" for b in BRANCHES],
            *[f"ssim_{b}" for b in BRANCHES],
            *[f"kl_{b}" for b in BRANCHES],
            "rel_dP", "rel_gSP", "rel_R", "g_mean_M", "g_mean_S",
            "cos_PS_0-1kHz", "cos_PS_1-4kHz", "cos_PS_4-12kHz", "cos_PS_12-22kHz",
            "mean_g_0-1kHz", "mean_g_1-4kHz", "mean_g_4-12kHz", "mean_g_12-22kHz",
        ]
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        for r in records:
            row = {
                "sample_id": r["sample_id"],
                "source_id": r["source_id"],
                "degradations": "|".join(r["degradations"]),
                "reverb_types": "|".join(r["reverb_types"]),
                "n_deg": r["n_deg"],
            }
            for b in BRANCHES:
                row[f"msd_{b}"] = r["msd"][b]
                row[f"pq_{b}"] = r["pq"][b]
                row[f"ssim_{b}"] = r["ssim"][b]
                row[f"kl_{b}"] = r["kl1000"][b]
            for dk in ("rel_dP", "rel_gSP", "rel_R", "g_mean_M", "g_mean_S",
                       "cos_PS_0-1kHz", "cos_PS_1-4kHz", "cos_PS_4-12kHz", "cos_PS_12-22kHz",
                       "mean_g_0-1kHz", "mean_g_1-4kHz", "mean_g_4-12kHz", "mean_g_12-22kHz"):
                row[dk] = r["diag"][dk]
            w.writerow(row)

    # Export 6 representative listening bundles: 2 big, 2 real, 1 mix, 1 small with largest PQ(A0) - PQ(Full)
    list_dir = audit_dir / "listening"
    if list_dir.exists():
        shutil.rmtree(list_dir)
    list_dir.mkdir(parents=True, exist_ok=True)
    picked_ids: set[str] = set()
    picks: list[tuple[str, dict]] = []
    for rt, cnt in (("big", 2), ("real", 2), ("mix", 1), ("small", 1)):
        cands = [r for r in records if rt in r["reverb_types"] and r["sample_id"] not in picked_ids]
        cands.sort(key=lambda r: r["pq"]["A0"] - r["pq"]["Full"], reverse=True)
        for r in cands[:cnt]:
            picked_ids.add(r["sample_id"])
            picks.append((rt, r))

    bundle_meta = []
    for rank, (rt, r) in enumerate(picks, 1):
        sid = r["sample_id"]
        frames = int(r["frames"])
        it = manifest_map[sid]
        bdir = list_dir / f"{rank}_{sid}_{rt}"
        bdir.mkdir(parents=True, exist_ok=True)
        sources = {
            "0_clean.wav": res(it["paths"]["clean"]),
            "1_input.wav": res(it["paths"]["input"]),
            "2_P_only.wav": audit_dir / "wav/P" / f"{sid}.wav",
            "3_Pg_noR.wav": audit_dir / "wav/Pg" / f"{sid}.wav",
            "4_Full_v4_4.wav": audit_dir / "wav/Full" / f"{sid}.wav",
            "5_A0_sonicmaster.wav": res(it["paths"]["A0"]),
        }
        for out_name, src_p in sources.items():
            arr = load_stereo_np(src_p, frames)
            sf.write(str(bdir / out_name), np.clip(arr.T, -1.0, 1.0).astype(np.float32), 44100, subtype="PCM_16")
        bundle_meta.append({
            "rank": rank, "subtype": rt, "sample_id": sid, "degradations": r["degradations"],
            "msd": r["msd"], "pq": r["pq"], "folder": str(bdir),
        })
    (list_dir / "bundles.json").write_text(json.dumps(bundle_meta, indent=2), encoding="utf-8")

    # Write reverb_branch_audit_table.md
    md_lines = [
        "# Reverb Branch Audit (`Dev Reverb, N = 106`, Fixed Common Protocol)",
        "",
        f"- **Mô hình:** `V4-4` ([best.pt](file:///d:/SonicMaster_Reproducing/artifacts/10_extension/v4/kaggle_out_v4_4/out/v4_4/best.pt), Ep 20)",
        f"- **Manifest cố định:** [manifest.json](file:///d:/SonicMaster_Reproducing/artifacts/10_extension/v4/matched_dev_recheck/manifest.json)",
        f"- **Spot-check PQ vs cached `residual_sweep_dev`:** `max_diff(Pg vs r0) = {max_diff_pg:.4f}`, `max_diff(Full vs r1) = {max_diff_full:.4f}`",
        "",
        "## 1. Bảng So Sánh 5 Nhánh Toàn Tập Reverb (`N = 106`)",
        "",
        "| Metric | `Input` | `P` ($F_W(Z_x)$ only) | `Pg` (no $R$) | `Full` (`V4-4`) | `A0` (SonicMaster) |",
        "|---|---:|---:|---:|---:|---:|",
        f"| **Official Reverb MSD ↓** | `{overall['Input']['msd']:.4f}` | `{overall['P']['msd']:.4f}` | `{overall['Pg']['msd']:.4f}` | `{overall['Full']['msd']:.4f}` | `{overall['A0']['msd']:.4f}` |",
        f"| **Audiobox PQ ↑** | `{overall['Input']['pq']:.4f}` | `{overall['P']['pq']:.4f}` | `{overall['Pg']['pq']:.4f}` | `{overall['Full']['pq']:.4f}` | `{overall['A0']['pq']:.4f}` |",
        f"| **ΔPQ vs A0 ↑** | `{overall['Input']['pq']-overall['A0']['pq']:+.4f}` | `{overall['P']['pq']-overall['A0']['pq']:+.4f}` | `{overall['Pg']['pq']-overall['A0']['pq']:+.4f}` | `{overall['Full']['pq']-overall['A0']['pq']:+.4f}` | `0.0000` |",
        f"| **Mel-SSIM ↑** | `{overall['Input']['ssim']:.4f}` | `{overall['P']['ssim']:.4f}` | `{overall['Pg']['ssim']:.4f}` | `{overall['Full']['ssim']:.4f}` | `{overall['A0']['ssim']:.4f}` |",
        f"| **Mel-KL × 1000 ↓** | `{overall['Input']['kl1000']:.4f}` | `{overall['P']['kl1000']:.4f}` | `{overall['Pg']['kl1000']:.4f}` | `{overall['Full']['kl1000']:.4f}` | `{overall['A0']['kl1000']:.4f}` |",
        "",
        "## 2. Phân Tách Theo Loại Reverb (`small / big / mix / real`) & Độ Phức Tạp",
        "",
        "| Nhóm | `n` | MSD `Input` ↓ | MSD `P` ↓ | MSD `Pg` ↓ | MSD `Full` ↓ | MSD `A0` ↓ | PQ `Input` ↑ | PQ `P` ↑ | PQ `Pg` ↑ | PQ `Full` ↑ | PQ `A0` ↑ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, g in [
        ("small", by_subtype["small"]),
        ("big", by_subtype["big"]),
        ("mix", by_subtype["mix"]),
        ("real", by_subtype["real"]),
        ("single (n_deg=1)", by_complexity["single"]),
        ("double+triple (n_deg>1)", by_complexity["double_triple"]),
    ]:
        md_lines.append(
            f"| **{label}** | {g['n']} | "
            + " | ".join(f"`{g[b]['msd']:.4f}`" for b in BRANCHES) + " | "
            + " | ".join(f"`{g[b]['pq']:.4f}`" for b in BRANCHES) + " |"
        )

    md_lines += [
        "",
        "## 3. Source-Clustered Paired Bootstrap 95% CI (`N = 106` Reverb)",
        "",
        "| So sánh | `Mean Δ` | `95% CI` |",
        "|---|---:|---|",
    ]
    for k, v in paired_cis.items():
        md_lines.append(f"| `{k}` | `{v['mean']:+.4f}` | `[{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]` |")

    md_lines += [
        "",
        "## 4. Chẩn Đoán Pha & Gate Theo Dải Tần (Trung Bình `106` Reverb Clips)",
        "",
        "| Dải tần | $\\cos(P, S)$ | $|0.5P+0.5S|/(0.5|P|+0.5|S|)$ | `mean(g)` |",
        "|---|---:|---:|---:|",
    ]
    for bname in ("0-1kHz", "1-4kHz", "4-12kHz", "12-22kHz"):
        md_lines.append(
            f"| `{bname}` | `{phase_gate_diag[f'cos_PS_{bname}']:+.3f}` | `{phase_gate_diag[f'cancel50_{bname}']:.3f}` | `{phase_gate_diag[f'mean_g_{bname}']:.3f}` |"
        )
    md_lines += [
        "",
        f"- **Relative RMS vs $|Z_x|$:** $|P - Z_x| = {phase_gate_diag['rel_dP']:.3f}$, $|g(S - P)| = {phase_gate_diag['rel_gSP']:.3f}$, $|R| = {phase_gate_diag['rel_R']:.3f}$",
        "",
        "## 5. 6 Bộ Nghe Đối Chứng Đại Diện (`listening/`)",
        "",
        "| # | Sample ID | Subtype | Degradations | MSD (`Input → P → Full → A0`) | PQ (`Input → P → Full → A0`) |",
        "|---:|---|---|---|---|---|",
    ]
    for bm in bundle_meta:
        m_s = f"{bm['msd']['Input']:.4f} → {bm['msd']['P']:.4f} → {bm['msd']['Full']:.4f} → {bm['msd']['A0']:.4f}"
        p_s = f"{bm['pq']['Input']:.2f} → {bm['pq']['P']:.2f} → {bm['pq']['Full']:.2f} → {bm['pq']['A0']:.2f}"
        md_lines.append(f"| {bm['rank']} | `{bm['sample_id']}` | `{bm['subtype']}` | `{'|'.join(bm['degradations'])}` | {m_s} | {p_s} |")

    md_path = audit_dir / "reverb_branch_audit_table.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    png_ws = ROOT / "artifacts/10_extension/figures/v4_reverb_branch_audit.png"
    png_art = BRAIN_ART / "v4_reverb_branch_audit.png"
    render_audit_png(summary, png_ws, png_art)
    print(f"ALL DONE in {time.time() - t0:.1f}s -> {md_path}", flush=True)


if __name__ == "__main__":
    main()
