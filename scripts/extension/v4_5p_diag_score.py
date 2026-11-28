"""Step 2 of V4-5p Diagnostic: Score Reverb Oracle WAVs (Audiobox PQ + Official MSD), aggregate stats, & render report."""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from audiobox_aesthetics.infer import initialize_predictor
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(r"d:\SonicMaster_Reproducing")
GDRIVE = Path(r"G:\My Drive\SonicMaster_Reproducing")
BRAIN_ART = Path(r"C:\Users\ADMIN\.gemini\antigravity\brain\bd54de32-517b-4715-954e-8096d7c41b00")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "SonicMaster" / "evaluation"))

import evaluate_control_multiple_degs_mass as attr_eval  # noqa: E402

CHECKPOINTS = ROOT / "tmp/rescore_failure_segments"
SR = 44100
DIAG_BANDS = ("0-1kHz", "1-4kHz", "4-12kHz", "12-22kHz", "bright_6-22kHz")
ORACLE_ARMS = ("Input", "default", "ph_S_midhi", "ph_S_all", "ph_S_all_noR", "mag_S_ph_P", "mag_hi_ph_S", "A0")
ORACLE_LABELS = {
    "Input": "Input (degraded)",
    "default": "V4-5p (default)",
    "ph_S_midhi": "ph_S_midhi (1-22k)",
    "ph_S_all": "ph_S_all (g_ph=1)",
    "ph_S_all_noR": "ph_S_all (noR)",
    "mag_S_ph_P": "mag_S_ph_P (noR)",
    "mag_hi_ph_S": "mag_hi>=0.7+ph_S",
    "A0": "A0 (SonicMaster)",
}
BRIGHT_ARMS = ("Input", "A0", "V4-4", "P_only", "noR", "default", "r1_0", "r_hf1_0")
BRIGHT_LABELS = {
    "Input": "Input (degraded)",
    "A0": "A0 (SonicMaster)",
    "V4-4": "V4-4 (prev baseline)",
    "P_only": "V4-5p P only",
    "noR": "V4-5p noR (r=0)",
    "default": "V4-5p default (r=0.5)",
    "r1_0": "V4-5p (r=1.0 all)",
    "r_hf1_0": "V4-5p (r_hf=1.0 >=6k)",
}


def res(p: str | Path) -> Path:
    dp = ROOT / p
    if dp.is_file():
        return dp
    gp = GDRIVE / p
    if gp.is_file():
        return gp
    raise FileNotFoundError(f"Missing required audio file on both D: and G: -> {p}")


def load_stereo_chfirst(path: Path, frames: int) -> np.ndarray:
    y, rate = sf.read(io.BytesIO(path.read_bytes()), dtype="float32", always_2d=True)
    assert rate == SR and len(y) >= frames, f"Extent mismatch for {path}"
    if y.shape[1] == 1:
        y = np.repeat(y, 2, axis=1)
    return y[:frames].T.copy()


def _fnt(size: int, bold: bool = False):
    path = r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf"
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


def render_diagnostic_png(summary: dict, out_ws: Path, out_art: Path) -> None:
    W = 1480
    pad_x = 34
    table_w = W - 2 * pad_x

    # Build table definitions: (section_title, headers, col_w, rows, highlight_fn)
    # 1. Band-wise gating table
    t1_cols = [200, 155, 130, 130, 180, 180, 145, 292]  # sum = 1412
    assert sum(t1_cols) == table_w
    t1_hdrs = ["Group (n)", "Band", "g_mag", "g_ph", "g_mag (M / S)", "g_ph (M / S)", "cos(P, S)", "rel_R (band)"]
    t1_rows: list[tuple[str, ...]] = []
    for gname, gdata in summary["band_summary"].items():
        gn = f"{gname} ({gdata['n']})"
        for b_idx, bname in enumerate(DIAG_BANDS):
            bd = gdata["bands"][bname]
            t1_rows.append(
                (
                    gn if b_idx == 0 else "",
                    bname,
                    f"{bd['g_mag']:.3f}",
                    f"{bd['g_ph']:.3f}",
                    f"{bd['g_mag_M']:.3f} / {bd['g_mag_S']:.3f}",
                    f"{bd['g_ph_M']:.3f} / {bd['g_ph_S']:.3f}",
                    f"{bd['cos_PS']:+.3f}",
                    f"{bd['rel_R']:.3f}",
                )
            )

    # 2. Stratified Reverb Oracle table
    t2_cols = [282, 165, 165, 195, 185, 200, 220]  # sum = 1412
    assert sum(t2_cols) == table_w
    t2_hdrs = ["Oracle Arm (N=24 Reverb)", "MSD ↓", "PQ ↑", "dPQ vs default", "dPQ vs A0", "Mel-SSIM ↑", "Mel-KLx1000 ↓"]
    ora = summary["reverb_oracle"]["arms"]
    def_pq = ora["default"]["pq"]
    a0_pq = ora["A0"]["pq"]
    best_ora_pq = max(ora[k]["pq"] for k in ("ph_S_midhi", "ph_S_all", "ph_S_all_noR", "mag_S_ph_P", "mag_hi_ph_S"))
    t2_rows: list[tuple[str, ...]] = []
    for k in ORACLE_ARMS:
        d = ora[k]
        dpq_def = f"{d['pq'] - def_pq:+.4f}" if k != "default" else "0.0000"
        dpq_a0 = f"{d['pq'] - a0_pq:+.4f}" if k != "A0" else "0.0000"
        t2_rows.append(
            (
                ORACLE_LABELS[k],
                f"{d['msd']:.4f}",
                f"{d['pq']:.4f}",
                dpq_def,
                dpq_a0,
                f"{d['ssim']:.4f}",
                f"{d['kl1000']:.4f}",
            )
        )

    # 3. bright (n=19) signed-error & attribution table
    t3_cols = [302, 180, 190, 250, 220, 270]  # sum = 1412
    assert sum(t3_cols) == table_w
    t3_hdrs = ["bright (n=19, 6-22 kHz)", "Mean Ratio", "Signed Err (Y-C)", "Too Bright / Too Dark", "Mean |Err|", "Norm Det vs A0"]
    br = summary["bright_attribution"]["arms"]
    t3_rows: list[tuple[str, ...]] = []
    for k in BRIGHT_ARMS:
        d = br[k]
        t3_rows.append(
            (
                BRIGHT_LABELS[k],
                f"{d['mean_ratio']:.5f}",
                f"{d['mean_signed_err']:+.5f}",
                f"{d['n_too_bright']}/19 ({d['pct_too_bright']:.0f}%) / {d['n_too_dark']}/19",
                f"{d['mean_abs_err']:.5f}",
                f"{d['norm_det_vs_A0']:+.4f}" if k != "A0" else "0.0000 (ref)",
            )
        )

    clean_br = summary["bright_attribution"]["mean_clean_ratio"]
    n_inp_bright = summary["bright_attribution"]["input_split"]["n_input_too_bright"]
    n_inp_dark = summary["bright_attribution"]["input_split"]["n_input_too_dark"]

    title_h = 74
    sec_hdr_h = 40
    tbl_hdr_h = 44
    r1_h, r2_h, r3_h = 31, 40, 38
    gap_h = 26
    footer_h = 56
    H = (
        28
        + title_h
        + sec_hdr_h + tbl_hdr_h + len(t1_rows) * r1_h + gap_h
        + sec_hdr_h + tbl_hdr_h + len(t2_rows) * r2_h + gap_h
        + sec_hdr_h + tbl_hdr_h + len(t3_rows) * r3_h + gap_h
        + footer_h
    )

    img = Image.new("RGB", (W, H), "#FFFFFF")
    d = ImageDraw.Draw(img)
    f_title = _fnt(25, True)
    f_sec = _fnt(19, True)
    f_hdr = _fnt(16, True)
    f_cell = _fnt(15, False)
    f_bold = _fnt(15, True)
    f_note = _fnt(14, False)

    d.text((W // 2, 44), "V4-5p Diagnostic: Band-Wise Gate (N=280), Reverb Oracle (N=24) & bright Attribution (n=19)", fill="#0F172A", font=f_title, anchor="mm")
    d.text((W // 2, 74), f"Fixed Common Protocol | Parity max_abs_diff={summary['parity']['gpu_forward_max_abs_diff']:.2e} | PQ max_diff={summary['parity']['pq_vs_csv_max_diff']:.4f}", fill="#475569", font=f_note, anchor="mm")

    y = 28 + title_h

    def draw_table(sec_title: str, hdrs: list[str], col_w: list[int], rows: list[tuple[str, ...]], row_h: int, row_tint_fn=None):
        nonlocal y
        d.text((pad_x, y + 18), sec_title, fill="#1E293B", font=f_sec, anchor="lm")
        y += sec_hdr_h
        top_y = y
        x = pad_x
        d.rectangle([pad_x, y, W - pad_x, y + tbl_hdr_h], fill="#1E293B")
        for j, (h, cw) in enumerate(zip(hdrs, col_w)):
            d.text((x + 12 if j == 0 else x + cw // 2, y + tbl_hdr_h // 2), h, fill="#FFFFFF", font=f_hdr, anchor="lm" if j == 0 else "mm")
            x += cw
        y += tbl_hdr_h
        for i, row in enumerate(rows):
            bg = "#F8FAFC" if i % 2 == 1 else "#FFFFFF"
            if row_tint_fn is not None:
                bg = row_tint_fn(i, row, bg)
            d.rectangle([pad_x, y, W - pad_x, y + row_h], fill=bg)
            d.line([(pad_x, y + row_h), (W - pad_x, y + row_h)], fill="#E2E8F0", width=1)
            x = pad_x
            for j, (val, cw) in enumerate(zip(row, col_w)):
                tx = x + 12 if j == 0 else x + cw // 2
                fnt = f_bold if (j == 0 and val != "") else f_cell
                d.text((tx, y + row_h // 2), val, fill="#0F172A", font=fnt, anchor="lm" if j == 0 else "mm")
                x += cw
            y += row_h
        d.rectangle([pad_x, top_y, W - pad_x, y], outline="#94A3B8", width=2)
        y += gap_h

    def t1_tint(i: int, row: tuple[str, ...], default_bg: str) -> str:
        if row[1] == "bright_6-22kHz":
            return "#FFFBEB"
        return "#F1F5F9" if (i // 5) % 2 == 1 else "#FFFFFF"

    def t2_tint(i: int, row: tuple[str, ...], default_bg: str) -> str:
        k = ORACLE_ARMS[i]
        if k == "default":
            return "#EFF6FF"
        if k != "A0" and k != "Input" and abs(ora[k]["pq"] - best_ora_pq) < 1e-9:
            return "#ECFDF5"
        if k == "A0":
            return "#F1F5F9"
        return default_bg

    def t3_tint(i: int, row: tuple[str, ...], default_bg: str) -> str:
        k = BRIGHT_ARMS[i]
        if k == "default":
            return "#FEF2F2"
        if k == "A0":
            return "#F1F5F9"
        return default_bg

    draw_table("1. Band-Wise Decoupled Gates (g_mag vs g_ph), Phase Cosine & Relative Residual across Dev (N = 280)", t1_hdrs, t1_cols, t1_rows, r1_h, t1_tint)
    draw_table("2. Stratified Reverb Oracle (N = 24: 6 small, 6 big, 6 mix, 6 real) — Phase vs Magnitude Teacher Overrides", t2_hdrs, t2_cols, t2_rows, r2_h, t2_tint)
    draw_table("3. bright Guard Failure Attribution (n = 19, 6,000-22,050 Hz) — Signed Energy Ratio Error", t3_hdrs, t3_cols, t3_rows, r3_h, t3_tint)

    d.text(
        (pad_x, y + 14),
        f"Clean bright mean ratio = {clean_br:.5f} | Input split: {n_inp_bright}/19 too bright, {n_inp_dark}/19 too dark | "
        f"bright mean |err| default={br['default']['mean_abs_err']:.4f} (norm={br['default']['norm_det_vs_A0']:+.4f})",
        fill="#334155",
        font=f_note,
        anchor="lm",
    )

    out_ws.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_ws)
    out_art.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_art)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-render", action="store_true", help="Render PNG directly from existing summary.json")
    args = ap.parse_args()

    t0 = time.time()
    diag_dir = ROOT / "artifacts/10_extension/v4/v4_5p_diagnostic"
    png_ws = ROOT / "artifacts/10_extension/figures/v4_5p_diagnostic.png"
    png_art = BRAIN_ART / "v4_5p_diagnostic.png"
    if args.only_render:
        summary = json.loads((diag_dir / "summary.json").read_text(encoding="utf-8"))
        render_diagnostic_png(summary, png_ws, png_art)
        print(f"RENDERED {png_ws} & {png_art} in {time.time() - t0:.2f}s", flush=True)
        return

    diag_gpu = json.loads((diag_dir / "diag_gpu.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "artifacts/10_extension/v4/matched_dev_recheck/manifest.json").read_text(encoding="utf-8"))
    manifest_map = {it["sample_id"]: it for it in manifest}
    pq_cached = json.loads((ROOT / "artifacts/10_extension/v4/matched_dev_recheck/pq.json").read_text(encoding="utf-8"))
    idx_pq = {sid: i for i, sid in enumerate(pq_cached["ids"])}

    v45p_csv: dict[str, float] = {}
    with (ROOT / "artifacts/10_extension/v4/matched_dev_v4_5p/per_sample.csv").open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            v45p_csv[row["sample_id"]] = float(row["pq"])

    oracle_recs = diag_gpu["oracle_reverb_ids"]
    assert len(oracle_recs) == 24

    # 1. Official Reverb MSD across all 8 arms on the 24 stratified Reverb clips
    print(f"Computing Official Reverb MSD for {len(oracle_recs)} oracle clips across 8 arms...", flush=True)
    for rec in oracle_recs:
        sid = rec["sample_id"]
        frames = int(rec["frames"])
        it = manifest_map[sid]
        c_mono = load_stereo_chfirst(res(it["paths"]["clean"]), frames).mean(axis=0)
        x_mono = load_stereo_chfirst(res(it["paths"]["input"]), frames).mean(axis=0)
        a0_mono = load_stereo_chfirst(res(it["paths"]["A0"]), frames).mean(axis=0)
        msd_map = {
            "Input": float(attr_eval.modulation_spectrum_distance(c_mono, x_mono, fs=SR)),
            "A0": float(attr_eval.modulation_spectrum_distance(c_mono, a0_mono, fs=SR)),
        }
        for mode in ("default", "ph_S_midhi", "ph_S_all", "ph_S_all_noR", "mag_S_ph_P", "mag_hi_ph_S"):
            m_mono = load_stereo_chfirst(diag_dir / "oracle_wav" / mode / f"{sid}.wav", frames).mean(axis=0)
            msd_map[mode] = float(attr_eval.modulation_spectrum_distance(c_mono, m_mono, fs=SR))
        rec["msd"] = msd_map

    # 2. Audiobox PQ scoring for Input + 6 oracle modes (A0 pulled from pq.json cache)
    print("Loading Audiobox Aesthetics predictor...", flush=True)
    predictor = initialize_predictor(str(CHECKPOINTS / "audiobox_checkpoint.pt"))

    def score_pq_arrs(items: list[tuple[Path, int]]) -> list[float]:
        scores: list[float] = []
        for i in range(0, len(items), 4):
            batch = []
            for p, fr in items[i : i + 4]:
                arr = load_stereo_chfirst(p, fr)
                assert arr.shape == (2, fr)
                batch.append({"path": torch.from_numpy(arr.astype(np.float32)), "sample_rate": SR})
            scores.extend(float(x["PQ"]) for x in predictor.forward(batch))
        return scores

    inp_pq = score_pq_arrs([(res(manifest_map[r["sample_id"]]["paths"]["input"]), int(r["frames"])) for r in oracle_recs])
    mode_pqs: dict[str, list[float]] = {"Input": inp_pq}
    for mode in ("default", "ph_S_midhi", "ph_S_all", "ph_S_all_noR", "mag_S_ph_P", "mag_hi_ph_S"):
        print(f"  Scoring PQ for oracle mode {mode} (24 clips)...", flush=True)
        mode_pqs[mode] = score_pq_arrs([(diag_dir / "oracle_wav" / mode / f"{r['sample_id']}.wav", int(r["frames"])) for r in oracle_recs])
    mode_pqs["A0"] = [float(pq_cached["scores"]["A0"][idx_pq[r["sample_id"]]]) for r in oracle_recs]

    pq_max_diff = 0.0
    for i, rec in enumerate(oracle_recs):
        sid = rec["sample_id"]
        rec["pq"] = {arm: mode_pqs[arm][i] for arm in ORACLE_ARMS}
        diff = abs(rec["pq"]["default"] - v45p_csv[sid])
        pq_max_diff = max(pq_max_diff, diff)
    print(f"PQ parity check default vs matched_dev_v4_5p/per_sample.csv max_abs_diff = {pq_max_diff:.5f}", flush=True)
    assert pq_max_diff < 0.01, f"PQ parity diff exceeded 0.01: {pq_max_diff}"

    ora_summary = {}
    for arm in ORACLE_ARMS:
        ora_summary[arm] = {
            "msd": float(np.mean([r["msd"][arm] for r in oracle_recs])),
            "pq": float(np.mean([r["pq"][arm] for r in oracle_recs])),
            "ssim": float(np.mean([r["ssim"][arm] for r in oracle_recs])),
            "kl1000": float(np.mean([r["kl1000"][arm] for r in oracle_recs])),
        }

    # By subtype breakdown for key oracle arms
    ora_by_subtype = {}
    for rt in ("small", "big", "mix", "real"):
        sub = [r for r in oracle_recs if r["primary_subtype"] == rt]
        ora_by_subtype[rt] = {
            "n": len(sub),
            "arms": {
                arm: {
                    "msd": float(np.mean([r["msd"][arm] for r in sub])),
                    "pq": float(np.mean([r["pq"][arm] for r in sub])),
                }
                for arm in ORACLE_ARMS
            },
        }

    # 3. Aggregate Band-Wise (g_mag, g_ph, cos_PS, rel_R) across All / Reverb / Non-Reverb / bright
    all_clips = diag_gpu["band_stats_per_clip"]
    groups = {
        "All": all_clips,
        "Reverb": [c for c in all_clips if c["is_reverb"]],
        "Non-Reverb": [c for c in all_clips if not c["is_reverb"]],
        "bright": [c for c in all_clips if c["is_bright"]],
    }
    band_summary: dict = {}
    for gname, clips in groups.items():
        bands_d = {}
        for bname in DIAG_BANDS:
            bands_d[bname] = {
                m: float(np.mean([c["diag"][f"{m}_{bname}"] for c in clips]))
                for m in ("g_mag", "g_ph", "g_mag_M", "g_mag_S", "g_ph_M", "g_ph_S", "cos_PS", "rel_R")
            }
        band_summary[gname] = {
            "n": len(clips),
            "rel_dP": float(np.mean([c["diag"]["rel_dP"] for c in clips])),
            "rel_R": float(np.mean([c["diag"]["rel_R"] for c in clips])),
            "bands": bands_d,
        }

    # 4. Aggregate bright (n=19) signed-error & branch attribution
    bright_recs = diag_gpu["bright_records"]
    assert len(bright_recs) == 19
    inp_err_b = float(np.mean([r["arms"]["Input"]["abs_err"] for r in bright_recs]))
    a0_err_b = float(np.mean([r["arms"]["A0"]["abs_err"] for r in bright_recs]))
    def_err_b = float(np.mean([r["arms"]["default"]["abs_err"] for r in bright_recs]))
    def_norm_b = (def_err_b - a0_err_b) / inp_err_b
    print(f"bright parity check: input={inp_err_b:.5f}, A0={a0_err_b:.5f}, default={def_err_b:.5f}, norm={def_norm_b:+.5f}", flush=True)
    assert abs(def_err_b - 0.0083) < 0.0005, f"bright default abs_err {def_err_b} deviates from 0.0083"

    bright_arms_summary: dict = {}
    for ak in BRIGHT_ARMS:
        ratios = [r["arms"][ak]["ratio"] for r in bright_recs]
        serrs = [r["arms"][ak]["signed_err"] for r in bright_recs]
        aerrs = [r["arms"][ak]["abs_err"] for r in bright_recs]
        n_bright = sum(1 for se in serrs if se > 0)
        n_dark = sum(1 for se in serrs if se < 0)
        mean_ae = float(np.mean(aerrs))
        bright_arms_summary[ak] = {
            "mean_ratio": float(np.mean(ratios)),
            "mean_signed_err": float(np.mean(serrs)),
            "median_signed_err": float(np.median(serrs)),
            "n_too_bright": n_bright,
            "n_too_dark": n_dark,
            "pct_too_bright": 100.0 * n_bright / len(bright_recs),
            "pct_too_dark": 100.0 * n_dark / len(bright_recs),
            "mean_abs_err": mean_ae,
            "norm_det_vs_A0": (mean_ae - a0_err_b) / inp_err_b,
        }

    # Split bright clips by input direction (too bright vs too dark)
    inp_too_bright = [r for r in bright_recs if r["arms"]["Input"]["signed_err"] > 0]
    inp_too_dark = [r for r in bright_recs if r["arms"]["Input"]["signed_err"] <= 0]

    def split_stats(sub: list[dict]) -> dict:
        if not sub:
            return {"n": 0}
        return {
            "n": len(sub),
            "mean_clean_ratio": float(np.mean([r["ratio_clean"] for r in sub])),
            "arms": {
                ak: {
                    "mean_ratio": float(np.mean([r["arms"][ak]["ratio"] for r in sub])),
                    "mean_signed_err": float(np.mean([r["arms"][ak]["signed_err"] for r in sub])),
                    "mean_abs_err": float(np.mean([r["arms"][ak]["abs_err"] for r in sub])),
                }
                for ak in BRIGHT_ARMS
            },
        }

    bright_summary = {
        "n": len(bright_recs),
        "mean_clean_ratio": float(np.mean([r["ratio_clean"] for r in bright_recs])),
        "arms": bright_arms_summary,
        "input_split": {
            "n_input_too_bright": len(inp_too_bright),
            "n_input_too_dark": len(inp_too_dark),
            "too_bright": split_stats(inp_too_bright),
            "too_dark": split_stats(inp_too_dark),
        },
    }

    summary = {
        "elapsed_sec": round(time.time() - t0 + diag_gpu["elapsed_sec"], 1),
        "parity": {
            "gpu_forward_max_abs_diff": diag_gpu["parity_max_abs_diff"],
            "pq_vs_csv_max_diff": pq_max_diff,
            "bright_default_abs_err": def_err_b,
            "bright_default_norm": def_norm_b,
        },
        "band_summary": band_summary,
        "reverb_oracle": {
            "n": len(oracle_recs),
            "arms": ora_summary,
            "by_subtype": ora_by_subtype,
            "per_clip": oracle_recs,
        },
        "bright_attribution": bright_summary,
    }
    (diag_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # 5. Write Markdown Table Report
    md = [
        "# Chẩn Đoán `V4-5p`: Gate Pha/Biên Độ (`N=280`), Oracle Reverb (`N=24`) & Dấu Sai Số `bright` (`n=19`)",
        "",
        f"- **Checkpoint:** [best.pt](file:///d:/SonicMaster_Reproducing/artifacts/10_extension/v4/kaggle_out_v4_5p/out/v4_5p/best.pt)",
        f"- **Manifest cố định:** [manifest.json](file:///d:/SonicMaster_Reproducing/artifacts/10_extension/v4/matched_dev_recheck/manifest.json)",
        f"- **Kiểm tra khớp chuẩn (Parity):** `GPU forward_track max_abs_diff = {diag_gpu['parity_max_abs_diff']:.2e}`, `Audiobox PQ vs CSV max_diff = {pq_max_diff:.5f}`, `bright |err| = {def_err_b:.4f} (norm = {def_norm_b:+.4f})`",
        "",
        "## 1. Phân Bố Gate Biên Độ (`g_mag`), Gate Pha (`g_ph`), Cosine Pha & Residual Theo Dải Tần (`N = 280` Dev)",
        "",
        "| Nhóm (`n`) | Dải tần | `g_mag` | `g_ph` | `g_mag` (`M / S`) | `g_ph` (`M / S`) | $\\cos(P, S)$ | `rel_R` (band) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for gname, gdata in band_summary.items():
        for bi, bname in enumerate(DIAG_BANDS):
            bd = gdata["bands"][bname]
            g_col = f"**{gname}** (`{gdata['n']}`)" if bi == 0 else ""
            md.append(
                f"| {g_col} | `{bname}` | `{bd['g_mag']:.3f}` | `{bd['g_ph']:.3f}` | `{bd['g_mag_M']:.3f} / {bd['g_mag_S']:.3f}` | `{bd['g_ph_M']:.3f} / {bd['g_ph_S']:.3f}` | `{bd['cos_PS']:+.3f}` | `{bd['rel_R']:.3f}` |"
            )

    md += [
        "",
        "## 2. Stratified Reverb Oracle (`N = 24`: `6 small`, `6 big`, `6 mix`, `6 real`)",
        "",
        "| Nhánh Oracle | Official MSD ↓ | Audiobox PQ ↑ | ΔPQ vs `default` | ΔPQ vs `A0` | Mel-SSIM ↑ | Mel-KL×1000 ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for k in ORACLE_ARMS:
        d = ora_summary[k]
        dp_def = "`0.0000`" if k == "default" else f"`{d['pq'] - ora_summary['default']['pq']:+.4f}`"
        dp_a0 = "`0.0000`" if k == "A0" else f"`{d['pq'] - ora_summary['A0']['pq']:+.4f}`"
        md.append(
            f"| **{ORACLE_LABELS[k]}** | `{d['msd']:.4f}` | `{d['pq']:.4f}` | {dp_def} | {dp_a0} | `{d['ssim']:.4f}` | `{d['kl1000']:.4f}` |"
        )

    md += [
        "",
        "### Phân tách Oracle `PQ` theo loại Reverb (`n = 6` mỗi loại)",
        "",
        "| Loại | `Input` | `default` | `ph_S_midhi` | `ph_S_all` | `ph_S_all_noR` | `mag_S_ph_P` | `mag_hi_ph_S` | `A0` |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rt in ("small", "big", "mix", "real"):
        sub_a = ora_by_subtype[rt]["arms"]
        vals = " | ".join(f"`{sub_a[k]['pq']:.3f}`" for k in ORACLE_ARMS)
        md.append(f"| **`{rt}`** (`6`) | {vals} |")

    md += [
        "",
        f"## 3. Phân Tích Dấu Sai Số & Quy Nhân `bright` (`n = 19`, `6,000–22,050 Hz`, Clean Mean Ratio = `{bright_summary['mean_clean_ratio']:.5f}`)",
        "",
        "| Nhánh | Mean Ratio | Signed Err (`Y - Clean`) | `Too Bright / Too Dark` | Mean `|Err|` | Norm Det vs `A0` |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for k in BRIGHT_ARMS:
        d = bright_arms_summary[k]
        nd = "`0.0000` (ref)" if k == "A0" else f"**`{d['norm_det_vs_A0']:+.4f}`**"
        md.append(
            f"| **{BRIGHT_LABELS[k]}** | `{d['mean_ratio']:.5f}` | `{d['mean_signed_err']:+.5f}` | `{d['n_too_bright']}/19 ({d['pct_too_bright']:.0f}%)` / `{d['n_too_dark']}/19` | `{d['mean_abs_err']:.5f}` | {nd} |"
        )

    sp = bright_summary["input_split"]
    md += [
        "",
        "### Tách theo chiều méo ban đầu của Input",
        "",
        "| Nhóm Input | `n` | Clean Ratio | Input Signed Err | A0 Signed Err | `P_only` Signed Err | `V4-5p default` Signed Err | `V4-5p r=1.0` Signed Err | `V4-5p r_hf=1.0` Signed Err |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, sd in (("Input quá chói (Too Bright)", sp["too_bright"]), ("Input bị tối (Too Dark)", sp["too_dark"])):
        if sd["n"] == 0:
            continue
        sa = sd["arms"]
        md.append(
            f"| **{label}** | {sd['n']} | `{sd['mean_clean_ratio']:.5f}` | `{sa['Input']['mean_signed_err']:+.5f}` | `{sa['A0']['mean_signed_err']:+.5f}` | `{sa['P_only']['mean_signed_err']:+.5f}` | `{sa['default']['mean_signed_err']:+.5f}` | `{sa['r1_0']['mean_signed_err']:+.5f}` | `{sa['r_hf1_0']['mean_signed_err']:+.5f}` |"
        )

    md_path = diag_dir / "v4_5p_diagnostic_table.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    png_ws = ROOT / "artifacts/10_extension/figures/v4_5p_diagnostic.png"
    png_art = BRAIN_ART / "v4_5p_diagnostic.png"
    render_diagnostic_png(summary, png_ws, png_art)
    print(f"ALL DONE in {time.time() - t0:.1f}s -> {md_path} & {png_ws}", flush=True)


if __name__ == "__main__":
    main()
