"""Phase 0 of V4-7: Zero-Dev-snooping Pilot Val (N=560, 80 sources) acoustic reverb separability check.

Extracts 6 length-invariant 2.0s-window cues across 24 psychoacoustic bands strictly from blind (x, s) pairs
and measures 5-fold source-grouped CV AUROC, Recall @ FPR <= 3%, per-subtype recall, and per-family FPR.
"""
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
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(r"d:\SonicMaster_Reproducing")
GDRIVE = Path(r"G:\My Drive\SonicMaster_Reproducing")
sys.path.insert(0, str(ROOT))

from scripts.extension.v4_alignment import align_teacher_to_input  # noqa: E402
from scripts.extension.v4_models import BAND_WIDTHS, compute_track_norm, lr_to_ms, stft_ms  # noqa: E402

REVERB_TYPES = ("small", "big", "mix", "real")
REVERB_SET = set(REVERB_TYPES)
SR = 44100


def res(p: str | Path) -> Path:
    dp = ROOT / p
    if dp.is_file():
        return dp
    gp = GDRIVE / p
    if gp.is_file():
        return gp
    raise FileNotFoundError(f"Missing required file on both D: and G: -> {p}")


class WindowedModCueExtractor(nn.Module):
    """Extract 6 length-invariant blind acoustic cues per 24 bands on fixed 2.0s sliding windows."""

    def __init__(self, band_slices: list[tuple[int, int]], win_frames: int = 172, stride: int = 86) -> None:
        super().__init__()
        self.band_slices = band_slices
        self.win_frames = win_frames
        self.stride = stride
        self.register_buffer("mod_win", torch.hann_window(win_frames, periodic=True), persistent=False)

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, z_x: torch.Tensor, z_s: torch.Tensor) -> torch.Tensor:
        """z_x, z_s: complex (B, 2, F=1025, T) -> float32 (B, 24, 6)."""
        zx = z_x.to(torch.complex64)
        zs = z_s.to(torch.complex64)
        bsz, _, _, n_frames = zx.shape

        amp_x_list, amp_s_list = [], []
        ex_list, es_list = [], []
        px_conj_ps_re, px_abs_ps_abs = [], []
        for s_b, e_b in self.band_slices:
            xb = zx[:, :, s_b:e_b, :]
            sb = zs[:, :, s_b:e_b, :]
            ex_b = (xb.real**2 + xb.imag**2).mean(dim=(1, 2))
            es_b = (sb.real**2 + sb.imag**2).mean(dim=(1, 2))
            ex_list.append(ex_b)
            es_list.append(es_b)
            amp_x_list.append(torch.sqrt(ex_b + 1e-8))
            amp_s_list.append(torch.sqrt(es_b + 1e-8))
            px_conj_ps_re.append(torch.real(xb * torch.conj(sb)).sum(dim=(1, 2)))
            px_abs_ps_abs.append((xb.abs() * sb.abs()).sum(dim=(1, 2)))

        ex = torch.stack(ex_list, dim=1)  # (B, 24, T)
        es = torch.stack(es_list, dim=1)
        ax = torch.stack(amp_x_list, dim=1)
        as_ = torch.stack(amp_s_list, dim=1)
        p_dot = torch.stack(px_conj_ps_re, dim=1)
        p_norm = torch.stack(px_abs_ps_abs, dim=1)

        w_len = self.win_frames
        if n_frames < w_len:
            pad_r = w_len - n_frames
            ex = F.pad(ex, (0, pad_r))
            es = F.pad(es, (0, pad_r))
            ax = F.pad(ax, (0, pad_r))
            as_ = F.pad(as_, (0, pad_r))
            p_dot = F.pad(p_dot, (0, pad_r))
            p_norm = F.pad(p_norm, (0, pad_r))

        ex_w = ex.unfold(-1, w_len, self.stride)
        es_w = es.unfold(-1, w_len, self.stride)
        ax_w = ax.unfold(-1, w_len, self.stride)
        as_w = as_.unfold(-1, w_len, self.stride)
        dot_w = p_dot.unfold(-1, w_len, self.stride).sum(dim=-1)
        nrm_w = p_norm.unfold(-1, w_len, self.stride).sum(dim=-1)

        # 1. d_mod & 2. delta_rho on Hann-windowed centered amplitude envelope (0.5 - 8.0 Hz = bins 1..16)
        mw = self.mod_win.to(ax_w.device, dtype=torch.float32).view(1, 1, 1, w_len)
        wx_c = (ax_w - ax_w.mean(dim=-1, keepdim=True)) * mw
        ws_c = (as_w - as_w.mean(dim=-1, keepdim=True)) * mw
        ms_x = torch.fft.rfft(wx_c, dim=-1).abs()[..., 1:17]
        ms_s = torch.fft.rfft(ws_c, dim=-1).abs()[..., 1:17]
        nx = ms_x / (torch.linalg.norm(ms_x, dim=-1, keepdim=True) + 1e-8)
        ns = ms_s / (torch.linalg.norm(ms_s, dim=-1, keepdim=True) + 1e-8)
        d_mod = torch.linalg.norm(nx - ns, dim=-1).mean(dim=-1)

        rho_x = ms_x[..., 7:16].square().sum(dim=-1) / (ms_x.square().sum(dim=-1) + 1e-8)
        rho_s = ms_s[..., 7:16].square().sum(dim=-1) / (ms_s.square().sum(dim=-1) + 1e-8)
        delta_rho = (rho_s - rho_x).mean(dim=-1)

        # 3. Envelope CV log ratio
        cv_x = ax_w.std(dim=-1, unbiased=False) / (ax_w.mean(dim=-1) + 1e-6)
        cv_s = as_w.std(dim=-1, unbiased=False) / (as_w.mean(dim=-1) + 1e-6)
        env_var_ratio = torch.log((cv_s + 1e-4) / (cv_x + 1e-4)).mean(dim=-1)

        # 4. Log band energy ratio
        log_band_ratio = torch.clamp(torch.log((es_w.mean(dim=-1) + 1e-6) / (ex_w.mean(dim=-1) + 1e-6)), -4.0, 4.0).mean(dim=-1)

        # 5. Phase decorrelation (zero when identical or silence)
        cos_w = torch.where(nrm_w > 1e-12, dot_w / (nrm_w + 1e-12), torch.ones_like(dot_w))
        phase_decorr = (1.0 - torch.clamp(cos_w, -1.0, 1.0)).mean(dim=-1)

        # 6. Temporal envelope Pearson correlation loss (zero when identical or flat)
        dx = ax_w - ax_w.mean(dim=-1, keepdim=True)
        ds = as_w - as_w.mean(dim=-1, keepdim=True)
        cov = (dx * ds).sum(dim=-1)
        denom_corr = torch.sqrt(dx.square().sum(dim=-1) * ds.square().sum(dim=-1))
        corr = torch.where(denom_corr > 1e-12, cov / (denom_corr + 1e-12), torch.ones_like(cov))
        env_corr_loss = (1.0 - torch.clamp(corr, -1.0, 1.0)).mean(dim=-1)

        return torch.stack([d_mod, delta_rho, env_var_ratio, log_band_ratio, phase_decorr, env_corr_loss], dim=-1)


def exact_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from scipy.stats import rankdata

    y_b = y_true.astype(bool)
    n_pos = int(y_b.sum())
    n_neg = int((~y_b).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = rankdata(y_score)
    return float((ranks[y_b].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def fit_eval_fold_logreg(x_tr: np.ndarray, y_tr: np.ndarray, x_va: np.ndarray, l2: float = 1e-2) -> np.ndarray:
    torch.manual_seed(0)
    mu = x_tr.mean(axis=0, keepdims=True)
    sd = np.maximum(x_tr.std(axis=0, keepdims=True), 1e-5)
    xt = torch.from_numpy(((x_tr - mu) / sd).astype(np.float32))
    yt = torch.from_numpy(y_tr.astype(np.float32))
    xv = torch.from_numpy(((x_va - mu) / sd).astype(np.float32))
    w = nn.Linear(xt.shape[1], 1)
    opt = torch.optim.LBFGS(w.parameters(), lr=0.5, max_iter=80)

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(w(xt).squeeze(-1), yt) + l2 * w.weight.square().sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        return torch.sigmoid(w(xv).squeeze(-1)).numpy()


def fit_eval_fold_mlp(x_tr: np.ndarray, y_tr: np.ndarray, x_va: np.ndarray, l2: float = 1e-3) -> np.ndarray:
    torch.manual_seed(20261128)
    mu = x_tr.mean(axis=0, keepdims=True)
    sd = np.maximum(x_tr.std(axis=0, keepdims=True), 1e-5)
    xt = torch.from_numpy(((x_tr - mu) / sd).astype(np.float32))
    yt = torch.from_numpy(y_tr.astype(np.float32))
    xv = torch.from_numpy(((x_va - mu) / sd).astype(np.float32))
    mlp = nn.Sequential(nn.Linear(xt.shape[1], 32), nn.GELU(), nn.Dropout(0.15), nn.Linear(32, 1))
    opt = torch.optim.AdamW(mlp.parameters(), lr=3e-3, weight_decay=l2)
    mlp.train()
    for _ in range(180):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(mlp(xt).squeeze(-1), yt)
        loss.backward()
        opt.step()
    mlp.eval()
    with torch.no_grad():
        return torch.sigmoid(mlp(xv).squeeze(-1)).numpy()


def main() -> None:
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Phase 0 Pilot Val Probe running on {device}", flush=True)

    splits = json.loads((ROOT / "artifacts/10_extension/v4/pilot_splits.json").read_text(encoding="utf-8"))
    val_srcs = sorted(splits["pilot"]["val_sources"])
    val_src_set = set(val_srcs)
    assert len(val_srcs) == 80

    rows = [
        r
        for r in csv.DictReader(res("artifacts/10_extension/inputs_manifest.csv").open("r", encoding="utf-8"))
        if r["source_id"] in val_src_set
    ]
    assert len(rows) == 560, f"Expected 560 Pilot Val clips, got {len(rows)}"

    offsets = [0]
    for w in BAND_WIDTHS:
        offsets.append(offsets[-1] + w)
    band_slices = list(zip(offsets[:-1], offsets[1:]))
    extractor = WindowedModCueExtractor(band_slices).to(device)
    extractor.eval()

    feats_list: list[np.ndarray] = []
    meta_rows: list[dict] = []
    for idx, r in enumerate(rows):
        px = res(r["path_x"])
        ps = res(r["path_s"])
        info_x = sf.info(px)
        info_s = sf.info(ps)
        n_avail = min(info_x.frames, info_s.frames, 1320960)
        start = 220500 if n_avail >= 441000 else 0
        frames = min(220500, n_avail - start)

        yx, sr_x = sf.read(str(px), start=start, frames=frames, dtype="float32", always_2d=True)
        ys, sr_s = sf.read(str(ps), start=start, frames=frames, dtype="float32", always_2d=True)
        assert sr_x == SR and sr_s == SR
        if yx.shape[1] == 1:
            yx = np.repeat(yx, 2, axis=1)
        if ys.shape[1] == 1:
            ys = np.repeat(ys, 2, axis=1)
        x_np = yx.T.copy()
        s_np = align_teacher_to_input(x_np, ys.T.copy())

        xt = torch.from_numpy(x_np).unsqueeze(0).to(device)
        st = torch.from_numpy(s_np).unsqueeze(0).to(device)
        with torch.no_grad():
            x_in, s_in, _, _ = compute_track_norm(xt, st)
            assert s_in is not None
            zx = stft_ms(lr_to_ms(x_in), n_fft=2048, hop=512)
            zs = stft_ms(lr_to_ms(s_in), n_fft=2048, hop=512)
            cues = extractor(zx, zs).squeeze(0).cpu().numpy()  # (24, 6)

        degs = [d.strip() for d in r["degradation_types"].split("|") if d.strip()]
        is_rev = int(bool(set(degs) & REVERB_SET))
        feats_list.append(cues)
        meta_rows.append({"sample_id": r["sample_id"], "source_id": r["source_id"], "degs": degs, "is_rev": is_rev})
        if (idx + 1) % 140 == 0 or (idx + 1) == len(rows):
            print(f"  Extracted cues {idx + 1}/{len(rows)} in {time.time() - t0:.1f}s", flush=True)

    cues_arr = np.stack(feats_list, axis=0)  # (560, 24, 6)
    y_true = np.array([m["is_rev"] for m in meta_rows], dtype=np.int32)
    n_pos = int(y_true.sum())
    print(f"Reverb positives: {n_pos}/560 ({100.0 * n_pos / len(y_true):.1f}%)", flush=True)

    # Single-feature AUROCs
    single_aurocs = {
        "d_mod_mean": exact_auroc(y_true, cues_arr[:, :, 0].mean(axis=1)),
        "d_mod_midhi_1-12k": exact_auroc(y_true, cues_arr[:, 8:21, 0].mean(axis=1)),
        "delta_rho_mean": exact_auroc(y_true, cues_arr[:, :, 1].mean(axis=1)),
        "env_var_ratio_mean": exact_auroc(y_true, cues_arr[:, :, 2].mean(axis=1)),
        "log_band_ratio_mean": exact_auroc(y_true, cues_arr[:, :, 3].mean(axis=1)),
        "phase_decorr_low_0-1k": exact_auroc(y_true, cues_arr[:, :8, 4].mean(axis=1)),
        "env_corr_loss_mean": exact_auroc(y_true, cues_arr[:, :, 5].mean(axis=1)),
    }

    # 5-fold source-grouped CV
    src_to_fold = {src: i % 5 for i, src in enumerate(val_srcs)}
    folds = np.array([src_to_fold[m["source_id"]] for m in meta_rows], dtype=np.int32)
    x_flat = cues_arr.reshape(len(rows), -1)

    oof_logreg = np.zeros(len(rows), dtype=np.float64)
    oof_mlp = np.zeros(len(rows), dtype=np.float64)
    for f in range(5):
        va_mask = folds == f
        tr_mask = ~va_mask
        oof_logreg[va_mask] = fit_eval_fold_logreg(x_flat[tr_mask], y_true[tr_mask], x_flat[va_mask])
        oof_mlp[va_mask] = fit_eval_fold_mlp(x_flat[tr_mask], y_true[tr_mask], x_flat[va_mask])

    def summarize_oof(probs: np.ndarray) -> dict:
        auroc = exact_auroc(y_true, probs)
        neg_probs = probs[y_true == 0]
        # Threshold giving FPR <= 3% on non-reverb
        thr = float(np.quantile(neg_probs, 0.97))
        preds = probs >= thr
        actual_fpr = float(preds[y_true == 0].mean())
        recall = float(preds[y_true == 1].mean())
        sub_rec = {}
        for rt in REVERB_TYPES:
            mask = np.array([rt in m["degs"] for m in meta_rows])
            sub_rec[rt] = {"n": int(mask.sum()), "recall": float(preds[mask].mean()) if mask.any() else 0.0}
        fam_fpr = {}
        for fam in ("comp", "punch", "clip", "xband", "stereo", "bright"):
            mask = (y_true == 0) & np.array([fam in m["degs"] for m in meta_rows])
            fam_fpr[fam] = {"n": int(mask.sum()), "fpr": float(preds[mask].mean()) if mask.any() else 0.0}
        eq_set = {"clarity", "bright", "dark", "airy", "warm", "boom", "mud", "vocal", "mic"}
        mask_eq = (y_true == 0) & np.array([bool(set(m["degs"]) & eq_set) for m in meta_rows])
        fam_fpr["EQ"] = {"n": int(mask_eq.sum()), "fpr": float(preds[mask_eq].mean()) if mask_eq.any() else 0.0}
        return {
            "auroc": auroc,
            "thr_fpr3pct": thr,
            "actual_fpr": actual_fpr,
            "recall_at_fpr3pct": recall,
            "per_subtype_recall": sub_rec,
            "per_nonreverb_fpr": fam_fpr,
        }

    res_logreg = summarize_oof(oof_logreg)
    res_mlp = summarize_oof(oof_mlp)

    summary = {
        "elapsed_sec": round(time.time() - t0, 1),
        "n_clips": len(rows),
        "n_sources": len(val_srcs),
        "n_reverb": n_pos,
        "single_feature_aurocs": single_aurocs,
        "cv5_source_grouped_logreg": res_logreg,
        "cv5_source_grouped_mlp": res_mlp,
    }
    out_dir = ROOT / "artifacts/10_extension/v4/v4_7_phase0_val_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== Single-Feature Pilot Val (N=560) AUROCs ===")
    for k, v in single_aurocs.items():
        print(f"  {k:24s}: {v:.4f}")
    print("\n=== 5-Fold Source-Grouped CV ===")
    print(f"  LogReg AUROC = {res_logreg['auroc']:.4f} | Recall@FPR<=3% ({res_logreg['actual_fpr']*100:.2f}%): {res_logreg['recall_at_fpr3pct']*100:.1f}%")
    print(f"    Subtypes: { {k: round(v['recall']*100, 1) for k, v in res_logreg['per_subtype_recall'].items()} }")
    print(f"    NonRev FPR: { {k: round(v['fpr']*100, 1) for k, v in res_logreg['per_nonreverb_fpr'].items()} }")
    print(f"  MLP    AUROC = {res_mlp['auroc']:.4f} | Recall@FPR<=3% ({res_mlp['actual_fpr']*100:.2f}%): {res_mlp['recall_at_fpr3pct']*100:.1f}%")
    print(f"    Subtypes: { {k: round(v['recall']*100, 1) for k, v in res_mlp['per_subtype_recall'].items()} }")
    print(f"    NonRev FPR: { {k: round(v['fpr']*100, 1) for k, v in res_mlp['per_nonreverb_fpr'].items()} }")
    print(f"SAVED {out_dir / 'summary.json'} in {summary['elapsed_sec']}s")


if __name__ == "__main__":
    main()
