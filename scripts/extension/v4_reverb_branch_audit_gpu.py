"""Step 1 of Reverb Branch Audit: GPU extraction of P, Pg, Full branches on 106 Dev Reverb clips.

Uses the exact fixed common protocol manifest (artifacts/10_extension/v4/matched_dev_recheck/manifest.json).
Computes Mel-SSIM, Mel-KLx1000, and band-wise phase/gate/branch diagnostics for Input, P, Pg, Full, A0,
and exports 16-bit PCM WAVs for downstream official Reverb MSD + Audiobox PQ scoring.
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

ROOT = Path(r"d:\SonicMaster_Reproducing")
GDRIVE = Path(r"G:\My Drive\SonicMaster_Reproducing")
sys.path.insert(0, str(ROOT))

from scripts.extension.eval_dev import kl_div, logmel, mel_ssim  # noqa: E402
from scripts.extension.v4_alignment import align_teacher_to_input  # noqa: E402
from scripts.extension.v4_models import (  # noqa: E402
    HybridV4Net,
    compress_stft,
    compute_track_norm,
    istft_ms,
    lr_to_ms,
    ms_to_lr,
    stft_ms,
)

REVERB_SET = {"small", "big", "mix", "real"}
BANDS = [("0-1kHz", 0, 47), ("1-4kHz", 47, 186), ("4-12kHz", 186, 558), ("12-22kHz", 558, 1025)]


def res(p: str | Path) -> Path:
    dp = ROOT / p
    return dp if dp.is_file() else GDRIVE / p


def load_stereo_np(path: Path, frames: int) -> np.ndarray:
    y, _ = sf.read(io.BytesIO(path.read_bytes()), dtype="float32", always_2d=True)
    if y.shape[1] == 1:
        y = np.repeat(y, 2, axis=1)
    return y[:frames].T.copy()


def decode_chunk_branches(
    model: HybridV4Net,
    xc: torch.Tensor,
    sc: torch.Tensor,
    raw_stats: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """Run one normalized chunk through V4-4 and return waveforms (P, Pg, Full) + diagnostics."""
    n_samples = xc.shape[-1]
    zx = stft_ms(lr_to_ms(xc), n_fft=model.n_fft, hop=model.hop)
    zs = stft_ms(lr_to_ms(sc), n_fft=model.n_fft, hop=model.hop)
    zc_x = compress_stft(zx)
    zc_s = compress_stft(zs)

    h = model.encoder(zc_x, zc_s, raw_stats)
    for blk in model.blocks:
        h = blk(h)

    dec = model.decoder
    hs = dec.shared_mlp(h.float())
    bsz, n_frames, _, _ = hs.shape
    zx_c = zx.to(torch.complex64)
    zs_c = zs.to(torch.complex64)

    dw0_list, gm_list, gs_list, rm_list, rs_list = [], [], [], [], []
    for bi, (sb, eb) in enumerate(dec.band_slices):
        wb = eb - sb
        ob = dec.per_bin_heads[bi](hs[:, :, bi, :])
        dw0_list.append(2.0 * torch.tanh(ob[:, :, : 8 * wb]).reshape(bsz, n_frames, wb, 8))
        gb = torch.sigmoid(ob[:, :, 8 * wb : 10 * wb].reshape(bsz, n_frames, wb, 2) + dec.gate_bias_init)
        gm_list.append(gb[..., 0].transpose(1, 2))
        gs_list.append(gb[..., 1].transpose(1, 2))

        rr = 2.0 * torch.tanh(ob[:, :, 10 * wb : 14 * wb]).reshape(bsz, n_frames, wb, 4)
        brms = torch.sqrt(
            torch.mean(zx_c[:, :, sb:eb, :].abs() ** 2 + zs_c[:, :, sb:eb, :].abs() ** 2, dim=(1, 2), keepdim=True) + 1e-6
        )
        rm_list.append(torch.complex(rr[..., 0].transpose(1, 2), rr[..., 1].transpose(1, 2)) * brms.squeeze(1))
        rs_list.append(torch.complex(rr[..., 2].transpose(1, 2), rr[..., 3].transpose(1, 2)) * brms.squeeze(1))

    dw0 = torch.cat(dw0_list, dim=2)
    g_ms = torch.stack([torch.cat(gm_list, dim=1), torch.cat(gs_list, dim=1)], dim=1)
    r_ms = torch.stack([torch.cat(rm_list, dim=1), torch.cat(rs_list, dim=1)], dim=1)

    z_p = zx_c + dec._apply_2x2_complex(dw0, zx_c)
    mf_sub = (2.0 * torch.tanh(dec.mf_head(hs))).reshape(bsz, n_frames, 96, dec.n_taps_other, 8)
    dwk_bins = mf_sub[:, :, dec.subband_map, :, :]
    for ti, ko in enumerate(dec.mf_offsets):
        if ko < 0:
            z_sh = F.pad(zx_c[..., : n_samples // model.hop + 1 + ko], (-ko, 0)) if False else F.pad(zx_c[..., : n_frames + ko], (-ko, 0))
        else:
            z_sh = F.pad(zx_c[..., ko:], (0, ko))
        z_p = z_p + dec._apply_2x2_complex(dwk_bins[:, :, :, ti, :], z_sh)

    z_pg = (1.0 - g_ms) * z_p + g_ms * zs_c
    z_full = z_pg + dec.residual_scale * r_ms

    y_p = ms_to_lr(istft_ms(z_p, length=n_samples, n_fft=model.n_fft, hop=model.hop))
    y_pg = ms_to_lr(istft_ms(z_pg, length=n_samples, n_fft=model.n_fft, hop=model.hop))
    y_full = ms_to_lr(istft_ms(z_full, length=n_samples, n_fft=model.n_fft, hop=model.hop))

    eps = 1e-8
    norm_zx = float(torch.sqrt(torch.mean(zx_c.abs() ** 2) + eps).item())
    diag: dict[str, float] = {
        "rel_dP": float(torch.sqrt(torch.mean((z_p - zx_c).abs() ** 2) + eps).item()) / norm_zx,
        "rel_gSP": float(torch.sqrt(torch.mean((g_ms * (zs_c - z_p)).abs() ** 2) + eps).item()) / norm_zx,
        "rel_R": float(torch.sqrt(torch.mean(r_ms.abs() ** 2) + eps).item()) / norm_zx,
        "g_mean_M": float(g_ms[:, 0].mean().item()),
        "g_mean_S": float(g_ms[:, 1].mean().item()),
    }
    for bname, lo, hi in BANDS:
        pb, sb_z, gb_z = z_p[:, :, lo:hi, :], zs_c[:, :, lo:hi, :], g_ms[:, :, lo:hi, :]
        diag[f"cos_PS_{bname}"] = float((torch.real(pb * torch.conj(sb_z)).sum() / ((pb.abs() * sb_z.abs()).sum() + eps)).item())
        mix50 = (0.5 * pb + 0.5 * sb_z).abs()
        ideal50 = 0.5 * pb.abs() + 0.5 * sb_z.abs()
        diag[f"cancel50_{bname}"] = float((mix50.sum() / (ideal50.sum() + eps)).item())
        diag[f"mean_g_{bname}"] = float(gb_z.mean().item())
    return y_p, y_pg, y_full, diag


def forward_track_branches(
    model: HybridV4Net,
    x: torch.Tensor,
    s: torch.Tensor,
    chunk_sec: float = 8.0,
    overlap_ratio: float = 0.5,
    sr: int = 44100,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Run track-level Hann OLA returning P, Pg, Full branches matched to rms(s)."""
    x_in, s_in, scale, raw_stats = compute_track_norm(x, s)
    assert s_in is not None
    n_samples = x_in.shape[-1]
    chunk_len = int(round(chunk_sec * sr))
    hop_len = max(1, int(round(chunk_len * (1.0 - overlap_ratio))))
    rms_s = torch.sqrt(torch.mean(s.float() ** 2) + 1e-12)

    def _rms_match(y_in: torch.Tensor) -> torch.Tensor:
        y_out = y_in * scale
        rms_y = torch.sqrt(torch.mean(y_out.float() ** 2) + 1e-12)
        if rms_s > 1e-6 and rms_y > 1e-6:
            y_out = y_out * (rms_s / rms_y).to(y_out.dtype)
        return y_out

    if n_samples <= chunk_len:
        yp, ypg, yf, diag = decode_chunk_branches(model, x_in, s_in, raw_stats)
        return {"P": _rms_match(yp), "Pg": _rms_match(ypg), "Full": _rms_match(yf)}, diag

    pad_l, pad_r = hop_len, hop_len
    x_pad = F.pad(x_in, (pad_l, pad_r), mode="reflect")
    s_pad = F.pad(s_in, (pad_l, pad_r), mode="reflect")
    total_pad = x_pad.shape[-1]
    win = torch.hann_window(chunk_len, periodic=True, device=x.device, dtype=x.dtype)
    acc = {k: torch.zeros_like(x_pad) for k in ("P", "Pg", "Full")}
    w_sum = torch.zeros((1, 1, total_pad), device=x.device, dtype=x.dtype)

    starts = list(range(0, max(1, total_pad - chunk_len + 1), hop_len))
    if starts[-1] + chunk_len < total_pad:
        starts.append(total_pad - chunk_len)

    diag_acc: dict[str, list[float]] = {}
    for st in starts:
        en = st + chunk_len
        yp, ypg, yf, diag = decode_chunk_branches(model, x_pad[:, :, st:en], s_pad[:, :, st:en], raw_stats)
        acc["P"][:, :, st:en] += yp * win[None, None, :]
        acc["Pg"][:, :, st:en] += ypg * win[None, None, :]
        acc["Full"][:, :, st:en] += yf * win[None, None, :]
        w_sum[:, :, st:en] += win[None, None, :]
        for k, v in diag.items():
            diag_acc.setdefault(k, []).append(v)

    w_clamp = torch.clamp(w_sum, min=1e-6)
    outs = {k: _rms_match((acc[k] / w_clamp)[:, :, pad_l : pad_l + n_samples]) for k in ("P", "Pg", "Full")}
    diag_mean = {k: sum(vs) / len(vs) for k, vs in diag_acc.items()}
    return outs, diag_mean


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    manifest_path = ROOT / "artifacts/10_extension/v4/matched_dev_recheck/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rev_items = [it for it in manifest if (set(it["degradations"]) & REVERB_SET)]
    print(f"Loaded {len(rev_items)} Reverb clips from fixed manifest", flush=True)
    assert len(rev_items) == 106, f"Expected 106 reverb clips, got {len(rev_items)}"

    ckpt_path = ROOT / "artifacts/10_extension/v4/kaggle_out_v4_4/out/v4_4/best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = HybridV4Net("v4_4", hop=512, gate_bias_init=-1.5).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    out_dir = ROOT / "artifacts/10_extension/v4/reverb_branch_audit"
    wav_root = out_dir / "wav"
    for b in ("P", "Pg", "Full"):
        (wav_root / b).mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    t0 = time.time()
    for idx, it in enumerate(rev_items):
        sid = it["sample_id"]
        frames = int(it["frames"])
        c_np = load_stereo_np(res(it["paths"]["clean"]), frames)
        x_np = load_stereo_np(res(it["paths"]["input"]), frames)
        s_np = load_stereo_np(res(it["paths"]["A0"]), frames)
        s_al = align_teacher_to_input(x_np, s_np)

        xt = torch.from_numpy(x_np).unsqueeze(0).to(device)
        st = torch.from_numpy(s_al).unsqueeze(0).to(device)
        with torch.no_grad():
            outs, diag = forward_track_branches(model, xt, st, chunk_sec=8.0, overlap_ratio=0.5)

        if idx == 0:
            with torch.no_grad():
                y_ref, _ = model.forward_track(xt, st, chunk_sec=8.0, overlap_ratio=0.5)
            max_diff = float(torch.max(torch.abs(outs["Full"] - y_ref)).item())
            print(f"Parity check Full vs model.forward_track max_abs_diff = {max_diff:.2e}", flush=True)

        branch_np: dict[str, np.ndarray] = {
            "Input": x_np,
            "A0": s_np,
        }
        for b in ("P", "Pg", "Full"):
            y_b = outs[b].squeeze(0).cpu().numpy()
            y_save = np.clip(y_b.T, -1.0, 1.0).astype(np.float32)
            wav_p = wav_root / b / f"{sid}.wav"
            sf.write(str(wav_p), y_save, 44100, subtype="PCM_16")
            branch_np[b] = load_stereo_np(wav_p, frames)

        mc = logmel(c_np.mean(axis=0))
        rec: dict = {
            "sample_id": sid,
            "source_id": it["source_id"],
            "frames": frames,
            "degradations": it["degradations"],
            "n_deg": len(it["degradations"]),
            "reverb_types": sorted(set(it["degradations"]) & REVERB_SET),
            "ssim": {},
            "kl1000": {},
            "diag": diag,
        }
        for b_name, arr in branch_np.items():
            n_f = min(c_np.shape[-1], arr.shape[-1])
            mb = logmel(arr[:, :n_f].mean(axis=0))
            rec["ssim"][b_name] = float(mel_ssim(mc[:, : mb.shape[1]], mb)) if mc.shape[1] != mb.shape[1] else float(mel_ssim(mc, mb))
            rec["kl1000"][b_name] = float(kl_div(mc, mb) * 1000.0)
        records.append(rec)

        if (idx + 1) % 25 == 0 or (idx + 1) == len(rev_items):
            print(f"[reverb_audit_gpu] {idx+1}/{len(rev_items)} in {time.time()-t0:.1f}s", flush=True)

    out_json = out_dir / "spectral_reverb.json"
    out_json.write_text(json.dumps({"elapsed_sec": round(time.time() - t0, 1), "records": records}, indent=2), encoding="utf-8")
    print(f"Saved {out_json}", flush=True)


if __name__ == "__main__":
    main()
