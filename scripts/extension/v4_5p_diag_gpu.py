"""Step 1 of V4-5p Diagnostic: GPU band stats (N=280), Reverb Oracle WAVs (N=24), & bright attribution (N=19).

Uses the exact fixed common protocol manifest (artifacts/10_extension/v4/matched_dev_recheck/manifest.json)
without modifying v4_models.py or any locked files.
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
from scripts.extension.v4_runtime import checkpoint_inference_config  # noqa: E402

REVERB_TYPES = ("small", "big", "mix", "real")
REVERB_SET = set(REVERB_TYPES)
DIAG_BANDS = (
    ("0-1kHz", 0, 47),
    ("1-4kHz", 47, 186),
    ("4-12kHz", 186, 558),
    ("12-22kHz", 558, 1025),
    ("bright_6-22kHz", 279, 1025),
)
ORACLE_MODES = ("default", "ph_S_midhi", "ph_S_all", "ph_S_all_noR", "mag_S_ph_P", "mag_hi_ph_S")
BRIGHT_MODES = ("P_only", "noR", "default", "r1_0", "r_hf1_0")
BRIGHT_BIN_START = 279  # bin 279 = 6007.76 Hz (>= 6000 Hz)
SR = 44100


def res(p: str | Path) -> Path:
    dp = ROOT / p
    if dp.is_file():
        return dp
    gp = GDRIVE / p
    if gp.is_file():
        return gp
    raise FileNotFoundError(f"Missing required audio file on both D: and G: -> {p}")


def load_stereo_np(path: Path, frames: int) -> np.ndarray:
    y, sr = sf.read(io.BytesIO(path.read_bytes()), dtype="float32", always_2d=True)
    if sr != SR or len(y) < frames:
        raise ValueError(f"Unexpected sr={sr} or len={len(y)}<{frames} in {path}")
    if y.shape[1] == 1:
        y = np.repeat(y, 2, axis=1)
    return y[:frames].T.copy()


def band_energy_ratio_exact(y_mono: np.ndarray, band_low: float, band_high: float) -> float:
    """Exact copy of SonicMaster/evaluation/evaluate_control_multiple_degs_mass.py band_energy_ratio."""
    import librosa

    spec = np.abs(librosa.stft(y_mono, n_fft=2048, hop_length=512)) ** 2
    freqs = librosa.fft_frequencies(sr=SR, n_fft=2048)
    band = np.logical_and(freqs >= band_low, freqs < band_high)
    band_energy = np.sum(spec[band, :])
    total_energy = np.sum(spec)
    return float(band_energy / (total_energy + 1e-10))


def select_stratified_reverb(manifest: list[dict], per_subtype: int = 6) -> list[tuple[str, dict]]:
    """Deterministically pick 6 clips per subtype (small, big, mix, real) without duplicates."""
    picked_ids: set[str] = set()
    selected: list[tuple[str, dict]] = []
    for rt in REVERB_TYPES:
        cands = sorted(
            [it for it in manifest if rt in it["degradations"] and it["sample_id"] not in picked_ids],
            key=lambda it: it["sample_id"],
        )
        if len(cands) < per_subtype:
            raise RuntimeError(f"Only {len(cands)} candidates for reverb subtype {rt!r}, expected >= {per_subtype}")
        idxs = np.round(np.linspace(0, len(cands) - 1, per_subtype)).astype(int)
        for idx in idxs:
            it = cands[int(idx)]
            assert it["sample_id"] not in picked_ids
            picked_ids.add(it["sample_id"])
            selected.append((rt, it))
    assert len(selected) == 4 * per_subtype
    return selected


def decode_chunk_v45p(
    model: HybridV4Net,
    xc: torch.Tensor,
    sc: torch.Tensor,
    raw_stats: torch.Tensor,
    modes: tuple[str, ...],
    use_autocast: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Exact replica of HybridV4Net('v4_5p') chunk forward returning requested override waveforms + diagnostics."""
    n_samples = xc.shape[-1]
    with torch.amp.autocast("cuda", enabled=use_autocast):
        z_x = stft_ms(lr_to_ms(xc), n_fft=model.n_fft, hop=model.hop)
        zc_x = compress_stft(z_x)
        z_s = stft_ms(lr_to_ms(sc), n_fft=model.n_fft, hop=model.hop)
        zc_s = compress_stft(z_s)
        h = model.encoder(zc_x, zc_s, raw_stats)
        for blk in model.blocks:
            h = blk(h)

    with torch.amp.autocast("cuda", enabled=False):
        dec = model.decoder
        bsz, n_frames, n_bands, _ = h.shape
        hs = dec.shared_mlp(h.float())
        zx_c = z_x.to(torch.complex64)
        zs_c = z_s.to(torch.complex64)

        dw0_list, gm_list, gs_list, rm_list, rs_list = [], [], [], [], []
        for bi, (sb, eb) in enumerate(dec.band_slices):
            wb = eb - sb
            ob = dec.per_bin_heads[bi](hs[:, :, bi, :])
            dw0_list.append(2.0 * torch.tanh(ob[:, :, : 8 * wb]).reshape(bsz, n_frames, wb, 8))
            gb = torch.sigmoid(ob[:, :, 8 * wb : 10 * wb].reshape(bsz, n_frames, wb, 2) + dec.gate_bias_init)
            gm_list.append(gb[..., 0].transpose(1, 2))
            gs_list.append(gb[..., 1].transpose(1, 2))

            rr = 2.0 * torch.tanh(ob[:, :, 10 * wb : 14 * wb]).reshape(bsz, n_frames, wb, 4)
            zx_b = zx_c[:, :, sb:eb, :]
            zs_b = zs_c[:, :, sb:eb, :]
            band_rms = torch.sqrt(
                torch.maximum(
                    zx_b.abs().square().mean(dim=(1, 2), keepdim=True),
                    zs_b.abs().square().mean(dim=(1, 2), keepdim=True),
                )
                + 1e-6
            )
            rm_list.append(torch.complex(rr[..., 0].transpose(1, 2), rr[..., 1].transpose(1, 2)) * band_rms.squeeze(1))
            rs_list.append(torch.complex(rr[..., 2].transpose(1, 2), rr[..., 3].transpose(1, 2)) * band_rms.squeeze(1))

        dw0 = torch.cat(dw0_list, dim=2)
        g_ms = torch.stack([torch.cat(gm_list, dim=1), torch.cat(gs_list, dim=1)], dim=1)
        r_ms = torch.stack([torch.cat(rm_list, dim=1), torch.cat(rs_list, dim=1)], dim=1)

        f_w = zx_c + dec._apply_2x2_complex(dw0, zx_c)
        mf_raw = 2.0 * torch.tanh(dec.mf_head(hs))
        dwk_bins = mf_raw.reshape(bsz, n_frames, 96, dec.n_taps_other, 8)[:, :, dec.subband_map, :, :]
        for tap_idx, k_off in enumerate(dec.mf_offsets):
            if abs(k_off) >= n_frames:
                continue
            if k_off < 0:
                pad_l = -k_off
                z_shifted = F.pad(zx_c[..., : n_frames - pad_l], (pad_l, 0))
            else:
                pad_r = k_off
                z_shifted = F.pad(zx_c[..., pad_r:], (0, pad_r))
            f_w = f_w + dec._apply_2x2_complex(dwk_bins[:, :, :, tap_idx, :], z_shifted)

        mag_p = torch.sqrt(f_w.real**2 + f_w.imag**2 + 1e-12)
        mag_s = torch.sqrt(zs_c.real**2 + zs_c.imag**2 + 1e-12)
        u_p = f_w / mag_p
        u_s = zs_c / mag_s

        ph_logits = dec.phase_gate_head(hs)
        ph_seq = ph_logits.permute(0, 2, 3, 1).reshape(bsz * n_bands, 2, n_frames)
        k_sm = min(9, n_frames if n_frames % 2 == 1 else max(1, n_frames - 1))
        ph_sm = F.avg_pool1d(ph_seq, kernel_size=k_sm, stride=1, padding=k_sm // 2, count_include_pad=False)
        g_ph_band = torch.sigmoid(ph_sm.reshape(bsz, n_bands, 2, n_frames).permute(0, 2, 1, 3) + dec.gate_bias_init)
        g_ph = g_ph_band[:, :, dec.band_map, :]

        def _synth_decoupled(gm_use: torch.Tensor, gp_use: torch.Tensor, r_scale: float | torch.Tensor) -> torch.Tensor:
            mag_f = (1.0 - gm_use) * mag_p + gm_use * mag_s
            u_mix = (1.0 - gp_use) * u_p + gp_use * u_s
            u_out = u_mix / torch.sqrt(u_mix.real**2 + u_mix.imag**2 + 1e-6)
            return mag_f * u_out + r_scale * r_ms

        z_modes: dict[str, torch.Tensor] = {}
        for mode in modes:
            if mode == "default":
                z_modes[mode] = _synth_decoupled(g_ms, g_ph, dec.residual_scale)
            elif mode == "P_only":
                z_modes[mode] = f_w
            elif mode == "noR":
                z_modes[mode] = _synth_decoupled(g_ms, g_ph, 0.0)
            elif mode == "r1_0":
                z_modes[mode] = _synth_decoupled(g_ms, g_ph, 1.0)
            elif mode == "r_hf1_0":
                rs_vec = torch.full((1, 1, r_ms.shape[2], 1), dec.residual_scale, device=r_ms.device, dtype=r_ms.dtype)
                rs_vec[:, :, BRIGHT_BIN_START:, :] = 1.0
                z_modes[mode] = _synth_decoupled(g_ms, g_ph, rs_vec)
            elif mode == "ph_S_midhi":
                gp = g_ph.clone()
                gp[:, :, 47:, :] = 1.0
                z_modes[mode] = _synth_decoupled(g_ms, gp, dec.residual_scale)
            elif mode == "ph_S_all":
                gp = torch.ones_like(g_ph)
                z_modes[mode] = _synth_decoupled(g_ms, gp, dec.residual_scale)
            elif mode == "ph_S_all_noR":
                z_modes[mode] = _synth_decoupled(g_ms, torch.ones_like(g_ph), 0.0)
            elif mode == "mag_S_ph_P":
                z_modes[mode] = _synth_decoupled(torch.ones_like(g_ms), g_ph, 0.0)
            elif mode == "mag_hi_ph_S":
                gm = g_ms.clone()
                gm[:, :, 47:, :] = torch.clamp(gm[:, :, 47:, :], min=0.7)
                gp = g_ph.clone()
                gp[:, :, 47:, :] = 1.0
                z_modes[mode] = _synth_decoupled(gm, gp, 0.0)
            else:
                raise ValueError(f"Unknown diagnostic mode: {mode}")

        y_modes: dict[str, torch.Tensor] = {}
        for mode, z_m in z_modes.items():
            y_lr = ms_to_lr(istft_ms(z_m, length=n_samples, n_fft=model.n_fft, hop=model.hop))
            y_modes[mode] = y_lr.float()

        eps = 1e-8
        norm_zx = float(torch.sqrt(torch.mean(zx_c.abs() ** 2) + eps).item())
        diag: dict[str, float] = {
            "rel_dP": float(torch.sqrt(torch.mean((f_w - zx_c).abs() ** 2) + eps).item()) / norm_zx,
            "rel_R": float(torch.sqrt(torch.mean(r_ms.abs() ** 2) + eps).item()) / norm_zx,
            "g_mag_M": float(g_ms[:, 0].mean().item()),
            "g_mag_S": float(g_ms[:, 1].mean().item()),
            "g_mag": float(g_ms.mean().item()),
            "g_ph_M": float(g_ph[:, 0].mean().item()),
            "g_ph_S": float(g_ph[:, 1].mean().item()),
            "g_ph": float(g_ph.mean().item()),
        }
        for bname, lo, hi in DIAG_BANDS:
            pb, sb_z = f_w[:, :, lo:hi, :], zs_c[:, :, lo:hi, :]
            gmb, gpb = g_ms[:, :, lo:hi, :], g_ph[:, :, lo:hi, :]
            rb, zxb = r_ms[:, :, lo:hi, :], zx_c[:, :, lo:hi, :]
            diag[f"g_mag_M_{bname}"] = float(gmb[:, 0].mean().item())
            diag[f"g_mag_S_{bname}"] = float(gmb[:, 1].mean().item())
            diag[f"g_mag_{bname}"] = float(gmb.mean().item())
            diag[f"g_ph_M_{bname}"] = float(gpb[:, 0].mean().item())
            diag[f"g_ph_S_{bname}"] = float(gpb[:, 1].mean().item())
            diag[f"g_ph_{bname}"] = float(gpb.mean().item())
            diag[f"cos_PS_{bname}"] = float((torch.real(pb * torch.conj(sb_z)).sum() / ((pb.abs() * sb_z.abs()).sum() + eps)).item())
            norm_b = float(torch.sqrt(torch.mean(zxb.abs() ** 2) + eps).item())
            diag[f"rel_R_{bname}"] = float(torch.sqrt(torch.mean(rb.abs() ** 2) + eps).item()) / norm_b

    return y_modes, diag


def forward_track_modes(
    model: HybridV4Net,
    x: torch.Tensor,
    s: torch.Tensor,
    modes: tuple[str, ...],
    chunk_sec: float = 8.0,
    overlap_ratio: float = 0.5,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Run exact track-level Hann OLA + rms(s) matching across requested override modes."""
    use_autocast = x.device.type == "cuda"
    with torch.amp.autocast("cuda", enabled=use_autocast):
        x_in, s_in, scale, raw_stats = compute_track_norm(x, s)
    assert s_in is not None
    n_samples = x_in.shape[-1]
    chunk_len = int(round(chunk_sec * SR))
    hop_len = max(1, int(round(chunk_len * (1.0 - overlap_ratio))))
    rms_s = torch.sqrt(torch.mean(s.float() ** 2) + 1e-12)

    def _rms_match(y_in: torch.Tensor) -> torch.Tensor:
        y_out = y_in * scale
        rms_y = torch.sqrt(torch.mean(y_out.float() ** 2) + 1e-12)
        if rms_s > 1e-6 and rms_y > 1e-6:
            y_out = y_out * (rms_s / rms_y).to(y_out.dtype)
        return y_out

    if n_samples <= chunk_len:
        ym, diag = decode_chunk_v45p(model, x_in, s_in, raw_stats, modes, use_autocast)
        return {m: _rms_match(ym[m]) for m in modes}, diag

    pad_l, pad_r = hop_len, hop_len
    x_pad = F.pad(x_in, (pad_l, pad_r), mode="reflect")
    s_pad = F.pad(s_in, (pad_l, pad_r), mode="reflect")
    total_pad = x_pad.shape[-1]
    win = torch.hann_window(chunk_len, periodic=True, device=x.device, dtype=torch.float32)
    acc = {m: torch.zeros((1, 2, total_pad), device=x.device, dtype=torch.float32) for m in modes}
    w_sum = torch.zeros((1, 1, total_pad), device=x.device, dtype=torch.float32)

    starts = list(range(0, max(1, total_pad - chunk_len + 1), hop_len))
    if starts[-1] + chunk_len < total_pad:
        starts.append(total_pad - chunk_len)

    diag_acc: dict[str, list[float]] = {}
    for st in starts:
        en = st + chunk_len
        ym, diag = decode_chunk_v45p(model, x_pad[:, :, st:en], s_pad[:, :, st:en], raw_stats, modes, use_autocast)
        for m in modes:
            acc[m][:, :, st:en] += ym[m] * win[None, None, :]
        w_sum[:, :, st:en] += win[None, None, :]
        for k, v in diag.items():
            diag_acc.setdefault(k, []).append(v)

    w_clamp = torch.clamp(w_sum, min=1e-6)
    outs = {m: _rms_match((acc[m] / w_clamp)[:, :, pad_l : pad_l + n_samples]) for m in modes}
    diag_mean = {k: sum(vs) / len(vs) for k, vs in diag_acc.items()}
    return outs, diag_mean


def main() -> None:
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    manifest_path = ROOT / "artifacts/10_extension/v4/matched_dev_recheck/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest) == 280, f"Expected 280 manifest items, got {len(manifest)}"

    ckpt_path = ROOT / "artifacts/10_extension/v4/kaggle_out_v4_5p/out/v4_5p/best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = checkpoint_inference_config(ckpt, "v4_5p")
    model = HybridV4Net("v4_5p", hop=cfg["hop"], gate_bias_init=cfg["gate_bias_init"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded V4-5p best.pt (ep={ckpt.get('epoch')}, cfg={cfg}, r_scale={model.decoder.residual_scale})", flush=True)

    strat_reverb = select_stratified_reverb(manifest, per_subtype=6)
    oracle_subtype_map = {it["sample_id"]: rt for rt, it in strat_reverb}
    oracle_set = set(oracle_subtype_map.keys())
    bright_set = {it["sample_id"] for it in manifest if "bright" in it["degradations"]}
    print(f"Selected {len(oracle_set)} stratified Reverb oracle clips and {len(bright_set)} bright clips", flush=True)

    out_dir = ROOT / "artifacts/10_extension/v4/v4_5p_diagnostic"
    wav_root = out_dir / "oracle_wav"
    for m in ORACLE_MODES:
        (wav_root / m).mkdir(parents=True, exist_ok=True)

    parity_max_abs_diff = -1.0
    band_stats_per_clip: list[dict] = []
    oracle_reverb_records: list[dict] = []
    bright_records: list[dict] = []

    for idx, it in enumerate(manifest):
        sid = it["sample_id"]
        frames = int(it["frames"])
        degs = it["degradations"]
        is_reverb = bool(set(degs) & REVERB_SET)
        is_bright = sid in bright_set
        is_oracle = sid in oracle_set

        x_np = load_stereo_np(res(it["paths"]["input"]), frames)
        s_np = load_stereo_np(res(it["paths"]["A0"]), frames)
        s_al = align_teacher_to_input(x_np, s_np)

        modes_req: list[str] = ["default"]
        if is_oracle:
            modes_req.extend(m for m in ORACLE_MODES if m not in modes_req)
        if is_bright:
            modes_req.extend(m for m in BRIGHT_MODES if m not in modes_req)

        xt = torch.from_numpy(x_np).unsqueeze(0).to(device)
        st = torch.from_numpy(s_al).unsqueeze(0).to(device)
        with torch.no_grad():
            outs, diag = forward_track_modes(
                model, xt, st, tuple(modes_req), chunk_sec=cfg["chunk_sec"], overlap_ratio=0.5
            )

        if idx == 0:
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                y_ref, _ = model.forward_track(xt, st, chunk_sec=cfg["chunk_sec"], overlap_ratio=0.5)
            parity_max_abs_diff = float(torch.max(torch.abs(outs["default"] - y_ref.float())).item())
            print(f"Parity check default vs model.forward_track max_abs_diff = {parity_max_abs_diff:.2e}", flush=True)
            assert parity_max_abs_diff < 1e-5, f"Parity diff too high: {parity_max_abs_diff}"

        band_stats_per_clip.append(
            {
                "sample_id": sid,
                "source_id": it["source_id"],
                "degradations": degs,
                "is_reverb": is_reverb,
                "is_bright": is_bright,
                "diag": diag,
            }
        )

        if is_oracle or is_bright:
            c_np = load_stereo_np(res(it["paths"]["clean"]), frames)
            c_mono = c_np.mean(axis=0)

        if is_oracle:
            mc = logmel(c_mono)
            arm_np: dict[str, np.ndarray] = {"Input": x_np, "A0": s_np}
            for m in ORACLE_MODES:
                y_m = outs[m].squeeze(0).cpu().numpy()
                y_save = np.clip(y_m.T, -1.0, 1.0).astype(np.float32)
                wp = wav_root / m / f"{sid}.wav"
                sf.write(str(wp), y_save, SR, subtype="PCM_16")
                arm_np[m] = load_stereo_np(wp, frames)
            ssim_d, kl_d = {}, {}
            for arm_k, arr in arm_np.items():
                mo = logmel(arr.mean(axis=0))
                ssim_d[arm_k] = float(mel_ssim(mo, mc))
                kl_d[arm_k] = float(1000.0 * kl_div(mo, mc))
            oracle_reverb_records.append(
                {
                    "sample_id": sid,
                    "source_id": it["source_id"],
                    "frames": frames,
                    "primary_subtype": oracle_subtype_map[sid],
                    "reverb_types": sorted(set(degs) & REVERB_SET),
                    "degradations": degs,
                    "n_deg": len(degs),
                    "ssim": ssim_d,
                    "kl1000": kl_d,
                    "diag": diag,
                }
            )

        if is_bright:
            v44_np = load_stereo_np(res(it["paths"]["v4_4"]), frames)
            r_clean = band_energy_ratio_exact(c_mono, 6000, SR // 2)
            arms_b: dict[str, np.ndarray] = {
                "Input": x_np,
                "A0": s_np,
                "V4-4": v44_np,
            }
            for m in BRIGHT_MODES:
                y_m = outs[m].squeeze(0).cpu().numpy()
                # Quantize identically via clip+float32 PCM_16 roundtrip in memory
                buf = io.BytesIO()
                sf.write(buf, np.clip(y_m.T, -1.0, 1.0).astype(np.float32), SR, format="WAV", subtype="PCM_16")
                buf.seek(0)
                yq, _ = sf.read(buf, dtype="float32", always_2d=True)
                arms_b[m] = yq.T
            b_metrics: dict[str, dict[str, float]] = {}
            for ak, arr in arms_b.items():
                r_val = band_energy_ratio_exact(arr.mean(axis=0), 6000, SR // 2)
                b_metrics[ak] = {
                    "ratio": r_val,
                    "signed_err": r_val - r_clean,
                    "abs_err": abs(r_val - r_clean),
                }
            bright_records.append(
                {
                    "sample_id": sid,
                    "source_id": it["source_id"],
                    "degradations": degs,
                    "ratio_clean": r_clean,
                    "arms": b_metrics,
                    "diag": diag,
                }
            )

        if (idx + 1) % 70 == 0 or (idx + 1) == len(manifest):
            print(f"  Processed {idx + 1}/{len(manifest)} clips in {time.time() - t0:.1f}s", flush=True)

    out_payload = {
        "parity_max_abs_diff": parity_max_abs_diff,
        "elapsed_sec": round(time.time() - t0, 1),
        "n_clips": len(band_stats_per_clip),
        "n_oracle_reverb": len(oracle_reverb_records),
        "n_bright": len(bright_records),
        "band_stats_per_clip": band_stats_per_clip,
        "oracle_reverb_ids": oracle_reverb_records,
        "bright_records": bright_records,
    }
    diag_json = out_dir / "diag_gpu.json"
    diag_json.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
    print(f"SAVED {diag_json} in {out_payload['elapsed_sec']}s", flush=True)


if __name__ == "__main__":
    main()
