"""Hybrid-v4 (Rev 2.2-Final) Models: V4-0 (NoGuide), V4-1 (DirectMap), V4-2 (MF7)."""
from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

# 24 psychoacoustic non-overlapping bands summing to 1,025 STFT bins (n_fft=2048, sr=44100)
BAND_WIDTHS: tuple[int, ...] = (
    6, 6, 6, 6, 6, 6, 6, 5,          # 0 - 1,012 Hz (47 bins)
    17, 17, 17, 17, 17, 18, 18, 18,  # 1,012 - 4,005 Hz (139 bins)
    74, 74, 74, 75, 75,              # 4,005 - 12,015 Hz (372 bins)
    155, 156, 156,                   # 12,015 - 22,050 Hz (467 bins)
)
assert sum(BAND_WIDTHS) == 1025

MF7_OFFSETS: tuple[int, ...] = (-5, -4, -3, -2, -1, 1)  # k=0 is per-bin
MF9_DILATED_OFFSETS: tuple[int, ...] = (-18, -12, -8, -5, -3, -2, -1, 1)  # 8 non-zero taps + k=0 = 9 taps (~220 ms)


def lr_to_ms(y: torch.Tensor) -> torch.Tensor:
    """Convert stereo (..., 2, N) Left/Right to Mid/Side."""
    inv_s2 = 1.0 / math.sqrt(2.0)
    l_ch, r_ch = y[..., 0, :], y[..., 1, :]
    m_ch = (l_ch + r_ch) * inv_s2
    s_ch = (l_ch - r_ch) * inv_s2
    return torch.stack([m_ch, s_ch], dim=-2)


def ms_to_lr(ms: torch.Tensor) -> torch.Tensor:
    """Convert (..., 2, N) Mid/Side back to Left/Right."""
    inv_s2 = 1.0 / math.sqrt(2.0)
    m_ch, s_ch = ms[..., 0, :], ms[..., 1, :]
    l_ch = (m_ch + s_ch) * inv_s2
    r_ch = (m_ch - s_ch) * inv_s2
    return torch.stack([l_ch, r_ch], dim=-2)


def stft_ms(ms: torch.Tensor, n_fft: int = 2048, hop: int = 512) -> torch.Tensor:
    """STFT of (B, 2, N) -> complex (B, 2, F, T)."""
    bsz, n_ch, n_samples = ms.shape
    flat = ms.reshape(bsz * n_ch, n_samples)
    win = torch.hann_window(n_fft, periodic=True, device=ms.device, dtype=ms.dtype)
    z = torch.stft(flat, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=win, center=True, return_complex=True)
    return z.reshape(bsz, n_ch, z.shape[-2], z.shape[-1])


def istft_ms(z: torch.Tensor, length: int, n_fft: int = 2048, hop: int = 512) -> torch.Tensor:
    """iSTFT of complex (B, 2, F, T) -> (B, 2, length)."""
    bsz, n_ch, n_freqs, n_frames = z.shape
    flat = z.reshape(bsz * n_ch, n_freqs, n_frames)
    win = torch.hann_window(n_fft, periodic=True, device=z.device, dtype=torch.float32)
    y = torch.istft(flat, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=win, center=True, length=length)
    return y.reshape(bsz, n_ch, length)


def compress_stft(z: torch.Tensor, power: float = 0.3, eps: float = 1e-7) -> torch.Tensor:
    """Power-compress complex STFT for encoder features (or DirectMap target)."""
    mag = z.abs()
    scale = (mag + eps).pow(power) / (mag + eps)
    return z * scale


def decompress_stft(zc: torch.Tensor, power: float = 0.3, eps: float = 1e-7) -> torch.Tensor:
    """Explicit inverse power-compression before iSTFT for DirectMap models."""
    mag_c = zc.abs()
    scale = (mag_c + eps).pow(1.0 / power) / (mag_c + eps)
    return zc * scale


def compute_track_norm(
    x: torch.Tensor,
    s: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """Track-level bounded normalization preserving volume cues.

    x, s: (B, 2, N).
    Returns: (x_in, s_in, scale, raw_stats (B, 3)).
    """
    rms_x = torch.sqrt(torch.mean(x.float() ** 2, dim=(-2, -1), keepdim=True) + eps)
    if s is None:
        scale = torch.clamp(rms_x, min=eps)
        x_in = x / scale
        log_rx = torch.log(rms_x.squeeze(-1).squeeze(-1) + eps)
        raw_stats = torch.stack([log_rx, torch.zeros_like(log_rx), torch.zeros_like(log_rx)], dim=-1)
        return x_in, None, scale, raw_stats

    rms_s = torch.sqrt(torch.mean(s.float() ** 2, dim=(-2, -1), keepdim=True) + eps)
    scale = torch.clamp(rms_s, min=eps)
    g_align = torch.clamp(rms_s / rms_x, min=0.1, max=16.0)  # +/- 24 dB bound
    x_in = (x * g_align) / scale
    s_in = s / scale

    log_rx = torch.log(rms_x.squeeze(-1).squeeze(-1) + eps)
    log_rs = torch.log(rms_s.squeeze(-1).squeeze(-1) + eps)
    gain_db_norm = torch.clamp(20.0 * torch.log10(rms_s.squeeze(-1).squeeze(-1) / rms_x.squeeze(-1).squeeze(-1)) / 24.0, -2.0, 2.0)
    raw_stats = torch.stack([log_rx, log_rs, gain_db_norm], dim=-1)
    return x_in, s_in, scale, raw_stats


class PsychoacousticBandEncoder(nn.Module):
    """24-band psychoacoustic encoder with Dual Scale-Aware conditioning."""

    def __init__(self, has_teacher: bool, embed_dim: int = 64, eps: float = 1e-7) -> None:
        super().__init__()
        self.has_teacher = has_teacher
        self.embed_dim = embed_dim
        self.eps = eps

        offsets = [0]
        for w in BAND_WIDTHS:
            offsets.append(offsets[-1] + w)
        self.band_slices = list(zip(offsets[:-1], offsets[1:]))

        self.band_projs = nn.ModuleList()
        n_scale = 7 if has_teacher else 4
        ch_mul = 8 if has_teacher else 4
        for w in BAND_WIDTHS:
            in_dim = ch_mul * w + n_scale
            self.band_projs.append(
                nn.Sequential(
                    nn.LayerNorm(in_dim),
                    nn.Linear(in_dim, embed_dim),
                    nn.SiLU(),
                )
            )

    def forward(
        self,
        zc_x: torch.Tensor,
        zc_s: torch.Tensor | None,
        raw_stats: torch.Tensor,
    ) -> torch.Tensor:
        """zc_x, zc_s: complex (B, 2, F=1025, T); raw_stats: (B, 3) -> (B, T, 24, D)."""
        bsz, _, _, n_frames = zc_x.shape
        raw_t = raw_stats.unsqueeze(1).expand(bsz, n_frames, 3)  # (B, T, 3)
        band_feats = []

        for b_idx, (s_b, e_b) in enumerate(self.band_slices):
            xb = zc_x[:, :, s_b:e_b, :]  # (B, 2, w_b, T)
            rx_b = torch.sqrt(torch.mean(xb.real**2 + xb.imag**2, dim=(1, 2)) + self.eps)  # (B, T)
            xb_norm = xb / rx_b[:, None, None, :]
            # (B, T, 4*w_b)
            xb_flat = torch.cat(
                [xb_norm[:, 0].real, xb_norm[:, 0].imag, xb_norm[:, 1].real, xb_norm[:, 1].imag],
                dim=1,
            ).transpose(1, 2)

            rx_m = torch.sqrt(torch.mean(xb[:, 0].real**2 + xb[:, 0].imag**2, dim=1) + self.eps)
            rx_s = torch.sqrt(torch.mean(xb[:, 1].real**2 + xb[:, 1].imag**2, dim=1) + self.eps)
            ms_x = torch.log((rx_s + self.eps) / (rx_m + self.eps))

            if self.has_teacher and zc_s is not None:
                sb = zc_s[:, :, s_b:e_b, :]
                rs_b = torch.sqrt(torch.mean(sb.real**2 + sb.imag**2, dim=(1, 2)) + self.eps)
                sb_norm = sb / rs_b[:, None, None, :]
                sb_flat = torch.cat(
                    [sb_norm[:, 0].real, sb_norm[:, 0].imag, sb_norm[:, 1].real, sb_norm[:, 1].imag],
                    dim=1,
                ).transpose(1, 2)

                rs_m = torch.sqrt(torch.mean(sb[:, 0].real**2 + sb[:, 0].imag**2, dim=1) + self.eps)
                rs_s = torch.sqrt(torch.mean(sb[:, 1].real**2 + sb[:, 1].imag**2, dim=1) + self.eps)
                ms_s = torch.log((rs_s + self.eps) / (rs_m + self.eps))

                sc = torch.cat(
                    [
                        torch.log(rx_b + self.eps).unsqueeze(-1),
                        torch.log(rs_b + self.eps).unsqueeze(-1),
                        torch.log((rs_b + self.eps) / (rx_b + self.eps)).unsqueeze(-1),
                        (ms_s - ms_x).unsqueeze(-1),
                        raw_t,
                    ],
                    dim=-1,
                )
                feat = torch.cat([xb_flat, sb_flat, sc], dim=-1)
            else:
                sc = torch.stack(
                    [
                        torch.log(rx_b + self.eps),
                        ms_x,
                        raw_t[:, :, 0],
                        torch.zeros_like(rx_b),
                    ],
                    dim=-1,
                )
                feat = torch.cat([xb_flat, sc], dim=-1)

            band_feats.append(self.band_projs[b_idx](feat))

        return torch.stack(band_feats, dim=2)  # (B, T, 24, D)


class DualAxisBlock(nn.Module):
    """Cross-Band Self-Attention + Temporal BiGRU + Dense FFN."""

    def __init__(self, embed_dim: int = 64, n_heads: int = 4) -> None:
        super().__init__()
        self.norm_band = nn.LayerNorm(embed_dim)
        self.band_attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)

        self.norm_time = nn.LayerNorm(embed_dim)
        self.time_gru = nn.GRU(embed_dim, embed_dim, bidirectional=True, batch_first=True)
        self.time_proj = nn.Linear(embed_dim * 2, embed_dim)

        self.norm_ffn = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, T, B_bands, D)."""
        bsz, n_frames, n_bands, dim = h.shape

        # 1. Cross-Band Attention
        hb = self.norm_band(h).reshape(bsz * n_frames, n_bands, dim)
        attn_out, _ = self.band_attn(hb, hb, hb, need_weights=False)
        h = h + attn_out.reshape(bsz, n_frames, n_bands, dim)

        # 2. Temporal BiGRU
        ht = self.norm_time(h).permute(0, 2, 1, 3).reshape(bsz * n_bands, n_frames, dim)
        gru_out, _ = self.time_gru(ht)
        ht_proj = self.time_proj(gru_out).reshape(bsz, n_bands, n_frames, dim).permute(0, 2, 1, 3)
        h = h + ht_proj

        # 3. Dense FFN
        h = h + self.ffn(self.norm_ffn(h))
        return h


class DirectMapDecoder(nn.Module):
    """Band-wise residual complex mapping in compressed domain -> explicit decompression."""

    def __init__(self, band_slices: list[tuple[int, int]], embed_dim: int = 64) -> None:
        super().__init__()
        self.band_slices = band_slices
        self.shared_mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
        )
        self.band_heads = nn.ModuleList()
        for s_b, e_b in band_slices:
            w_b = e_b - s_b
            lin = nn.Linear(embed_dim, 4 * w_b)
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
            self.band_heads.append(lin)

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, h: torch.Tensor, zc_x: torch.Tensor) -> torch.Tensor:
        """h: (B, T, 24, D), zc_x: complex (B, 2, F, T) -> linear complex Z_hat (B, 2, F, T)."""
        hs = self.shared_mlp(h.float())
        zc_x = zc_x.to(torch.complex64)
        deltas = []
        for b_idx, (s_b, e_b) in enumerate(self.band_slices):
            w_b = e_b - s_b
            out_b = self.band_heads[b_idx](hs[:, :, b_idx, :])  # (B, T, 4*w_b)
            out_b = out_b.transpose(1, 2).reshape(zc_x.shape[0], 4, w_b, zc_x.shape[-1])
            d_m = torch.complex(out_b[:, 0], out_b[:, 1])
            d_s = torch.complex(out_b[:, 2], out_b[:, 3])
            deltas.append(torch.stack([d_m, d_s], dim=1))
        delta_zc = torch.cat(deltas, dim=2)
        z_hat_c = zc_x + delta_zc
        return decompress_stft(z_hat_c)


class MF7Decoder(nn.Module):
    """Identity-centered 2x2 M/S 7-frame complex filter + teacher gate on LINEAR STFT."""

    def __init__(
        self,
        band_slices: list[tuple[int, int]],
        embed_dim: int = 64,
        gate_bias_init: float = -5.0,
        mf_offsets: tuple[int, ...] = MF7_OFFSETS,
        use_additive_residual: bool = False,
    ) -> None:
        super().__init__()
        self.band_slices = band_slices
        self.gate_bias_init = gate_bias_init
        self.mf_offsets = mf_offsets
        self.n_taps_other = len(mf_offsets)
        self.use_additive_residual = use_additive_residual
        self.residual_scale = 1.0
        self.residual_sweep: tuple[float, ...] | None = None

        self.shared_mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
        )

        self.per_bin_heads = nn.ModuleList()
        # 4 sub-bands per psychoacoustic band -> 96 sub-bands total
        sub_map = []
        ch_per_bin = 14 if use_additive_residual else 10
        for b_idx, (s_b, e_b) in enumerate(band_slices):
            w_b = e_b - s_b
            lin = nn.Linear(embed_dim, ch_per_bin * w_b)  # 8 dW0 + 2 g (+ 4 additive complex residual)
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
            self.per_bin_heads.append(lin)
            for local_i in range(w_b):
                sb_local = min(3, (local_i * 4) // w_b)
                sub_map.append(b_idx * 4 + sb_local)

        self.register_buffer("subband_map", torch.tensor(sub_map, dtype=torch.long), persistent=False)

        # Shared sub-band multi-frame head: predicts 4 sub-bands x 6 taps x 8 (2x2 complex)
        self.mf_head = nn.Linear(embed_dim, 4 * self.n_taps_other * 8)
        nn.init.zeros_(self.mf_head.weight)
        nn.init.zeros_(self.mf_head.bias)

    @staticmethod
    def _apply_2x2_complex(w_mat: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """w_mat: (B, T, F, 8) -> (2x2 complex), z: complex (B, 2, F, T) -> complex (B, 2, F, T)."""
        # w_mat channels: [MM_r, MM_i, MS_r, MS_i, SM_r, SM_i, SS_r, SS_i]
        w_bt = w_mat.permute(0, 2, 1, 3)  # (B, F, T, 8)
        w_mm = torch.complex(w_bt[..., 0], w_bt[..., 1])
        w_ms = torch.complex(w_bt[..., 2], w_bt[..., 3])
        w_sm = torch.complex(w_bt[..., 4], w_bt[..., 5])
        w_ss = torch.complex(w_bt[..., 6], w_bt[..., 7])
        zm, zs = z[:, 0], z[:, 1]
        out_m = w_mm * zm + w_ms * zs
        out_s = w_sm * zm + w_ss * zs
        return torch.stack([out_m, out_s], dim=1)

    @torch.amp.autocast("cuda", enabled=False)
    def forward(
        self,
        h: torch.Tensor,
        z_x: torch.Tensor,
        z_s: torch.Tensor,
        force_identity: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """h: (B, T, 24, D), z_x/z_s: uncompressed linear complex (B, 2, F, T)."""
        bsz, n_frames, n_bands, _ = h.shape
        if force_identity:
            zero_t = z_x.new_zeros((), dtype=torch.float32)
            return z_x, {"g_mean_M": zero_t, "g_mean_S": zero_t, "dw0_rms": zero_t, "dwk_rms": zero_t}

        hs = self.shared_mlp(h.float())  # (B, T, 24, D)
        z_x = z_x.to(torch.complex64)
        z_s = z_s.to(torch.complex64)

        dw0_list = []
        g_m_list = []
        g_s_list = []
        r_m_list = []
        r_s_list = []
        for b_idx, (s_b, e_b) in enumerate(self.band_slices):
            w_b = e_b - s_b
            out_b = self.per_bin_heads[b_idx](hs[:, :, b_idx, :])  # (B, T, (10|14)*w_b)
            dw0_b = 2.0 * torch.tanh(out_b[:, :, : 8 * w_b]).reshape(bsz, n_frames, w_b, 8)
            g_raw = out_b[:, :, 8 * w_b : 10 * w_b].reshape(bsz, n_frames, w_b, 2)
            g_b = torch.sigmoid(g_raw + self.gate_bias_init)
            dw0_list.append(dw0_b)
            g_m_list.append(g_b[..., 0].transpose(1, 2))  # (B, w_b, T)
            g_s_list.append(g_b[..., 1].transpose(1, 2))
            if self.use_additive_residual:
                r_raw = 2.0 * torch.tanh(out_b[:, :, 10 * w_b : 14 * w_b]).reshape(bsz, n_frames, w_b, 4)
                zx_b = z_x[:, :, s_b:e_b, :]
                zs_b = z_s[:, :, s_b:e_b, :]
                band_rms = torch.sqrt(
                    torch.mean(zx_b.abs() ** 2 + zs_b.abs() ** 2, dim=(1, 2), keepdim=True) + 1e-6
                )  # (B, 1, 1, T)
                r_m_b = torch.complex(r_raw[..., 0].transpose(1, 2), r_raw[..., 1].transpose(1, 2)) * band_rms.squeeze(1)
                r_s_b = torch.complex(r_raw[..., 2].transpose(1, 2), r_raw[..., 3].transpose(1, 2)) * band_rms.squeeze(1)
                r_m_list.append(r_m_b)
                r_s_list.append(r_s_b)

        dw0 = torch.cat(dw0_list, dim=2)  # (B, T, F, 8)
        g_m = torch.cat(g_m_list, dim=1)  # (B, F, T)
        g_s = torch.cat(g_s_list, dim=1)  # (B, F, T)
        g_ms = torch.stack([g_m, g_s], dim=1)  # (B, 2, F, T)

        # Filtered k=0: Z_x + dW0(Z_x)
        f_w = z_x + self._apply_2x2_complex(dw0, z_x)

        # Sub-band multi-frame taps k in {-5..-1, +1}
        mf_raw = 2.0 * torch.tanh(self.mf_head(hs))  # (B, T, 24, 4*6*8)
        mf_sub = mf_raw.reshape(bsz, n_frames, 96, self.n_taps_other, 8)
        dwk_bins = mf_sub[:, :, self.subband_map, :, :]  # (B, T, F, 6, 8)

        for tap_idx, k_off in enumerate(self.mf_offsets):
            if abs(k_off) >= n_frames:
                continue
            if k_off < 0:
                pad_l = -k_off
                z_shifted = F.pad(z_x[..., : n_frames - pad_l], (pad_l, 0))
            else:
                pad_r = k_off
                z_shifted = F.pad(z_x[..., pad_r:], (0, pad_r))
            f_w = f_w + self._apply_2x2_complex(dwk_bins[:, :, :, tap_idx, :], z_shifted)

        z_hat = (1.0 - g_ms) * f_w + g_ms * z_s
        if self.use_additive_residual:
            r_ms = torch.stack([torch.cat(r_m_list, dim=1), torch.cat(r_s_list, dim=1)], dim=1)
            if self.residual_sweep is None:
                z_hat = z_hat + self.residual_scale * r_ms
            else:
                if bsz != 1:
                    raise ValueError("A residual sweep requires one input clip")
                scales = r_ms.real.new_tensor(self.residual_sweep).view(-1, 1, 1, 1)
                z_hat = z_hat + scales * r_ms
        diag = {
            # The training loss uses these terms. Convert to scalars at logging time.
            "g_mean_M": g_m.mean(),
            "g_mean_S": g_s.mean(),
            "dw0_rms": torch.sqrt(torch.mean(dw0**2) + 1e-12),
            "dwk_rms": torch.sqrt(torch.mean(dwk_bins**2) + 1e-12),
        }
        if self.use_additive_residual:
            diag["res_rms"] = torch.sqrt(torch.mean(r_ms.abs() ** 2) + 1e-12).detach()
        return z_hat, diag


class HybridV4Net(nn.Module):
    """Unified Hybrid-v4 Network supporting V4-0 (NoGuide), V4-1 (DirectMap), V4-2 (MF7), and V4-3 (Dilated MF9)."""

    def __init__(
        self,
        variant: Literal["v4_0", "v4_1", "v4_2", "v4_3", "v4_4"],
        embed_dim: int = 64,
        n_blocks: int = 4,
        n_fft: int = 2048,
        hop: int = 512,
        gate_bias_init: float = -5.0,
    ) -> None:
        super().__init__()
        self.variant = variant
        self.has_teacher = variant in ("v4_1", "v4_2", "v4_3", "v4_4")
        self.n_fft = n_fft
        self.hop = hop

        self.encoder = PsychoacousticBandEncoder(has_teacher=self.has_teacher, embed_dim=embed_dim)
        self.blocks = nn.ModuleList([DualAxisBlock(embed_dim=embed_dim, n_heads=4) for _ in range(n_blocks)])

        if variant in ("v4_0", "v4_1"):
            self.decoder: nn.Module = DirectMapDecoder(self.encoder.band_slices, embed_dim=embed_dim)
        elif variant == "v4_2":
            self.decoder = MF7Decoder(
                self.encoder.band_slices,
                embed_dim=embed_dim,
                gate_bias_init=gate_bias_init,
                mf_offsets=MF7_OFFSETS,
            )
        elif variant == "v4_3":
            self.decoder = MF7Decoder(
                self.encoder.band_slices,
                embed_dim=embed_dim,
                gate_bias_init=gate_bias_init,
                mf_offsets=MF9_DILATED_OFFSETS,
            )
        elif variant == "v4_4":
            self.decoder = MF7Decoder(
                self.encoder.band_slices,
                embed_dim=embed_dim,
                gate_bias_init=gate_bias_init,
                mf_offsets=MF9_DILATED_OFFSETS,
                use_additive_residual=True,
            )
        else:
            raise ValueError(f"Unknown variant: {variant}")

    def forward_normalized(
        self,
        x_in: torch.Tensor,
        s_in: torch.Tensor | None,
        raw_stats: torch.Tensor,
        force_identity: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Forward on track-normalized chunks (B, 2, N)."""
        n_samples = x_in.shape[-1]
        z_x = stft_ms(lr_to_ms(x_in), n_fft=self.n_fft, hop=self.hop)
        zc_x = compress_stft(z_x)

        if self.has_teacher:
            assert s_in is not None
            z_s = stft_ms(lr_to_ms(s_in), n_fft=self.n_fft, hop=self.hop)
            zc_s = compress_stft(z_s)
        else:
            z_s = None
            zc_s = None

        h = self.encoder(zc_x, zc_s, raw_stats)
        for blk in self.blocks:
            h = blk(h)

        if self.variant in ("v4_0", "v4_1"):
            z_hat = self.decoder(h, zc_x)
            diag: dict[str, torch.Tensor] = {}
        else:
            assert z_s is not None
            z_hat, diag = self.decoder(h, z_x, z_s, force_identity=force_identity)

        y_ms = istft_ms(z_hat, length=n_samples, n_fft=self.n_fft, hop=self.hop)
        y_lr = ms_to_lr(y_ms)
        return y_lr, diag

    def forward_track(
        self,
        x: torch.Tensor,
        s: torch.Tensor | None = None,
        chunk_sec: float = 8.0,
        overlap_ratio: float = 0.5,
        sr: int = 44100,
        force_identity: bool = False,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Full-track inference with track-level normalization & Hann overlap-add."""
        x_in, s_in, scale, raw_stats = compute_track_norm(x, s if self.has_teacher else None)
        n_samples = x_in.shape[-1]
        chunk_len = int(round(chunk_sec * sr))
        hop_len = max(1, int(round(chunk_len * (1.0 - overlap_ratio))))

        if n_samples <= chunk_len:
            y_in, diag = self.forward_normalized(x_in, s_in, raw_stats, force_identity=force_identity)
            y_out = y_in * scale
            if self.variant in ("v4_3", "v4_4") and s is not None and not force_identity:
                rms_s = torch.sqrt(torch.mean(s.float() ** 2) + 1e-12)
                if self.decoder.residual_sweep is not None:
                    rms_y = torch.sqrt(y_out.float().square().mean(dim=(1, 2), keepdim=True) + 1e-12)
                    y_out = y_out * torch.where((rms_s > 1e-6) & (rms_y > 1e-6), rms_s / rms_y, 1.0)
                else:
                    rms_y = torch.sqrt(torch.mean(y_out.float() ** 2) + 1e-12)
                    if rms_s > 1e-6 and rms_y > 1e-6:
                        y_out = y_out * (rms_s / rms_y).to(y_out.dtype)
            return y_out, {k: float(v.item()) for k, v in diag.items()}

        # Pad by hop_len on both sides (reflect) so Hann OLA sum_w == 1.0 everywhere
        pad_l = hop_len
        pad_r = hop_len
        x_pad = F.pad(x_in, (pad_l, pad_r), mode="reflect")
        s_pad = F.pad(s_in, (pad_l, pad_r), mode="reflect") if s_in is not None else None
        total_pad = x_pad.shape[-1]

        win = torch.hann_window(chunk_len, periodic=True, device=x.device, dtype=x.dtype)
        sweep = getattr(self.decoder, "residual_sweep", None)
        acc = torch.zeros_like(x_pad) if sweep is None else x_pad.new_zeros(len(sweep), x_pad.shape[1], total_pad)
        w_sum = torch.zeros((1, 1, total_pad), device=x.device, dtype=x.dtype)

        starts = list(range(0, max(1, total_pad - chunk_len + 1), hop_len))
        if starts[-1] + chunk_len < total_pad:
            starts.append(total_pad - chunk_len)

        diag_acc: dict[str, list[float]] = {}
        for st in starts:
            en = st + chunk_len
            xc = x_pad[:, :, st:en]
            sc = s_pad[:, :, st:en] if s_pad is not None else None
            yc, diag = self.forward_normalized(xc, sc, raw_stats, force_identity=force_identity)
            acc[:, :, st:en] += yc * win[None, None, :]
            w_sum[:, :, st:en] += win[None, None, :]
            for k, v in diag.items():
                diag_acc.setdefault(k, []).append(float(v.item()))

        y_pad = acc / torch.clamp(w_sum, min=1e-6)
        y_in = y_pad[:, :, pad_l : pad_l + n_samples]
        y_out = y_in * scale
        if self.variant in ("v4_3", "v4_4") and s is not None and not force_identity:
            rms_s = torch.sqrt(torch.mean(s.float() ** 2) + 1e-12)
            if sweep is not None:
                rms_y = torch.sqrt(y_out.float().square().mean(dim=(1, 2), keepdim=True) + 1e-12)
                y_out = y_out * torch.where((rms_s > 1e-6) & (rms_y > 1e-6), rms_s / rms_y, 1.0)
            else:
                rms_y = torch.sqrt(torch.mean(y_out.float() ** 2) + 1e-12)
                if rms_s > 1e-6 and rms_y > 1e-6:
                    y_out = y_out * (rms_s / rms_y).to(y_out.dtype)
        diag_mean = {k: sum(vs) / len(vs) for k, vs in diag_acc.items()}
        return y_out, diag_mean


def count_parameters_breakdown(model: HybridV4Net) -> dict[str, int]:
    """Exact PyTorch parameter count by module and shared vs band-specific."""
    enc = sum(p.numel() for p in model.encoder.parameters())
    bb = sum(p.numel() for p in model.blocks.parameters())
    dec = sum(p.numel() for p in model.decoder.parameters())
    total = sum(p.numel() for p in model.parameters())
    return {"encoder": enc, "backbone": bb, "decoder": dec, "total": total}
