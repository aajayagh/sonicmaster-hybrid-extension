"""Loss functions for Hybrid-v4 (Stage 1: MR-STFT + CompComplex L1 + Wav L1; Stage 2: + Stereo + Reg)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from scripts.extension.v4_models import compress_stft, lr_to_ms, stft_ms


class MelKLLoss(nn.Module):
    """Differentiable 128-bin Mel spectral KL + log-mel L1 loss matching eval_dev mel specs."""

    def __init__(self, sr: int = 44100, n_fft: int = 2048, hop: int = 512, n_mels: int = 128, eps: float = 1e-7) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop
        self.eps = eps
        fb = torchaudio.functional.melscale_fbanks(
            n_freqs=n_fft // 2 + 1,
            f_min=0.0,
            f_max=float(sr) / 2.0,
            n_mels=n_mels,
            sample_rate=sr,
            norm="slaney",
            mel_scale="slaney",
        )
        self.register_buffer("mel_fb", fb, persistent=False)  # (F, M)

    def _mel_power(self, y: torch.Tensor) -> torch.Tensor:
        y_mono = 0.5 * (y[:, 0] + y[:, 1])
        win = torch.hann_window(self.n_fft, periodic=True, device=y.device, dtype=torch.float32)
        z = torch.stft(
            y_mono.float(),
            n_fft=self.n_fft,
            hop_length=self.hop,
            win_length=self.n_fft,
            window=win,
            center=True,
            return_complex=True,
        )
        pow_spec = z.real**2 + z.imag**2
        return torch.einsum("fm,bft->bmt", self.mel_fb, pow_spec)

    def forward(self, y_hat: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mh = self._mel_power(y_hat)
        mt = self._mel_power(target)
        ph_raw = mh.mean(dim=-1) + self.eps
        pt_raw = mt.mean(dim=-1) + self.eps
        ph = ph_raw / torch.clamp(ph_raw.sum(dim=-1, keepdim=True), min=self.eps)
        pt = pt_raw / torch.clamp(pt_raw.sum(dim=-1, keepdim=True), min=self.eps)
        kl = (pt * (torch.log(pt) - torch.log(ph))).sum(dim=-1).mean()
        log_mel_l1 = F.l1_loss(torch.log(mh + self.eps), torch.log(mt + self.eps))
        return kl + 0.1 * log_mel_l1


class HighFreqBrightLoss(nn.Module):
    """Differentiable >= 6 kHz energy-ratio + log-mag loss targeting the 'bright' guard."""

    def __init__(self, sr: int = 44100, n_fft: int = 2048, hop: int = 512, f_lo: float = 6000.0, eps: float = 1e-8) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop
        self.bin_lo = int(round(f_lo * n_fft / float(sr)))
        self.eps = eps

    def forward(self, y_hat: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        yh_m = 0.5 * (y_hat[:, 0] + y_hat[:, 1]).float()
        yt_m = 0.5 * (target[:, 0] + target[:, 1]).float()
        win = torch.hann_window(self.n_fft, periodic=True, device=y_hat.device, dtype=torch.float32)
        zh = torch.stft(yh_m, n_fft=self.n_fft, hop_length=self.hop, win_length=self.n_fft, window=win, center=True, return_complex=True)
        zt = torch.stft(yt_m, n_fft=self.n_fft, hop_length=self.hop, win_length=self.n_fft, window=win, center=True, return_complex=True)
        ph = zh.real**2 + zh.imag**2
        pt = zt.real**2 + zt.imag**2
        eh_ratio = ph[:, self.bin_lo :, :].sum(dim=(-2, -1)) / (ph.sum(dim=(-2, -1)) + self.eps)
        et_ratio = pt[:, self.bin_lo :, :].sum(dim=(-2, -1)) / (pt.sum(dim=(-2, -1)) + self.eps)
        l_ratio = F.l1_loss(eh_ratio, et_ratio) * 50.0
        l_hf_mag = F.l1_loss(
            torch.log(torch.sqrt(ph[:, self.bin_lo :, :] + self.eps)),
            torch.log(torch.sqrt(pt[:, self.bin_lo :, :] + self.eps)),
        )
        return l_ratio + 0.2 * l_hf_mag


class SymmetricBandRatio6kLoss(nn.Module):
    """Symmetric linear-power energy-ratio L1 (+ optional quad/log HF terms) in [6,000, 22,050) Hz."""

    def __init__(
        self,
        sr: int = 44100,
        n_fft: int = 2048,
        hop: int = 512,
        f_lo: float = 6000.0,
        f_hi: float = 22050.0,
        scale: float = 20.0,
        eps: float = 1e-8,
        quad_scale: float = 0.0,
        log_weight: float = 0.0,
        log_eps: float = 1e-4,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop
        self.scale = scale
        self.eps = eps
        self.quad_scale = quad_scale
        self.log_weight = log_weight
        self.log_eps = log_eps
        freqs = torch.linspace(0.0, float(sr) / 2.0, n_fft // 2 + 1)
        band_mask = (freqs >= f_lo) & (freqs < f_hi)
        self.register_buffer("band_mask", band_mask, persistent=False)

    def _hf_ratio(self, y: torch.Tensor) -> torch.Tensor:
        y_m = 0.5 * (y[:, 0] + y[:, 1]).float()
        win = torch.hann_window(self.n_fft, periodic=True, device=y.device, dtype=torch.float32)
        z = torch.stft(y_m, n_fft=self.n_fft, hop_length=self.hop, win_length=self.n_fft, window=win, center=True, return_complex=True)
        pw = z.real**2 + z.imag**2
        return pw[:, self.band_mask, :].sum(dim=(-2, -1)) / (pw.sum(dim=(-2, -1)) + self.eps)

    def _ratio_loss(
        self,
        eh_ratio: torch.Tensor,
        et_ratio: torch.Tensor,
        quad_scale: float = 0.0,
        log_weight: float = 0.0,
    ) -> torch.Tensor:
        diff = eh_ratio - et_ratio
        loss = self.scale * diff.abs().mean()
        if quad_scale > 0.0:
            loss = loss + float(quad_scale) * diff.square().mean()
        if log_weight > 0.0:
            loss = loss + float(log_weight) * F.l1_loss(
                torch.log(eh_ratio + self.log_eps),
                torch.log(et_ratio + self.log_eps),
            )
        return loss

    def forward(
        self,
        y_hat: torch.Tensor,
        target: torch.Tensor,
        quad_scale: float | None = None,
        log_weight: float | None = None,
    ) -> torch.Tensor:
        qs = self.quad_scale if quad_scale is None else float(quad_scale)
        lw = self.log_weight if log_weight is None else float(log_weight)
        return self._ratio_loss(self._hf_ratio(y_hat), self._hf_ratio(target), quad_scale=qs, log_weight=lw)


class ModSpecLoss(nn.Module):
    """Differentiable Table-2 Modulation Spectrum Distance on full-band and 1-12 kHz sub-band."""

    def __init__(self, n_fft: int = 1024, hop: int = 512, n_mod_bins: int = 20, eps: float = 1e-8) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop
        self.n_mod_bins = n_mod_bins
        self.eps = eps

    def _norm_mod(self, mag: torch.Tensor) -> torch.Tensor:
        env = mag - mag.mean(dim=-1, keepdim=True)
        spec_c = torch.fft.fft(env, dim=-1)
        spec = torch.sqrt(spec_c.real**2 + spec_c.imag**2 + 1e-12)[..., : self.n_mod_bins]
        flat = spec.reshape(spec.shape[0], -1)
        norm = torch.sqrt(flat.square().sum(dim=-1, keepdim=True) + 1e-12)
        return flat / norm

    def forward(self, y_hat: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        yh_m = 0.5 * (y_hat[:, 0] + y_hat[:, 1]).float()
        yt_m = 0.5 * (target[:, 0] + target[:, 1]).float()
        win = torch.hann_window(self.n_fft, periodic=True, device=y_hat.device, dtype=torch.float32)
        zh = torch.stft(yh_m, n_fft=self.n_fft, hop_length=self.hop, win_length=self.n_fft, window=win, center=True, return_complex=True)
        zt = torch.stft(yt_m, n_fft=self.n_fft, hop_length=self.hop, win_length=self.n_fft, window=win, center=True, return_complex=True)
        mh = torch.sqrt(zh.real**2 + zh.imag**2 + self.eps)
        mt = torch.sqrt(zt.real**2 + zt.imag**2 + self.eps)
        if mh.shape[-1] < self.n_mod_bins:
            return y_hat.new_zeros(())
        d_full = torch.sqrt(((self._norm_mod(mh) - self._norm_mod(mt)) ** 2).sum(dim=-1) + 1e-12).mean()
        d_midhi = torch.sqrt(((self._norm_mod(mh[:, 24:280, :]) - self._norm_mod(mt[:, 24:280, :])) ** 2).sum(dim=-1) + 1e-12).mean()
        return 0.5 * d_full + 0.5 * d_midhi


class MultiResSTFTLoss(nn.Module):
    """Multi-resolution STFT spectral convergence + log-magnitude L1 on L/R and M/S."""

    def __init__(self, fft_sizes: tuple[int, ...] = (512, 1024, 2048), eps: float = 1e-7) -> None:
        super().__init__()
        self.fft_sizes = fft_sizes
        self.eps = eps

    def _stft_mag(self, y: torch.Tensor, n_fft: int) -> torch.Tensor:
        hop = n_fft // 4
        bsz, n_ch, n_samples = y.shape
        flat = y.reshape(bsz * n_ch, n_samples)
        win = torch.hann_window(n_fft, periodic=True, device=y.device, dtype=torch.float32)
        z = torch.stft(flat.float(), n_fft=n_fft, hop_length=hop, win_length=n_fft, window=win, center=True, return_complex=True)
        return torch.sqrt(z.real**2 + z.imag**2 + self.eps)

    def forward(self, y_hat: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = y_hat.new_zeros(())
        y_ms = lr_to_ms(y_hat)
        t_ms = lr_to_ms(target)
        for sig_h, sig_t in ((y_hat, target), (y_ms, t_ms)):
            for n_fft in self.fft_sizes:
                mh = self._stft_mag(sig_h, n_fft)
                mt = self._stft_mag(sig_t, n_fft)
                sc = torch.linalg.norm(mt - mh, dim=(-2, -1)) / torch.clamp(torch.linalg.norm(mt, dim=(-2, -1)), min=1e-1)
                sc = torch.clamp(sc, max=5.0)
                l_mag = F.l1_loss(torch.log(mh + self.eps), torch.log(mt + self.eps))
                loss = loss + 0.5 * sc.mean() + l_mag
        return loss / (2.0 * len(self.fft_sizes))


class HybridV4Loss(nn.Module):
    """Combined Stage 1 + Stage 2 loss for Hybrid-v4."""

    def __init__(self) -> None:
        super().__init__()
        self.mrstft = MultiResSTFTLoss()
        self.mel_kl = MelKLLoss()
        self.hf_bright = HighFreqBrightLoss()
        self.band_ratio_6k = SymmetricBandRatio6kLoss()
        self.mod_spec = ModSpecLoss()

    def forward(
        self,
        y_hat: torch.Tensor,
        target: torch.Tensor,
        diag: dict[str, torch.Tensor],
        stage: int = 1,
        lambda_ms: float = 0.2,
        lambda_g: float = 0.002,
        lambda_w: float = 0.005,
        lambda_kl: float = 0.0,
        lambda_hf: float = 0.0,
        lambda_br: float = 0.0,
        w_comp: float = 0.5,
        w_wav: float = 0.5,
        lambda_mod: float = 0.0,
        rev_target: torch.Tensor | None = None,
        lambda_rev: float = 0.0,
        rev_fp_weight: float = 2.0,
        lambda_br_p: float = 0.0,
        br_quad_scale: float = 0.0,
        br_log_weight: float = 0.0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        y_f = y_hat.float()
        t_f = target.float()
        l_stft = self.mrstft(y_f, t_f)
        loss = 1.0 * l_stft
        keep_phase = (1.0 - rev_target.to(y_f.device, dtype=torch.float32)) if rev_target is not None else None

        if w_wav > 0.0:
            if keep_phase is None:
                l_wav = F.l1_loss(y_f, t_f)
            else:
                l_wav = (keep_phase * torch.abs(y_f - t_f).mean(dim=(-2, -1))).mean()
            loss = loss + w_wav * l_wav
            wav_val = float(l_wav.item())
        else:
            wav_val = float(F.l1_loss(y_f.detach(), t_f).item())

        if w_comp > 0.0:
            zh_c = compress_stft(stft_ms(lr_to_ms(y_f), n_fft=2048, hop=512))
            zt_c = compress_stft(stft_ms(lr_to_ms(t_f), n_fft=2048, hop=512))
            if keep_phase is None:
                l_comp = F.l1_loss(zh_c.real, zt_c.real) + F.l1_loss(zh_c.imag, zt_c.imag)
            else:
                l_comp_per = torch.abs(zh_c.real - zt_c.real).mean(dim=(-3, -2, -1)) + torch.abs(zh_c.imag - zt_c.imag).mean(dim=(-3, -2, -1))
                l_comp = (keep_phase * l_comp_per).mean()
            loss = loss + w_comp * l_comp
            comp_val = float(l_comp.item())
        else:
            comp_val = 0.0

        parts = {
            "l_stft": float(l_stft.item()),
            "l_comp": comp_val,
            "l_wav": wav_val,
        }
        if lambda_kl > 0.0:
            l_kl = self.mel_kl(y_f, t_f)
            loss = loss + lambda_kl * l_kl
            parts["l_kl"] = float(l_kl.item())
        if lambda_hf > 0.0:
            l_hf = self.hf_bright(y_f, t_f)
            loss = loss + lambda_hf * l_hf
            parts["l_hf"] = float(l_hf.item())
        if lambda_br > 0.0:
            l_br = self.band_ratio_6k(y_f, t_f, quad_scale=br_quad_scale, log_weight=br_log_weight)
            loss = loss + lambda_br * l_br
            parts["l_br"] = float(l_br.item())
        if lambda_br_p > 0.0 and "p_hf_ratio" in diag:
            et_ratio = self.band_ratio_6k._hf_ratio(t_f)
            l_br_p = self.band_ratio_6k._ratio_loss(
                diag["p_hf_ratio"].float(),
                et_ratio,
                quad_scale=br_quad_scale,
                log_weight=br_log_weight,
            )
            loss = loss + lambda_br_p * l_br_p
            parts["l_br_p"] = float(l_br_p.item())
        if lambda_mod > 0.0:
            l_mod = self.mod_spec(y_f, t_f)
            loss = loss + lambda_mod * l_mod
            parts["l_mod"] = float(l_mod.item())
        if rev_target is not None and lambda_rev > 0.0 and "rev_logit" in diag:
            rt_f = rev_target.to(diag["rev_logit"].device, dtype=torch.float32)
            bce_per = F.binary_cross_entropy_with_logits(diag["rev_logit"].float(), rt_f, reduction="none")
            wts = torch.where(rt_f > 0.5, torch.ones_like(rt_f), torch.full_like(rt_f, float(rev_fp_weight)))
            l_rev = (bce_per * wts).mean()
            loss = loss + lambda_rev * l_rev
            parts["l_rev"] = float(l_rev.item())

        if stage >= 2:
            y_ms = lr_to_ms(y_f)
            t_ms = lr_to_ms(t_f)
            ry = torch.sqrt(torch.mean(y_ms[:, 1] ** 2, dim=-1) + 1e-8) / (
                torch.sqrt(torch.mean(y_ms[:, 0] ** 2, dim=-1) + 1e-8) + 1e-6
            )
            rt = torch.sqrt(torch.mean(t_ms[:, 1] ** 2, dim=-1) + 1e-8) / (
                torch.sqrt(torch.mean(t_ms[:, 0] ** 2, dim=-1) + 1e-8) + 1e-6
            )
            l_ms = F.l1_loss(ry, rt)
            loss = loss + lambda_ms * l_ms
            parts["l_ms"] = float(l_ms.item())

            if "g_mean_M" in diag:
                l_g = 0.5 * (diag["g_mean_M"] + diag["g_mean_S"])
                l_w = diag["dw0_rms"] ** 2 + diag["dwk_rms"] ** 2
                loss = loss + lambda_g * l_g + lambda_w * l_w
                parts["l_g"] = float(l_g.item())
                parts["l_w"] = float(l_w.item())

        parts["total"] = float(loss.item())
        return loss, parts
