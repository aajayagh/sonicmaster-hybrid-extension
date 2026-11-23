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
    ) -> tuple[torch.Tensor, dict[str, float]]:
        y_f = y_hat.float()
        t_f = target.float()
        l_stft = self.mrstft(y_f, t_f)
        l_wav = F.l1_loss(y_f, t_f)

        zh_c = compress_stft(stft_ms(lr_to_ms(y_f), n_fft=2048, hop=512))
        zt_c = compress_stft(stft_ms(lr_to_ms(t_f), n_fft=2048, hop=512))
        l_comp = F.l1_loss(zh_c.real, zt_c.real) + F.l1_loss(zh_c.imag, zt_c.imag)

        loss = 1.0 * l_stft + 0.5 * l_comp + 0.5 * l_wav
        parts = {
            "l_stft": float(l_stft.item()),
            "l_comp": float(l_comp.item()),
            "l_wav": float(l_wav.item()),
        }
        if lambda_kl > 0.0:
            l_kl = self.mel_kl(y_f, t_f)
            loss = loss + lambda_kl * l_kl
            parts["l_kl"] = float(l_kl.item())

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
