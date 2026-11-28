"""Self-contained Dual-Domain Discriminators (Multi-Scale Waveform + Multi-Resolution Complex-STFT) for Hybrid-v4 Stage 3."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm

from scripts.extension.v4_models import compress_stft, lr_to_ms


class ScaleWaveformDiscriminator(nn.Module):
    """Single-scale 1D M/S waveform discriminator sensitive to phase smearing & temporal coherence."""

    def __init__(self) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            [
                spectral_norm(nn.Conv1d(2, 16, kernel_size=15, stride=1, padding=7)),
                spectral_norm(nn.Conv1d(16, 32, kernel_size=41, stride=4, groups=4, padding=20)),
                spectral_norm(nn.Conv1d(32, 64, kernel_size=41, stride=4, groups=8, padding=20)),
                spectral_norm(nn.Conv1d(64, 128, kernel_size=41, stride=4, groups=16, padding=20)),
                spectral_norm(nn.Conv1d(128, 128, kernel_size=5, stride=1, padding=2)),
                spectral_norm(nn.Conv1d(128, 1, kernel_size=3, stride=1, padding=1)),
            ]
        )

    def forward(self, y_ms: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = y_ms
        fmaps: list[torch.Tensor] = []
        for idx, conv in enumerate(self.convs):
            x = conv(x)
            if idx < len(self.convs) - 1:
                x = F.leaky_relu(x, 0.1)
                fmaps.append(x)
        return x, fmaps


class ResComplexSTFTDiscriminator(nn.Module):
    """2D Complex-STFT + log-magnitude M/S discriminator at a fixed FFT resolution."""

    def __init__(self, n_fft: int) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop = n_fft // 4
        self.convs = nn.ModuleList(
            [
                spectral_norm(nn.Conv2d(6, 32, kernel_size=(3, 9), stride=(1, 1), padding=(1, 4))),
                spectral_norm(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(2, 2), padding=(1, 4))),
                spectral_norm(nn.Conv2d(32, 64, kernel_size=(3, 7), stride=(2, 2), padding=(1, 3))),
                spectral_norm(nn.Conv2d(64, 64, kernel_size=(3, 5), stride=(2, 2), padding=(1, 2))),
                spectral_norm(nn.Conv2d(64, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
            ]
        )

    def _complex_features(self, y_ms: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast("cuda", enabled=False):
            bsz, n_ch, n_samples = y_ms.shape
            flat = y_ms.float().reshape(bsz * n_ch, n_samples)
            win = torch.hann_window(self.n_fft, periodic=True, device=y_ms.device, dtype=torch.float32)
            z = torch.stft(
                flat,
                n_fft=self.n_fft,
                hop_length=self.hop,
                win_length=self.n_fft,
                window=win,
                center=True,
                return_complex=True,
            ).reshape(bsz, n_ch, self.n_fft // 2 + 1, -1)
            zc = compress_stft(z, power=0.3)
            log_mag = torch.log(z.abs() + 1e-5)
            return torch.stack(
                [zc[:, 0].real, zc[:, 0].imag, log_mag[:, 0], zc[:, 1].real, zc[:, 1].imag, log_mag[:, 1]],
                dim=1,
            )

    def forward(self, y_ms: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self._complex_features(y_ms).to(y_ms.dtype)
        fmaps: list[torch.Tensor] = []
        for idx, conv in enumerate(self.convs):
            x = conv(x)
            if idx < len(self.convs) - 1:
                x = F.leaky_relu(x, 0.1)
                fmaps.append(x)
        return x, fmaps


class HybridV4Discriminator(nn.Module):
    """Dual-Domain Discriminator: 3 Multi-Scale Waveform + 3 Multi-Resolution Complex-STFT sub-discriminators."""

    def __init__(self, stft_ffts: tuple[int, ...] = (512, 1024, 2048)) -> None:
        super().__init__()
        self.wave_discs = nn.ModuleList([ScaleWaveformDiscriminator() for _ in range(3)])
        self.stft_discs = nn.ModuleList([ResComplexSTFTDiscriminator(n_fft=n_fft) for n_fft in stft_ffts])

    def forward(self, y_lr: torch.Tensor) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        y_ms = lr_to_ms(y_lr.float()).to(y_lr.dtype)
        logits: list[torch.Tensor] = []
        fmaps: list[list[torch.Tensor]] = []

        w_in = y_ms
        for idx, disc in enumerate(self.wave_discs):
            if idx > 0:
                w_in = F.avg_pool1d(w_in, kernel_size=4, stride=2, padding=1)
            lg, fm = disc(w_in)
            logits.append(lg)
            fmaps.append(fm)

        for disc in self.stft_discs:
            lg, fm = disc(y_ms)
            logits.append(lg)
            fmaps.append(fm)

        return logits, fmaps


def hinge_disc_loss(real_logits: list[torch.Tensor], fake_logits: list[torch.Tensor]) -> torch.Tensor:
    """Mean Hinge adversarial loss across all sub-discriminators for updating D."""
    total = real_logits[0].new_zeros((), dtype=torch.float32)
    for rl, fl in zip(real_logits, fake_logits):
        total = total + (F.relu(1.0 - rl.float()).mean() + F.relu(1.0 + fl.float()).mean())
    return total / float(len(real_logits))


def hinge_gen_loss(fake_logits: list[torch.Tensor]) -> torch.Tensor:
    """Mean Hinge adversarial loss across all sub-discriminators for updating G."""
    total = fake_logits[0].new_zeros((), dtype=torch.float32)
    for fl in fake_logits:
        total = total + F.relu(1.0 - fl.float()).mean()
    return total / float(len(fake_logits))


def normalized_feature_matching_loss(
    real_fmaps: list[list[torch.Tensor]],
    fake_fmaps: list[list[torch.Tensor]],
    eps: float = 1e-4,
) -> torch.Tensor:
    """EnCodec-style normalized L1 feature matching loss across all sub-discriminators & feature layers."""
    total = real_fmaps[0][0].new_zeros((), dtype=torch.float32)
    count = 0
    for r_list, f_list in zip(real_fmaps, fake_fmaps):
        for rf, ff in zip(r_list, f_list):
            rf_det = rf.detach()
            denom = rf_det.abs().mean().float() + eps
            total = total + F.l1_loss(ff, rf_det).float() / denom
            count += 1
    return total / float(max(1, count))
