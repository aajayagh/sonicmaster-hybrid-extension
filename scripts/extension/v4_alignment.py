"""Teacher waveform alignment for Hybrid-v4."""
from __future__ import annotations

import numpy as np

from scripts.extension.branch_r import estimate_lag_samples


def align_teacher_to_input(x: np.ndarray, s: np.ndarray, max_lag: int = 512) -> np.ndarray:
    """Align channel-first teacher audio with zero padding at the exposed edge."""
    lag, _ = estimate_lag_samples(x.T, s.T, max_lag=max_lag)
    if lag == 0:
        return s
    aligned = np.zeros_like(s)
    if 0 < lag < s.shape[-1]:
        aligned[..., :-lag] = s[..., lag:]
    elif -s.shape[-1] < lag < 0:
        aligned[..., -lag:] = s[..., :lag]
    return aligned
