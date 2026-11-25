"""Deterministic, shared CLAP input for matched music restoration comparisons."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def clap_window(frames: int) -> tuple[int, int]:
    length = min(frames, 10 * 44100)
    return (frames - length) // 2, length


def read_clap_input(path: Path | str, frames: int) -> np.ndarray:
    start, length = clap_window(frames)
    audio, rate = sf.read(path, start=start, frames=length, dtype="float32", always_2d=True)
    if rate != 44100 or len(audio) != length:
        raise ValueError(f"Expected the common crop at 44.1 kHz: {path}")
    # CLAP receives <=480,000 mono samples, so its random truncation does not run.
    return resample_poly(audio.mean(axis=1), 160, 147)


def matched_embeddings(model, paths, frame_counts, batch_size: int = 8) -> np.ndarray:
    result = []
    for start in range(0, len(paths), batch_size):
        end = start + batch_size
        batch = [read_clap_input(path, frames)
                 for path, frames in zip(paths[start:end], frame_counts[start:end], strict=True)]
        result.append(model.get_audio_embedding_from_data(batch, use_tensor=False))
    return np.concatenate(result)
