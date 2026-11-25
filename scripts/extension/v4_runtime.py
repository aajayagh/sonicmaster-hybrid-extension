"""Shared inference settings and metrics for Hybrid-v4."""
from __future__ import annotations

import numpy as np

from scripts.extension.eval_dev import kl_div, logmel, mel_ssim


def checkpoint_inference_config(
    checkpoint: dict, variant: str, gate_bias: float | None = None, chunk_sec: float | None = None
) -> dict:
    saved_variant = checkpoint.get("variant")
    if saved_variant is not None and saved_variant != variant:
        raise ValueError(f"Checkpoint variant {saved_variant!r} does not match {variant!r}")
    saved = checkpoint.get("inference_config", {})
    # Historical checkpoints predate this metadata. Match the recorded Dev runners.
    legacy_bias = -1.5 if variant in {"v4_3", "v4_4"} else -3.5
    return {
        "hop": int(saved.get("hop", 512)),
        "chunk_sec": float(chunk_sec if chunk_sec is not None else saved.get("chunk_sec", 8.0)),
        "gate_bias_init": float(gate_bias if gate_bias is not None else saved.get("gate_bias_init", legacy_bias)),
    }


def spectral_scores(reference_mel: np.ndarray, output: np.ndarray) -> tuple[float, float]:
    """Score channel-first audio in the same direction as the project evaluator."""
    output_mel = logmel(output.mean(axis=0))
    return mel_ssim(output_mel, reference_mel), 1000.0 * kl_div(output_mel, reference_mel)
