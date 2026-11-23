"""Run Hybrid-v2 on all 280 Dev samples.

Routing:
- S (keep A0): real, clip, xband
- vocal_mask_P: vocal (M * P(x, s) + (1 - M) * s using bs_roformer_1297 stems)
- deverb_P: small, big, mix (apply_branch_p(deverb_audio, s))
- full_mix_P: all other linear degradations (apply_branch_p(x, s))
"""
import csv
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(r"d:\SonicMaster_Reproducing")
GDRIVE_ROOT = Path(r"G:\My Drive\SonicMaster_Reproducing")
sys.path.insert(0, str(ROOT))

from scripts.extension.branch_p import apply_branch_p  # noqa: E402
from scripts.extension.mask import compute_separator_mask  # noqa: E402
from scripts.extension.mix import mix_complementary_stft  # noqa: E402


def resolve_path(rel_or_abs: str | Path) -> Path:
    p = Path(rel_or_abs)
    if p.is_absolute():
        if p.exists():
            return p
        try:
            rel = p.relative_to(ROOT)
            gp = GDRIVE_ROOT / rel
            if gp.exists():
                return gp
        except ValueError:
            pass
        return p
    dp = ROOT / p
    if dp.exists():
        return dp
    gp = GDRIVE_ROOT / p
    if gp.exists():
        return gp
    return dp


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    for attempt in range(3):
        try:
            data = path.read_bytes()
            audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
            return audio, sr
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1.0)
    raise RuntimeError(f"Failed to load {path}")


def main():
    splits = json.loads(resolve_path("artifacts/10_extension/splits.json").read_text(encoding="utf-8"))
    dev_sources = set(splits["dev_sources"])

    manifest_path = resolve_path("artifacts/10_extension/inputs_manifest.csv")
    rows = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["source_id"] in dev_sources:
                rows.append(r)

    stems_dir = resolve_path("artifacts/10_extension/kaggle_kernel_p1_out/out/dev_stems_flac")
    deverb_dir = ROOT / "artifacts/10_extension/dev/hybrid_v2/deverb_flac"
    out_dir = ROOT / "artifacts/10_extension/dev/hybrid_v2/out_wav"
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = {"S": 0, "vocal_mask_P": 0, "deverb_P": 0, "full_mix_P": 0}
    for i, r in enumerate(rows):
        sid = r["sample_id"]
        degs = [d.strip() for d in r["degradation_types"].split("|") if d.strip()]
        deg_set = set(degs)

        s_path = resolve_path(r["path_s"])
        x_path = resolve_path(r["path_x"])
        s, sr = load_audio(s_path)

        if deg_set & {"real", "clip", "xband", "small", "big", "mix", "mud"}:
            route = "S"
            y = s
        else:
            x, _ = load_audio(x_path)
            n_xs = min(len(x), len(s))
            rx = float(np.sqrt(np.mean(x[:n_xs].astype(np.float64) ** 2)))
            rs = float(np.sqrt(np.mean(s[:n_xs].astype(np.float64) ** 2)))
            gain_db = 20.0 * np.log10((rs + 1e-9) / (rx + 1e-9))
            if gain_db > 24.0:
                route = "S"
                y = s
            elif "vocal" in deg_set:
                route = "vocal_mask_P"
                p_out, _ = apply_branch_p(x[:n_xs], s[:n_xs], sr=sr)
                v_stem, _ = load_audio(stems_dir / f"{sid}_voc.flac")
                a_stem, _ = load_audio(stems_dir / f"{sid}_acc.flac")
                n = min(len(p_out), len(s), len(v_stem), len(a_stem))
                mask = compute_separator_mask(v_stem[:n], a_stem[:n], sr=sr, clamp_min=0.2, clamp_max=0.8)
                y, _ = mix_complementary_stft(p_out[:n], s[:n], mask, sr=sr)
            else:
                route = "full_mix_P"
                y, _ = apply_branch_p(x[:n_xs], s[:n_xs], sr=sr)

        counts[route] += 1
        y = np.clip(y, -1.0, 1.0)
        sf.write(str(out_dir / f"{sid}.wav"), y, sr, subtype="PCM_16")
        if (i + 1) % 50 == 0 or (i + 1) == len(rows):
            print(f"[{i + 1}/{len(rows)}] counts={counts}", flush=True)

    print("DONE:", counts)


if __name__ == "__main__":
    main()
