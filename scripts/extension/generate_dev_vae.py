"""Generate VAE roundtrip r = D(E(x)) for Dev split (280 samples) using local AutoencoderOobleck."""
from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from diffusers import AutoencoderOobleck

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "artifacts" / "10_extension" / "inputs_manifest.csv"
OUT_DIR = ROOT / "artifacts" / "10_extension" / "dev" / "vae_wav"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    vae = AutoencoderOobleck.from_pretrained(str(ROOT / "tmp" / "decoder_assets"), local_files_only=True).to(device).eval()

    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    updated = 0
    with torch.no_grad():
        for i, row in enumerate(rows):
            if row["split_ext"] != "dev":
                continue
            sid = row["sample_id"]
            out_wav = OUT_DIR / f"{sid}_vae.wav"
            rel_out = str(out_wav.relative_to(ROOT)).replace("\\", "/")

            if not out_wav.exists():
                x, sr = sf.read(ROOT / row["path_x"], dtype="float32", always_2d=True)
                if x.shape[1] == 1:
                    x = np.repeat(x, 2, axis=1)
                n_trunc = (x.shape[0] // 2048) * 2048
                x = x[:n_trunc]
                xt = torch.from_numpy(x.T.copy()).unsqueeze(0).to(device)
                z = vae.encode(xt).latent_dist.mode()
                r = vae.decode(z).sample[0].cpu().numpy().T
                tmp = out_wav.with_suffix(".part.wav")
                sf.write(tmp, r.astype(np.float32), sr, format="WAV", subtype="FLOAT")
                os.replace(tmp, out_wav)

            if row["path_r"] != rel_out:
                row["path_r"] = rel_out
                updated += 1
            if (i + 1) % 20 == 0:
                print(f"Processed {sid} (updated={updated})", flush=True)

    tmp_csv = MANIFEST_PATH.with_suffix(".tmp.csv")
    with tmp_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp_csv, MANIFEST_PATH)
    print(f"Done. Updated {updated} dev rows in {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
