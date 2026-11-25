"""Score fixed-window CLAP and common-span PQ on the verified Dev exports."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.extension.v4_clap import matched_embeddings
from scripts.extension.v4_rescore_dev import OUT, ROOT, SYSTEMS, read_audio, save

CHECKPOINTS = ROOT / "tmp/rescore_failure_segments"


def load_clap(torch, device):
    from transformers import BartTokenizer, BertTokenizer, RobertaConfig, RobertaModel, RobertaTokenizer

    config = RobertaConfig.from_json_file(str(CHECKPOINTS / "roberta_config.json"))
    with (
        patch.object(RobertaModel, "from_pretrained", side_effect=lambda *a, **k: RobertaModel(config)),
        patch.object(RobertaTokenizer, "from_pretrained", return_value=None),
        patch.object(BertTokenizer, "from_pretrained", return_value=None),
        patch.object(BartTokenizer, "from_pretrained", return_value=None),
    ):
        from laion_clap import CLAP_Module
        model = CLAP_Module(enable_fusion=False, device=device)
    checkpoint = CHECKPOINTS / "630k-audioset-best.pt"
    with checkpoint.open("rb") as stream:
        use_mmap = stream.read(4) == b"PK\x03\x04"
    torch_load = torch.load
    with patch.object(torch, "load", side_effect=lambda *a, **k: torch_load(*a, **k, mmap=use_mmap)):
        model.load_ckpt(ckpt=str(checkpoint), verbose=False)
    model.eval()
    return model


def score_clap(rows, ids, torch, device, *, out=OUT, arms=("clean", *SYSTEMS)):
    model = load_clap(torch, device)
    embeddings = {}
    with torch.inference_mode():
        for arm in arms:
            # The same clean crop often serves all seven variants of a source.
            keys = [(r["sha256"][arm], r["frames"]) for r in rows]
            first = {}
            for key, row in zip(keys, rows):
                first.setdefault(key, row)
            batch_rows = list(first.values())
            result = []
            for i in range(0, len(batch_rows), 8):
                batch = batch_rows[i:i+8]
                result.append(matched_embeddings(model, [ROOT / r["paths"][arm] for r in batch],
                                                 [r["frames"] for r in batch]))
                if i % 80 == 0:
                    print(f"CLAP {arm} {i+len(batch)}/{len(batch_rows)}", flush=True)
            mapping = dict(zip(first, np.concatenate(result), strict=True))
            embeddings[arm] = np.stack([mapping[key] for key in keys]).astype(np.float64)
            if not np.isfinite(embeddings[arm]).all():
                raise ValueError(f"Nonfinite CLAP features for {arm}")
    np.savez(out / "clap_embeddings.npz", ids=np.array(ids), **embeddings)


def score_pq(rows, ids, torch, *, out=OUT, arms=SYSTEMS):
    from audiobox_aesthetics.infer import initialize_predictor

    predictor = initialize_predictor(str(CHECKPOINTS / "audiobox_checkpoint.pt"))
    pq_cache = {}
    scores = {}
    for arm in arms:
        keys = [(r["sha256"][arm], r["frames"]) for r in rows]
        needed = {}
        for key, row in zip(keys, rows):
            if key not in pq_cache:
                needed.setdefault(key, row)
        pending = list(needed.items())
        for i in range(0, len(pending), 3):
            batch = pending[i:i+3]
            inputs = [dict(path=torch.from_numpy(read_audio(row, arm).T.astype(np.float32)), sample_rate=44100)
                      for _, row in batch]
            predictions = predictor.forward(inputs)
            for (key, _), result in zip(batch, predictions, strict=True):
                pq_cache[key] = float(result["PQ"])
            if i % 60 == 0:
                print(f"PQ {arm} {i+len(batch)}/{len(pending)}", flush=True)
        scores[arm] = [pq_cache[key] for key in keys]
        save(out / "pq.json", dict(ids=ids, scores=scores))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-torch", action="store_true")
    parser.add_argument("--only", choices=("clap", "pq"))
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--arms", nargs="+", default=SYSTEMS)
    parser.add_argument("--reuse-clean", action="store_true", help="Reuse clean embeddings from an identical frozen manifest.")
    args = parser.parse_args()
    if args.cuda_torch:
        sys.path.insert(0, str(ROOT / "tools/extension-env/Lib/site-packages"))
        import torch
        import torchvision
        import torchaudio
        sys.path.pop(0)
    else:
        import torch
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = json.loads((args.out / "manifest.json").read_text())
    ids = [r["sample_id"] for r in rows]
    if args.only != "pq":
        arms = tuple(args.arms) if args.reuse_clean else ("clean", *args.arms)
        score_clap(rows, ids, torch, device, out=args.out, arms=arms)
        if device == "cuda":
            torch.cuda.empty_cache()
    if args.only != "clap":
        score_pq(rows, ids, torch, out=args.out, arms=args.arms)


if __name__ == "__main__":
    main()
