"""Sweep the additive residual on the frozen Hybrid-v4.4 Dev checkpoint."""
import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np
import soundfile as sf

from scripts.extension.v4_rescore_dev import OUT as MATCHED, ROOT, fad, read_audio, save, sha256

OUT = MATCHED.parent / "residual_sweep_dev"
CHECKPOINT = OUT.parent / "kaggle_out_v4_4/out/v4_4/best.pt"
SCALES = {"r0": 0.0, "r05": 0.5, "r1": 1.0}
ROOMS = {"small", "big", "mix", "real"}


def infer(rows, resume=False):
    import torch

    from scripts.extension.v4_alignment import align_teacher_to_input
    from scripts.extension.v4_models import HybridV4Net
    from scripts.extension.v4_runtime import checkpoint_inference_config

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the matched inference run")
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    cfg = checkpoint_inference_config(ckpt, "v4_4")
    model = HybridV4Net("v4_4", hop=cfg["hop"], gate_bias_init=cfg["gate_bias_init"]).cuda().eval()
    model.load_state_dict(ckpt["model"])
    model.decoder.residual_sweep = tuple(SCALES.values())

    @lru_cache(maxsize=None)
    def verify(relative, digest):
        if sha256(ROOT / relative) != digest:
            raise ValueError(f"Frozen WAV hash mismatch for {relative}")

    for arm in SCALES:
        (OUT / arm).mkdir(parents=True, exist_ok=True)
    reused = 0
    with torch.inference_mode(), torch.autocast("cuda"):
        for i, row in enumerate(rows, 1):
            for arm in ("input", "A0", "clean"):
                verify(row["paths"][arm], row["sha256"][arm])
            paths = {arm: OUT / arm / f"{row['sample_id']}.wav" for arm in SCALES}
            if resume and all(p.is_file() and sf.info(p).frames == row["frames"] for p in paths.values()):
                reused += 1
            else:
                audio = [sf.read(ROOT / row["paths"][arm], frames=1320960, dtype="float32", always_2d=True)
                         for arm in ("input", "A0")]
                n = min(len(y) for y, _ in audio)
                if n < row["frames"] or any(sr != 44100 for _, sr in audio):
                    raise ValueError(f"Unexpected input extent for {row['sample_id']}")
                x, s = [np.repeat(y[:n], 2, axis=1).T.copy() if y.shape[1] == 1 else y[:n].T.copy()
                        for y, _ in audio]
                xt = torch.from_numpy(x).cuda().unsqueeze(0)
                st = torch.from_numpy(align_teacher_to_input(x, s)).cuda().unsqueeze(0)
                result, _ = model.forward_track(xt, st, chunk_sec=cfg["chunk_sec"])
                waves = result[:, :, :row["frames"]].float().cpu().numpy().transpose(0, 2, 1)
                if not np.isfinite(waves).all():
                    raise ValueError(f"Nonfinite result for {row['sample_id']}")
                for path, wave in zip(paths.values(), waves, strict=True):
                    sf.write(path, np.clip(wave, -1, 1), 44100, subtype="PCM_16")
            for arm, path in paths.items():
                row["paths"][arm] = str(path.relative_to(ROOT))
                row["sha256"][arm] = sha256(path)
            if i % 10 == 0:
                print(f"Inference {i}/{len(rows)}", flush=True)
    save(OUT / "manifest.json", rows)
    save(OUT / "inference.json", dict(checkpoint=str(CHECKPOINT.relative_to(ROOT)),
                                      sha256=sha256(CHECKPOINT), scales=SCALES, config=cfg,
                                      epoch=ckpt.get("epoch"), overlap_ratio=0.5, reused=reused))


def spectral(rows):
    from scripts.extension.eval_dev import logmel
    from scripts.extension.v4_runtime import spectral_scores

    @lru_cache(maxsize=48)
    def reference(relative, frames):
        wave, sr = sf.read(ROOT / relative, frames=frames, always_2d=True)
        if len(wave) != frames or sr != 44100:
            raise ValueError(f"Clean extent changed for {relative}")
        return logmel(wave.mean(axis=1))

    def score(row):
        ref = reference(row["paths"]["clean"], row["frames"])
        values = {}
        for arm in SCALES:
            ssim, kl = spectral_scores(ref, read_audio(row, arm).T)
            values[arm] = dict(ssim=ssim, kl1000=kl)
        return values

    with ThreadPoolExecutor(max_workers=4) as pool:
        scores = list(pool.map(score, rows))
    save(OUT / "spectral.json", dict(ids=[r["sample_id"] for r in rows], scores=scores))


def summarize(rows):
    from scripts.extension.eval_dev import source_bootstrap_ci

    ids = [row["sample_id"] for row in rows]
    frozen = json.loads((MATCHED / "manifest.json").read_text(encoding="utf-8"))
    if any(any(r[k] != f[k] for k in ("sample_id", "frames", "clap_start", "clap_frames"))
           or any(r["sha256"][a] != f["sha256"][a] for a in ("A0", "clean"))
           for r, f in zip(rows, frozen, strict=True)):
        raise ValueError("Baseline audio or crop changed")
    pq = json.loads((OUT / "pq.json").read_text())
    spectral = json.loads((OUT / "spectral.json").read_text())
    embeddings = np.load(OUT / "clap_embeddings.npz", allow_pickle=False)
    original = np.load(MATCHED / "clap_embeddings.npz", allow_pickle=False)
    if any(list(found) != ids for found in (pq["ids"], spectral["ids"], embeddings["ids"], original["ids"])):
        raise ValueError("Score IDs do not match manifest")
    with (MATCHED / "paired_scores.csv").open(newline="", encoding="utf-8") as stream:
        baseline = {r["sample_id"]: r for r in csv.DictReader(stream) if r["arm"] == "A0"}
    results = []
    for i, row in enumerate(rows):
        keys = dict(sample_id=ids[i], source_id=row["source_id"],
                    reverb=bool(ROOMS.intersection(row["degradations"])))
        results.append(dict(**keys, arm="A0", **{k: float(baseline[ids[i]][k]) for k in ("ssim", "kl1000", "pq")}))
        for arm in SCALES:
            results.append(dict(**keys, arm=arm, **spectral["scores"][i][arm], pq=pq["scores"][arm][i]))

    with (OUT / "paired_scores.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    summary = {}
    for arm in ("A0", *SCALES):
        group = [r for r in results if r["arm"] == arm]
        reverb = [r for r in group if r["reverb"]]
        other = [r for r in group if not r["reverb"]]
        item = dict(
            **{k: float(np.mean([r[k] for r in group])) for k in ("ssim", "kl1000", "pq")},
            fad=fad(original["clean"], original["A0"] if arm == "A0" else embeddings[arm]),
            pq_reverb=float(np.mean([r["pq"] for r in reverb])),
            pq_nonreverb=float(np.mean([r["pq"] for r in other])),
        )
        for reference_arm in ("A0", "r1"):
            refs = {r["sample_id"]: r for r in results if r["arm"] == reference_arm}
            for name, subset in (("all", group), ("reverb", reverb)):
                deltas = [dict(source_id=r["source_id"], delta=r["pq"] - refs[r["sample_id"]]["pq"]) for r in subset]
                mean, low, high = source_bootstrap_ci(deltas, "delta")
                item[f"pq_{name}_vs_{reference_arm}"] = dict(source_mean=mean, ci95=[low, high])
        summary[arm] = item
    save(OUT / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("infer", "spectral", "summarize"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    phase = args.phase
    manifest = (MATCHED if phase == "infer" else OUT) / "manifest.json"
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    assert len({r["sample_id"] for r in rows}) == len(rows) == 280
    assert len({r["source_id"] for r in rows}) == 40
    assert sum(bool(ROOMS.intersection(r["degradations"])) for r in rows) == 106
    if phase == "infer":
        infer(rows, resume=args.resume)
    else:
        {"spectral": spectral, "summarize": summarize}[phase](rows)


if __name__ == "__main__":
    main()
