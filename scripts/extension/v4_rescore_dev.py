"""Build and score a common Dev manifest from frozen WAV exports."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / "artifacts/10_extension"
OUT = EXT / "v4/matched_dev_recheck"
ARMS = {
    "v3_label": EXT / "dev/hybrid_v3_v3c_minphase_nojoint/out_wav",
    "v3_blind": EXT / "dev/hybrid_v3_blind/e1_tight_wav",
    "v4_3": EXT / "v4/pilot_dev_eval/v4_3/out_wav",
    "v4_4": EXT / "v4/pilot_dev_eval/v4_4/out_wav",
}
SYSTEMS = ("A0", *ARMS)


def save(path, value):
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare():
    from scripts.extension.v4_clap import clap_window

    manifest = ROOT / "artifacts/pa1_audio_router/pilot/inputs_manifest.csv"
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r["split_ext"] == "dev"]
    expected = {r["sample_id"] for r in rows}
    sources = {r["source_id"] for r in rows}
    split = json.loads((EXT / "v4/pilot_splits.json").read_text())["pilot"]
    if len(rows) != 280 or len(expected) != 280 or sources != set(split["test_sources"]):
        raise ValueError("Expected the 280 unique clips from the V4 Dev split")
    for name in ("v4_3", "v4_4"):
        meta = json.loads((ARMS[name].parent / "infer_meta.json").read_text())["per_sample"]
        if len(meta) != 280 or {r["sample_id"] for r in meta} != expected:
            raise ValueError(f"{name} export metadata does not match Dev IDs")

    def staged(relative):
        path = ROOT / relative
        return path if path.is_file() else OUT / "staged" / relative

    records, hashes = [], {}
    for row in rows:
        sid = row["sample_id"]
        paths = {"clean": staged(row["path_clean"]), "input": staged(row["path_x"]),
                 "A0": staged(row["path_s"])}
        paths.update({name: folder / f"{sid}.wav" for name, folder in ARMS.items()})
        audio = {key: sf.info(path) for key, path in paths.items()}
        if any(meta.samplerate != 44100 or meta.channels not in (1, 2) for meta in audio.values()):
            raise ValueError(f"Unexpected WAV format for {sid}")
        for path in paths.values():
            if path not in hashes:
                hashes[path] = sha256(path)
        if hashes[paths["A0"]] != row["sha256_s"]:
            raise ValueError(f"SonicMaster cache hash mismatch for {sid}")
        frames = min(1320960, *(meta.frames for meta in audio.values()))
        start, length = clap_window(frames)
        records.append({
            "sample_id": sid, "source_id": row["source_id"], "prompt": row["prompt"],
            "degradations": row["degradation_types"].split("|"), "frames": frames,
            "clap_start": start, "clap_frames": length,
            "paths": {key: str(path.relative_to(ROOT)) for key, path in paths.items()},
            "sha256": {key: hashes[path] for key, path in paths.items()},
        })
    save(OUT / "manifest.json", records)
    print(f"Matched {len(records)} clips, {len(sources)} sources and {len(SYSTEMS)} systems", flush=True)


def read_audio(record, arm):
    audio, sr = sf.read(ROOT / record["paths"][arm], frames=record["frames"], always_2d=True)
    if len(audio) != record["frames"] or sr != 44100:
        raise ValueError(f"Audio extent changed for {record['sample_id']}/{arm}")
    return np.repeat(audio, 2, axis=1) if audio.shape[1] == 1 else audio


def spectral(rows, workers):
    from scripts.extension.eval_dev import logmel
    from scripts.extension.v4_runtime import spectral_scores
    from scripts.neural_codec_bypass.fallback_attributes import evaluator

    evaluate = evaluator()
    @lru_cache(maxsize=48)
    def clean_reference(relative_path, frames):
        audio, rate = sf.read(ROOT / relative_path, frames=frames, always_2d=True)
        if rate != 44100 or len(audio) != frames:
            raise ValueError(f"Clean reference extent changed: {relative_path}")
        if audio.shape[1] == 1:
            audio = np.repeat(audio, 2, axis=1)
        mono = audio.mean(axis=1)
        return audio, mono, logmel(mono)

    def score_clip(record):
        c, cm, reference = clean_reference(record["paths"]["clean"], record["frames"])
        x = read_audio(record, "input")
        xm = x.mean(axis=1)
        # The existing Dev guards only take abs_error_after or cosine_abs_error_after.
        # Reverb RT60/MSD are not entered into those guards.
        active = [deg for deg in record["degradations"] if deg not in ("small", "big", "mix", "real")]
        results = []
        for arm in ("input", *SYSTEMS):
            y = x if arm == "input" else read_audio(record, arm)
            ssim, kl = spectral_scores(reference, y.T)
            details = evaluate(cm, xm, y.mean(axis=1), active, c, x, y)
            errors = {}
            for deg, fields in details.items():
                value = fields.get("abs_error_after", fields.get("cosine_abs_error_after"))
                if value is not None and np.isfinite(value):
                    errors[deg] = float(value)
            results.append(dict(sample_id=record["sample_id"], source_id=record["source_id"],
                               arm=arm, ssim=ssim, kl1000=kl, errors=errors))
        return results

    output = score_clip(rows[0])
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, results in enumerate(pool.map(score_clip, rows[1:]), 2):
            output.extend(results)
            if index % 20 == 0:
                print(f"Spectral/attributes {index}/{len(rows)}", flush=True)
    save(OUT / "spectral.json", output)


def fad(clean, output):
    a, b = clean - clean.mean(0), output - output.mean(0)
    tr = (np.sum(a*a) + np.sum(b*b) - 2*np.linalg.svd(a@b.T, compute_uv=False).sum()) / (len(a)-1)
    return float(np.sum((clean.mean(0)-output.mean(0))**2) + tr)


def summarize(rows):
    from scripts.extension.eval_dev import source_bootstrap_ci

    ids = [r["sample_id"] for r in rows]
    spectral = json.loads((OUT / "spectral.json").read_text())
    pq = json.loads((OUT / "pq.json").read_text())
    if pq["ids"] != ids:
        raise ValueError("PQ ID order differs from the frozen manifest")
    values = {(r["sample_id"], r["arm"]): r for r in spectral}
    if len(spectral) != len(values) or set(values) != {(sid, arm) for sid in ids for arm in ("input", *SYSTEMS)}:
        raise ValueError("Spectral scores do not cover exactly the frozen manifest")
    embeddings = np.load(OUT / "clap_embeddings.npz", allow_pickle=False)
    if list(embeddings["ids"]) != ids:
        raise ValueError("CLAP ID order differs from the frozen manifest")
    for arm in SYSTEMS:
        for sid, score in zip(ids, pq["scores"][arm], strict=True):
            values[sid, arm]["pq"] = score
    families = {"all": ids, "single": [], "multiple": [], "reverb": [], "clip": []}
    for row in rows:
        deg = set(row["degradations"])
        sid = row["sample_id"]
        families["single" if len(deg) == 1 else "multiple"].append(sid)
        if deg & {"small", "big", "mix", "real"}:
            families["reverb"].append(sid)
        if "clip" in deg:
            families["clip"].append(sid)
    summary = {}
    comparisons = {}
    for group, subset in families.items():
        section = {}
        selected = set(subset)
        ix = [i for i, sid in enumerate(ids) if sid in selected]
        for arm in SYSTEMS:
            scores = [values[sid, arm] for sid in subset]
            item = {key: float(np.mean([r[key] for r in scores])) for key in ("ssim", "kl1000", "pq")}
            item["fad_clap"] = fad(embeddings["clean"][ix], embeddings[arm][ix])
            item["delta_vs_A0"] = {}
            for metric in ("ssim", "kl1000", "pq"):
                pairs = [{"source_id": r["source_id"], "delta": r[metric] - values[r["sample_id"], "A0"][metric]}
                         for r in scores]
                mean, low, high = source_bootstrap_ci(pairs, "delta")
                item["delta_vs_A0"][metric] = {"source_mean": mean, "ci95": [low, high]}
            section[arm] = item
        summary[group] = {"n_clips": len(subset), "n_sources": len({values[sid,"A0"]["source_id"] for sid in subset}),
                          "systems": section}
        comparisons[group] = {}
        for arm, baseline in (("v4_4", "v4_3"), ("v4_3", "v3_label"), ("v4_4", "v3_label"),
                              ("v4_3", "v3_blind"), ("v4_4", "v3_blind")):
            deltas = {}
            for metric in ("ssim", "kl1000", "pq"):
                pairs = [{"source_id": values[sid, arm]["source_id"],
                          "delta": values[sid, arm][metric] - values[sid, baseline][metric]} for sid in subset]
                mean, low, high = source_bootstrap_ci(pairs, "delta")
                deltas[metric] = {"source_mean": mean, "ci95": [low, high]}
            comparisons[group][f"{arm}-{baseline}"] = deltas
    guards = {}
    for deg in sorted({key for row in spectral for key in row["errors"]}):
        matched = [sid for sid in ids if all(deg in values[sid, arm]["errors"] for arm in ("input", *SYSTEMS))]
        if not matched:
            continue
        error = {arm: float(np.mean([values[sid, arm]["errors"][deg] for sid in matched]))
                 for arm in ("input", *SYSTEMS)}
        eligible = len(matched) >= 10 and not deg.endswith("_msd") and error["input"] > 1e-12
        guards[deg] = {"n": len(matched), "eligible": eligible, "errors": error,
                       "normalized_vs_A0": {arm: (error[arm]-error["A0"])/error["input"]
                                            for arm in SYSTEMS} if error["input"] > 1e-12 else {}}
    for arm in SYSTEMS:
        summary["all"]["systems"][arm]["guard_failures"] = [
            deg for deg, entry in guards.items() if entry["eligible"] and entry["normalized_vs_A0"][arm] > .05
        ]
    save(OUT / "summary.json", dict(groups=summary, attributes=guards, paired_comparisons=comparisons))
    fields = ("sample_id", "source_id", "arm", "ssim", "kl1000", "pq")
    with (OUT / "paired_scores.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(spectral)
    print(json.dumps(summary["all"], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("prepare", "spectral", "summarize"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    step = args.step
    OUT.mkdir(parents=True, exist_ok=True)
    if step == "prepare":
        prepare()
    else:
        rows = json.loads((OUT / "manifest.json").read_text())
        if step == "spectral":
            spectral(rows, args.workers)
        else:
            summarize(rows)


if __name__ == "__main__":
    main()
