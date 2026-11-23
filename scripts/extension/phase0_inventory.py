"""Phase 0: Inventory inputs and lock Dev/Confirm splits for 10_extension."""
from __future__ import annotations

import csv
import glob
import hashlib
import json
import os
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "artifacts" / "10_extension"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load 7000 test manifest
    manifest_rows = []
    with (ROOT / "data" / "manifests" / "test_manifest.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            manifest_rows.append(json.loads(line))
    assert len(manifest_rows) == 7000, f"Expected 7000, got {len(manifest_rows)}"

    # 2. Load verified SonicMaster outputs (s)
    s_inv_path = ROOT / "artifacts" / "14_run7000" / "autopilot" / "sonicmaster7000_verified_inventory.json"
    s_inv = {item["sample_id"]: item for item in json.loads(s_inv_path.read_text(encoding="utf-8"))}

    # 3. Scan local x, clean, r in artifacts/14_run7000
    x_map: dict[str, Path] = {}
    for p in glob.glob(str(ROOT / "artifacts" / "14_run7000" / "**" / "*_input.wav"), recursive=True):
        sid = Path(p).name.replace("_input.wav", "")
        x_map[sid] = Path(p)

    clean_map: dict[str, Path] = {}
    for p in glob.glob(str(ROOT / "artifacts" / "14_run7000" / "**" / "*_clean.wav"), recursive=True):
        stem = Path(p).name.replace("_clean.wav", "")
        # handle both <source_id>_clean.wav and <sample_id>_clean.wav
        src_id = stem.split("_")[0]
        clean_map[src_id] = Path(p)

    r_map: dict[str, Path] = {}
    for p in glob.glob(str(ROOT / "artifacts" / "14_run7000" / "**" / "*_vae.wav"), recursive=True):
        sid = Path(p).name.replace("_vae.wav", "")
        r_map[sid] = Path(p)

    # 4. Select 40 Dev sources from test sources that already have local x, s, and clean
    # (Note: VRG 40 selection sources are in val_manifest, not test_manifest, so they lack run7000 s outputs.)
    all_sources = sorted({str(r["source_id"]) for r in manifest_rows})
    assert len(all_sources) == 1000

    by_source: dict[str, list[str]] = {}
    for row in manifest_rows:
        by_source.setdefault(str(row["source_id"]), []).append(row["id"])

    local_complete_sources = sorted(
        src_id for src_id, sids in by_source.items()
        if src_id in clean_map
        and all(sid in x_map and sid in s_inv and Path(s_inv[sid]["file"]).exists() for sid in sids)
    )
    assert len(local_complete_sources) >= 40, f"Only {len(local_complete_sources)} local complete sources"
    dev_sources = sorted(random.Random(20261119).sample(local_complete_sources, 40))
    dev_set = set(dev_sources)
    confirm_sources = sorted(set(all_sources) - dev_set)
    assert len(confirm_sources) == 960
    assert len(dev_set & set(confirm_sources)) == 0

    # 5. Build inputs_manifest.csv
    csv_path = OUT_DIR / "inputs_manifest.csv"
    counts = {"total": 0, "has_s": 0, "has_x": 0, "has_r": 0, "has_clean": 0,
              "dev_samples": 0, "dev_has_x": 0, "dev_has_clean": 0, "dev_has_r": 0}

    fieldnames = [
        "sample_id", "source_id", "variant", "split_ext", "n_degradations",
        "degradation_groups", "degradation_types", "vocalinstrumental", "prompt",
        "path_x", "path_s", "path_r", "path_clean", "sha256_s"
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in manifest_rows:
            sid = row["id"]
            src_id = str(row["source_id"])
            variant = sid.split("_")[1]
            split_ext = "dev" if src_id in dev_set else "confirm"

            s_item = s_inv.get(sid)
            path_s = str(Path(s_item["file"]).relative_to(ROOT)).replace("\\", "/") if s_item and Path(s_item["file"]).exists() else ""
            sha_s = s_item["sha256"] if s_item else ""

            px = x_map.get(sid)
            path_x = str(px.relative_to(ROOT)).replace("\\", "/") if px else ""

            pr = r_map.get(sid)
            path_r = str(pr.relative_to(ROOT)).replace("\\", "/") if pr else ""

            pc = clean_map.get(src_id)
            path_clean = str(pc.relative_to(ROOT)).replace("\\", "/") if pc else ""

            counts["total"] += 1
            if path_s: counts["has_s"] += 1
            if path_x: counts["has_x"] += 1
            if path_r: counts["has_r"] += 1
            if path_clean: counts["has_clean"] += 1
            if split_ext == "dev":
                counts["dev_samples"] += 1
                if path_x: counts["dev_has_x"] += 1
                if path_clean: counts["dev_has_clean"] += 1
                if path_r: counts["dev_has_r"] += 1

            w.writerow({
                "sample_id": sid,
                "source_id": src_id,
                "variant": variant,
                "split_ext": split_ext,
                "n_degradations": len(row["degradations_specifics"]),
                "degradation_groups": "|".join(row["degradations"]),
                "degradation_types": "|".join(row["degradations_specifics"]),
                "vocalinstrumental": row.get("vocalinstrumental", ""),
                "prompt": row["prompt"],
                "path_x": path_x,
                "path_s": path_s,
                "path_r": path_r,
                "path_clean": path_clean,
                "sha256_s": sha_s,
            })

    splits = {
        "dev_source_origin": "40 test sources with complete local (x, s, clean) sampled with seed=20261119 (VRG 40 were in val split without run7000 s)",
        "dev_sources": dev_sources,
        "confirm_sources": confirm_sources,
        "listen_sources": [],
        "listen_status": "planned_after_phase1_separator",
        "counts": counts,
    }
    splits_path = OUT_DIR / "splits.json"
    splits_path.write_text(json.dumps(splits, indent=2), encoding="utf-8")

    summary = {
        "inputs_manifest_csv": str(csv_path.relative_to(ROOT)).replace("\\", "/"),
        "inputs_manifest_sha256": sha256_file(csv_path),
        "splits_json": str(splits_path.relative_to(ROOT)).replace("\\", "/"),
        "splits_sha256": sha256_file(splits_path),
        "counts": counts,
    }
    (OUT_DIR / "phase0_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
