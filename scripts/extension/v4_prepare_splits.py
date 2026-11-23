"""Create source-disjoint splits for Hybrid-v4 Step 1 (Smoke 28/6/6 on Dev + Statistical Pilot 400/80/Dev)."""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

ROOT = Path(r"d:\SonicMaster_Reproducing")
GDRIVE = Path(r"G:\My Drive\SonicMaster_Reproducing")


def res(p: str | Path) -> Path:
    dp = ROOT / p
    return dp if dp.is_file() else GDRIVE / p


def main() -> None:
    orig_splits = json.loads(res("artifacts/10_extension/splits.json").read_text(encoding="utf-8"))
    dev_srcs = sorted(orig_splits["dev_sources"])
    confirm_srcs = sorted(orig_splits["confirm_sources"])

    # Verify manifest counts
    rows = list(csv.DictReader(res("artifacts/10_extension/inputs_manifest.csv").open("r", encoding="utf-8")))
    valid_srcs = set()
    by_src: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        if r["path_x"] and r["path_s"] and r["path_clean"]:
            by_src.setdefault(r["source_id"], []).append(r)
    for src, srows in by_src.items():
        if len(srows) > 0:
            valid_srcs.add(src)

    rng = random.Random(20261122)

    # 1. Smoke split on Dev (28 train / 6 val / 6 test) - sanity check only
    dev_pool = [s for s in dev_srcs if s in valid_srcs]
    rng.shuffle(dev_pool)
    smoke_train = sorted(dev_pool[:28])
    smoke_val = sorted(dev_pool[28:34])
    smoke_test = sorted(dev_pool[34:40])
    assert not (set(smoke_train) & set(smoke_val))
    assert not (set(smoke_train) & set(smoke_test))
    assert not (set(smoke_val) & set(smoke_test))

    # 2. Statistical Internal Pilot: 400 Confirm train / 80 Confirm val / 40 Dev test / 480 Confirm locked
    conf_pool = [s for s in confirm_srcs if s in valid_srcs]
    rng.shuffle(conf_pool)
    pilot_train = sorted(conf_pool[:400])
    pilot_val = sorted(conf_pool[400:480])
    pilot_locked = sorted(conf_pool[480:])
    pilot_test = sorted(dev_pool)

    sets = [set(pilot_train), set(pilot_val), set(pilot_test), set(pilot_locked)]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            assert not (sets[i] & sets[j]), f"Source overlap between {i} and {j}!"

    out = {
        "smoke": {
            "train_sources": smoke_train,
            "val_sources": smoke_val,
            "test_sources": smoke_test,
        },
        "pilot": {
            "train_sources": pilot_train,
            "val_sources": pilot_val,
            "test_sources": pilot_test,
            "locked_confirm_sources": pilot_locked,
        },
    }
    out_p = ROOT / "artifacts" / "10_extension" / "v4" / "pilot_splits.json"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(
        f"Saved {out_p}\n"
        f"  Smoke: train={len(smoke_train)}, val={len(smoke_val)}, test={len(smoke_test)}\n"
        f"  Pilot: train={len(pilot_train)}, val={len(pilot_val)}, test(Dev)={len(pilot_test)}, locked={len(pilot_locked)}"
    )


if __name__ == "__main__":
    main()
