"""Build Dev 280 test set combining locked Hybrid-v3 V3-best with TD_declip on clip-without-reverb samples."""
from __future__ import annotations

import csv
import hashlib
import io
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.ndimage import binary_dilation, uniform_filter1d
from scipy.signal import correlate

ROOT = Path(r"d:\SonicMaster_Reproducing")
GDRIVE = Path(r"G:\My Drive\SonicMaster_Reproducing")
sys.path.insert(0, str(ROOT))

from scripts.extension.branch_p import match_volume_lufs  # noqa: E402
from scripts.extension.branch_r import soft_limit  # noqa: E402


def res(p: str | Path) -> Path:
    dp = ROOT / p
    return dp if dp.is_file() else GDRIVE / p


def load_stereo(path: Path, frames: int = 1320960) -> tuple[np.ndarray, int]:
    for attempt in range(3):
        try:
            y, sr = sf.read(io.BytesIO(path.read_bytes()), dtype="float32", always_2d=True)
            if y.shape[1] == 1:
                y = np.repeat(y, 2, axis=1)
            return y[:frames], sr
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1.0)
    raise RuntimeError(f"Failed to load {path}")


def process_one(r: dict[str, str], v3_dir_str: str, cand_dir_str: str, out_dir_str: str) -> tuple[str, str]:
    sid = r["sample_id"]
    out_p = Path(out_dir_str) / f"{sid}.wav"
    degs = {d.strip() for d in r["degradation_types"].split("|") if d.strip()}
    reverb_set = {"real", "small", "big", "mix"}

    if "clip" in degs and not (degs & reverb_set) and r["path_x"]:
        x, sr = load_stereo(res(r["path_x"]))
        s, _ = load_stereo(res(r["path_s"]))
        p, _ = load_stereo(Path(cand_dir_str) / f"{sid}.wav")
        n = min(len(x), len(s), len(p))
        x, s, p = x[:n], s[:n], p[:n]

        xm = np.max(np.abs(x), axis=1)
        peak = float(np.max(xm)) + 1e-9
        clip_mask = binary_dilation(xm >= 0.96 * peak, iterations=64).astype(np.float32)
        w_t = uniform_filter1d(clip_mask, size=128)[:, None]

        pm, sm = p.mean(axis=1), s.mean(axis=1)
        seg = min(len(pm), 44100 * 5)
        cc = correlate(pm[:seg], sm[:seg], mode="full")
        mid = seg - 1
        win = cc[mid - 256 : mid + 257]
        idx_max = int(np.argmax(np.abs(win)))
        lag = idx_max - 256
        pol = 1.0 if win[idx_max] >= 0 else -1.0
        s_al = np.roll(s, lag, axis=0) * pol
        y_td = (1.0 - w_t) * p + w_t * s_al

        y_td, _ = match_volume_lufs(y_td, s, sr=sr)
        y_td, _ = soft_limit(y_td, ceiling_db=-1.0)
        sf.write(str(out_p), np.clip(y_td, -1.0, 1.0), sr, subtype="PCM_16")
        return sid, "td_declip"

    shutil.copy2(Path(v3_dir_str) / f"{sid}.wav", out_p)
    return sid, "v3_copied"


def main() -> None:
    v3_dir = ROOT / "artifacts" / "10_extension" / "dev" / "hybrid_v3_v3c_minphase_nojoint" / "out_wav"
    cand_dir = ROOT / "artifacts" / "10_extension" / "dev" / "hybrid_v3_blind" / "cand_wav"
    out_dir = ROOT / "artifacts" / "10_extension" / "dev" / "exp_v35" / "td_declip_v3_wav"
    out_dir.mkdir(parents=True, exist_ok=True)

    with res("artifacts/10_extension/inputs_manifest.csv").open("r", encoding="utf-8") as f:
        dev_rows = [r for r in csv.DictReader(f) if r["split_ext"] == "dev"]

    td_sids = []
    copied_sids = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(process_one, r, str(v3_dir), str(cand_dir), str(out_dir)) for r in dev_rows]
        for fut in as_completed(futs):
            sid, kind = fut.result()
            if kind == "td_declip":
                td_sids.append(sid)
            else:
                copied_sids.append(sid)

    td_sids.sort()
    first_manifest_sid = dev_rows[0]["sample_id"]
    sha_manifest0 = hashlib.sha256((out_dir / f"{first_manifest_sid}.wav").read_bytes()).hexdigest()
    sha_td0 = hashlib.sha256((out_dir / f"{td_sids[0]}.wav").read_bytes()).hexdigest() if td_sids else ""
    print(f"TD_declip count={len(td_sids)}, V3-best copied={len(copied_sids)}, total={len(td_sids)+len(copied_sids)}")
    print(f"LOCAL_SHA256[manifest[0]={first_manifest_sid}]={sha_manifest0}")
    print(f"LOCAL_SHA256[td[0]={td_sids[0]}]={sha_td0}")


if __name__ == "__main__":
    main()
