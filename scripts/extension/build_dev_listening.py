"""Build an internal Dev blinded listening page (10 vocal clips: Clean ref + blinded A0 / A5[0.2,0.8] / A5[0.0,1.0])."""
from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.extension.branch_p import match_volume_lufs
from scripts.extension.branch_r import soft_limit
from scripts.extension.routing import route_degradations
from scripts.extension.run_pipeline import process_sample

EXT_DIR = ROOT / "artifacts" / "10_extension"
STEMS_DIR = EXT_DIR / "kaggle_kernel_p1_out" / "out" / "dev_stems_flac"
OUT_DIR = EXT_DIR / "dev" / "listening_check"


def best_vocal_window(voc: np.ndarray, sr: int = 44100, sec: float = 8.0) -> tuple[int, int]:
    win = int(sec * sr)
    hop = sr // 2  # 0.5s steps
    mono = np.mean(voc, axis=1).astype(np.float64)
    best_e, best_i = -1.0, 0
    for start in range(0, max(1, len(mono) - win + 1), hop):
        seg = mono[start : start + win]
        e = float(np.mean(seg * seg))
        if e > best_e:
            best_e = e
            best_i = start
    return best_i, best_i + win


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audio_out = OUT_DIR / "audio"
    audio_out.mkdir(parents=True, exist_ok=True)

    with (EXT_DIR / "inputs_manifest.csv").open("r", encoding="utf-8") as f:
        dev_rows = [r for r in csv.DictReader(f) if r["split_ext"] == "dev" and r["vocalinstrumental"] == "vocal"]

    out_mf = {
        r["sample_id"]: r
        for r in csv.DictReader((EXT_DIR / "dev" / "final_dev_a5" / "outputs_manifest.csv").open(encoding="utf-8"))
        if r["variant"] == "A5"
    }

    # Group by mode (exclude S where A5 == A0), distinct sources
    buckets: dict[str, list[dict[str, str]]] = {"P": [], "R_eq": [], "R_reverb": [], "R_then_P": [], "low_corr": []}
    for r in dev_rows:
        sid = r["sample_id"]
        if sid not in out_mf:
            continue
        degs = [d for d in r["degradation_types"].split("|") if d]
        mode = route_degradations(degs).mode
        if mode == "S":
            continue
        corr = float(out_mf[sid]["phase_corr_xs"])
        if corr < 0.35:
            buckets["low_corr"].append(r)
        elif mode == "P":
            buckets["P"].append(r)
        elif mode == "R_then_P":
            buckets["R_then_P"].append(r)
        elif mode == "R":
            if any(d in {"small", "big", "mix", "real"} for d in degs):
                buckets["R_reverb"].append(r)
            else:
                buckets["R_eq"].append(r)

    rng = random.Random(20261120)
    picked: list[tuple[str, dict[str, str]]] = []
    used_src: set[str] = set()
    quotas = [("P", 2), ("R_eq", 3), ("R_reverb", 2), ("R_then_P", 1), ("low_corr", 2)]
    for bname, q in quotas:
        cands = list(buckets[bname])
        rng.shuffle(cands)
        n_got = 0
        for r in cands:
            if r["source_id"] in used_src:
                continue
            picked.append((bname, r))
            used_src.add(r["source_id"])
            n_got += 1
            if n_got >= q:
                break

    key_records = []
    html_items = []

    for idx, (bucket, row) in enumerate(picked, 1):
        sid = row["sample_id"]
        voc, sr = sf.read(STEMS_DIR / f"{sid}_voc.flac", dtype="float32", always_2d=True)
        i0, i1 = best_vocal_window(voc, sr=sr, sec=8.0)

        clean, _ = sf.read(ROOT / row["path_clean"], dtype="float32", always_2d=True)
        a0, _ = sf.read(ROOT / row["path_s"], dtype="float32", always_2d=True)
        a5_28, _ = sf.read(EXT_DIR / "dev" / "final_dev_a5" / "A5" / f"{sid}.wav", dtype="float32", always_2d=True)
        # Generate A5 [0.0, 1.0] on the fly
        a5_010, _ = process_sample(row, "A5", STEMS_DIR, eq_clamp_db=12.0, smooth_ms=50.0, clamp_min=0.0, clamp_max=1.0)

        # Slice 8s window and loudness-match all 4 to A0 excerpt
        ref_seg = a0[i0:i1]
        clean_seg, _ = match_volume_lufs(clean[i0:i1], ref_seg, sr=sr)
        clean_seg, _ = soft_limit(clean_seg, -1.0)
        a0_seg, _ = soft_limit(ref_seg, -1.0)
        a5_28_seg, _ = match_volume_lufs(a5_28[i0:i1], ref_seg, sr=sr)
        a5_28_seg, _ = soft_limit(a5_28_seg, -1.0)
        a5_010_seg, _ = match_volume_lufs(a5_010[i0:i1], ref_seg, sr=sr)
        a5_010_seg, _ = soft_limit(a5_010_seg, -1.0)

        ref_fname = f"clip_{idx:02d}_ref_clean.wav"
        sf.write(audio_out / ref_fname, clean_seg, sr, subtype="PCM_16")

        cands = [("A0", a0_seg), ("A5_c2_8", a5_28_seg), ("A5_c0_10", a5_010_seg)]
        rng.shuffle(cands)

        labels = ["A", "B", "C"]
        mapping = {}
        cand_files = {}
        for lab, (cname, arr) in zip(labels, cands):
            h = hashlib.sha256(f"{sid}_{cname}_{idx}".encode()).hexdigest()[:10]
            fname = f"clip_{idx:02d}_{lab}_{h}.wav"
            sf.write(audio_out / fname, arr, sr, subtype="PCM_16")
            mapping[lab] = cname
            cand_files[lab] = f"audio/{fname}"

        key_records.append({
            "clip_num": idx,
            "sample_id": sid,
            "source_id": row["source_id"],
            "bucket": bucket,
            "degradation_types": row["degradation_types"],
            "phase_corr_xs": float(out_mf[sid]["phase_corr_xs"]),
            "window_sec": [round(i0 / sr, 2), round(i1 / sr, 2)],
            "mapping": mapping,
        })

        html_items.append({
            "clip_num": idx,
            "bucket": bucket,
            "degradation_types": row["degradation_types"],
            "ref": f"audio/{ref_fname}",
            "cands": cand_files,
        })
        print(f"Prepared clip {idx:02d}: {sid} ({bucket}, {row['degradation_types']}, corr={float(out_mf[sid]['phase_corr_xs']):.2f})")

    (OUT_DIR / "key_DO_NOT_OPEN_BEFORE_LISTENING.json").write_text(
        json.dumps(key_records, indent=2), encoding="utf-8"
    )

    html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8">
<title>Dev Blind Listening Check — A0 vs A5[0.2,0.8] vs A5[0.0,1.0]</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 960px; margin: 24px auto; padding: 0 16px; background: #111; color: #eee; }}
  h1 {{ font-size: 1.4rem; }}
  .card {{ background: #1c1c1c; border: 1px solid #333; border-radius: 10px; padding: 16px; margin-bottom: 18px; }}
  .row {{ display: flex; align-items: center; gap: 12px; margin: 8px 0; }}
  .lbl {{ width: 90px; font-weight: 600; }}
  audio {{ flex: 1; height: 34px; }}
  .q {{ margin-top: 10px; display: flex; gap: 20px; flex-wrap: wrap; align-items: center; }}
  input[type=text] {{ background: #222; color: #eee; border: 1px solid #444; border-radius: 6px; padding: 6px 10px; width: 320px; }}
  button {{ background: #2563eb; color: #fff; border: none; padding: 10px 18px; border-radius: 8px; cursor: pointer; font-size: 1rem; }}
  button:hover {{ background: #1d4ed8; }}
  .tag {{ font-size: 0.85rem; color: #9ca3af; }}
</style>
</head>
<body>
<h1>Nghe mù nội bộ tập Dev (10 đoạn vocal 8 giây — đã cân loudness)</h1>
<p class="tag">Mỗi đoạn gồm bản gốc sạch (Clean) để làm mốc và 3 bản ẩn danh A / B / C (thứ tự ngẫu nhiên giữa SonicMaster gốc, A5 [0.2, 0.8], A5 [0.0, 1.0]). Nghe kỹ tiếng kim loại ở giọng và xem có bị rỗng/lệch pha không.</p>
<div id="container"></div>
<button onclick="exportJson()">Tải kết quả phiếu nghe (JSON)</button>
<pre id="out" style="background:#181818;padding:12px;border-radius:8px;margin-top:12px;overflow:auto;"></pre>
<script>
const ITEMS = {json.dumps(html_items, ensure_ascii=False)};
const cont = document.getElementById('container');
ITEMS.forEach(it => {{
  const d = document.createElement('div');
  d.className = 'card';
  d.innerHTML = `
    <div><b>Đoạn #${{it.clip_num}}</b> <span class="tag">[nhóm: ${{it.bucket}} | lỗi: ${{it.degradation_types}}]</span></div>
    <div class="row"><span class="lbl">Clean (mốc)</span><audio controls src="${{it.ref}}"></audio></div>
    <div class="row"><span class="lbl">Bản A</span><audio controls src="${{it.cands.A}}"></audio></div>
    <div class="row"><span class="lbl">Bản B</span><audio controls src="${{it.cands.B}}"></audio></div>
    <div class="row"><span class="lbl">Bản C</span><audio controls src="${{it.cands.C}}"></audio></div>
    <div class="q">
      <span>Ít kim loại nhất:
        <label><input type="radio" name="metal_${{it.clip_num}}" value="A"> A</label>
        <label><input type="radio" name="metal_${{it.clip_num}}" value="B"> B</label>
        <label><input type="radio" name="metal_${{it.clip_num}}" value="C"> C</label>
        <label><input type="radio" name="metal_${{it.clip_num}}" value="tie"> Như nhau</label>
      </span>
      <span>Tốt nhất tổng thể:
        <label><input type="radio" name="overall_${{it.clip_num}}" value="A"> A</label>
        <label><input type="radio" name="overall_${{it.clip_num}}" value="B"> B</label>
        <label><input type="radio" name="overall_${{it.clip_num}}" value="C"> C</label>
        <label><input type="radio" name="overall_${{it.clip_num}}" value="tie"> Như nhau</label>
      </span>
      <input type="text" id="note_${{it.clip_num}}" placeholder="Ghi chú (rỗng tiếng, pha, rè...)">
    </div>`;
  cont.appendChild(d);
}});
function exportJson() {{
  const res = ITEMS.map(it => ({{
    clip_num: it.clip_num,
    less_metallic: (document.querySelector(`input[name="metal_${{it.clip_num}}"]:checked`) || {{}}).value || null,
    best_overall: (document.querySelector(`input[name="overall_${{it.clip_num}}"]:checked`) || {{}}).value || null,
    note: document.getElementById(`note_${{it.clip_num}}`).value
  }}));
  const txt = JSON.stringify(res, null, 2);
  document.getElementById('out').textContent = txt;
  const blob = new Blob([txt], {{type: 'application/json'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'dev_listening_response.json';
  a.click();
}}
</script>
</body>
</html>
"""
    html_path = OUT_DIR / "listen_dev.html"
    html_path.write_text(html, encoding="utf-8")
    print("Saved listening page:", html_path)


if __name__ == "__main__":
    main()
