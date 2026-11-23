"""Build 20-clip (6s vocal) blinded listening check: Clean vs SonicMaster vs Hybrid-v3 on robot/autotune samples."""
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

from scripts.extension.biquad_eq import fit_biquad_eq
from scripts.extension.branch_p import match_dynamics, match_stereo, match_volume_lufs
from scripts.extension.branch_r import soft_limit
from scripts.extension.xband_crossover import apply_xband_crossover

AUDIO_SRC = ROOT / "outputs" / "random_listening_190" / "audio"
PHIEU_PATH = ROOT / "outputs" / "random_listening_190" / "phieu_nghe_100_da_tong_hop.csv"
MANIFEST_PATH = ROOT / "tmp" / "inputs_manifest.csv"
OUT_DIR = ROOT / "artifacts" / "10_extension" / "robot_check_20"

S_FALLBACK = {"real", "clip", "small", "big", "mix"}


def load_stereo(path: Path) -> tuple[np.ndarray, int]:
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if y.shape[1] == 1:
        y = np.repeat(y, 2, axis=1)
    return y, sr


def run_v3_on_clip(x: np.ndarray, s: np.ndarray, sr: int, route: str) -> tuple[np.ndarray, str]:
    n = min(len(x), len(s))
    x, s = x[:n], s[:n]
    rx = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    rs = float(np.sqrt(np.mean(s.astype(np.float64) ** 2)))
    if 20.0 * np.log10((rs + 1e-9) / (rx + 1e-9)) > 24.0:
        return s, "S_guard"
    if route == "xband_crossover":
        y, _ = apply_xband_crossover(x, s, sr=sr, eq_clamp_db=18.0, eq_phase="minphase")
        return y, "xband_crossover"
    y, _ = fit_biquad_eq(x, s, sr=sr, eq_clamp_db=18.0, eq_phase="minphase")
    y, _ = match_dynamics(y, s, sr=sr)
    y, _ = match_stereo(y, s)
    y, _ = match_volume_lufs(y, s, sr=sr)
    return y, "peq_P"


def main() -> None:
    mf = {r["sample_id"]: r for r in csv.DictReader(MANIFEST_PATH.open("r", encoding="utf-8"))}
    rows = list(csv.DictReader(PHIEU_PATH.open("r", encoding="utf-8-sig")))

    candidates: dict[str, dict[str, object]] = {}
    for r in rows:
        note_cat = (r["Nhóm lỗi Output"] + " " + r["Lỗi nghe Output"]).lower()
        if "robot" not in note_cat and "autotune" not in note_cat:
            continue
        if r["Nhãn vocal nguồn"].strip().lower() != "vocal":
            continue
        if "nhạc" in r["Lỗi nghe Output"].lower():
            continue
        sid = r["Mã mẫu"]
        degs = {d.strip() for d in mf[sid]["degradation_types"].split("|") if d.strip()}
        if degs & S_FALLBACK or "vocal" in degs:
            continue  # exclude S fallback and vocal_mask_P (needs external stems)
        route = "xband_crossover" if "xband" in degs else "peq_P"
        if sid not in candidates:
            candidates[sid] = {
                "sample_id": sid,
                "seg": int(r["Đoạn"]),
                "start_s": float(r["Bắt đầu s"]),
                "end_s": float(r["Kết thúc s"]),
                "degradations": mf[sid]["degradation_types"],
                "route": route,
                "orig_note": r["Lỗi nghe Output"],
            }

    xband_list = sorted([v for v in candidates.values() if v["route"] == "xband_crossover"], key=lambda v: str(v["sample_id"]))
    peq_list = sorted([v for v in candidates.values() if v["route"] == "peq_P"], key=lambda v: str(v["sample_id"]))
    picked = peq_list[:17] + xband_list[:3]
    assert len(picked) == 20, f"Expected 20 samples, got {len(picked)}"

    audio_out = OUT_DIR / "audio"
    audio_out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20261123)

    key_records = []
    html_items = []

    for idx, item in enumerate(picked, 1):
        sid = str(item["sample_id"])
        seg = int(item["seg"])
        c_arr, sr = load_stereo(AUDIO_SRC / f"{sid}_{seg:02d}_clean_raw.wav")
        x_arr, _ = load_stereo(AUDIO_SRC / f"{sid}_{seg:02d}_input_raw.wav")
        s_arr, _ = load_stereo(AUDIO_SRC / f"{sid}_{seg:02d}_sonicmaster_raw.wav")

        v3_arr, actual_route = run_v3_on_clip(x_arr, s_arr, sr, str(item["route"]))

        # Loudness-match clean and hybrid_v3 to sonicmaster_raw so volume does not bias listening
        c_m, _ = match_volume_lufs(c_arr, s_arr, sr=sr)
        c_m, _ = soft_limit(c_m, -1.0)
        s_m, _ = soft_limit(s_arr, -1.0)
        v3_m, _ = match_volume_lufs(v3_arr, s_arr, sr=sr)
        v3_m, _ = soft_limit(v3_m, -1.0)

        ref_fname = f"clip_{idx:02d}_ref_clean.wav"
        sf.write(str(audio_out / ref_fname), c_m, sr, subtype="PCM_16")

        cands = [("SonicMaster", s_m), ("Hybrid_v3", v3_m)]
        mapping = {}
        cand_files = {}
        for cname, arr in cands:
            h = hashlib.sha256(f"{sid}_{cname}_{idx}".encode()).hexdigest()[:10]
            fname = f"clip_{idx:02d}_{cname}_{h}.wav"
            sf.write(str(audio_out / fname), arr, sr, subtype="PCM_16")
            mapping[cname] = cname
            cand_files[cname] = f"audio/{fname}"

        key_records.append({
            "clip_num": idx,
            "sample_id": sid,
            "seg": seg,
            "window_sec": [item["start_s"], item["end_s"]],
            "degradations": item["degradations"],
            "actual_route": actual_route,
            "orig_note": item["orig_note"],
            "mapping": mapping,
        })
        html_items.append({
            "clip_num": idx,
            "sample_id": sid,
            "seg": seg,
            "degradations": item["degradations"],
            "actual_route": actual_route,
            "orig_note": item["orig_note"],
            "ref": f"audio/{ref_fname}",
            "cands": cand_files,
        })
        print(f"[{idx:02d}/20] {sid} seg={seg} ({item['degradations']}) -> {actual_route} | note: {item['orig_note']}")

    (OUT_DIR / "key_DO_NOT_OPEN.json").write_text(json.dumps(key_records, indent=2, ensure_ascii=False), encoding="utf-8")

    html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8">
<title>Kiểm tra 20 đoạn Vocal Robot/Auto-Tune: Clean vs SonicMaster vs Hybrid-v3</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 980px; margin: 24px auto; padding: 0 16px; background: #111; color: #eee; }}
  h1 {{ font-size: 1.35rem; }}
  .card {{ background: #1c1c1c; border: 1px solid #333; border-radius: 10px; padding: 16px; margin-bottom: 16px; }}
  .row {{ display: flex; align-items: center; gap: 12px; margin: 8px 0; }}
  .lbl {{ width: 95px; font-weight: 600; }}
  audio {{ flex: 1; height: 34px; }}
  .q {{ margin-top: 10px; display: flex; gap: 20px; flex-wrap: wrap; align-items: center; }}
  input[type=text] {{ background: #222; color: #eee; border: 1px solid #444; border-radius: 6px; padding: 6px 10px; width: 320px; }}
  button {{ background: #2563eb; color: #fff; border: none; padding: 10px 18px; border-radius: 8px; cursor: pointer; font-size: 1rem; }}
  button:hover {{ background: #1d4ed8; }}
  .tag {{ font-size: 0.85rem; color: #9ca3af; }}
</style>
</head>
<body>
<h1>Đối chiếu 20 đoạn Vocal 6 giây từng bị note Robot / Auto-Tune: Clean vs SonicMaster vs Hybrid-v3</h1>
<p class="tag">Mỗi đoạn gồm <b>Clean (mốc)</b>, <b>SonicMaster</b> và <b>Hybrid-v3</b> (đã cân bằng âm lượng LUFS). Nghe trực tiếp để kiểm tra xem Hybrid-v3 có giảm hiện tượng giọng robot/autotune so với SonicMaster hay không.</p>
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
    <div><b>Đoạn #${{it.clip_num}}</b> — <code>${{it.sample_id}}</code> (đoạn ${{it.seg}}) <span class="tag">[lỗi: ${{it.degradations}} | nhánh v3: ${{it.actual_route}} | note cũ: "${{it.orig_note}}"]</span></div>
    <div class="row"><span class="lbl">Clean (mốc)</span><audio controls src="${{it.ref}}"></audio></div>
    <div class="row"><span class="lbl">SonicMaster</span><audio controls src="${{it.cands.SonicMaster}}"></audio></div>
    <div class="row"><span class="lbl">Hybrid-v3</span><audio controls src="${{it.cands.Hybrid_v3}}"></audio></div>
    <div class="q">
      <span>Ít robot/autotune hơn:
        <label><input type="radio" name="robot_${{it.clip_num}}" value="Hybrid_v3"> Hybrid-v3</label>
        <label><input type="radio" name="robot_${{it.clip_num}}" value="SonicMaster"> SonicMaster</label>
        <label><input type="radio" name="robot_${{it.clip_num}}" value="tie"> Như nhau</label>
      </span>
      <span>Tốt hơn tổng thể:
        <label><input type="radio" name="overall_${{it.clip_num}}" value="Hybrid_v3"> Hybrid-v3</label>
        <label><input type="radio" name="overall_${{it.clip_num}}" value="SonicMaster"> SonicMaster</label>
        <label><input type="radio" name="overall_${{it.clip_num}}" value="tie"> Như nhau</label>
      </span>
      <input type="text" id="note_${{it.clip_num}}" placeholder="Ghi chú thêm...">
    </div>`;
  cont.appendChild(d);
}});
function exportJson() {{
  const res = ITEMS.map(it => ({{
    clip_num: it.clip_num,
    less_robot: (document.querySelector(`input[name="robot_${{it.clip_num}}"]:checked`) || {{}}).value || null,
    best_overall: (document.querySelector(`input[name="overall_${{it.clip_num}}"]:checked`) || {{}}).value || null,
    note: document.getElementById(`note_${{it.clip_num}}`).value
  }}));
  const txt = JSON.stringify(res, null, 2);
  document.getElementById('out').textContent = txt;
  const blob = new Blob([txt], {{type: 'application/json'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'robot20_listening_response.json';
  a.click();
}}
</script>
</body>
</html>
"""
    (OUT_DIR / "listen_robot20.html").write_text(html, encoding="utf-8")
    print("Saved listening page:", OUT_DIR / "listen_robot20.html")


if __name__ == "__main__":
    main()
