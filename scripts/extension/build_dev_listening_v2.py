"""Build Hybrid-v2 (100% ZERO-VAE: BS-RoFormer De-Reverb + full-mix Branch P) 10-clip blind listening page."""
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

from scripts.extension.branch_p import apply_branch_p, match_volume_lufs
from scripts.extension.branch_r import soft_limit
from scripts.extension.eval_dev import kl_div, logmel, mel_ssim

EXT_DIR = ROOT / "artifacts" / "10_extension"
GDRIVE_ROOT = Path("G:/My Drive/SonicMaster_Reproducing")


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
    return gp if gp.exists() else dp


KEY_V1 = resolve_path("artifacts/10_extension/dev/listening_check/key_DO_NOT_OPEN_BEFORE_LISTENING.json")
DEVERB_DIR = resolve_path("artifacts/10_extension/dev/listening_v2/deverb_out/out")
OUT_DIR = EXT_DIR / "dev" / "listening_v2"


def main() -> None:
    audio_out = OUT_DIR / "audio"
    audio_out.mkdir(parents=True, exist_ok=True)

    keys_v1 = json.loads(KEY_V1.read_text(encoding="utf-8"))
    manifest = {r["sample_id"]: r for r in csv.DictReader(resolve_path("artifacts/10_extension/inputs_manifest.csv").open(encoding="utf-8"))}

    rng = random.Random(2026112002)
    key_records = []
    html_items = []
    metrics_rows = []

    for k in keys_v1:
        idx = k["clip_num"]
        sid = k["sample_id"]
        row = manifest[sid]
        sr = 44100
        i0 = int(round(k["window_sec"][0] * sr))
        i1 = int(round(k["window_sec"][1] * sr))

        clean, _ = sf.read(resolve_path(row["path_clean"]), dtype="float32", always_2d=True)
        x, _ = sf.read(resolve_path(row["path_x"]), dtype="float32", always_2d=True)
        a0, _ = sf.read(resolve_path(row["path_s"]), dtype="float32", always_2d=True)
        n = len(a0)
        clean, x = clean[:n], x[:n]

        deverb_p = DEVERB_DIR / f"{sid}_deverb.wav"
        if deverb_p.exists():
            u, _ = sf.read(deverb_p, dtype="float32", always_2d=True)
            u = u[:n]
            method = "deverb_bs_roformer + full_mix_P"
        else:
            u = x
            method = "full_mix_P"

        hv2, p_info = apply_branch_p(u, a0, sr=sr, eq_clamp_db=12.0)
        hv2, _ = soft_limit(hv2, -1.0)

        # Full-clip SSIM & KL vs clean
        mc = logmel(np.mean(clean.astype(np.float64), axis=1))
        m0 = logmel(np.mean(a0.astype(np.float64), axis=1))
        m2 = logmel(np.mean(hv2.astype(np.float64), axis=1))
        metrics_rows.append({
            "clip": idx, "sid": sid, "degs": row["degradation_types"], "method": method,
            "a0_ssim": round(mel_ssim(m0, mc), 4), "v2_ssim": round(mel_ssim(m2, mc), 4),
            "a0_kl": round(kl_div(m0, mc) * 1000, 4), "v2_kl": round(kl_div(m2, mc) * 1000, 4),
        })

        # 8s excerpt, loudness-matched to A0 excerpt
        ref_seg = a0[i0:i1]
        clean_seg, _ = match_volume_lufs(clean[i0:i1], ref_seg, sr=sr)
        clean_seg, _ = soft_limit(clean_seg, -1.0)
        a0_seg, _ = soft_limit(ref_seg, -1.0)
        v2_seg, _ = match_volume_lufs(hv2[i0:i1], ref_seg, sr=sr)
        v2_seg, _ = soft_limit(v2_seg, -1.0)

        ref_fname = f"clip_{idx:02d}_ref_clean.wav"
        sf.write(audio_out / ref_fname, clean_seg, sr, subtype="PCM_16")

        cands = [("A0", a0_seg), ("Hybrid_v2", v2_seg)]
        rng.shuffle(cands)
        mapping = {}
        cand_files = {}
        for lab, (cname, arr) in zip(["A", "B"], cands):
            h = hashlib.sha256(f"v2_{sid}_{cname}_{idx}".encode()).hexdigest()[:10]
            fname = f"clip_{idx:02d}_{lab}_{h}.wav"
            sf.write(audio_out / fname, arr, sr, subtype="PCM_16")
            mapping[lab] = cname
            cand_files[lab] = f"audio/{fname}"

        key_records.append({
            "clip_num": idx, "sample_id": sid, "bucket": k["bucket"],
            "degradation_types": row["degradation_types"], "method": method,
            "mapping": mapping,
        })
        html_items.append({
            "clip_num": idx, "bucket": k["bucket"],
            "degradation_types": row["degradation_types"],
            "ref": f"audio/{ref_fname}", "cands": cand_files,
        })

    (OUT_DIR / "key_v2_DO_NOT_OPEN.json").write_text(json.dumps(key_records, indent=2), encoding="utf-8")
    (OUT_DIR / "metrics_v2_10clips.json").write_text(json.dumps(metrics_rows, indent=2), encoding="utf-8")

    html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8">
<title>Hybrid-v2 (100% Zero-VAE) Blind Listening — A0 vs Hybrid-v2</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 960px; margin: 24px auto; padding: 0 16px; background: #111; color: #eee; }}
  h1 {{ font-size: 1.4rem; }}
  .card {{ background: #1c1c1c; border: 1px solid #333; border-radius: 10px; padding: 16px; margin-bottom: 18px; }}
  .row {{ display: flex; align-items: center; gap: 12px; margin: 8px 0; }}
  .lbl {{ width: 95px; font-weight: 600; }}
  audio {{ flex: 1; height: 34px; }}
  .q {{ margin-top: 10px; display: flex; gap: 20px; flex-wrap: wrap; align-items: center; }}
  input[type=text] {{ background: #222; color: #eee; border: 1px solid #444; border-radius: 6px; padding: 6px 10px; width: 320px; }}
  button {{ background: #16a34a; color: #fff; border: none; padding: 10px 18px; border-radius: 8px; cursor: pointer; font-size: 1rem; }}
  button:hover {{ background: #15803d; }}
  .tag {{ font-size: 0.85rem; color: #9ca3af; }}
</style>
</head>
<body>
<h1>Nghe mù Hybrid-v2 (100% không qua VAE: BS-RoFormer De-Reverb + DSP toàn bài)</h1>
<p class="tag">Cùng 10 đoạn vocal 8 giây như lần trước (đã cân loudness). So A / B ẩn danh giữa SonicMaster gốc và Hybrid-v2. Chú ý đặc biệt #6, #7, #9, #10 (reverb) và #1, #2 (P).</p>
<div id="container"></div>
<button onclick="exportJson()">Tải kết quả phiếu nghe v2 (JSON)</button>
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
    <div class="q">
      <span>Ít kim loại hơn:
        <label><input type="radio" name="metal_${{it.clip_num}}" value="A"> A</label>
        <label><input type="radio" name="metal_${{it.clip_num}}" value="B"> B</label>
        <label><input type="radio" name="metal_${{it.clip_num}}" value="tie"> Như nhau</label>
      </span>
      <span>Tốt hơn tổng thể (gần Clean hơn):
        <label><input type="radio" name="overall_${{it.clip_num}}" value="A"> A</label>
        <label><input type="radio" name="overall_${{it.clip_num}}" value="B"> B</label>
        <label><input type="radio" name="overall_${{it.clip_num}}" value="tie"> Như nhau</label>
      </span>
      <input type="text" id="note_${{it.clip_num}}" placeholder="Ghi chú (rõ ràng / nhẹ / lỗi...)">
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
  a.download = 'dev_listening_v2_response.json';
  a.click();
}}
</script>
</body>
</html>
"""
    html_path = OUT_DIR / "listen_v2.html"
    html_path.write_text(html, encoding="utf-8")
    print("Saved:", html_path)
    for m in metrics_rows:
        print(f"#{m['clip']:02d} {m['degs']:16s} | SSIM: A0={m['a0_ssim']:.4f} -> v2={m['v2_ssim']:.4f} | KL: A0={m['a0_kl']:.3f} -> v2={m['v2_kl']:.3f} [{m['method']}]")


if __name__ == "__main__":
    main()
