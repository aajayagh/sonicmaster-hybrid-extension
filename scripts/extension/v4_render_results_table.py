"""Render the updated Hybrid-v4 matched common-protocol progression table PNG (through V4-7 & V4-7c)."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(r"d:\SonicMaster_Reproducing")
BRAIN_ART = Path(r"C:\Users\ADMIN\.gemini\antigravity\brain\bd54de32-517b-4715-954e-8096d7c41b00")


def _fnt(size: int, bold: bool = False):
    path = r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf"
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


def main() -> None:
    recheck = json.loads((ROOT / "artifacts/10_extension/v4/matched_dev_recheck/summary.json").read_text(encoding="utf-8"))
    s4bs = json.loads((ROOT / "artifacts/10_extension/v4/matched_dev_v4_4bs/summary.json").read_text(encoding="utf-8"))
    s5p = json.loads((ROOT / "artifacts/10_extension/v4/matched_dev_v4_5p/summary.json").read_text(encoding="utf-8"))
    s6 = json.loads((ROOT / "artifacts/10_extension/v4/matched_dev_v4_6/summary.json").read_text(encoding="utf-8"))
    s7 = json.loads((ROOT / "artifacts/10_extension/v4/matched_dev_v4_7/summary.json").read_text(encoding="utf-8"))
    s7c = json.loads((ROOT / "artifacts/10_extension/v4/matched_dev_v4_7c/summary.json").read_text(encoding="utf-8"))

    rec_all = recheck["groups"]["all"]["systems"]
    rec_rev = recheck["groups"]["reverb"]["systems"]
    nonrev_a0_pq = s5p["groups"]["nonreverb"]["pq"] - s5p["groups"]["nonreverb"]["versus"]["A0"]["pq"]["source_mean"]
    nonrev_v44_pq = s5p["groups"]["nonreverb"]["pq"] - s5p["groups"]["nonreverb"]["versus"]["v4_4"]["pq"]["source_mean"]
    rev_a0_pq_src = s5p["groups"]["reverb"]["pq"] - s5p["groups"]["reverb"]["versus"]["A0"]["pq"]["source_mean"]
    rev_v44_pq_src = s5p["groups"]["reverb"]["pq"] - s5p["groups"]["reverb"]["versus"]["v4_4"]["pq"]["source_mean"]

    headers = [
        "Metric (Matched Dev N = 280)",
        "A0 (SonicMaster)",
        "V4-2 (MF7)",
        "V4-3 (MF9)",
        "V4-4 (+Res R)",
        "V4-4bs (BS)",
        "V4-5p (Decoupled)",
        "V4-6 (Stage 3)",
        "V4-7 (Rev-Route)",
        "V4-7c (+HF-Gate)",
    ]

    rows = [
        ("Parameters", "—", "1.68M", "1.69M", "1.95M", "1.95M", "1.95M", "1.95M", "1.97M", "1.97M"),
        ("Mel-SSIM ↑", f"{rec_all['A0']['ssim']:.4f}", "0.8325", f"{rec_all['v4_3']['ssim']:.4f}", f"{rec_all['v4_4']['ssim']:.4f}", f"{s4bs['all']['ssim']:.4f}", f"{s5p['groups']['all']['ssim']:.4f}", f"{s6['groups']['all']['ssim']:.4f}", f"{s7['groups']['all']['ssim']:.4f}", f"{s7c['groups']['all']['ssim']:.4f}"),
        ("Mel-KL x 1000 ↓", f"{rec_all['A0']['kl1000']:.4f}", "1.2963", f"{rec_all['v4_3']['kl1000']:.4f}", f"{rec_all['v4_4']['kl1000']:.4f}", f"{s4bs['all']['kl1000']:.4f}", f"{s5p['groups']['all']['kl1000']:.4f}", f"{s6['groups']['all']['kl1000']:.4f}", f"{s7['groups']['all']['kl1000']:.4f}", f"{s7c['groups']['all']['kl1000']:.4f}"),
        ("FAD-CLAP ↓ (Matched)", f"{rec_all['A0']['fad_clap']:.4f}", "—", f"{rec_all['v4_3']['fad_clap']:.4f}", f"{rec_all['v4_4']['fad_clap']:.4f}", f"{s4bs['all']['fad_clap']:.4f}", f"{s5p['groups']['all']['fad_clap']:.4f}", f"{s6['groups']['all']['fad_clap']:.4f}", f"{s7['groups']['all']['fad_clap']:.4f}", f"{s7c['groups']['all']['fad_clap']:.4f}"),
        ("Audiobox PQ ↑", f"{rec_all['A0']['pq']:.4f}", "7.3530", f"{rec_all['v4_3']['pq']:.4f}", f"{rec_all['v4_4']['pq']:.4f}", f"{s4bs['all']['pq']:.4f}", f"{s5p['groups']['all']['pq']:.4f}", f"{s6['groups']['all']['pq']:.4f}", f"{s7['groups']['all']['pq']:.4f}", f"{s7c['groups']['all']['pq']:.4f}"),
        ("Reverb (106) PQ ↑", f"{rev_a0_pq_src:.4f}", "6.8875", f"{rec_rev['v4_3']['pq']:.4f}", f"{rev_v44_pq_src:.4f}", f"{s4bs['reverb']['pq']:.4f}", f"{s5p['groups']['reverb']['pq']:.4f}", f"{s6['groups']['reverb']['pq']:.4f}", f"{s7['groups']['reverb']['pq']:.4f}", f"{s7c['groups']['reverb']['pq']:.4f}"),
        ("Reverb (106) Official MSD ↓", f"{s5p['groups']['reverb']['msd_A0']:.4f}", "—", "—", f"{s5p['groups']['reverb']['msd_v4_4']:.4f}", "—", f"{s5p['groups']['reverb']['msd']:.4f}", f"{s6['groups']['reverb']['msd']:.4f}", f"{s7['groups']['reverb']['msd']:.4f}", f"{s7c['groups']['reverb']['msd']:.4f}"),
        ("Non-Reverb (174) PQ ↑", f"{nonrev_a0_pq:.4f}", "—", "—", f"{nonrev_v44_pq:.4f}", "—", f"{s5p['groups']['nonreverb']['pq']:.4f}", f"{s6['groups']['nonreverb']['pq']:.4f}", f"{s7['groups']['nonreverb']['pq']:.4f}", f"{s7c['groups']['nonreverb']['pq']:.4f}"),
        ("Attribute Guard Fails / 12 ↓", "0 / 12", "6 / 12", "2 / 12", "2 / 12", "2 / 12* (1 leg.)", "1 / 12", "1 / 12", f"{len(s7['violations'])} / 12", f"{len(s7c['violations'])} / 12"),
        ("  - stereo (n=40) norm_det", "0.000 [ok]", "+0.159 [FAIL]", "+0.177 [FAIL]", "+0.097 [FAIL]", "+0.052* (+0.050 leg.)", f"+{s5p['guards']['stereo']['norm']:.3f} [PASS]", f"+{s6['guards']['stereo']['norm']:.3f} [PASS]", f"+{s7['guards']['stereo']['norm']:.3f} [PASS]", f"+{s7c['guards']['stereo']['norm']:.3f} [PASS]"),
        ("  - bright (n=19) norm_det", "0.000 [ok]", "+0.198 [FAIL]", "+0.194 [FAIL]", "+0.108 [FAIL]", "+0.129 [FAIL]", f"+{s5p['guards']['bright']['norm']:.3f} [FAIL]", f"+{s6['guards']['bright']['norm']:.3f} [FAIL]", f"+{s7['guards']['bright']['norm']:.3f} [FAIL]", f"+{s7c['guards']['bright']['norm']:.3f} [PASS]"),
    ]

    # Best column index among V4 models (cols 2..9) per row
    best_col = {
        1: 7,  # SSIM 0.8414
        2: 9,  # KL 0.9045
        3: 8,  # FAD 0.1042
        4: 9,  # PQ 7.5525
        5: 9,  # Reverb PQ 7.4044
        6: 6,  # Reverb MSD 0.3790
        7: 9,  # Non-Reverb PQ 7.6427
        8: 9,  # Guard Fails 0/12
        9: 7,  # stereo +0.027 [PASS]
        10: 9, # bright +0.021 [PASS]
    }
    also_best: set[tuple[int, int]] = set()

    col_w = [310, 170, 120, 120, 135, 185, 175, 150, 165, 175]  # sum = 1705
    pad_x, pad_y, title_h, hdr_h, row_h, foot_h = 36, 30, 62, 54, 46, 68
    W = pad_x * 2 + sum(col_w)
    H = pad_y * 2 + title_h + hdr_h + len(rows) * row_h + foot_h

    img = Image.new("RGB", (W, H), "#FFFFFF")
    d = ImageDraw.Draw(img)
    f_title = _fnt(25, True)
    f_hdr = _fnt(17, True)
    f_cell = _fnt(16, False)
    f_bold = _fnt(16, True)
    f_note = _fnt(14, False)

    d.text((W // 2, pad_y + 24), "Hybrid-v4 Progression on Common Dev Protocol (N = 280)", fill="#0F172A", font=f_title, anchor="mm")

    y = pad_y + title_h
    x = pad_x
    d.rectangle([pad_x, y, W - pad_x, y + hdr_h], fill="#1E293B")
    for j, (h, cw) in enumerate(zip(headers, col_w)):
        d.text((x + 16 if j == 0 else x + cw // 2, y + hdr_h // 2), h, fill="#FFFFFF", font=f_hdr, anchor="lm" if j == 0 else "mm")
        x += cw

    y += hdr_h
    for i, r in enumerate(rows):
        row_bg = "#F8FAFC" if i % 2 == 1 else "#FFFFFF"
        d.rectangle([pad_x, y, W - pad_x, y + row_h], fill=row_bg)
        d.line([(pad_x, y + row_h), (W - pad_x, y + row_h)], fill="#E2E8F0", width=1)
        x = pad_x
        for j, (val, cw) in enumerate(zip(r, col_w)):
            is_best = (best_col.get(i) == j) or ((i, j) in also_best)
            if is_best:
                d.rectangle([x + 3, y + 3, x + cw - 3, y + row_h - 3], fill="#ECFDF5", outline="#10B981", width=2)
            fill_c = "#065F46" if is_best else "#0F172A"
            if not is_best:
                if "[FAIL]" in val:
                    fill_c = "#DC2626"
                elif "[PASS]" in val:
                    fill_c = "#059669"
                elif "leg." in val:
                    fill_c = "#C2410C"
            tx = x + 16 if j == 0 else x + cw // 2
            fnt = f_bold if (is_best or (j == 0 and not val.startswith(" "))) else f_cell
            d.text((tx, y + row_h // 2), val, fill=fill_c, font=fnt, anchor="lm" if j == 0 else "mm")
            x += cw
        y += row_h

    d.rectangle([pad_x, pad_y + title_h, W - pad_x, y], outline="#94A3B8", width=2)
    d.text(
        (pad_x, y + 20),
        "Common protocol: matched_dev_recheck / matched_dev_v4_4bs / matched_dev_v4_5p / matched_dev_v4_6 / matched_dev_v4_7 / matched_dev_v4_7c.",
        fill="#475569",
        font=f_note,
        anchor="lm",
    )
    d.text(
        (pad_x, y + 44),
        "V4-7c reaches 0/12 guard fails (bright=+0.021 PASS, stereo=+0.040 PASS), Reverb PQ=7.4044 (+0.455 vs V4-6), All PQ=7.5525, All KL=0.9045.",
        fill="#065F46",
        font=f_note,
        anchor="lm",
    )

    out_ws = ROOT / "artifacts/10_extension/figures/v4_results_table.png"
    out_art = BRAIN_ART / "v4_results_table.png"
    out_ws.parent.mkdir(parents=True, exist_ok=True)
    out_art.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_ws)
    img.save(out_art)
    print(f"SAVED {out_ws} & {out_art}")


if __name__ == "__main__":
    main()
