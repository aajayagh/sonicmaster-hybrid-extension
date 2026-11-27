# Bảng Kết Quả Chi Tiết `V4-4bs` Trên Dev Độc Lập (`N = 280`)

**Nguồn dữ liệu kiểm toán:**
- [v4_4bs_summary.json](file:///d:/SonicMaster_Reproducing/artifacts/10_extension/v4/pilot_dev_eval/v4_4bs_summary.json)
- [v4_4bs_tables123.json](file:///d:/SonicMaster_Reproducing/artifacts/10_extension/v4/pilot_dev_eval/v4_4bs_tables123.json)
- [matched_dev_v4_4bs/summary.json](file:///d:/SonicMaster_Reproducing/artifacts/10_extension/v4/matched_dev_v4_4bs/summary.json) *(giao thức chấm chung khớp chiều dài frame trên manifest cố định giữa `v4_4bs`, `v4_4` cũ và `A0`)*
- [residual_sweep_dev/summary.json](file:///d:/SonicMaster_Reproducing/artifacts/10_extension/v4/residual_sweep_dev/summary.json) *(mốc matched CLAP FAD cho `A0` và `V4-4 r1`)*
- [matched_dev_recheck/README.md](file:///d:/SonicMaster_Reproducing/artifacts/10_extension/v4/matched_dev_recheck/README.md) *(mô tả giao thức chấm chung)*

---

## 1. Bảng So Sánh Tổng Hợp (`A0` vs `V4-4 (r1 recheck)` vs `V4-4bs`)

| Nhóm / Metric (Dev `280`) | `A0` (SonicMaster) | `V4-4` (`r1` recheck) | `V4-4bs` (`Bright+Stereo`) | Ghi chú |
|---|---:|---:|---:|---|
| **All (`280`) Mel-SSIM ↑** | `0.6064` | `0.8260` | `0.8076` | PASS (`>= 0.74`) |
| **All (`280`) Mel-KL×1000 ↓** | `1.1064` | `1.0475` | `1.1229` | Gần tương đương `A0` (`1.015× A0`) |
| **All (`280`) FAD-CLAP ↓** *(matched)* | `0.1674` | `0.1304` | `0.1315` | Tốt hơn `A0` `-21.5%` |
| **All (`280`) Audiobox PQ ↑** | `7.6452` | `7.3340` | `7.3322` | — |
| **All (`280`) ΔPQ vs `A0`** | `0.0000` | `-0.3113` | `-0.3131` | CI95 `[-0.3966, -0.2364]` |
| **Single (`160`) SSIM / KL** | — | — | `0.8634` / `0.7509` | — |
| **Single (`160`) FAD / PQ** | — | — | `0.0932` / `7.5469` | — |
| **Double+Triple (`120`) SSIM / KL** | — | — | `0.7332` / `1.6190` | — |
| **Double+Triple (`120`) FAD / PQ** | — | — | `0.1962` / `7.0458` | — |
| **Reverb (`106`) SSIM / KL** | — | — | `0.6676` / `1.6534` | — |
| **Reverb (`106`) PQ / ΔPQ vs `A0`** | `7.5946` / `0.0000` | `6.8800` / `-0.7058` | `6.8888` / `-0.7059` | `g_reverb = 0.1597` |
| **Clip (`48`) SSIM / KL** | — | — | `0.8178` / `0.8941` | — |
| **Clip (`48`) PQ / ΔPQ vs `A0`** | — | — | `7.1849` / `-0.3982` | `g_clip = 0.2567` |
| **Attribute Guard Fails (`n >= 10`)** | `0 / 12` | `2 / 12` | **`1 / 12` (legacy) \| `2 / 12` (matched)** | Matched common protocol: `['bright', 'stereo']` (`stereo = +0.05184`, sát ngưỡng `0.05`) |

### Đối chiếu ghép cặp `v4_4bs` vs `v4_4` (`matched_dev_v4_4bs/summary.json`)
- **All (`280`):** `ΔSSIM = -0.0184` (CI95 `[-0.0214, -0.0154]`), `ΔKL×1000 = +0.0754` (CI95 `[+0.0235, +0.1262]`), `ΔPQ = -0.0017` (CI95 `[-0.0173, +0.0135]`, gần như tương đương hoàn toàn về `PQ`).
- **Reverb (`106`):** `ΔSSIM = -0.0144` (CI95 `[-0.0179, -0.0110]`), `ΔKL×1000 = +0.1161` (CI95 `[+0.0015, +0.2338]`), `ΔPQ = +0.0040` (CI95 `[-0.0223, +0.0300]`).
- **Giải trình khác biệt guard `Stereo` giữa 2 giao thức:** Công thức guard `(sai số Hybrid − sai số A0) / sai số input` giữ nguyên, sai số thô của `v4_4bs` (`0.09453`, giảm **`-16.4%`** so với `v4_4` cũ `0.11310`) và `A0` gần như không đổi; điểm khác nằm ở **mẫu số sai số input gốc** do script cũ (`eval_tables123.py`) đọc mỗi WAV tới giới hạn riêng còn giao thức chung cắt cùng số frame từ manifest cố định:

| Lượt chấm (Guard Stereo, `n=40`) | Sai số `v4_4bs` | Sai số `v4_4` cũ | Mẫu số `input` | Điểm guard `Stereo` (`v4_4bs`) | Ngưỡng `<= 0.05` |
|---|---:|---:|---:|---:|---|
| Bảng ban đầu (`eval_tables123.py`) | `0.09453` | `0.11310` | `0.42272` | **`+0.04994`** | Vừa đạt (`[ok]`) |
| Giao thức chấm chung (`matched_dev_v4_4bs`) | `0.09453` | `0.11310` | `0.40720` | **`+0.05184`** | Vừa trượt (cách ngưỡng `+0.00184`) |

> **Kết luận chuẩn khi báo cáo:** *"Stereo cải thiện rõ (~16.4% về sai số thô so với `V4-4` cũ), còn cách ngưỡng khoảng `0.0018` theo giao thức chấm chung."*

### Chẩn đoán nội bộ `V4-4bs` (`internal_diagnostics`)
- **`g_by_family`:** `all = 0.1609`, `linear_eq_dyn = 0.1330`, `reverb = 0.1597`, `clip = 0.2567`, `xband = 0.1803`
- **Biên độ bộ lọc & tín hiệu:** `dw0_rms = 0.3772`, `dwk_rms = 0.0195`, `mean_clip_rate = 0.000818`, `mean_stereo_corr = 0.8050`

---

## 2. Bảng Đầy Đủ 15 Attribute Guards của `V4-4bs` (`Dev N = 280`)

| Attribute | `n` | `input_err` (legacy) | `A0_err` ↓ | `v4_4bs_err` ↓ | `v4_4_err` (old) ↓ | `legacy_norm` (`eval_tables123`) | `matched_norm` (common protocol) | Status (Matched / Legacy) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `airy` | 6 | `0.0040` | `0.0017` | `0.0020` | `0.0013` | `+0.0874` | `+0.0938` | `low-n` (`eligible: false`) |
| `boom` | 13 | `0.4056` | `0.0726` | `0.0775` | `0.0614` | `+0.0120` | `+0.0135` | **PASS** |
| `bright` | 19 | `0.0100` | `0.0061` | `0.0077` | `0.0075` | **`+0.1619`** | **`+0.1286`** | **FAIL** (`> +0.050`) |
| `clarity` | 8 | `0.0362` | `0.0123` | `0.0121` | `0.0113` | `-0.0058` | `-0.0071` | `low-n` *(tốt hơn A0)* |
| `clip` | 48 | `2.512e-06` | `1.659e-07` | `9.167e-08` | `9.304e-08` | `-0.0296` | **`-0.0114`** | **PASS** *(tốt hơn A0)* |
| `comp` | 51 | `0.0472` | `0.0249` | `0.0213` | `0.0213` | `-0.0765` | **`-0.0789`** | **PASS** *(tốt hơn A0)* |
| `dark` | 14 | `0.0872` | `0.0073` | `0.0086` | `0.0082` | `+0.0152` | `+0.0180` | **PASS** |
| `mic` | 26 | `0.3327` | `0.0110` | `0.0113` | `0.0139` | `+0.0009` | `+0.0010` | **PASS** |
| `mud` | 14 | `0.4264` | `0.0538` | `0.0585` | `0.0542` | `+0.0110` | `+0.0109` | **PASS** |
| `punch` | 20 | `0.1065` | `0.0883` | `0.0758` | `0.0722` | `-0.1168` | **`-0.1023`** | **PASS** *(tốt hơn A0)* |
| `stereo` | 40 | `0.4227` | `0.0734` | `0.0945` | `0.1131` | `+0.0499` | **`+0.0518`** | **Borderline** (`PASS` legacy / cách `+0.0018` matched; `v4_4 old = +0.0974`) |
| `vocal` | 12 | `0.2462` | `0.0492` | `0.0534` | `0.0475` | `+0.0169` | `+0.0166` | **PASS** |
| `volume` | 22 | `0.1738` | `0.0466` | `0.0470` | `0.0469` | `+0.0028` | `+0.0028` | **PASS** |
| `warm` | 9 | `0.4472` | `0.0397` | `0.0338` | `0.0327` | `-0.0132` | `-0.0123` | `low-n` *(tốt hơn A0)* |
| `xband` | 32 | `0.2138` | `0.0729` | `0.0720` | `0.0674` | `-0.0042` | **`-0.0040`** | **PASS** *(tốt hơn A0)* |
