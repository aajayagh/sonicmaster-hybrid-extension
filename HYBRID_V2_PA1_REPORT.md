# Báo Cáo Tổng Kết Thử Nghiệm & Nghiệm Thu Phương Án Hybrid-v2 PA1

**Dự án:** Mở rộng khắc phục suy giảm codec VAE cho SonicMaster (`10_extension`)  
**Cấu hình khóa:** [`lock_v2.json`](file:///d:/SonicMaster_Reproducing/artifacts/10_extension/lock_v2.json)  
**Tài liệu gốc:** [`PLAN_EXTENSION_CODEC_BYPASS.md`](file:///d:/SonicMaster_Reproducing/PLAN_EXTENSION_CODEC_BYPASS.md)

---

## 1. Bối Cảnh & Mục Tiêu

Mô hình SonicMaster (`A0`) tái tạo toàn bộ âm thanh qua bộ nén/giải nén **Stable Audio Open 1.0 VAE**, gây ra hiện tượng méo tiếng kim loại (*metallic vocal artifact*) và làm giảm độ tương đồng phổ Mel (`SSIM ≈ 0.609`).

Mục tiêu của nhánh mở rộng là xây dựng một pipeline **zero-retraining** (không huấn luyện lại backbone, không đọc tín hiệu `clean` khi suy luận) để loại bỏ suy giảm do VAE trên các lỗi tuyến tính và lỗi giọng hát, đồng thời vượt qua cả **5 cổng nghiệm thu (Gates)** so với `A0`:

1. **SSIM ↑:** `ΔSSIM ≥ +0.01`
2. **KL×1000 ↓:** `KL ≤ 1.100× A0`
3. **FAD-CLAP ↓:** `FAD ≤ 1.100× A0`
4. **Audiobox PQ ↑:** `CI95_lo(ΔPQ) ≥ -0.05`
5. **15+ Attribute Guards:** `0` vi phạm (`n ≥ 10`, `norm_deterioration ≤ +0.05`)

Tập dữ liệu 7.000 mẫu được khóa cố định tại `splits.json` (source-disjoint): **Dev** (`40` nguồn / `280` mẫu), **Confirm** (`960` nguồn / `6.720` mẫu), và **Listen** (`20` nguồn vocal giữ kín).

---

## 2. Hành Trình Thử Nghiệm & Chẩn Đoán Kỹ Thuật (Từ `A5` Đến `PA1`)

### Giai đoạn 1: `A5` — VRG-style Residual + Mặt nạ `bs_roformer_1297`
- **Thiết kế:** Dùng `model_bs_roformer_ep_317_sdr_12.9755.ckpt` (được chọn qua Phase 1 với rò rỉ trung vị `-32.62 dB`) tạo mặt nạ mềm `M` (`clamp = [0.2, 0.8]`), trộn `x + (s - r)` với phần bù từ VAE.
- **Kết quả Dev (`280` mẫu):** `SSIM: 0.6064 → 0.6279`, `KL×1000: 1.1064 → 1.1490` (`1.038×`), `FAD: 0.2129 → 0.2174`, `PQ: 7.6452 → 7.6388`, `0` vi phạm Attribute Guards.
- **Hạn chế:** Nghe mù 10 clip Dev vocal cho thấy `A5` chỉ thắng `4/10` (hòa `5/10`, thua `1/10`) và sự khác biệt so với `A0` khá nhỏ do vẫn giữ phần dư `s - r` đi qua VAE.

### Giai đoạn 2: `Hybrid-v2` (Thử nghiệm 100% Zero-VAE + `deverb_bs_roformer`)
- **Thiết kế:** Bỏ hoàn toàn thành phần VAE ở các lỗi tuyến tính (`full_mix_P`) và dùng `deverb_bs_roformer_8_384dim_10depth.ckpt` cho các lỗi vang (`small`, `big`, `mix`, `real`).
- **Nghe mù 10 clip Dev vocal:** `Hybrid-v2` thắng **`6/10`**, hòa `4/10`, thua **`0/10`** về độ ít kim loại (`6/6 = 100%` cặp phân thắng bại); thắng `6/10`, thua `1/10` (`#9 boom|real`) về tổng thể.
- **Lỗi lộ ra trên 10 clip (`metrics_v2_10clips.json`):** `KL×1000` vọt lên `2.250` (`+170%`) do:
  1. `real` reverb qua `deverb_bs_roformer` bị tụt SSIM và thua nghe → đưa `real` về `S` (`A0`).
  2. `vocal` EQ chạy `full_mix_P` làm lệch phân bố năng lượng Mel phần nhạc đệm (KL vọt `6.5–11×`) → chuyển `vocal` sang `vocal_mask_P`: `M * P(x, s) + (1 - M) * s`.

### Giai đoạn 3: Đánh giá `Hybrid-v2` trên toàn bộ 280 mẫu Dev & Phát hiện nghẽn `deverb_P`
- Chạy `deverb_bs_roformer` trên Kaggle (`aajayagh/sonicmaster-ext-deverb-dev-run`) cho 50 clip `small`/`big`/`mix` và chấm toàn bộ 280 mẫu Dev:
  - `vocal_mask_P` (`n=10`): `KL×1000 = 1.067`, `SSIM = 0.652` — **đã sửa triệt để lỗi vọt KL trên vocal**.
  - `full_mix_P` (`n=115`): `KL×1000 = 1.365`, `SSIM = 0.940`.
  - **`deverb_P` (`n=48`):** **`KL×1000 = 13.909`, `SSIM = 0.386`** — `deverb_bs_roformer` phá phân bố phổ Mel, kéo `KL×1000` toàn tập Dev lên **`3.5217`** (`3.183× A0`, rớt cổng `KL`).

### Giai đoạn 4: Chẩn đoán sâu `full_mix_P` & Hình thành `PA1`
Khi bỏ `deverb_P → S` trên Dev (`SSIM = 0.7332`), `KL×1000` vẫn ở mức `1.4088` (`1.273× A0`, vượt ngưỡng `1.2171`). Phân tích trực tiếp các mẫu có `ΔKL` cao nhất trong `full_mix_P` phát hiện **3 điểm chốt hạ**:

1. **Sửa thứ tự DSP trong [`branch_p.py`](file:///d:/SonicMaster_Reproducing/scripts/extension/branch_p.py) (`volFirst` + `eq_clamp_db=18.0`):**
   - *Nguyên nhân:* `match_eq_fir(u, s)` trước đây chạy **trước** cân bằng âm lượng, khiến bộ lọc EQ `±12 dB` bị kẹp cho phần chênh lệch gain băng rộng và gây hard-clipping (`peak = 10.42` ở `1842342_deg5 [mud|comp]`).
   - *Khắc phục:* Cân bằng RMS của `u` theo `s` **trước** khi ước lượng FIR EQ (`eq_clamp_db = 18.0 dB`). Kết quả: `1842342_deg5` giảm KL `6.63 → 1.31`, `1903190_deg2 [warm]` giảm KL `3.15 → 0.80` (`SSIM = 0.996`).
2. **Guard mù chống nền nhiễu lượng tử 16-bit PCM (`rms_dB > 24 dB → S`):**
   - *Nguyên nhân:* Một số mẫu `volume` bị suy giảm `-58` đến `-61 dB` (`1884673_deg6`, `1077462_deg4`, `1784406_deg7`) nằm sát quantization floor 16-bit, nên mọi khuếch đại tuyến tính đều khuếch đại nhiễu lượng tử (`KL = 4.07–20.78`).
   - *Khắc phục:* Thêm điều kiện mù (không đọc prompt hay `clean`): nếu `20·log10(rms(s) / rms(x)) > 24.0 dB` thì fallback về `S` (`A0`).
3. **Loại bỏ vi phạm `mud` trên Attribute Guard (`eval_tables123.py`):**
   - Khi chạy đủ 15 thuộc tính trên Dev, `mud` (`n=14`) có `norm_deterioration = +0.064 > 0.05` (`[FAIL]`). Đưa `mud` về `S` (`A0`) giúp xóa sổ hoàn toàn vi phạm (`Violations: []`) và giảm thêm `KL×1000`.
4. **Xác thực SHA-256 Kaggle dataset (`sonicmaster-ext-dev-pq-fad-v2`):**
   - Kiểm tra đối chiếu phát hiện bản chạy Kaggle đầu tiên mount nhầm Version 1 (`A5` cũ, `SHA-256: cc02f079...`). Đã tạo dataset mới `sonicmaster-ext-dev-pq-fad-v2` và xác nhận khớp 100% `SHA-256: 682ce104ce0e31d13c556c903b91a8a6b3307d4683ee736a8fd609c7ded9a0da`.

---

## 3. Kiến Trúc Chốt Của `Hybrid-v2 PA1`

Pipeline gồm **3 nhánh routing** (không dùng `deverb`, chỉ cần chạy `bs_roformer_1297` trên đúng nhóm `vocal`):

1. **Nhánh `S` (Fallback giữ `A0`):**
   - Các mẫu có ít nhất một lỗi thuộc `{real, clip, xband, small, big, mix, mud}`, **hoặc** có `20·log10(rms(s)/rms(x)) > 24.0 dB`, **hoặc** thiếu `path_x`.
   - *(Tỷ lệ: `180/280` trên Dev; `4.832/6.720` trên Confirm.)*
2. **Nhánh `vocal_mask_P` (cho lỗi `vocal` không thuộc nhóm `S`):**
   - Tính `p_out = apply_branch_p(x, s)`.
   - Dùng stem từ `model_bs_roformer_ep_317_sdr_12.9755.ckpt` ước lượng mặt nạ mềm `M` (`clamp = [0.2, 0.8]`) và trộn bù STFT: `y = M * p_out + (1 - M) * s`.
   - *(Tỷ lệ: `7/280` trên Dev; `135/6.720` trên Confirm — tách 141 clip trên Kaggle.)*
3. **Nhánh `full_mix_P` (các lỗi tuyến tính còn lại):**
   - Áp dụng trực tiếp `apply_branch_p(x, s)` theo đúng thứ tự: **RMS-match → 1/3-octave FIR EQ (`±18 dB`) → Feed-forward Compressor → Side/Mid Stereo Match → Integrated LUFS Match**.
   - *(Tỷ lệ: `93/280` trên Dev; `1.753/6.720` trên Confirm.)*

---

## 4. Kết Quả Cuối Cùng Trên Dev & Confirm

### 4.1. Kết quả toàn tập Dev (`N = 280`) & Confirm (`N = 6.720`)

| Tập dữ liệu | Chỉ số | `A0` (SonicMaster) | **Hybrid-v2 PA1** | Chênh lệch / Tỷ lệ | Ngưỡng cổng | Trạng thái |
|---|---|---:|---:|---:|---|---|
| **Dev** (`N=280`) | **SSIM ↑** | `0.6064` | **`0.7120`** | **`+0.1056`** | `Δ ≥ +0.01` | **PASS** |
| | **KL×1000 ↓** | `1.1064` | **`1.1215`** | `1.014×` | `≤ 1.100×` | **PASS** |
| | **FAD-CLAP ↓** | `0.2145` | **`0.1744`** | **`0.813×`** *(-18.7%)* | `≤ 1.100×` | **PASS** |
| | **Audiobox PQ ↑** | `7.6452` | **`7.6302`** | `Δ = -0.0151` (`CI95_lo = -0.0397`) | `CI95_lo ≥ -0.05` | **PASS** |
| | **Attribute Guards** | — | **`0` vi phạm** | `Violations: []` | `0` vi phạm | **PASS** |
| **Confirm** (`N=6.720`) | **SSIM ↑** | `0.6092` | **`0.6912`** | **`+0.0820`** | `Δ ≥ +0.01` | **PASS** |
| | *(Single / D+T)* | `0.6242 / 0.5891` | **`0.7442 / 0.6205`** | `+0.1200 / +0.0314` | — | — |
| | **KL×1000 ↓** | `0.8839` | **`0.9711`** | `1.099×` | `≤ 1.100×` (`0.9723`) | **PASS** |
| | *(Single / D+T)* | `0.6923 / 1.1393` | `0.7825 / 1.2226` | `1.130× / 1.073×` | — | — |
| | **FAD-CLAP ↓** *(`n=5.306`)* | `0.0756` | **`0.0466`** | **`0.616×`** *(-38.4%)* | `≤ 1.100×` | **PASS** |
| | *(Single / D+T)* | `0.0723 / 0.0837` | **`0.0337 / 0.0716`** | `-53.4% / -14.5%` | — | — |
| | **Audiobox PQ ↑** *(`n=5.306`)* | `7.7103` | **`7.7061`** | `Δ = -0.0042` (`CI95_lo = -0.0104`) | `CI95_lo ≥ -0.05` | **PASS** |
| | *(Single / D+T)* | `7.7481 / 7.6599` | **`7.7522 / 7.6446`** | `+0.0041 / -0.0153` | — | — |
| | **Attribute Guards** | — | **`0` vi phạm** | Max `det = +0.0189` (`clarity`) | `norm_det ≤ +0.05` | **PASS** |

*\*Ghi chú phạm vi FAD-CLAP & PQ trên Confirm: tính trên `n = 5.306` mẫu (`758/960` nguồn) có sẵn CLAP embedding `Clean`/`A0` trong `14_run7000/s01..s16` (`Clean PQ = 7.8894`); `SSIM`, `KL×1000` và `19 Attribute Guards` tính đủ 100% trên `6.720` mẫu.*

### 4.2. Hiệu năng riêng trên nhóm mẫu không fallback `S` (trực tiếp qua `full_mix_P` / `vocal_mask_P`)

| Tập | Nhóm | `n` | `A0` SSIM → **V2 SSIM** | `A0` KL×1000 → **V2 KL×1000** | `A0` FAD → **V2 FAD** | `A0` PQ → **V2 PQ** |
|---|---|---:|---|---|---|---|
| **Dev** | **All** | `100` | `0.6384 →` **`0.9340`** (`+0.2955`) | `0.6374 → 0.6796` (`1.066×`) | — | — |
| | Single / D+T | `83 / 17` | `0.6370→0.9386` / `0.6453→0.9115` | `0.6193→0.6586` / `0.7258→0.7818` | — | — |
| **Confirm** | **All** | `2.083` *(2.077)* | `0.6423 →` **`0.9068`** (`+0.2645`) | `0.5370 → 0.8183` (`1.524×`) | `0.0694 →` **`0.0178`** (**`-74.3%`**) | `7.7848 → 7.7740` (`-0.0107`) |
| | Single | `1.651` *(1.647)* | `0.6440 →` **`0.9229`** (`+0.2789`) | `0.5021 → 0.7118` (`1.418×`) | `0.0697 →` **`0.0163`** (**`-76.6%`**) | `7.7810 →` **`7.7885`** (**`+0.0075`**) |
| | Double+Triple | `432` *(430)* | `0.6359 →` **`0.8453`** (`+0.2094`) | `0.6702 → 1.2254` (`1.828×`) | `0.0914 →` **`0.0484`** (**`-47.0%`**) | `7.7994 → 7.7188` (`-0.0806`) |

### 4.3. Kết quả riêng của nhánh `vocal_mask_P` (khắc phục lỗi kim loại trên giọng hát)

| Tập | Nhóm | `n` | Chỉ số `A0` → **Hybrid-v2 PA1** | Ghi chú |
|---|---|---:|---|---|
| **Dev** | `vocal_mask_P` | `7` | **SSIM:** `0.5961 →` **`0.6746`** (`+0.0785`) \| **KL×1000:** `0.7540 → 0.8864` (`1.176×`) | Giữ KL `< 1.0` (trước đây `full_mix_P` vọt `~4.2–4.9`) |
| **Confirm** | `vocal_mask_P` *(135 chạy + 6 guard)* | `141` | **FAD-CLAP:** `0.1558 →` **`0.1222`** (**`-21.5%`**) \| **PQ:** `7.4950 →` **`7.5987`** (**`+0.1037`**) | Tăng PQ mạnh nhất trong tất cả các nhánh |
| **Confirm** | `vocal` Attribute Guard | `304` | Lỗi thuộc tính `vocal`: `0.06291 → 0.06263` (`norm_det = -0.0011`) | Cải thiện nhẹ so với `A0` |

---

## 5. Kết Luận & Hồ Sơ Lưu Trữ

- **Kết luận:** **Hybrid-v2 PA1** giải quyết triệt để suy giảm phổ và suy giảm nhận thức do VAE trên các phép xử lý tuyến tính và giọng hát mà không cần huấn luyện lại mô hình. Trên Confirm (`6.720` mẫu), phương án đạt **PASS 5/5 cổng**, nâng `SSIM` toàn tập lên **`0.6912`** (`+0.0820`), giảm `FAD-CLAP` **`-38.4%`** (riêng nhóm không fallback giảm **`-74.3%`** và `SSIM` đạt **`0.9068`**), giữ nguyên `Audiobox PQ` (`Δ = -0.0042`) và **0 vi phạm** trên 19 thuộc tính.
- **Các tệp lưu trữ chính:**
  - Cấu hình khóa: [`lock_v2.json`](file:///d:/SonicMaster_Reproducing/artifacts/10_extension/lock_v2.json)
  - Báo cáo Dev: [`tables12_dev.json`](file:///d:/SonicMaster_Reproducing/artifacts/10_extension/dev/hybrid_v2/tables12_dev.json), [`dev_pq_fad.json`](file:///d:/SonicMaster_Reproducing/artifacts/10_extension/dev/hybrid_v2/pq_fad_v2/out/dev_pq_fad.json)
  - Báo cáo Confirm: [`confirm_eval_summary.json`](file:///d:/SonicMaster_Reproducing/artifacts/10_extension/confirm/hybrid_v2/eval/confirm_eval_summary.json), [`confirm_pq_fad_summary.json`](file:///d:/SonicMaster_Reproducing/artifacts/10_extension/confirm/hybrid_v2/eval/confirm_pq_fad_summary.json)
  - Mã nguồn chính: [`branch_p.py`](file:///d:/SonicMaster_Reproducing/scripts/extension/branch_p.py), [`run_hybrid_v2_dev.py`](file:///d:/SonicMaster_Reproducing/scripts/extension/run_hybrid_v2_dev.py), [`run_hybrid_v2_confirm.py`](file:///d:/SonicMaster_Reproducing/scripts/extension/run_hybrid_v2_confirm.py), [`eval_confirm_fast.py`](file:///d:/SonicMaster_Reproducing/scripts/extension/eval_confirm_fast.py)

---

## 7. Tài Liệu Tham Khảo & Nguồn Gốc Kỹ Thuật (References & Provenance)

| Thành phần / Chỉ số | Nguồn gốc (Paper / Repo) | Ghi chú áp dụng trong `Hybrid-v2` |
|---|---|---|
| **Baseline `A0`, tập test 7.000 mẫu, EQ/Dynamics/Reverb/Clip/Stereo/KL/SSIM/FAD/PQ** | J. Melechovsky, A. Mehrish, A. Roy, D. Herremans, *"SonicMaster: Towards Controllable All-in-One Music Restoration and Mastering"*, ICML 2026 ([arXiv:2508.03448](https://arxiv.org/abs/2508.03448); [GitHub](https://github.com/AMAAI-Lab/SonicMaster)) | Nguồn cung cấp tín hiệu suy hao `x`, đầu ra generative `s` (`A0`), và toàn bộ bộ chỉ số đánh giá Bảng 1–3 (`evaluation/`). |
| **PA-1 (Phase-Aligned Mixing & Crossover Band Routing)** | Phân tích & thiết kế DSP tuyến tính / Linkwitz-Riley crossover chuẩn âm thanh (không dùng mạng neural bổ sung) | Trộn giữa `x` (bảo toàn waveform/phase) và `s` (`A0`) để tránh suy hao codec của VAE trong `Stable Audio Open 1.0` ở các bài EQ tuyến tính. |
| **Perceptual Quality (`PQ`)** | A. Tjandra et al. (Meta FAIR), *"Meta Audiobox Aesthetics: Unified Automatic Quality Assessment for Speech, Music, and Sound"*, 2025 ([arXiv:2502.05139](https://arxiv.org/abs/2502.05139)) | Chấm điểm chất lượng cảm nhận `PQ` độc lập theo chuẩn `audiobox_aesthetics`. |
| **FAD-CLAP** | Y. Wu et al., *"Large-Scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation"*, IEEE ICASSP 2023 ([arXiv:2211.06687](https://arxiv.org/abs/2211.06687)); A. Gui et al., *"Adapting Fréchet Audio Distance for Generative Music Evaluation"*, IEEE ICASSP 2024 ([arXiv:2311.01616](https://arxiv.org/abs/2311.01616)) | Trích xuất embedding CLAP và tính Fréchet Audio Distance trên toàn tập gộp. |
| **Mel-Spectrogram SSIM** | Z. Wang et al., *"Image Quality Assessment: From Error Visibility to Structural Similarity"*, IEEE TIP 2004 (áp dụng trên log-Mel spectrogram theo `extract_kl_ssim_mass.py` của SonicMaster) | Đo mức độ bảo toàn cấu trúc phổ thời gian–tần số so với `Clean`. |
