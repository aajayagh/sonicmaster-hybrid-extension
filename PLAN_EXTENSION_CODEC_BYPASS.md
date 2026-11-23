# PLAN — Mở rộng `10_extension`: Pipeline "SonicMaster điều khiển, xử lý trên sóng gốc" (Codec-Bypass Hybrid)

> Tài liệu bàn giao cho agent thực thi. Đọc hết trước khi chạy bất kỳ lệnh nào.
> Ngày lập: 2026-11-19. Trạng thái: **planned**.

---

## 0. Bối cảnh & lý do (đọc để hiểu, không cần làm lại)

Dự án đã tái lập M1 trên 7.000 mẫu (`artifacts/14_run7000/`, báo cáo `BAO_CAO_REPRODUCE_7000.md`). Vấn đề còn lại: **giọng hát nghe "kim loại"**. Bằng chứng trong `outputs/`:

| Thư mục | Kết quả | Kết luận |
|---|---|---|
| `vocal_clean_vae_diagnostic` | Clean → VAE đã kim loại | Lỗi do codec (VAE Stable Audio Open), không do flow |
| `decoder_mixed_latent_pilot` | Fine-tune decoder: nghe không khác | Loại hướng sửa decoder nhẹ |
| `latent_inversion_1874206`, `flow_inversion_1874206` | Tối ưu latent không cải thiện phổ | Trần là tầm biểu diễn của decoder → loại hướng "học z\*" và "loss audio giải mã" |
| `codec_listen_*` | DAC ≈ không khác; Music2Latent tệ nhất | Loại hướng đổi codec có sẵn |
| `waveform_residual/1283374_deg2` | **Residual ít kim loại hơn SM**; gain-only làm đệm rè | Tín hiệu dương duy nhất |
| `stem_bypass_*` | Demucs rò; `339425` split-sum error 14% | Không ghép stem cứng |
| VRG (`PLAN_CLIP_JOINS_VRG.md`, nhật ký) | Blend α=0.1 qua gate CPU, ΔPQ −0.017 | Giữ một phần sóng gốc có lợi |
| `decoded_flow_pilot` | Đang chạy/đã chạy — **không phụ thuộc plan này** | Ghi kết quả riêng, không trộn |

**Ý tưởng:** giữ SonicMaster để *quyết định sửa gì*, nhưng thành phần giọng ở đầu ra lấy từ nhánh không đi qua decoder VAE.

---

## 1. Ràng buộc bắt buộc (từ `GEMINI.md`)

1. Đây là **mở rộng của nhóm**, không phải tái lập. Mọi output nằm trong `artifacts/10_extension/` và báo cáo riêng `BAO_CAO_EXTENSION_CODEC_BYPASS.md`. **Không** sửa/ghi đè `artifacts/14_run7000/` hay các bảng tái lập.
2. Không sửa mã tác giả trong `SonicMaster/`. Code mới đặt ở `scripts/extension/`.
3. Không chỉnh tham số sau khi nhìn kết quả tập xác nhận. Mọi tham số/ngưỡng khóa trong `artifacts/10_extension/lock.json` (kèm SHA-256 của chính file lock và các script) **trước** khi chạy tập xác nhận.
4. Không dùng clean ground-truth trong pipeline suy luận (chỉ dùng trong chấm điểm). Kiểm tra: grep script pipeline không được đọc đường dẫn clean.
5. Không loại mẫu lỗi im lặng: mọi mẫu fail ghi vào `failures.csv` với nguyên nhân.
6. Không thay phiếu nghe bằng PQ/LLM. Kết quả nghe chưa có ghi `planned`.
7. Mọi trọng số mới (bộ tách) ghi URL + revision + SHA-256 vào `artifacts/10_extension/source-lock-extension.json`.
8. Không in token. Tải mô hình qua cách đã dùng trong dự án (Kaggle Secrets nếu chạy Kaggle).

---

## 2. Ký hiệu

- `x`: input lỗi (stereo, 44.1 kHz), `s`: output SonicMaster đã có từ run7000, `r = D(E(x))`: VAE roundtrip của input (hàng "Reconstructed").
- `V`: nhánh giọng ít/không qua codec. `M`: mặt nạ giọng mềm trên STFT. `out`: đầu ra cuối.
- Mọi tín hiệu phải cùng độ dài và lag = 0 (kiểm bằng cross-correlation, ghi lag vào log; nếu |lag| > 0 → dừng, điều tra, không tự căn chỉnh theo clean).

---

## 3. Phase 0 — Kiểm kê đầu vào (CPU, ~1 giờ)

1. Tìm và ghi đường dẫn thực tế của: `x`, `s`, `r`, clean cho 7.000 mẫu trong `artifacts/14_run7000/` (và nơi lưu audio local/Kaggle dataset). Đọc `artifacts/14_run7000/` README/manifest; không đoán.
2. Lấy prompt chính thức + nhãn loại lỗi (degradation type) cho từng mẫu từ manifest `01_manifest`.
3. Xuất `artifacts/10_extension/inputs_manifest.csv`: `sample_id, source_id, variant(deg1..7), n_degradations, degradation_types, prompt, path_x, path_s, path_r, path_clean, sha256_*`.
4. Nếu thiếu `r` cho mẫu nào → ghi lại; tạo bằng VAE mean mode đúng như job `04_infer` (cùng code, cùng version), ghi riêng.
5. Chia nguồn:
   - **Dev** = 40 nguồn validation đã dùng ở VRG (dùng lại danh sách cũ nếu có, ghi nguồn danh sách). Nếu không có, lấy 40 `source_id` ngẫu nhiên seed=20261119 **từ test set**, rồi loại 40 nguồn này khỏi tập xác nhận.
   - **Confirm** = toàn bộ nguồn còn lại (~960 nguồn × 7 variants).
   - **Listen** = 20 nguồn trong Confirm có giọng (lọc bằng năng lượng stem vocal của bộ tách trên `s` > ngưỡng, ngưỡng ghi trước), seed cố định. Không ai nghe trước tập này.
   - Kiểm tra source-disjoint giữa Dev/Confirm, ghi vào `splits.json`.

**Done khi:** manifest đủ 7.000 dòng (hoặc liệt kê rõ thiếu), `splits.json` có hash.

---

## 4. Phase 1 — Chọn bộ tách giọng (GPU ngắn, ~2–3 giờ)

Ứng viên (qua package `audio-separator`, ghi version):
1. `htdemucs_ft` (mốc).
2. BS-RoFormer `model_bs_roformer_ep_317_sdr_12.9755`.
3. Mel-Band RoFormer vocals (bản Kim).
4. Ensemble trung bình biên độ 2+3.

Thí nghiệm, **chỉ trên Dev**:
- Chạy mỗi bộ tách trên 3 tín hiệu: clean, `x`, `s`.
- "Mốc giọng" = stem vocal tách từ clean bằng ensemble (ghi rõ đây là mốc xấp xỉ, không phải ground truth).
- Đo: (a) **rò nhạc** = năng lượng của `acc_clean_ref` chiếu vào stem vocal (dB tương đối), (b) SI-SDR stem vocal so với mốc, (c) thời gian/clip, VRAM.
- Quy tắc chọn (khóa trước): chọn model có rò nhạc trung vị thấp nhất **khi tách trên `s`**; hòa (chênh < 0.5 dB) → chọn model nhanh hơn.

**Done khi:** `artifacts/10_extension/phase1_separator/summary.csv` + model được chọn ghi vào `lock.json`.

---

## 5. Phase 2 — Cài đặt pipeline (`scripts/extension/`)

Tạo các module, mỗi module có unit test trong `scripts/extension/tests/`:

### 5.1 `routing.py`
Ánh xạ loại lỗi → nhánh:
- **P (tham số):** clarity, boom, airy, bright, dark, muddy, warm, vocals (EQ), mic, xband, volume, stereo, compression, punch.
- **R (residual):** reverb (small/big/mixed/real), noise (nếu có), clip.
- Lỗi kép/ba: áp **R trước rồi P** nếu có cả hai; chỉ P → chuỗi P; chỉ R → R.
- Đọc danh sách loại lỗi từ manifest/prompt chính thức; loại lạ → `failures.csv`, fallback `out = s`, đếm số lượng.

### 5.2 `branch_r.py`
`V_R = x + (s − r)` cho từng kênh. Kiểm tra lag = 0. Giới hạn true-peak bằng soft-limiter (ceiling −1 dBTP) chỉ khi cần, ghi số mẫu bị limit.

### 5.3 `branch_p.py` (ước lượng tham số từ `s` so với đầu vào nhánh, áp DSP lên đầu vào nhánh)
- **EQ:** phổ trung bình dài hạn (STFT 4096, hop 1024, Hann) của `s` và `u` (u = x hoặc V_R). Gain(f) = 10·log10(P_s/P_u), gộp theo dải 1/3 octave 20 Hz–20 kHz, làm mượt 3 dải, kẹp ±12 dB. Thiết kế FIR pha tuyến tính 4097 tap (`scipy.signal.firwin2`), bù trễ chính xác.
- **Volume:** khớp LUFS (`pyloudnorm`) của `u` về `s`.
- **Stereo:** M/S; nhân hệ số k = RMS(S_s)/RMS(S_u) × RMS(M_u)/RMS(M_s) cho kênh side, kẹp k ∈ [0, 4].
- **Compression/punch:** ước lượng từ phân bố RMS khung (2048/1024): chọn tham số compressor (threshold, ratio ∈ [1, 8], attack 10 ms, release 100 ms) bằng grid nhỏ cực tiểu |std_RMS(out) − std_RMS(s)| — **mục tiêu là `s`, không phải clean**. Nếu `s` có độ động cao hơn `u` (expansion) → dùng upward expander tương tự hoặc bỏ qua và ghi lại.
- Thứ tự P: EQ → dynamics → stereo → volume.

### 5.4 `mask.py`
- Chạy bộ tách đã khóa trên `s` → `voc`, `acc`.
- STFT (2048, hop 512) cùng tham số cho mọi tín hiệu. `M = |VOC| / (|VOC| + |ACC| + 1e-8)`, mỗi kênh.
- Làm mượt: trung bình trượt 50 ms theo thời gian, 1/3 octave theo tần số. Kẹp [0.1, 0.9].
- Biến thể không model (A6): `M = 0.9` trong 1–8 kHz với chuyển tiếp cosine 1/2 octave, `0.1` ngoài dải; nhân thêm cờ khung có giọng = (năng lượng vocal_stem > ngưỡng) — **A6 thuần** không dùng bộ tách: thay cờ bằng 1 toàn bộ.

### 5.5 `mix.py`
`OUT = M·STFT(V) + (1−M)·STFT(s)` → ISTFT. Rồi khớp LUFS tích hợp của `out` về `s` và true-peak ≤ −1 dBTP. Ghi gain áp dụng.

### 5.6 `run_pipeline.py`
CLI: `--split dev|confirm --variants A0..A6 --manifest ... --out artifacts/10_extension/<run_id>/`. Ghi WAV 44.1 kHz float32 stereo, `outputs_manifest.csv` (sha256, độ dài, lag, gain, cờ limiter, nhánh dùng), `failures.csv`, log đầy đủ, `environment-lock.txt`. Ghi nguyên tử (`.part` → rename), resume được theo sample_id.

### 5.7 Unit test bắt buộc
- `x = s = r` ⇒ A2/A4/A5 trả về ≈ `s` (sai số < 1e-5).
- M ≡ 1 ⇒ out = V; M ≡ 0 ⇒ out = s.
- Branch P với `s` = `x` đã EQ bằng bộ lọc biết trước ⇒ khôi phục gain đúng ±0.5 dB.
- Tín hiệu bị dịch 1 mẫu ⇒ phát hiện lag và dừng.
- Script pipeline không mở bất kỳ file clean nào (test bằng monkeypatch `open`/kiểm tra đường dẫn).

---

## 6. Các biến thể (ablation)

| ID | Định nghĩa |
|---|---|
| A0 | `s` (SonicMaster gốc) |
| A1 | `0.9·s + 0.1·x` (đối chứng VRG) |
| A2 | `V_R` cho mọi mẫu, không mặt nạ |
| A3 | P áp lên `x` cho mọi mẫu, không mặt nạ |
| A4 | Định tuyến P/R (mục 5.1), không mặt nạ |
| **A5** | **A4 + mặt nạ bộ tách (ứng viên chính)** |
| A6 | A4 + mặt nạ dải 1–8 kHz không model |

Kèm hàng tham chiếu Input và Reconstructed (đã có).

---

## 7. Phase 3 — Dev (40 nguồn × 7 = 280 mẫu)

1. Pilot 1 mẫu → 1 nguồn (7 mẫu) → đo thời gian/VRAM/dung lượng → ước tính cho 7.000; ghi `budget.json`.
2. Chạy A0–A6 trên Dev, chấm bằng **đúng code chấm của run7000** (`SonicMaster/evaluation/` + script đã dùng; ghi hash): Bảng 1, 2, 3 (FAD gộp một lần theo nhóm, KL×1000, SSIM, PQ Audiobox).
3. Sanity: chấm A0 trên Dev phải trùng điểm run7000 cho cùng mẫu (sai < 1e-6 với metric tất định). Không trùng → dừng.
4. Được phép chỉnh **chỉ trên Dev**, tối đa các knob sau, grid nhỏ ghi trước:
   - Kẹp mặt nạ: {[0.1,0.9], [0.2,0.8], [0,1]}
   - Độ mượt thời gian: {20, 50, 100 ms}
   - Kẹp EQ: {±6, ±12 dB}
   Tiêu chí chọn: tối đa ΔSSIM so với A0 với điều kiện ΔPQ ≥ −0.05 và không guard thuộc tính nào vi phạm (mục 8). Ghi toàn bộ grid, không chỉ cấu hình thắng.
5. Nghe nội bộ trên Dev cho phép (để bắt lỗi rõ như click, pha, pumping), ghi nhận xét. **Không** dùng tập Listen.
6. Khóa: ghi `lock.json` (tham số, model tách + hash, hash các script, danh sách Confirm/Listen, ngưỡng mục 8). Commit/snapshot.

**Gate chuyển Phase 4:** A5 (hoặc A6) đạt tiêu chí mục 8 trên Dev. Nếu không biến thể nào đạt → dừng, viết báo cáo âm tính (vẫn có giá trị), không chạy Confirm.

---

## 8. Ngưỡng thành công (khóa trước Phase 4)

Trên Confirm, so ứng viên đã khóa với A0, báo cáo tách **Single / Double+Triple / All**:

1. **SSIM:** Δ ≥ +0.01 (CI95% bootstrap 2.000 lần theo source_id, cận dưới > 0).
2. **PQ:** Δ ≥ −0.05 (cận dưới CI95%).
3. **FAD-CLAP:** không tệ hơn A0 quá 10% tương đối.
4. **KL×1000:** không tệ hơn A0 quá 10% tương đối.
5. **Guard thuộc tính (Bảng 1–2):** với mọi thuộc tính có ≥ 10 mẫu, sai số chuẩn hóa xấu đi không quá 5% so với A0. Thuộc tính < 10 mẫu: chỉ báo cáo, đánh dấu "ít mẫu".
6. **Nghe mù (tập Listen, 20 nguồn có giọng):** ≥ 3 người nghe, A/B ngẫu nhiên thứ tự, cân loudness, đoạn 8–10 s chứa giọng (vị trí chọn bằng năng lượng stem vocal của `s`, không dựa trên điểm). Hai câu: "bản nào ít tiếng kim loại ở giọng hơn" và "bản nào tốt hơn tổng thể". 2 cặp lặp lại để kiểm độ tin cậy người nghe. Thành công: ứng viên thắng ≥ 65% câu "kim loại", binomial một phía p < 0.05 (gộp người nghe, kiểm tra thêm theo từng người), và câu "tổng thể" không thua có ý nghĩa.

Kết luận chỉ "thành công" khi đạt 1–5 **và** 6. Đạt 1–5 mà chưa có 6 → `partial`.

---

## 9. Phase 4 — Confirm (≈ 6.720 mẫu)

1. Kiểm tra hash `lock.json` và các script trùng bản khóa; khác → dừng.
2. Chạy chỉ A0 (đã có), A1, và ứng viên đã khóa (+A4 để giải thích). Chạy theo shard giữ đủ 7 variants của một nguồn trong cùng shard (tương tự run7000; có thể tái dùng hạ tầng Kaggle nhiều tài khoản, ghi rõ tài khoản/version notebook).
3. Chấm đầy đủ Bảng 1–3 + per-sample CSV + bootstrap CI.
4. Xuất 3–5 bộ minh họa: `clean`, `input`, `sonicmaster`, `extension` cho từng sample ID.

---

## 10. Phase 5 — Nghe mù

1. Tạo trang nghe giống `outputs/vocal_calibration/` (HTML tĩnh, file tên ngẫu nhiên, key tách riêng). Thêm 2 cặp lặp lại.
2. Người dùng tuyển người nghe; agent **không** tự điền phiếu. Lưu phiếu ẩn danh JSON.
3. Phân tích theo mục 8.6, báo cáo cả từng người nghe.

---

## 11. Sản phẩm bàn giao

```
artifacts/10_extension/
  inputs_manifest.csv, splits.json, lock.json, source-lock-extension.json, budget.json
  phase1_separator/summary.csv
  dev/<run_id>/{outputs_manifest.csv, failures.csv, scores_per_sample.csv, summary.csv, grid.csv, logs/}
  confirm/<run_id>/{...same..., bootstrap_ci.csv}
  listening/{page/, key.json (riêng), responses/, analysis.json}
  demos/<sample_id>/{clean,input,sonicmaster,extension}.wav
scripts/extension/ (code + tests)
BAO_CAO_EXTENSION_CODEC_BYPASS.md
```

Báo cáo phải có: bảng `Metric | A0 (SonicMaster đo lại) | Ứng viên | Delta (abs, %) | N | CI95% | Evidence path` cho Single / Double+Triple / All; bảng ablation A0–A6 trên Dev; grid đầy đủ; số mẫu fail và lý do; kết quả nghe hoặc `planned`; mục "Giới hạn" (bộ tách train trên mix sạch, mốc giọng xấp xỉ, nhánh P ước lượng từ `s` nên kế thừa sai của SM, số người nghe).

**Cấm** ghi bất kỳ câu nào kiểu "vượt paper SonicMaster" — so sánh là với SonicMaster **đo lại** trong cùng protocol, và đây là pipeline hậu xử lý có thêm mô hình tách.

---

## 12. Rủi ro & cách xử lý

| Rủi ro | Dấu hiệu | Xử lý |
|---|---|---|
| Residual giữ lại lỗi gốc (dark/mud xấu đi như VRG λ=0.25) | Guard thuộc tính vi phạm | Ghi nhận; chỉ dùng P cho nhóm đó **nếu** quyết định trên Dev |
| Mặt nạ gây pumping / nhạc phụt | Nghe Dev, phương sai năng lượng khung tăng | Tăng độ mượt trong grid đã định |
| EQ match khuếch đại nhiễu ở dải năng lượng thấp | Gain chạm ±12 dB nhiều | Kẹp ±6 dB (đã trong grid) |
| Lỗi kép phức tạp sai thứ tự | Double+Triple tệ hơn Single rõ rệt | Báo cáo; không đổi thứ tự sau Confirm |
| Lệch pha giữa `s` và `x` | Comb filtering khi trộn | Kiểm lag/tương quan pha mỗi mẫu, ghi `phase_corr`; mẫu tương quan < 0 → fallback `s`, đếm |
| Thiếu quota GPU | — | Phase 1 & mask chạy được trên CPU chậm hơn; Dev trước, Confirm sau |

---

## 13. Thứ tự thực thi tóm tắt

Phase 0 → Phase 1 → Phase 2 (+tests xanh) → Phase 3 (pilot → Dev → khóa) → **Gate** → Phase 4 → Phase 5 → Báo cáo.
Cập nhật `SONICMASTER_DIARY.md` sau mỗi phase (ngày giờ, việc làm, kết quả, đường dẫn bằng chứng).
