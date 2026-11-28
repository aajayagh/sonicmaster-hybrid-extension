# Thiết Kế Kỹ Thuật `Hybrid-v4.6` (`v4_6`: Stage 3 Dual-Domain Discriminator & Symmetric `bright` Fix)

**Mục tiêu:** Thực hiện **Stage 3 Fine-Tuning** trực tiếp từ checkpoint đã hội tụ [`v4_5p/best.pt`](file:///d:/SonicMaster_Reproducing/artifacts/10_extension/v4/kaggle_out_v4_5p/out/v4_5p/best.pt) (`1,953,200` tham số, nạp `strict=True`, **không sửa kiến trúc Generator**) nhằm gỡ dứt điểm **đúng 2 nút thắt cuối cùng** trên Dev `280` trước khi chạy `TF-MoE` / `Locked Confirm`:
1. **Mở khóa `phase_gate_head` (`g_ph`) & chọn lọc `g_mag` trên `Reverb (106)`** để kéo `Reverb Audiobox PQ` (`6.9494` → gần mốc `A0 = 7.59`, khép khoảng cách `-0.6442`) mà vẫn bảo toàn `Mel-SSIM` (`0.8396`), `Mel-KL` (`0.9345`) và `Official MSD` (`0.3790`).
2. **Sửa lỗi guard duy nhất còn lại `bright` (`n = 19`, `norm = +0.1710`)** bằng phạt tỷ lệ năng lượng dải `6–22 kHz` đối xứng trực tiếp, không gây hại `SSIM`/`KL` như `V4-4bs`.

---

## 1. Cơ Sở Thực Nghiệm Từ Chẩn Đoán `V4-5p` ([v4_5p_diagnostic_table.md](file:///d:/SonicMaster_Reproducing/artifacts/10_extension/v4/v4_5p_diagnostic/v4_5p_diagnostic_table.md))

| Phát hiện đo trực tiếp trên `V4-5p` | Con số bằng chứng | Hệ quả thiết kế cho `V4-6` |
|---|---|---|
| **1. `g_ph` vẫn đóng kín trên Reverb (`1–12 kHz`)** | `Reverb (106)` có `g_mag = 0.222–0.456`, nhưng `g_ph = 0.160` (`0–1k`), **`0.035` (`1–4k`)**, **`0.031` (`4–12k`)** | Chỉ có `w_comp=0.25` và `w_wav=0.15` đẩy gradient vào pha; cả 2 hàm loss đều đối chiếu từng điểm với `Clean` nên ưu tiên giữ 95% pha còn vang của `P`. Phải **tắt hoàn toàn `w_comp = 0.0, w_wav = 0.0`** ở Stage 3! |
| **2. Mở pha `S` thu hồi ngay 53%–72% PQ Reverb** | Trên 24 mẫu Reverb stratified: `default PQ = 6.7589` → **`ph_S_all_noR = 7.1500` (`+0.3911`, `Mel-SSIM = 0.6763`)** → **`mag_hi>=0.7 + ph_S (noR) = 7.2899` (`+0.5310`, `72%` khoảng cách tới `A0 = 7.4971`)** | `Audiobox PQ` (dựa trên encoder **WavLM** miền sóng) nghe rất rõ **vệt méo pha** mà `ModSpecLoss`/`MelKL` (thuần biên độ) và các bộ nhúng mel như VGGish/CLAP không phân biệt được. Cần **Dual-Domain Discriminator (Waveform + Complex-STFT)** để trực tiếp nhận diện pha còn vang so với sóng/STFT thật! |
| **3. Lỗi `bright (19)` là CHÓI QUÁ ĐÀ, không phải thiếu sáng** | `19/19` `Input` bị tối (`ratio = 0.00368` vs Clean `0.01624`), nhưng `P_only` vọt lên **`0.02806` (`+73%` quá sáng, `norm = +0.9156`)**; `default` còn `+0.00302` (`norm = +0.1710`) | Loại bỏ `l_hf_mag` của `V4-4bs`; dùng **`SymmetricBandRatio6kLoss`** thuần tỷ lệ năng lượng `[6,000, 22,050)` Hz (`scale = 20.0`, `λ_br = 0.25` $\Rightarrow$ hệ số hữu hiệu `5.0`) để kìm chính xác cú vọt dải cao của `P` về đúng `Clean`. |

---

## 2. Nguồn Gốc Bài Báo (Verified Paper Provenance)

| Thành phần kỹ thuật trong `V4-6` | Bài báo gốc (Đã kiểm chứng) | Cách áp dụng |
|---|---|---|
| **Dual-Domain Adversarial + Feature Matching (Time & Time–Frequency)** | J. Su, Z. Jin, A. Finkelstein, *"HiFi-GAN: High-Fidelity Denoising and Dereverberation Based on Speech Deep Features in Adversarial Networks"*, Interspeech 2020 | Cơ sở trực tiếp dùng **Multi-Scale Time + STFT Discriminator & Deep Feature Matching** cho bài toán khử nhiễu/khử vang (dereverberation) thay vì dùng perceptual loss thuần mel. |
| **Multi-Scale Waveform Discriminator (`MSD`)** | J. Kong, J. Kim, J. Bae, *"HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis"*, NeurIPS 2020 ([arXiv:2010.05646](https://arxiv.org/abs/2010.05646)) | 3 tầng phân giải `1× / 2× / 4×` avg-pool Conv1d trên sóng M/S để bắt méo pha thời gian thực mà không cần căn pha tuyệt đối với `Clean`. |
| **Multi-Resolution Complex-STFT Discriminator (`MRCD`) & Normalized Feature Matching** | A. Défossez, J. Copet, G. Synnaeve, Y. Adi, *"High Fidelity Neural Audio Compression"* (**EnCodec**), TMLR 2023 ([arXiv:2210.13438](https://arxiv.org/abs/2210.13438)) | Discriminator 2D trên phức STFT đa phân giải (`512, 1024, 2048`) và phép chuẩn hóa feature matching `L1 / (mean|rf| + 1e-4)` giúp cân bằng gradient giữa nhiều tầng. |
| **Ổn định huấn luyện GAN audio** | S. Lee, W. Ping, B. Ginsburg, B. Catanzaro, S. Yoon, *"BigVGAN: A Universal Neural Vocoder with Large-Scale Training"*, ICLR 2023 ([arXiv:2206.04658](https://arxiv.org/abs/2206.04658)) | `spectral_norm`, Hinge GAN nhẹ kết hợp reconstruction loss giữ ổn định không sụp mode. |
| **Decoupled Magnitude–Phase Fusion** | Y.-X. Lu, Y. Ai, Z.-H. Ling, *"MP-SENet"*, Interspeech 2023 ([arXiv:2308.08926](https://arxiv.org/abs/2308.08926)) | Giữ nguyên decoder tách `g_mag` và `g_ph` của `V4-5p`. |
| **Loại bỏ `Wav/Comp L1` khi cần mở pha** | Z.-Q. Wang, G. Wichern, J. Le Roux, *"On the Compensation Between Magnitude and Phase in Speech Separation"*, [MERL TR2021-137](https://www.merl.com/publications/TR2021-137) / IEEE SPL 2021 | Khóa `w_comp = 0.0, w_wav = 0.0` ở Stage 3 để giải phóng `phase_gate_head`. |
| **Đánh giá & Nền tảng `A0` (`s`)** | J. Melechovsky et al., *"SonicMaster"*, ICML 2026 ([arXiv:2508.03448](https://arxiv.org/abs/2508.03448)) | Đánh giá Bảng 1–3 & `Audiobox PQ` giữ hoàn toàn **độc lập (unseen)** khỏi quá trình train. |

---

## 3. Chi Tiết Kiến Trúc & Hàm Mất Mát

### 3.1. Generator (`v4_6` trong [v4_models.py](file:///d:/SonicMaster_Reproducing/scripts/extension/v4_models.py))
- Cấu trúc giống hệt `v4_5p` (`1,953,200` tham số: `encoder = 553,824`, `backbone = 433,152`, `decoder = 966,224`; `MF9_DILATED_OFFSETS`, `decoupled_fusion=True`, `residual_scale=0.5`, `gate_bias_init=-1.5`).
- Cho phép nạp **chặt 100% (`strict=True`)** từ [`v4_5p/best.pt`](file:///d:/SonicMaster_Reproducing/artifacts/10_extension/v4/kaggle_out_v4_5p/out/v4_5p/best.pt).

### 3.2. Dual-Domain Discriminator (`HybridV4Discriminator` trong [v4_discriminators.py](file:///d:/SonicMaster_Reproducing/scripts/extension/v4_discriminators.py))
Hoạt động trên **2 kênh Mid/Side** (`lr_to_ms(y)`) với **`spectral_norm`** trên mọi lớp conv (tự chứa 100% trong PyTorch, chạy hoàn toàn offline trên Kaggle):
- **3 × `ScaleWaveformDiscriminator` (`359,667` tham số):** Độ phân giải `1×`, `avg_pool1d(k=4, s=2, p=1)`, và `2× avg_pool1d`; mỗi nhánh gồm 6 lớp `Conv1d` + `LeakyReLU(0.1)` thu 5 tầng feature map.
- **3 × `ResComplexSTFTDiscriminator` (`414,147` tham số):** Tại `n_fft ∈ {512, 1024, 2048}` (`hop = n_fft // 4`, tính STFT ở `float32`), ghép 6 kênh `[Re(Zc_M), Im(Zc_M), log|Z_M|, Re(Zc_S), Im(Zc_S), log|Z_S|]` qua 5 lớp `Conv2d` + `LeakyReLU(0.1)` thu 4 tầng feature map.
- **Tổng tham số Discriminator:** **`773,814` tham số** (~0.77M, rất gọn nhẹ cho Kaggle T4 16 GB).

### 3.3. Tổ Hợp Loss Stage 3 ([v4_losses.py](file:///d:/SonicMaster_Reproducing/scripts/extension/v4_losses.py), [v4_discriminators.py](file:///d:/SonicMaster_Reproducing/scripts/extension/v4_discriminators.py))
$$\mathcal{L}_G = \mathcal{L}_{\text{MR-STFT}} + 2.0\,\mathcal{L}_{\text{MelKL}} + 2.0\,\mathcal{L}_{\text{ModSpec}} + 1.0\,\mathcal{L}_{\text{M/S}} + 0.25\,\mathcal{L}_{\text{Ratio6k}} + \alpha_{\text{warm}}(e)\,\big(0.05\,\mathcal{L}_{\text{adv}}^{\text{hinge}} + 0.50\,\mathcal{L}_{\text{FM}}^{\text{EnCodec}}\big)$$
trong đó:
- `w_comp = 0.0`, `w_wav = 0.0`, `lambda_g = 0.0` (không phạt pha theo điểm và không ép đóng gate).
- **Lịch Warm-up GAN (`gan_warmup_epochs = 2`):**
  - Epoch 1: $\alpha_{\text{warm}} = 0.0$ (*chỉ train Discriminator nhận diện `Clean` vs `v4_5p`, giữ `G` ổn định*).
  - Epoch 2: $\alpha_{\text{warm}} = 0.5$.
  - Epoch 3–12: $\alpha_{\text{warm}} = 1.0$.
- **Chính sách chọn checkpoint (`best.pt`):** Chỉ xét các epoch có $\alpha_{\text{warm}} = 1.0$ (`ep >= 3`) theo tổ hợp validation `l_stft + 0.5 * l_kl + 0.5 * l_mod + 0.25 * l_fm`, đồng thời lưu cố định `last.pt` và các mốc `ep06.pt`, `ep09.pt`, `ep12.pt`.

---

## 4. Kiểm Chứng Cục Bộ Trước Khi Đẩy Kaggle

- **Unit tests (`tools\extension-env\Scripts\python.exe -m pytest scripts/extension/tests/test_v4_step0.py -v`):**
  - **`18 passed in 48.00s`** (`0` lỗi, `0` cảnh báo).
  - Test mới `test_v4_6_stage3_discriminator_and_band_ratio` xác nhận: `HybridV4Net("v4_6")` đúng `1,953,200` tham số; nạp **`strict=True`** từ [`v4_5p/best.pt`](file:///d:/SonicMaster_Reproducing/artifacts/10_extension/v4/kaggle_out_v4_5p/out/v4_5p/best.pt); `SymmetricBandRatio6kLoss` cho `0` trên tín hiệu trùng, đối xứng, gradient hữu hạn; vòng cập nhật 2 bước `D + G` thực tế với `w_comp=0.0, w_wav=0.0` đưa gradient khác `0` qua `phase_gate_head` và giữ **bất biến 100%** trọng số của `D` trong bước `G`.
- **Kiểm tra sinh script Kaggle (`--dry-run` + `py_compile`):**
  - Chạy `--dry-run` sinh [`train_pilot.py`](file:///d:/SonicMaster_Reproducing/artifacts/10_extension/v4/kaggle_kernel_pilot_train/train_pilot.py) và [`kernel-metadata.json`](file:///d:/SonicMaster_Reproducing/artifacts/10_extension/v4/kaggle_kernel_pilot_train/kernel-metadata.json) (đã gắn `kernel_sources: ["aajayagh/sonicmaster-v4-pilot-v4-5p-run"]`, `enable_internet: false`), kiểm tra biên dịch `python -m py_compile` đạt hoàn toàn.

---

## 5. Lệnh Khởi Chạy Stage 3 Trên Kaggle T4 (Chưa Chạy — Chờ Xác Nhận)

```powershell
python scripts/extension/v4_kaggle_train_pilot.py --account-line 0 --ds-slug aajayagh/sonicmaster-v4-pilot-5s --init-kernel-slug aajayagh/sonicmaster-v4-pilot-v4-5p-run --k-slug sonicmaster-v4-pilot-v4-6-run --variants v4_6 --epochs 12 --stage2-epoch 1 --batch-size 8 --lr 1.5e-4 --lr-d 2e-4 --gate-bias-init -1.5 --lambda-ms 1.0 --lambda-g 0.0 --lambda-kl 2.0 --w-comp 0.0 --w-wav 0.0 --lambda-mod 2.0 --lambda-br 0.25 --use-gan --gan-warmup-epochs 2 --lambda-adv 0.05 --lambda-fm 0.50
```
*(Nếu `aajayagh` đang bận hoặc hết quota, đổi `--account-line 4` `maitvng` hoặc `--account-line 5` `nthkhanh` — lưu ý `--init-kernel-slug aajayagh/sonicmaster-v4-pilot-v4-5p-run` vẫn giữ nguyên vì đã public hoặc gắn quyền đọc tương ứng.)*
