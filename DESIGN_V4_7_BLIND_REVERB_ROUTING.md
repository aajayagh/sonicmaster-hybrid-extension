# DESIGN_V4_7_BLIND_REVERB_ROUTING.md — V4-7: Blind Reverb-Conditioned Phase/Teacher Routing & Dual-Point `bright` Control

> **Trạng thái:** Phase 0 (`Pilot Val N = 560`) và Phase 1 (triển khai mã nguồn, unit test `19/19`, đóng gói Kaggle `--dry-run` + `py_compile`) **đã hoàn tất và xác minh cục bộ**. Tài liệu này khóa thiết kế kỹ thuật trước khi người dùng xác nhận chạy Phase 2 trên Kaggle (`aajayagh/sonicmaster-v4-pilot-v4-7-run`).

---

## 1. Động Lực & Chẩn Đoán Gốc Rễ Từ `V4-5p` và `V4-6`

Sau nghiệm thu `V4-6` trên tập **Dev (`280` nguồn / `280` mẫu matched-protocol)** ([v4_6_results_table.md](file:///d:/SonicMaster_Reproducing/artifacts/10_extension/v4/matched_dev_v4_6/v4_6_results_table.md)):

- `V4-6` giải quyết dứt điểm nhánh `stereo`: chuẩn hóa lệch giảm xuống `+0.0274 [PASS]` (`11/12` eligible guards đạt `PASS`), toàn tập đạt `All SSIM = 0.8414`, `KL = 0.9467`, `FAD = 0.1149`, `PQ = 7.3764`.
- Tuy nhiên, còn đúng **2 điểm nghẽn cuối cùng**:
  1. **Nhánh `Reverb` (`n = 106` trong đánh giá `V4-6`) thiếu `Audiobox PQ`:** đạt `6.9496` so với teacher `A0 = 7.5936`. Cổng pha `g_ph` suy giảm từ `0.0588` (`V4-5p`) xuống còn `0.0415` (`V4-6`) do **cả loss điểm-mẫu (`w_wav`, `w_comp`) lẫn feature matching theo cặp (`L_FM`) đều phạt sai lệch pha cục bộ của `S`**, dù pha của VAE teacher `S` không đồng bộ mẫu với `clean` nhưng cho âm học sạch đuôi vang.
  2. **Guard `bright` (`n = 19`) còn `+0.0869 [FAIL]`** so với ngưỡng `<= +0.05` (dù đã giảm một nửa từ `+0.1710` của `V4-5p`).

Chẩn đoán trên `V4-5p` ([v4_5p_diagnostic_table.md](file:///d:/SonicMaster_Reproducing/artifacts/10_extension/v4/v4_5p_diagnostic/v4_5p_diagnostic_table.md)) đã chứng minh định lượng:
- Mở `g_ph = 1` + tắt nhánh dư cộng tính `R` (`ph_S_all (noR)` / `ph_S_all_noR`) giúp `Reverb PQ` hồi phục `+0.3911` (`6.7589 -> 7.1500`).
- Mở thêm biên độ `g_mag >= 0.7` (`mag_hi>=0.7+ph_S`) hồi phục `+0.5310` PQ (`7.2899` so với `A0 = 7.4971`).
- Lỗi `bright` xuất phát từ vọt năng lượng tần số cao trong nhánh lọc tuyến tính `P_only` (`+73%`: `0.02806` so với Clean `0.01624`; sai số mặc định `|err| = 0.00827`, `norm = +0.17102`).

---

## 2. Phase 0: Kiểm Chứng Phân Tách Mù Trên `Pilot Val` (`N = 560`, Zero Dev Snooping)

Để xác minh `x` và `s` tự mang đủ tín hiệu vật lý phân biệt mẫu có vang (`small`, `big`, `mix`, `real`) với mẫu không vang **mà không chạm tới Dev `280` hay `Locked Confirm`**, Phase 0 chạy [v4_7_phase0_val_probe.py](file:///d:/SonicMaster_Reproducing/scripts/extension/v4_7_phase0_val_probe.py) trên toàn bộ **`560` clip (`80` nguồn độc lập)** của `Pilot Val`, lưu kết quả tại [summary.json](file:///d:/SonicMaster_Reproducing/artifacts/10_extension/v4/v4_7_phase0_val_probe/summary.json):

- **Thời gian chạy & tỷ lệ nhãn:** `1480.2s`, `n_clips = 560`, `n_sources = 80`, `n_reverb = 214` (`38.2%`).
- **6 tín hiệu vật lý cửa sổ `2.0s` trên 24 dải tâm âm (không dùng trọng số backbone `h`):**
  - `delta_rho_mean` (chênh lệch tỷ trọng điều chế nhanh `rho_s - rho_x`): AUROC = **`0.9442`**
  - `env_corr_loss_mean` (`1 - Pearson corr` bao biên thời gian `x` vs `s`): AUROC = **`0.8678`**
  - `env_var_ratio_mean` (`log(CV_s / CV_x)`): AUROC = **`0.8126`**
  - `d_mod_mean` (khoảng cách phổ điều chế chuẩn hóa): AUROC = **`0.8051`**
  - `phase_decorr_low_0-1k` (khử tương quan pha vùng `0–1 kHz`): AUROC = **`0.7798`**
  - `d_mod_midhi_1-12k`: AUROC = **`0.7607`**
  - `log_band_ratio_mean`: AUROC = **`0.6120`**
- **5-fold Source-Grouped Cross-Validation:**

| Mô hình (chỉ 6 cues × 24 dải) | CV AUROC | Ngưỡng (`thr`) | Actual FPR | Recall @ FPR `3.18%` | Ngưỡng cổng Phase 0 |
|---|---|---|---|---|---|
| Logistic Regression (`L2`) | **`0.9817`** | `0.6084` | `3.18%` | `85.5%` | `AUROC >= 0.93` (**PASS**) |
| MLP (`144 -> 32 -> 1`) | **`0.9831`** | `0.7639` | `3.18%` | `85.5%` | `AUROC >= 0.93` (**PASS**) |

- **Chi tiết Recall theo từng tiểu loại Reverb (`LogReg` / `MLP`):**
  - `small (n=37)`: `73.0% / 67.6%`
  - `big (n=35)`: `91.4% / 88.6%`
  - `mix (n=54)`: `83.3% / 87.0%`
  - `real (n=88)`: `89.8% / 90.9%`
- **Chi tiết False Positive Rate trên các nhóm không vang (`LogReg` / `MLP`):**
  - `comp (n=75)`: `9.3% / 8.0%`
  - `punch (n=27)`: `0.0% / 0.0%`
  - `clip (n=49)`: `4.1% / 8.2%`
  - `xband (n=60)`: `1.7% / 1.7%`
  - `stereo (n=64)`: `0.0% / 0.0%`
  - `bright (n=22)`: `4.5% / 4.5%`
  - `EQ (n=169)`: `3.0% / 3.6%`

---

## 3. Kiến Trúc `V4-7` (`v4_models.py` & `v4_runtime.py`)

Toàn bộ suy luận **không dùng bất kỳ nhãn degradation hay thông tin `clean` nào** (`blind` 100%). Mã nguồn được cài đặt trong [v4_models.py](file:///d:/SonicMaster_Reproducing/scripts/extension/v4_models.py) và cấu hình suy luận trong [v4_runtime.py](file:///d:/SonicMaster_Reproducing/scripts/extension/v4_runtime.py):

### 3.1. `WindowedModCueExtractor` (Bất biến theo độ dài clip)
- Cửa sổ trượt cố định `win_frames = 172` (`2.0s` ở `44.1 kHz / hop = 512`), `stride = 86` (`1.0s`), dùng cửa sổ Hann `torch.hann_window(win_frames, periodic=True)` -> đồng nhất đặc trưng giữa crop huấn luyện (`4s`) và chunk suy luận (`8s`).
- Tính 6 đặc trưng vật lý per-band (`24` dải) từ `(z_x, z_s)`:
  1. `d_mod`: khoảng cách $L_2$ phổ điều chế `rfft` đã chuẩn hóa trên bins `1..16` (`0.5–8.0 Hz`).
  2. `delta_rho`: `rho_s - rho_x` với `rho` là tỷ số năng lượng bins `7..16` trên toàn dải `1..16`.
  3. `env_var_ratio`: `log((CV_s + 1e-4) / (CV_x + 1e-4))`.
  4. `log_band_ratio`: `clamp(log((E_s + 1e-6) / (E_x + 1e-6)), -4.0, 4.0)`.
  5. `phase_decorr`: `1 - cos_w`, bảo vệ điểm câm/bằng nhau `torch.where(nrm_w > 1e-9, dot_w / clamp(nrm_w, min=1e-9), ones)`.
  6. `env_corr_loss`: `1 - Pearson(ax_w, as_w)`, bảo vệ tương tự với `denom_corr > 1e-9`.

### 3.2. `BlindReverbHead` & Soft Blind Routing trong `MF7Decoder`
- **`BlindReverbHead`:** ghép `h.float().mean(dim=1)` (`D = 64`) với `cues.float()` (`6`) trên từng dải `(B, 24, 70)` -> `LayerNorm(70)` -> `Linear(70, 8)` + `GELU` -> flatten `192` -> `Linear(192, 64)` + `GELU` -> `Linear(64, 1)` (khởi tạo trọng số `0`, `bias_init = -4.0`, xác suất khởi đầu `p_rev ≈ 0.018` để bảo toàn hành vi `V4-6` lúc warm-start).
- **Noisy-OR Phase Gate:**
  $$g_{\text{ph,eff}} = 1 - (1 - g_{\text{ph}})\,(1 - p_{\text{rev}})$$
- **Per-band Magnitude Boost:** `gamma_bins = (0.90 * sigmoid(gamma_band_logits))[band_map]`, khởi tạo `0.0` cho 8 dải thấp `0..7` (`0.45`) và `1.2657` cho 16 dải trung/cao `8..23` (`~0.702`):
  $$g_{\text{m,eff}} = g_{\text{ms}} + p_{\text{rev}}\,\gamma_{\text{bins}}\,(1 - g_{\text{ms}})$$
- **Residual Suppression:**
  $$r_{\text{gate}} = 1 - p_{\text{rev}}, \qquad \hat{Z} = Z_{\text{fused}} + 0.5\,r_{\text{gate}}\,R_{\text{ms}}$$
- **Dual-point `bright` Expose:** `MF7Decoder` xuất `diag["p_hf_ratio"]` trực tiếp trên kênh Mid của nhánh lọc tuyến tính `f_w[:, 0]` trong dải `[6000.0, 22050.0) Hz` (`hf_band_mask`).

### 3.3. Quy Mô Tham Số (`count_parameters_breakdown`)

| Thành phần | Số tham số (`V4-7`) | Ghi chú |
|---|---|---|
| `encoder` | `553,824` | Giữ nguyên từ `V4-6` |
| `backbone` | `433,152` | Giữ nguyên từ `V4-6` |
| `decoder` | `979,373` | Gồm `reverb_head = 13,125` + `gamma_band_logits = 24` |
| **Tổng (`total`)** | **`1,966,349`** | Chỉ tăng **`+13,149`** (`+0.67%`) so với `V4-6` (`1,953,200`) |

---

## 4. Hàm Loss & Giao Thức Huấn Luyện (`v4_losses.py` & `v4_kaggle_train_pilot.py`)

Cài đặt trong [v4_losses.py](file:///d:/SonicMaster_Reproducing/scripts/extension/v4_losses.py) và [v4_kaggle_train_pilot.py](file:///d:/SonicMaster_Reproducing/scripts/extension/v4_kaggle_train_pilot.py):

1. **Warm-start an toàn từ `V4-6` (`--init-allow-new-head`):** nạp checkpoint `aajayagh/sonicmaster-v4-pilot-v4-6-run` (`v4_6/best.pt`) với `strict=False`, kiểm tra cứng `len(unexpected_keys) == 0` và toàn bộ `missing_keys` chỉ được phép nằm trong `decoder.reverb_head.*` và `decoder.gamma_band_logits`.
2. **Tắt GAN & Paired `L_FM` (`USE_GAN = False`, `--lambda-g 0.0`):** loại bỏ hoàn toàn áp lực ép pha theo cặp từ discriminator lên nhánh Reverb.
3. **Che loss phụ thuộc pha trên mẫu Reverb (`keep_phase = 1 - rev_target`):**
   - Trong lúc huấn luyện, `rev_target` lấy từ `degradation_types` với `REVERB_TAGS = {"small", "big", "mix", "real"}` (không dùng lúc suy luận).
   - `l_wav` (`w_wav = 0.10`) và `l_comp` (`w_comp = 0.15`) được nhân per-sample với `keep_phase` (`= 0` trên mẫu Reverb, `= 1` trên mẫu không Reverb).
   - Các loss bất biến pha (`MR-STFT = 1.0`, `Mel-KL = 2.0`, `ModSpec = 2.0`, `M/S ratio = 1.0`) chạy đồng đều trên mọi mẫu.
4. **Giám sát bất đối xứng cho `BlindReverbHead`:** `lambda_rev = 0.50` với weighted BCE (`rev_fp_weight = 2.0` cho mẫu không vang) nhằm khóa cứng `val_nonrev_fpr <= 0.04`.
5. **Kiểm soát `bright` hai điểm (Dual-Point `bright` Control):** `lambda_br = 0.60` trên `y_hat` + `lambda_br_p = 0.25` trực tiếp trên `diag["p_hf_ratio"]` theo chuẩn `SymmetricBandRatio6kLoss(scale=20.0)`.
6. **Lịch trình tối ưu (`12` epochs, `batch_size = 12`, `grad_accum = 1`, `stage2_epoch = 1`):** 2 nhóm learning rate AdamW (`lr = 1e-4` cho toàn mạng `V4-6`, `lr_head = 5e-4` cho `reverb_head` + `gamma_band_logits`), `CosineAnnealingLR` (`eta_min = 1e-5`); lựa chọn `best.pt` ưu tiên epoch có `val_nonrev_fpr <= 0.04`.

---

## 5. Cơ Sở Khoa Học Đã Xác Minh (Provenance)

- Melechovsky et al., *"SonicMaster: Towards Controllable All-in-One Music Restoration and Mastering"*, ICML 2026 (`arXiv:2508.03448`).
- Lu, Ai, Ling, *"Explicit Estimation of Magnitude and Phase Spectra in Parallel for High-Quality Speech Enhancement"* (`MP-SENet`), Interspeech 2023 (`arXiv:2308.08926`).
- Wang, Wichern, Le Roux, *"On the Compensation Between Magnitude and Phase in Speech Separation"*, MERL TR2021-137 / IEEE SPL 2021.
- Schröter et al., *"DeepFilterNet"*, IEEE ICASSP 2022 (`arXiv:2110.05588`).
- Luo & Yu, *"BSRNN"*, IEEE/ACM TASLP 2023 (`arXiv:2209.15174`).
- Wang et al., *"TF-GridNet"*, IEEE/ACM TASLP 2023 (`arXiv:2211.12433`).
- Nakatani et al., *"Speech Dereverberation Based on Variance-Normalized Delayed Linear Prediction"* (`WPE`), IEEE TASLP 2010.
- Tiền lệ blind routing nội bộ (zero-label E1/E2 modulation spectrum): `Hybrid-v3` ([blind_router.py](file:///d:/SonicMaster_Reproducing/scripts/extension/blind_router.py)).

---

## 6. Kiểm Chứng Cục Bộ (Phase 1) & Lệnh Khởi Chạy Phase 2 (Chờ Xác Nhận)

- **Unit tests ([test_v4_step0.py](file:///d:/SonicMaster_Reproducing/scripts/extension/tests/test_v4_step0.py)):**
  - `d:\SonicMaster_Reproducing\tools\extension-env\Scripts\python.exe -m pytest d:\SonicMaster_Reproducing\scripts\extension\tests\test_v4_step0.py -v` -> **`19 passed in 66.25s`** (`0` failures; gồm cả `test_v4_7_blind_reverb_routing_and_dual_bright`).
- **Đóng gói Kaggle & kiểm tra cú pháp ([train_pilot.py](file:///d:/SonicMaster_Reproducing/artifacts/10_extension/v4/kaggle_kernel_pilot_train/train_pilot.py)):**
  - `--dry-run` -> `DRY_RUN`, `python -m py_compile` -> exit code `0`, cờ nhúng xác nhận `V4_7_SCRIPT_VERIFIED` (`USE_INIT_CKPT = True`, `USE_GAN = False`, `INIT_ALLOW_NEW_HEAD = True`).
- **Lệnh Phase 2 Kaggle (chưa push — chỉ chạy sau khi người dùng xác nhận):**

```powershell
python scripts/extension/v4_kaggle_train_pilot.py --account-line 0 --ds-slug aajayagh/sonicmaster-v4-pilot-5s --init-kernel-slug aajayagh/sonicmaster-v4-pilot-v4-6-run --init-allow-new-head --k-slug sonicmaster-v4-pilot-v4-7-run --variants v4_7 --epochs 12 --stage2-epoch 1 --batch-size 12 --grad-accum 1 --lr 1e-4 --lr-head 5e-4 --gate-bias-init -1.5 --lambda-ms 1.0 --lambda-g 0.0 --lambda-kl 2.0 --w-comp 0.15 --w-wav 0.10 --lambda-mod 2.0 --lambda-br 0.60 --lambda-br-p 0.25 --lambda-rev 0.50 --rev-fp-weight 2.0
```
