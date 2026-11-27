# Thiết Kế Kỹ Thuật `Hybrid-v4.5p` (`v4_5p`: Decoupled Magnitude–Phase Fusion)

**Mục tiêu:** Phá vỡ "rào cản triệt tiêu pha" `-24%` trên dải `1–22 kHz` giữa nhánh lọc `P = F_W(Z_x)` và teacher SonicMaster `S = Z_s` — giải quyết đồng thời nút thắt **`Reverb PQ`** (`ΔPQ = -0.71` ở `V4-4`) và guard **`bright`** (`6–22 kHz`), mà vẫn giữ nguyên đường tắt `Z_x` bảo toàn độ sắc nét (`SSIM ~0.81–0.83`).

---

## 1. Động Lực Thực Nghiệm (Từ `Reverb Branch Audit`, `N = 106`)

Phép kiểm toán nhánh [reverb_branch_audit_table.md](file:///d:/SonicMaster_Reproducing/artifacts/10_extension/v4/reverb_branch_audit/reverb_branch_audit_table.md) và đo đạc pha đã chứng minh:
1. Trên `Dev Reverb (N=106)`, nhánh lọc đơn lẻ **`P` không tự khử được vang** (`Official MSD = 0.5695`, tệ hơn `Input = 0.5309`; `PQ = 6.3464` vs `6.3466`), gần như toàn bộ cải thiện `MSD` (`0.5695 → 0.3985`) và `PQ` (`+0.5519`) đến từ `Pg` mở gate lấy `S`.
2. Tại `0–1 kHz` (nơi $\cos(P, S) = +0.365$), `mean(g)` mở lên `0.342`. Nhưng suốt dải **`1–22 kHz`**, VAE làm pha của `S` mất tương quan với `P` ($\cos(P, S) \approx 0.001$), khiến phép trộn phức cũ $(1 - g)P + gS$ bị **triệt tiêu ~24% biên độ** ở `g = 0.5` (`|0.5P+0.5S| / (0.5|P|+0.5|S|) = 0.763`).
3. Hàng rào năng lượng này buộc optimizer đóng `mean(g)` xuống **`0.038–0.050`** ở `1–12 kHz`, giữ ~95% dải trung–cao ở nhánh `P` (gây hụt `Reverb PQ = 6.88` vs `A0 = 7.59`, và chặn `S` tiếp ứng năng lượng dải cao cho guard `bright`).

---

## 2. Cơ Sở Bài Báo (Paper Provenance)

- **Tách xử lý Biên độ & Pha (`Decoupled Magnitude–Phase`):** Y.-X. Lu, Y. Ai, Z.-H. Ling, *"Explicit Estimation of Magnitude and Phase Spectra in Parallel for High-Quality Speech Enhancement"* (**MP-SENet**, Interspeech 2023 / [arXiv:2308.08926](https://arxiv.org/abs/2308.08926)). Thay vì trộn phức trực tiếp khi pha lệch nhau, biên độ và pha phải được điều khiển qua 2 nhánh tách biệt.
- **Giảm bù méo biên độ–pha từ `Wav/Complex L1`:** Z.-Q. Wang, G. Wichern, J. Le Roux, *"On the Compensation Between Magnitude and Phase in Speech Separation"* ([MERL TR2021-137](https://www.merl.com/publications/TR2021-137) / IEEE SPL 2021).
- **Bảo toàn đường tắt cho nhạc & metric chuẩn:** *SonicMaster* ([arXiv:2508.03448](https://arxiv.org/abs/2508.03448)), *NPU-ASLP* ([arXiv:2610.15464](https://arxiv.org/abs/2610.15464)), *MusicFilterNet-MS* ([arXiv:2607.12872](https://arxiv.org/abs/2607.12872)), và *Early Reflections & Phase Recovery* (Luo et al., JASA 2024, `PMID: 38240664`).

---

## 3. Kiến Trúc & Công Thức Toán Học Của `V4-5p` ([v4_models.py](file:///d:/SonicMaster_Reproducing/scripts/extension/v4_models.py))

Giữ nguyên toàn bộ **PsychoacousticBandEncoder**, **4×DualAxisBlock**, **9-tap Dilated Multi-Frame `MF9`**, và chuẩn hóa **Output RMS Match**; chỉ nâng cấp phép ghép tại **Decoder (`decoupled_fusion=True`)**:

1. **Tách biên độ và pha đơn vị của 2 nguồn $P = F_W(Z_x)$ và $S = Z_s$:**
   $$|P| = \sqrt{\text{Re}(P)^2 + \text{Im}(P)^2 + 10^{-12}}, \quad u_P = \frac{P}{|P|}, \qquad |S| = \sqrt{\text{Re}(S)^2 + \text{Im}(S)^2 + 10^{-12}}, \quad u_S = \frac{S}{|S|}$$

2. **Trộn biên độ không triệt tiêu pha (`Magnitude Gate` $g_m \in (0,1)$ theo từng bin & kênh `M/S`):**
   $$M_{\text{fused}} = (1 - g_m) |P| + g_m |S|$$
   - Khi pha lệch $90^\circ$ ($\cos \Delta\phi = 0$), $|M_{\text{fused}}|$ vẫn đúng bằng $(1-g_m)|P| + g_m|S|$, **triệt tiêu hoàn toàn hàng rào `-24%`**!

3. **Cổng chọn pha mượt theo 24 dải tần & thời gian (`Phase Gate` $g_{\phi} \in (0,1)$):**
   - Nhánh mới `self.phase_gate_head = nn.Linear(64, 2)` (zero-init, chỉ thêm **130 tham số**) dự đoán logit trên 24 dải psychoacoustic, được làm mượt qua `avg_pool1d(kernel_size <= 9)` theo trục thời gian để tránh giật pha giữa các khung liên tiếp:
     $$u_{\text{mix}} = (1 - g_\phi) u_P + g_\phi u_S, \qquad u_{\text{out}} = \frac{u_{\text{mix}}}{\sqrt{|u_{\text{mix}}|^2 + 10^{-6}}}$$
   - Tín hiệu nền tái tạo: $Z_{\text{base}} = M_{\text{fused}} \cdot u_{\text{out}}$.

4. **Phần dư cộng $R_\theta$ có điều tiết (`residual_scale = 0.5`, nâng cấp `band_rms`):**
   - Dùng `band_rms = sqrt(max(mean(|Z_x|^2), mean(|Z_s|^2)) + 1e-6)` thay vì `mean(|Z_x|^2 + |Z_s|^2)` để các bài bị cắt dải cao (`bright`, $|Z_x| \approx 0$) không bị bóp hệ số khuếch đại của $R$, đồng thời giới hạn `residual_scale = 0.5` theo kết quả `residual_sweep_dev` (`r05` cho `PQ` và `FAD` tốt hơn `r1`).
   $$\hat{Z} = M_{\text{fused}} \cdot u_{\text{out}} + 0.5 \, R_\theta(Z_x, Z_s)$$

---

## 4. Thay Đổi Hàm Mất Mát ([v4_losses.py](file:///d:/SonicMaster_Reproducing/scripts/extension/v4_losses.py))

1. **Thêm `ModSpecLoss` (`--lambda-mod 2.0`):** Phiên bản khả vi PyTorch của **Official Table 2 Modulation Spectrum Distance** (`n_fft=1024, hop=512, n_mod_bins=20`), tính trung bình trên **cả toàn dải (`0–22 kHz`) và dải trung–cao (`1–12 kHz`, bins `24..280`)** — ép trực tiếp bộ lọc và gate khử đuôi điều chế vang ở `1–12 kHz`.
2. **Giảm nhạy pha (theo Wang et al. 2021):** Hạ `--w-comp` từ `0.5 → 0.25` và `--w-wav` từ `0.5 → 0.15` để tránh phạt `S` vì lệch pha vi mô với `Clean`.
3. **Giữ `l_ms` từ Epoch 1 (`--stage2-epoch 1 --lambda-ms 1.0`) + `--lambda-kl 2.0` + Free Gate (`--lambda-g 0.0`, `--gate-bias-init -1.5`).**

---

## 5. Kiểm Chứng Cục Bộ (`Unit Tests` & Tham Số)

- **Số tham số `v4_5p`** (`count_parameters_breakdown`):
  - `encoder = 553,824` | `backbone = 433,152` | `decoder = 966,224` | **Tổng = `1,953,200`** *(chỉ tăng `+130` tham số so với `V4-4` `1,953,070`)*.
- **Unit tests (`tools\extension-env\Scripts\python.exe -m pytest scripts/extension/tests/test_v4_step0.py -v`):**
  - **`17 passed in 41.24s`** (`0` failures), bao gồm test mới `test_v4_5p_decoupled_fusion_and_modspec_loss` đã xác minh: không hủy biên độ khi pha vuông góc $90^\circ$, khớp hoàn hảo `force_identity=True` và `x == s`, hữu hạn với tín hiệu im lặng, khớp RMS theo `rms(s)`, và gradient chảy đầy đủ qua `phase_gate_head` + `ModSpecLoss`.

---

## 6. Lệnh Chạy Pilot Trên Kaggle T4 (Chưa Chạy — Chờ Xác Nhận)

```powershell
python scripts/extension/v4_kaggle_train_pilot.py --account-line 0 --ds-slug aajayagh/sonicmaster-v4-pilot-5s --k-slug sonicmaster-v4-pilot-v4-5p-run --variants v4_5p --epochs 20 --stage2-epoch 1 --batch-size 12 --gate-bias-init -1.5 --lambda-ms 1.0 --lambda-g 0.0 --lambda-kl 2.0 --w-comp 0.25 --w-wav 0.15 --lambda-mod 2.0
```
*(Nếu `aajayagh` đang queue, kiểm tra/đổi `--account-line 4` `maitvng` hoặc `--account-line 5` `nthkhanh`.)*
