# Hướng Dẫn Dễ Hiểu Về Kiến Trúc Hybrid-v4 (`V4-0` → `V4-4`)

Tài liệu này giải thích toàn bộ kiến trúc **Hybrid-v4** theo cách trực quan nhất (không nặng công thức), từ bài toán gốc cho tới từng khối bên trong mạng.

![Sơ đồ kiến trúc Hybrid-v4](file:///d:/SonicMaster_Reproducing/artifacts/10_extension/figures/hybrid_v4_architecture.jpg)

---

## 1. Tại sao phải sinh ra `Hybrid-v4`? (Ý tưởng cốt lõi trong 30 giây)

Khi nghe lại đầu ra của **SonicMaster (`A0`, gọi là `s`)**:
- **Điểm mạnh của `s`:** SonicMaster hiểu được câu lệnh ngôn ngữ tự nhiên (*"làm sáng hơn"*, *"khử vang"*, *"bớt ù"*), biết cách **khử reverb**, **sửa clipping**, **phục hồi dải tần bị mất (`xband`)**.
- **Điểm yếu chí mạng của `s`:** SonicMaster là mô hình khuếch tán chạy qua **VAE của Stable Audio Open 1.0**. Mỗi lần âm thanh đi qua bộ nén/giải nén VAE, chi tiết pha và kết cấu sóng gốc bị biến dạng (làm **`SSIM` rớt từ ~0.83 xuống ~0.58–0.61**, chi tiết nhạc cụ nghe hơi "nhựa"/mờ).

Ngược lại, **nhạc gốc bị lỗi (`x`)** tuy bị ù/tối/vang nhưng **vẫn giữ nguyên 100% chi tiết sắc nét của sóng gốc** (không bị VAE làm mờ).

> **Ý tưởng vàng của `Hybrid-v4` (Codec-Bypass):**
> Đừng dùng trực tiếp âm thanh `s` bị mờ từ VAE! Hãy xem **`s` như một "bản nháp/kim chỉ nam"** để một mạng neural siêu nhẹ (~1.7M tham số) **nhìn `s` và học cách chỉnh trực tiếp trên `x`** — chỗ nào `x` tự sửa được bằng lọc tuyến tính thì sửa thẳng trên `x` (không qua VAE), chỗ nào `x` mất hẳn thông tin (như bị clipping nát bét, mất cả kênh Stereo Side, hoặc reverb quá nặng) mới mượn/tổng hợp từ `s`.

---

## 2. Đi từ đầu vào đến đầu ra: 5 bước trong `Hybrid-v4`

### Bước 1: Chuẩn hóa theo bài & Căn thời gian (`Track Norm + Lag Align`)
- Trước khi đưa `x` và `s` vào mạng, ta **căn lại độ trễ mẫu** giữa `s` và `x` (`estimate_lag_samples`) để 2 sóng khớp nhau từng mẫu âm thanh (tránh lệch pha).
- Đưa `x` và `s` về cùng thang đo RMS của bài (`compute_track_norm`), đồng thời giữ lại **3 con số thống kê toàn bài** (`raw_stats`: độ to của `x`, độ to của `s`, và độ chênh dB giữa chúng) để mạng không bị "mù âm lượng".

### Bước 2: Đổi sang Mid/Side STFT & Chia 24 Dải Tai Người (`PsychoacousticBandEncoder`)
1. **Đổi `L/R` → `Mid/Side` (`lr_to_ms`):**
   - `Mid = (L + R) / √2`: phần âm thanh nằm giữa (giọng hát, kick, bass).
   - `Side = (L - R) / √2`: phần tạo **độ rộng Stereo** (nhạc cụ trái/phải, không gian).
   - Tách `Mid/Side` giúp mạng kiểm soát độ rộng stereo trực tiếp mà không làm hỏng phần giữa.
2. **Biến đổi STFT (`n_fft=2048, hop=512`):** Cắt âm thanh thành `1,025` tần số theo từng khung thời gian (~11.6 ms/khung).
3. **Nén biên độ `|Z|^0.3` cho Encoder:** Giúp mạng nhìn rõ các tần số cao vốn có năng lượng rất nhỏ so với bass. *(Lưu ý: chỉ nén cho Encoder "nhìn", còn phần lọc tín hiệu ở Bước 4 vẫn chạy trên STFT tuyến tính nguyên gốc!)*
4. **Chia 24 dải tần không đều (kiểu BSRNN / DeepFilterNet):**
   - Tai người nghe rất tinh ở dải trầm/trung (20 Hz – 4 kHz) và thô hơn ở dải rất cao (10 – 22 kHz).
   - Vì vậy `1,025` tần số được gom thành **24 dải** (dải thấp chỉ vài bin, dải cao gom nhiều bin), chiếu thành vector `64` chiều cho mỗi dải tại mỗi khung thời gian: `(B, T, 24, 64)`.

### Bước 3: Bộ Não Xử Lý 2 Trục (`4 × DualAxisBlock`)
Mạng dùng 4 block giống **BSRNN / TF-GridNet**, mỗi block nhìn âm thanh theo **2 chiều**:
1. **Chiều Tần số (`Cross-Band Self-Attention`, 4 heads):** Ở cùng 1 thời điểm, 24 dải tần nói chuyện với nhau (*"dải bass đang ù thì dải treble cần xử lý thế nào"*).
2. **Chiều Thời gian (`Temporal BiGRU`):** Ở cùng 1 dải tần, nhìn xuyên suốt quá khứ và tương lai (*"đây là tiếng trống gõ ngắn hay là đuôi reverb kéo dài?"*).
3. **Dense FFN:** Tổng hợp phi tuyến để xuất ra lệnh điều khiển cho Decoder.

### Bước 4: Bộ Giải Mã Lọc Phức Đa Khung & Phần Dư (`Decoder` — Trái Tim Của `V4`)
Đây là nơi các phiên bản `V4-0` → `V4-4` khác nhau:

- **`V4-0` (`NoGuide`):** Không nhìn `s`, chỉ nhìn `x` và tự đoán phổ sạch. *(Thất bại: `SSIM=0.7495, KL=5.17`, chứng minh bắt buộc phải có `s` dẫn đường).*
- **`V4-1` (`DirectMap`):** Nhìn cả `(x, s)` rồi tự vẽ lại toàn bộ phổ sạch từ con số 0. *(Khá hơn nhưng vẫn bị "mờ" do tự vẽ lại toàn bộ: `10/12` lỗi Attribute Guard).*
- **`V4-2` (`MF7` — Identity-Centered 2×2 M/S Multi-Frame Filtering):**
  - Thay vì vẽ lại từ đầu, mạng **lấy thẳng `Z_x` làm gốc (`Identity-Centered`)**, rồi chỉ học **độ chỉnh nhỏ `ΔW`** xung quanh 0:
    $$F_W = Z_x + \Delta W_0(Z_x) + \sum_{k \in \text{taps}} \Delta W_k(Z_{x, t+k})$$
  - Mỗi $\Delta W$ là **ma trận phức 2×2 trên `(Mid, Side)`**, tức là vừa chỉnh được **biên độ + pha** (EQ, nén), vừa trộn chéo được `Mid ↔ Side` (stereo).
  - Cộng thêm **Gate $g \in (0, 1)$**:
    $$\hat{Z} = (1 - g) \odot F_W + g \odot Z_s$$
    *(Khi $g \approx 0$: giữ nguyên sóng sắc nét từ $x$; khi $g$ mở: mượn $Z_s$ từ SonicMaster để vá chỗ hỏng nặng).*
  - *Kết quả `V4-2`:* `SSIM` nhảy vọt lên **`0.8325`**, `FAD` đánh bại `A0` (`0.1848` vs `0.2083`), nhưng rớt `6/12` Guard do: gate bị phạt `λ_g` nên đóng quá chặt (`g ≈ 0.1`), không có `Mel-KL` loss, và lệch `volume`.
- **`V4-3` (`Dilated MF9 + Free Gate + Mel-KL + RMS Match`):**
  - Mở rộng tầm nhìn đa khung từ `7` tap (~70 ms) thành **9 tap giãn cách `{-18, -12, -8, -5, -3, -2, -1, 0, +1}` (~220 ms)** kiểu WPE để bắt đuôi reverb dài.
  - **Free Gate** (`gate_bias=-1.5`, `λ_g=0.0`): bỏ phạt ép đóng gate.
  - Thêm **`MelKLLoss`** (`λ_kl=2.0`) và tăng **`λ_ms=0.5`**: sửa cân bằng phổ.
  - **Output RMS Match:** ép RMS đầu ra khớp `rms(s)` để triệt tiêu lỗi `volume`.
  - *Kết quả `V4-3`:* Giảm từ `6/12` lỗi xuống còn **`2/12` lỗi** (`boom`, `mic`, `vocal`, `volume` đều PASS; `KL` giảm `-14.1%` còn `1.1135`; `FAD` còn `0.1775`). Chỉ còn 2 lỗi: **`stereo` (`+0.177`)** và **`bright` (`+0.194`)**.
- **`V4-4` (`V4-3` + Band-Scaled Additive Complex Residual — Đang train):**
  - Thêm nhánh **cộng trực tiếp phần dư phức** $R_\theta(Z_x, Z_s)$:
    $$\hat{Z} = (1 - g) \odot F_W + g \odot Z_s + R_\theta(Z_x, Z_s)$$
  - Tại sao cần $R_\theta$? Khi `x` bị **sụp kênh Side** (`S_x ≈ 0`, tức lỗi `stereo`) hoặc **mất sạch dải cao** (`bright`), phép nhân $\Delta W \cdot Z_x$ không thể tự "đẻ" ra thông tin từ số 0, còn $g \cdot Z_s$ chỉ nhân vô hướng thực nên không sửa được pha của $Z_s$. Nhánh $R_\theta$ (được **scale theo `band_rms` của từng dải** và **zero-init** lúc đầu) có thể tổng hợp trực tiếp các thành phần phức còn thiếu cho kênh Side và dải cao >8 kHz.

### Bước 5: Tái Tạo Sóng Âm (`iSTFT + M/S → L/R + RMS Match`)
- Ghép các khung STFT tuyến tính $\hat{Z}$ lại thành sóng `(Mid, Side)` bằng `iSTFT`, đổi ngược về `(Left, Right)`, rồi khớp RMS đầu ra với `rms(s)` để đảm bảo độ to chuẩn xác.

---

## 3. Tóm tắt 1 câu cho từng phiên bản

| Phiên bản | Bản chất kiến trúc | Điểm mạnh / Kết quả thực tế (Dev 280) |
|---|---|---|
| **`V4-0`** | Không có `s`, tự đoán từ `x` | `SSIM=0.7495, KL=5.17` — Chứng minh cần teacher `s`. |
| **`V4-1`** | Có `(x, s)`, tự vẽ lại toàn bộ STFT | `SSIM=0.8023, KL=2.62, FAD=0.251`, rớt `10/12` Guard. |
| **`V4-2`** | Lọc `ΔW` 7-frame quanh `Z_x` + Gate bị ép đóng | `SSIM=0.8325, KL=1.30, FAD=0.185`, rớt `6/12` Guard. |
| **`V4-3`** | Dilated 9-frame (~220 ms) + Free Gate + `Mel-KL` + RMS Match | `SSIM=0.8236, KL=1.11, FAD=0.178`, chỉ còn **`2/12` Guard** (`bright`, `stereo`). |
| **`V4-4`** | `V4-3` + **Additive Complex Residual** $R_\theta$ + `λ_ms=1.0` | **Đang train trên Kaggle T4** — mục tiêu dứt điểm `stereo` & `bright`. |
