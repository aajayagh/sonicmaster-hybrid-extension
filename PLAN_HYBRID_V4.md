# PLAN_HYBRID_V4.md — Kế Hoạch Chi Tiết Hybrid-v4 (Rev 2.2-Final)

**Hướng nghiên cứu:** *Teacher-Guided Codec-Bypass Unified Music Restoration*  
**Ngày lập kế hoạch:** 2026-11-22  
**Nguyên tắc bất biến:** **Tuyệt đối không chỉnh sửa** bất kỳ mã nguồn, trọng số hay cấu hình đã khóa của `Hybrid-v3` ([`lock_v3.json`](file:///d:/SonicMaster_Reproducing/artifacts/10_extension/lock_v3.json), `biquad_eq.py`, `xband_crossover.py`, `blind_router.py`, `run_hybrid_v3_confirm.py`). Toàn bộ mã nguồn mới nằm trong thư mục/file riêng `scripts/extension/v4_*.py` và artifacts riêng `artifacts/10_extension/v4/`.

---

## 1. Mục Tiêu, Câu Hỏi Nghiên Cứu & Bằng Chứng Thực Nghiệm Nền Tảng

### 1.1. Câu hỏi nghiên cứu cốt lõi
> *Liệu một mạng neural nhẹ được hướng dẫn bởi SonicMaster `(x, s) -> y` có thể thay thế hoàn toàn hệ thống định tuyến thủ công theo nhãn (`Hybrid-v3`) và xử lý thống nhất cả suy hao tuyến tính, phi tuyến, thời gian (`reverb`) và hỗn hợp (`Double+Triple`) — vừa bảo toàn độ trung thực tín hiệu gốc (codec-bypass), vừa giữ vững hoặc cải thiện chất lượng cảm nhận (`Audiobox PQ`)?*

### 1.2. Các mốc Baseline đã đo & khóa ([`HYBRID_V3_REPORT.md`](file:///d:/SonicMaster_Reproducing/HYBRID_V3_REPORT.md))

| Metric | A0 (SonicMaster) | Hybrid-v3 Label-Routed | Hybrid-v3-Blind (E1-tight) | Mục tiêu Hybrid-v4 |
|---|---:|---:|---:|---|
| **Confirm SSIM (n=6,720)** ↑ | 0.6092 | **0.7192** | 0.7118 | `>= 0.720` (vượt V3-Blind, tiệm cận/vượt V3-Label) |
| **Confirm KL×1000 (n=6,720)** ↓ | 0.8839 | 0.8902 | **0.8692** | `<= 0.900` (`<= 1.02× A0`) |
| **Confirm FAD-CLAP (n=5,306)** ↓ | 0.0756 | **0.0398** | 0.0487 | `<= 0.045` (`<= 0.60× A0`) |
| **Confirm Audiobox PQ (n=5,306)** ↑ | 7.7103 | **7.7132** (`Δ=+0.0029`) | 7.6093 (`Δ=-0.1010`) | **`CI95_lo(ΔPQ) >= -0.05`** (khắc phục lỗi `-0.101` của V3-Blind) |
| **Attribute Guards (19/19)** | — | **0 vi phạm** | **0 vi phạm** | **0 vi phạm** |

### 1.3. Ba bằng chứng thực nghiệm trên Dev (`280` mẫu) định hình kiến trúc V4
1. **Ngưỡng tay không tách nổi `linear` vs `reverb` (`exp_v35_dev.py`):** `flat_top_ratio < 1e-4` diệt 100% mẫu `clip` lọt (`17 → 0`), nhưng siết ngưỡng tay chặn `reverb` (`1–3` mẫu lọt) làm cắt nhầm nửa số mẫu `linear`, khiến `Dev SSIM` sập từ `0.7282` xuống `0.6335–0.6678` → **bắt buộc dùng mô hình học**.
2. **Dereverb thủ công giữ pha gốc thất bại nặng về `PQ` (`exp_dereverb_dev.py`, 106 mẫu Dev reverb):** `WPE` (`ΔPQ = -1.3647`), `Guided_Wiener` (`ΔPQ = -0.9151`), `WPE_Guided` (`ΔPQ = -1.3378`) → **xử lý `reverb` trên âm nhạc bắt buộc phải hiệu chỉnh pha phức đa khung theo thời gian**.
3. **Dư địa rất lớn trên `clip` (`run_dev_td_declip.py`, 36 mẫu Dev `clip` không reverb):** `V3 + TD_declip` đẩy `Dev SSIM` lên **`0.7606`** (`+0.0208` vs V3), `KL = 1.0641`, `FAD = 0.1422` (`0.690× A0` local, `-26% FAD` trên 36 mẫu clip), `0/15` vi phạm guards (`clip det = -0.010`, `punch det = -0.159`), nhưng trên `Double+Triple`/clip sâu bị kéo `CI95_lo(ΔPQ) = -0.0569` → cần cơ chế học chọn lọc giữa bảo toàn `x` và tái tạo từ `s`.

---

## 2. Kiến Trúc Chi Tiết `Hybrid-v4 (Rev 2.2-Final)` cho `V4-0`, `V4-1`, `V4-2`

> **Nguyên tắc kỷ luật Pilot:** Chỉ triển khai và huấn luyện **3 cấu hình nền tảng (`V4-0`, `V4-1`, `V4-2`)** dùng **Dense FFN** trong Pilot đầu tiên. Chưa bật `TF-MoE`, chưa bật `Delayed/Dilated MF`, chưa bật `Additive Residual`, và chưa bật `Perceptual Fine-Tuning` cho tới khi có số liệu ablation từ 3 bản nền tảng này.

### 2.1. Tiền xử lý, Căn pha & Chuẩn hóa toàn Track (Khắc phục lỗi `volume` & chunk boundary)
1. **Đầu vào:** Stereo `44.1 kHz` (`x`, `s`).
2. **Căn lag nguyên (chỉ cho `V4-1`, `V4-2`, không dùng `clean`):** Dùng `estimate_lag_samples(x, s, max_lag=512)` (từ [`branch_r.py`](file:///d:/SonicMaster_Reproducing/scripts/extension/branch_r.py)) để căn `s` theo `x`, triệt tiêu độ lệch pha tuyến tính từ VAE.
3. **Chuẩn hóa toàn Track 1 lần duy nhất trước khi cắt chunk** *(tránh loudness jump ở biên chunk và giữ trọn khả năng khôi phục `volume`)*:
   - Với **`V4-1` & `V4-2`** (có `s`):
     ```python
     rms_x = sqrt(mean(x**2)) + eps
     rms_s = sqrt(mean(s**2)) + eps
     scale = max(rms_s, eps)
     g_align = clip(rms_s / rms_x, 0.1, 16.0)   # kẹp ±24 dB tránh khuếch đại quantization noise
     x_in = (x * g_align) / scale               # unit-RMS (trừ ca >24 dB bị kẹp an toàn)
     s_in = s / scale                           # unit-RMS
     # Training target: c_in = c / scale
     # Output waveform: y_hat = model(x_in, s_in, raw_stats) * scale
     ```
     - **Kiểm chứng tính nhất quán:** Khi `rms_x ≈ rms_s` (`g_align ≈ 1`) và `ΔW = 0, g = 0` → `y_hat = x_in * scale = x` (`max_abs_err < 1e-5`). Khi `x` bị nhỏ `-15 dB` (`volume`), `g_align` nâng tuyến tính sạch lên `1.0` mà không ép `MF7` đẩy `g → 1`. Khi `x` bị nhỏ cực đoan `-35 dB`, `g_align` dừng ở `16.0` và `raw_stats` báo `gain_db = +35 dB` cho Encoder để mở cổng `g` sang `s`.
   - Với **`V4-0` (`NoGuide`, chỉ có `x`):** `scale = max(rms_x, eps)`, `g_align = 1.0`, `x_in = x / scale`, `y_hat = model(x_in, raw_stats) * scale`.
4. **Chunking & Overlap-Add (OLA):**
   - **Train:** Cắt ngẫu nhiên chunk **`4.0 giây`** (`176,400` mẫu stereo).
   - **Inference (30 giây):** Hỗ trợ cả **`4s / hop 2s`** và **`8s / hop 4s`** (benchmark cả 2 ở Step 0), dùng cửa sổ Hann và **chuẩn hóa theo tổng trọng số cửa sổ `sum_w`** (xử lý chuẩn 2 biên đầu/cuối track để `sum_w > 0` mọi mẫu).
5. **STFT & Mid/Side:**
   - `n_fft = 2048`, `hop = 512` (benchmark thêm `hop = 1024` ở Step 0), cửa sổ Hann, `center = True` → `F = 1,025` bin.
   - Chuyển `(L, R) ↔ (Mid, Side)`: $M = (L+R)/\sqrt{2},\ S = (L-R)/\sqrt{2}$.
   - **Tách miền nén và miền tuyến tính (BẮT BUỘC):**
     - **Encoder features:** Dùng phổ phức nén $Z^c = (|Z| + \epsilon)^{0.3} \cdot \frac{Z}{|Z| + \epsilon}$.
     - **Decoder & iSTFT của `V4-2 (MF7)`:** Thực hiện hoàn toàn trên **Linear Complex STFT chưa nén** $Z_x, Z_s$ rồi vào thẳng iSTFT.
     - **Decoder của `V4-0` & `V4-1 (DirectMap)`:** Dự đoán $\hat{Z}^c$ trong miền nén rồi **giải nén tường minh** $\hat{Z} = (|\hat{Z}^c| + \epsilon)^{1/0.3} \cdot \frac{\hat{Z}^c}{|\hat{Z}^c| + \epsilon}$ trước khi vào iSTFT (không bao giờ đưa phổ nén thẳng vào iSTFT).

---

### 2.2. Khối 1 — 24-Band Psychoacoustic Scale-Aware Encoder (`D = 64`)
Chia `1,025` bin (`0 – 22,050 Hz`, `Δf ≈ 21.53 Hz/bin`) thành **24 dải không chồng lấn**:
- **8 dải trầm (`0 – 1,012 Hz`, 47 bin):** độ rộng `[6, 6, 6, 6, 6, 6, 6, 5]` → độ phân giải ~110–130 Hz cho `boom` (20–120 Hz) và `mud` (200–500 Hz).
- **8 dải trung (`1,012 – 4,005 Hz`, 139 bin):** độ rộng `[17, 17, 17, 17, 17, 18, 18, 18]` → cho `warm`, `vocal`, `clarity`.
- **5 dải cao (`4,005 – 12,015 Hz`, 372 bin):** độ rộng `[74, 74, 74, 75, 75]` → cho `bright`, `airy`, `mic`.
- **3 dải siêu cao (`12,015 – 22,050 Hz`, 467 bin):** độ rộng `[155, 156, 156]` → cho `xband`, air.
- *(Tổng kiểm tra: `47 + 139 + 372 + 467 = 1,025` bin chính xác!)*

**Đặc trưng đầu vào từng dải $b$ (bề rộng $w_b$ bin) tại khung $t$:**
- Phổ phức nén đã chuẩn hóa theo RMS dải:
  - `V4-0`: `4·w_b` giá trị (`M_re, M_im, S_re, S_im` của `x_in`) + `4` scale features (`log_rms_x_b`, `log_ms_ratio_x_b`, `log_rms_x_raw`, `0.0`).
  - `V4-1 / V4-2`: `8·w_b` giá trị (`x_in` + `s_in`) + `7` Dual Scale-Aware features:
    $$\mathbf{e}_b(t) = \Big[\log r_{x,b},\ \log r_{s,b},\ \log\frac{r_{s,b}}{r_{x,b}},\ \Delta\log\frac{r^{\text{Side}}_b}{r^{\text{Mid}}_b},\ \log r^{\text{raw}}_x,\ \log r^{\text{raw}}_s,\ \text{clip}\!\left(\frac{\text{gain\_dB}}{24}, -2, 2\right)\Big]$$
- Chiếu qua lớp **band-specific** `LayerNorm -> Linear(C_in_b, 64) -> SiLU` → tensor `(Batch, T, B=24, D=64)`.

---

### 2.3. Khối 2 — Shared Dual-Axis Backbone (`N = 4` blocks, Dense FFN)
Dùng chung 100% giữa `V4-0`, `V4-1`, `V4-2` để so sánh Decoder công bằng:
1. **Cross-Band Multi-Head Self-Attention** (`4` heads, `D = 64`, Pre-LN, residual) dọc theo trục `B = 24`.
2. **Temporal BiGRU** (`input=64, hidden=64` 2 chiều → `Linear(128 → 64)`, Pre-LN, residual) dọc theo trục `T`.
3. **Dense FFN** (`Pre-LN → Linear(64 → 256) → GELU → Linear(256 → 64)`, residual).
- *(Chưa bật `TF-MoE` ở bước này; `TF-MoE` được tách riêng sang ablation `V4-5`.)*

---

### 2.4. Khối 3 — Decoder riêng cho `V4-0`, `V4-1` và `V4-2`

| Model | Kiến trúc Decoder | Khởi tạo & Ràng buộc | Tham số mục tiêu |
|---|---|---|---:|
| **`V4-0` (`NoGuide`)** | Shared MLP (`64→128→64`) → Band-specific `Linear(64 → 4·w_b)` dự đoán $\hat{Z}^c$ (M/S thực/ảo) → giải nén $\|\hat{Z}^c\|^{1/0.3}$ → iSTFT | Khởi tạo chuẩn (hoặc residual trên $Z^c_x$: $\hat{Z}^c = Z^c_x + \Delta Z^c$ với zero-init) | `~1.1–1.3M` |
| **`V4-1` (`DirectMap`)** | Giống hệt `V4-0`, nhận đặc trưng từ `(x_in, s_in)` → dự đoán $\hat{Z}^c = Z^c_x + \Delta Z^c$ → giải nén $\|\hat{Z}^c\|^{1/0.3}$ → iSTFT | Zero-init lớp cuối $\Delta Z^c \implies \hat{y} \approx x$ tại init | `~1.4–1.7M` |
| **`V4-2` (`MF7`)** | Shared MLP (`64→128→64`):<br>1. **Per-bin (`1,025` bin):** Band-specific `Linear(64 → 10·w_b)` xuất $\Delta W_0 \in \mathbb{C}^{2\times 2}$ (`8·w_b`) và gate logits `g_M, g_S` (`2·w_b`).<br>2. **Sub-band Multi-Frame (`96` sub-band mịn, chia 4 sub-band trong mỗi band của 24 band):** `Linear(64 → 4 × 6 × 8)` xuất 6 taps $k \in \{-5,-4,-3,-2,-1,+1\}$ ma trận `2×2` M/S, map lên 1,025 bin.<br>3. Áp trực tiếp lên **Linear STFT chưa nén** $Z_x, Z_s$: | • **Zero-init** weights & bias của $\Delta W_0$ và $\Delta W_{k\neq 0}$ (`W_0 = I + 2·tanh(raw)` → khởi tạo đúng $I$, các tap khác `= 0`).<br>• **Gate bias = `-5.0`** (`g = σ(logit - 5.0) ≈ 0.0067` tại init — gần đóng hoàn toàn mà vẫn còn gradient; tách riêng test *forced identity* `g=0` và *actual init*). | `~1.9–2.3M` *(đếm chính xác bằng PyTorch ở Step 0)* |

**Phương trình tái tạo Linear STFT của `V4-2 (MF7)`:**
$$F_W(Z_x)(t, f) = Z_x(t, f) + \sum_{k=-5}^{+1} \Delta W_k(t, f)\, Z_x(t + k, f)$$
$$\hat{Z}_c(t, f) = \big(1 - g_c(t, f)\big)\, F_{W,c}(Z_x)(t, f) + g_c(t, f)\, Z_{s,c}(t, f), \qquad c \in \{\text{Mid}, \text{Side}\}$$

---

## 3. Lịch Hàm Loss (3 Giai Đoạn, Theo Dõi Hội Tụ)

Giám sát hoàn toàn bằng **clean reference `c_in = c / scale`** (tuyệt đối không ép sao chép waveform VAE của `s`):

1. **Stage 1 — Initial Training (tới khi val MR-STFT hội tụ, dự kiến ~12–15 epochs):**
   $$\mathcal{L}_1 = 1.0\,\mathcal{L}_{\text{MR-STFT}}(\hat{y}, c) + 0.5\,\mathcal{L}_{\text{CompComplex-L1}}(\hat{Z}^c, Z_c^c) + 0.5\,\mathcal{L}_{\text{Wav-L1}}(\hat{y}, c)$$
   *(MR-STFT 3 phân giải `n_fft ∈ {512, 1024, 2048}`, tính cả spectral convergence + log-mag L1 trên cả L/R và Mid/Side).*
2. **Stage 2 — Preservation & Stereo (khi Stage 1 ổn định, dự kiến ~8–10 epochs):**
   $$\mathcal{L}_2 = \mathcal{L}_1 + \lambda_{\text{ms}}\,\mathcal{L}_{\text{M/S-Ratio}}(\hat{y}, c) + \lambda_g\,\|g\|_1 + \lambda_W \sum_{k} \|\Delta W_k\|_2^2$$
   *(Chỉ áp dụng $\lambda_g, \lambda_W$ cho `V4-2` để ngăn mạng vừa đóng `g` vừa đẩy biến dạng quá mạnh sang $\Delta W$).*
3. **Stage 3 — Optional Perceptual Fine-Tuning (chỉ chạy SAU KHI có số liệu Stage 2):**
   - Thử frozen **VGGish** (warm-up tuyến tính 2,000 bước kiểu NPU-ASLP) hoặc **CLAP** trọng số nhỏ.
   - **Không mặc định cải thiện FAD/PQ:** Chỉ giữ Stage 3 nếu trên tập Validation cải thiện đồng thời `FAD-CLAP` và `Audiobox PQ` mà không vi phạm Attribute Guards.

---

## 4. Lộ Trình Ablation Đầy Đủ (Không Bỏ Bước)

| ID | Kiến trúc | Giai đoạn chạy | Câu hỏi kiểm chứng |
|---|---|---|---|
| **V4-0** | `NoGuide`: `x` only, DirectMap (`Dense FFN`) | **Pilot 1 (ngay sau Step 0)** | Mạng ~1.2M không có teacher làm được tới đâu? |
| **V4-1** | `DirectMap`: `(x, s)` (`Dense FFN`) | **Pilot 1** | SonicMaster guidance `s` tăng thêm bao nhiêu so với `V4-0`? |
| **V4-2** | `MF7`: `(x, s)` Identity-centered 2×2 M/S 7-frame + gate (`Dense FFN`) | **Pilot 1** | Lọc phức đa khung trên linear STFT có vượt `V4-1` và `Hybrid-v3` không? |
| **V4-3** | `Delayed/Dilated MF`: taps giãn cách phủ ~250–300 ms (kiểu WPE delayed prediction) | **Vòng 2 (nếu V4-2 yếu ở `big/real` reverb)** | Tăng tầm phủ thời gian có cải thiện `reverb` dài không? |
| **V4-4** | `MF7 + Additive Complex Residual` $R_\theta(Z_x, Z_s)$ | **Vòng 2 (nếu V4-2 yếu ở `clip`/`xband`/`stereo` collapse)** | Nhánh additive có cứu được các bin bằng 0 không? |
| **V4-5** | Best decoder + **TF-MoE** (`4` experts, Top-1) vs **compute-matched Dense FFN** | **Vòng 3 (chỉ khi V4-2/3/4 PASS)** | MoE có thực sự hơn Dense FFN cùng ngân sách FLOPs/params không? |

---

## 5. Step 0 — Unit Tests & Compute Benchmark (Bắt Buộc Trước Khi Train)

Tạo `scripts/extension/v4_models.py` và `scripts/extension/tests/test_v4_step0.py`, kiểm chứng toàn bộ **11 hạng mục**:

1. **STFT → iSTFT roundtrip:** `max_abs_err < 1e-5` trên tín hiệu stereo 44.1 kHz (cả `hop=512` và `hop=1024`).
2. **Mid/Side ↔ Left/Right roundtrip:** `max_abs_err < 1e-6`.
3. **Forced Identity Test (`V4-2`):** Ép `ΔW = 0, g = 0` trên track chuẩn (`rms_x ≈ rms_s`) → `max_abs_err(y_hat, x) < 1e-5`.
4. **Actual Initialization Test (`V4-1`, `V4-2`):** Chạy model mới khởi tạo chưa train trên audio thật → đo `max_abs_err`, SNR so với `x`, xác nhận `mean(g) ≈ 0.0067` và **không tạo phase artifacts**.
5. **Extreme Silence & Low-Volume Stability:** Test với input `0` tuyệt đối (`silence`) và `volume -35 dB` → **0 NaN / 0 Inf**, `g_align` kẹp đúng `+24 dB`.
6. **Chunked Overlap-Add Boundary Test:** So sánh full-track vs OLA (`4s/2s` và `8s/4s`) ở trạng thái forced identity → `max_abs_err < 1e-5` kể cả 2 biên đầu và cuối 30s (0 loudness jump).
7. **Exact PyTorch Parameter Count:** In bảng chi tiết từng module (`Encoder`, `Backbone`, `Per-bin Decoder`, `Sub-band MF Decoder`) cho `V4-0`, `V4-1`, `V4-2` (xác nhận rõ lớp nào shared / band-specific).
8. **Compute & Memory Benchmark (Local + Kaggle T4 16 GB):**
   - Đo **GMACs/FLOPs** (chunk `4s` và full `30s`).
   - Đo **Peak VRAM** (training batch=4/8 với AMP fp16, và inference 30s).
   - Đo **Wall-clock latency cho full clip 30s (BAO GỒM cả STFT + iSTFT + OLA)**.
   - So sánh trực tiếp ma trận 4 cấu hình: `hop=512` vs `hop=1024`, và OLA `4s/2s` vs `8s/4s`.

---

## 6. Step 1 — Thiết Kế Data Split & Pilot Source-Disjoint

Tuyệt đối tuân thủ tách nguồn (`source_id`-disjoint), **0% rò rỉ**:

1. **Giai đoạn 1A — Smoke Overfit Check (`5 phút`, Dev 40 bài):**
   - Chia `28` bài Train / `6` bài Val / `6` bài Test trên Dev **CHỈ để xác nhận** loss giảm ổn định, gradient không NaN, checkpoint/resume hoạt động.
   - **CẤM dùng kết quả của split 6 bài (42 clip) để đánh giá hay so sánh mô hình** (vì 42 clip không tính nổi FAD-CLAP 512-d và hầu hết Attribute Guards có `n < 10`).
2. **Giai đoạn 1B — Statistical Internal Pilot (`0` giờ chạy thêm SonicMaster):**
   - Sử dụng kho `(x, s, c)` đã có sẵn từ `14_run7000`:
     - **`Pilot-Train`:** `400` bài từ Confirm (`2,800` clip, ~23.3 giờ âm thanh).
     - **`Pilot-Val`:** `80` bài từ Confirm (`560` clip, chọn checkpoint & early stopping).
     - **`Pilot-Test` (hoàn toàn độc lập):** Toàn bộ **`Dev 280` (`40` bài)** — cho phép so sánh trực tiếp 1-1 với các con số Dev đã khóa của `A0`, `V3-Label`, `V3-Blind`, `V3 + TD_declip`!
     - **`Confirm-Locked`:** Giữ nguyên **`480` bài Confirm (`3,360` clip)** còn lại hoàn toàn không đụng tới (làm kiểm chứng chéo thứ 2 nếu cần).
   - Lưu danh sách `source_id` cố định tại `artifacts/10_extension/v4/pilot_splits.json` (kiểm tra giao của 4 tập = `∅`).
   - Ghi rõ mọi kết quả Giai đoạn 1B là **Internal Pilot**.
3. **Giai đoạn 2 (Chỉ khi Internal Pilot PASS Go/No-Go) — Official Zero-Leakage Training:**
   - Tải **`400–500` bài mới từ `train` chính thức của `SonicMasterDataset`** (0% trùng với 1,000 bài test), chạy SonicMaster sinh `s` trên Kaggle, train lại bản thắng cuộc từ đầu và đánh giá chính thức trên **toàn bộ `Dev (280)` + `Confirm (6,720)` nguyên vẹn**.

---

## 7. Giao Thức Đánh Giá, Chẩn Đoán Nội Bộ & Kiểm Chứng Prompt Controllability

### 7.1. Bảng so sánh bắt buộc trên `Pilot-Test (Dev 280)`
So sánh trực tiếp **`A0` | `V3-Label` | `V3-Blind` | `V4-0 (NoGuide)` | `V4-1 (DirectMap)` | `V4-2 (MF7)`** trên:
- **Global metrics:** `SSIM ↑`, `KL×1000 ↓`, `FAD-CLAP ↓`, `Audiobox PQ ↑` (`ΔPQ`, `CI95_lo`).
- **15 Attribute Guards (Dev) / 19 Guards (Confirm):** `0` vi phạm (`norm_det <= 0.05`).
- **Tách nhóm bắt buộc:** `Single` vs `Double+Triple`, và 5 họ suy hao riêng: **EQ (`mud/boom/warm/bright/dark/clarity/airy/mic`)**, **Reverb (`small/big/mix/real`)**, **Clipping (`clip`)**, **Stereo (`stereo`)**, **Volume & Dynamics (`volume/comp/punch`)**.
- **Tín hiệu kỹ thuật:** Waveform/spectral fidelity, Stereo L/R correlation, Output clipping rate (`|y| >= 0.999`), Runtime & Peak VRAM.

### 7.2. Log chẩn đoán nội bộ bắt buộc (sau mỗi epoch & tại test)
- `mean(g_M)`, `mean(g_S)` toàn tập và **phân bố `mean(g)` theo từng họ suy hao** *(chỉ dùng nhãn khi phân tích chẩn đoán, không đưa vào model)*:
  - *Cảnh báo Mode Collapse:* Nếu `mean(g) > 0.85` trên nhóm EQ tuyến tính → model đang lười biếng copy `s`.
  - *Cảnh báo Ignored Teacher:* Nếu `mean(g) < 0.05` trên `xband` / `real reverb` → kiểm tra xem teacher `s` có thực sự đóng góp không.
- Phân bố biên độ `||ΔW_0||`, `||ΔW_{k≠0}||`, tỷ lệ output clipping, và sai số tái tạo STFT/phase.

### 7.3. Kiểm chứng `Prompt Controllability` (Mục 10)
- Chọn `10` mẫu `x` từ `Pilot-Test`, chạy SonicMaster với **2 prompt đối lập nhau** (ví dụ: `"make the audio brighter and clearer"` → `s_1` vs `"make the audio warmer and less bright"` → `s_2`).
- Chạy `V4(x, s_1)` và `V4(x, s_2)`, đo chênh lệch năng lượng dải cao/thấp giữa `y_1` và `y_2` so với `s_1` và `s_2` để **chứng minh thực nghiệm** mức độ kế thừa tính điều khiển bằng ngôn ngữ tự nhiên (không suy diễn suông).

---

## 8. Cổng Quyết Định Go / No-Go Sau Pilot 1 (`V4-0 / V4-1 / V4-2`)

| Tiêu chí trên `Pilot-Test (Dev 280)` | Ngưỡng PASS để đi tiếp sang Vòng 2 / Official Train | Hành động nếu FAIL |
|---|---|---|
| **Vượt `V3-Blind` về `Audiobox PQ`** | `CI95_lo(ΔPQ vs A0) >= -0.05` *(so với V3-Blind `-0.137` trên Dev)* | Kiểm tra `mean(g)` và phân tích lỗi theo họ (`reverb` vs `clip`); cân nhắc Stage 3 perceptual loss hoặc chuyển hướng Learned 3-Way Router `{p, TD_declip, s}`. |
| **Chứng minh giá trị Teacher `s`** | `V4-1` hoặc `V4-2` vượt rõ `V4-0 (NoGuide)` trên `Reverb`, `xband`, `FAD`, `PQ` | Nếu `V4-0` ngang `V4-1/2`, xem lại thiết kế Dual Scale-Aware conditioning. |
| **Chứng minh Codec-Bypass (không collapse sang `s`)** | `V4-2 SSIM >= 0.735`, `FAD <= 0.170`, `mean(g) < 0.4` trên nhóm EQ tuyến tính, và trên `Reverb (106)` giữ `PQ >= A0 - 0.05` | Nếu `V4-2` thua `V4-1`, giữ `V4-1` làm baseline; nếu `V4-2` tốt trên EQ/clip nhưng yếu trên `big/real reverb`, mở `V4-3` (Dilated MF). |
| **Attribute Guards** | **0 vi phạm** trên 15/15 thuộc tính đủ mẫu (`n >= 10`) | Điều tra đúng thuộc tính trượt trước khi mở rộng. |

---

## 9. Red Lines & Provenance (`GEMINI.md`)
- Không mở rộng lên toàn bộ `Confirm` hoặc bật `MoE` / `Perceptual Fine-Tuning` trước khi hoàn thành **Step 0** và **Pilot 1 (`V4-0 / V4-1 / V4-2`)**.
- Không dùng `clean` ở bất kỳ bước inference/alignment/normalization nào.
- Không sửa đổi bất kỳ file nào của `Hybrid-v3` đã khóa; toàn bộ log, trọng số và đánh giá của `Hybrid-v4` lưu cô lập trong `artifacts/10_extension/v4/`.
- Mọi kết quả báo cáo cuối cùng phải tách rõ 3 cột **Baseline (`A0` / `V3`) | Measured (`V4`) | Delta**, kèm mã kiểm toán file/log và mẫu âm thanh minh chứng (`Clean`, `Degraded`, `A0`, `V3`, `V4`).

---

## 10. Tài Liệu Tham Khảo & Nguồn Gốc Kỹ Thuật (References & Provenance)

| Thành phần trong `Hybrid-v4` | Nguồn gốc (Paper / Repo) | Lý do / Cách kế thừa |
|---|---|---|
| **Bài toán, SonicMaster `s`, Dataset & Bộ đánh giá Bảng 1–3** | J. Melechovsky, A. Mehrish, A. Roy, D. Herremans, *"SonicMaster: Towards Controllable All-in-One Music Restoration and Mastering"*, ICML 2026 ([arXiv:2508.03448](https://arxiv.org/abs/2508.03448)) | Baseline `A0` (`s`), dữ liệu train/val/dev/confirm và toàn bộ protocol chấm điểm. |
| **Complex Multi-Frame Filtering (`MF7` / `MF9`) & Psychoacoustic Bands** | H. Schröter et al., *"DeepFilterNet: A Low Complexity Speech Enhancement Framework for Full-Band Audio based on Deep Filtering"*, IEEE ICASSP 2022 ([arXiv:2110.05588](https://arxiv.org/abs/2110.05588)) | Cơ sở cho bộ lọc phức đa khung `ΔW_k` trên STFT tuyến tính thay cho mặt nạ điểm đơn. |
| **Delayed / Dilated Multi-Frame Taps (`V4-3`)** | T. Nakatani et al., *"Speech Dereverberation Based on Variance-Normalized Delayed Linear Prediction"* (WPE), IEEE TASLP 2010 | Cơ sở cho các tap trễ giãn cách (`-18..-1, +1`) phủ đuôi reverb ~220–300 ms trong miền STFT. |
| **Scale-Aware Band Encoder, Complex Mapping (`V4-0/1`) & TF-MoE (`V4-5`)** | ASLP Group (NPU-ASLP), *"The NPU-ASLP System for Music Restoration in the CCF-AATC 2026 Challenge"*, Oct 2026 | Cơ sở cho chuẩn hóa dải tần giữ log-RMS (`Scale-Aware`), ánh xạ phức trực tiếp (`V4-0/1`), `TF-MoE` (`V4-5`) và warm-up VGGish. |
| **24-Band Split & Dual-Path Time–Frequency Blocks (`V4` Backbone)** | Y. Luo & J. Yu, *"Music Source Separation with Band-split RNN"* (BSRNN), IEEE/ACM TASLP 2023 ([arXiv:2209.15174](https://arxiv.org/abs/2209.15174)); Z.-Q. Wang et al., *"TF-GridNet: Integrating Full- and Sub-Band Modeling for Speech Separation"*, IEEE/ACM TASLP 2023 ([arXiv:2211.12433](https://arxiv.org/abs/2211.12433)) | Thiết kế chia 24 dải tần không đều (psychoacoustic) và xử lý luân phiên theo trục Thời gian + Tần số. |
| **Multi-Resolution STFT Loss** | R. Yamamoto, E. Song, J.-M. Kim, *"Parallel WaveGAN"*, IEEE ICASSP 2020 ([arXiv:1910.11480](https://arxiv.org/abs/1910.11480)) | Loss đa phân giải STFT (mở rộng thêm kênh Mid/Side). |
| **Audiobox Aesthetics (`PQ`) & FAD-CLAP** | A. Tjandra et al., *"Meta Audiobox Aesthetics"*, 2025 ([arXiv:2502.05139](https://arxiv.org/abs/2502.05139)); Y. Wu et al., ICASSP 2023 ([arXiv:2211.06687](https://arxiv.org/abs/2211.06687)); A. Gui et al., ICASSP 2024 ([arXiv:2311.01616](https://arxiv.org/abs/2311.01616)) | Chỉ số chất lượng cảm nhận `PQ` và khoảng cách phân phối `FAD-CLAP` cho cổng Go/No-Go. |
