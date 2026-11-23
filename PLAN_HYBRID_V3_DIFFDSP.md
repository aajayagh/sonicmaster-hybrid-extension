# PLAN_HYBRID_V3_DIFFDSP.md — Kế Hoạch Nâng Cấp Hybrid-v2 PA1 Lên Hybrid-v3 (Differentiable DSP)

Tài liệu này là **bản đặc tả kỹ thuật và lộ trình thực thi chi tiết (handoff plan)** để một AI Agent khác triển khai độc lập nhánh nâng cấp **Hybrid-v3** cho dự án tái lập & mở rộng SonicMaster (`d:\SonicMaster_Reproducing`).

Mọi quy tắc khoa học và giới hạn an toàn trong [`GEMINI.md`](file:///d:/SonicMaster_Reproducing/GEMINI.md) đều bắt buộc áp dụng.

---

## 1. Điểm Xuất Phát & Bài Toán Cần Giải Quyết

### 1.1. Kết quả chuẩn đã khóa của **Hybrid-v2 PA1** ([`lock_v2.json`](file:///d:/SonicMaster_Reproducing/artifacts/10_extension/lock_v2.json))
Xem chi tiết đầy đủ tại [`HYBRID_V2_PA1_REPORT.md`](file:///d:/SonicMaster_Reproducing/HYBRID_V2_PA1_REPORT.md). Tóm tắt mốc đối chiếu:

| Tập dữ liệu | Chỉ số | `A0` (SonicMaster) | **PA1 (Baseline cần vượt)** | Cổng nghiệm thu |
|---|---|---:|---:|---|
| **Dev** (`N=280`) | SSIM ↑ / KL×1000 ↓ | `0.6064` / `1.1064` | `0.7120` / `1.1215` (`1.014×`) | `ΔSSIM ≥ +0.01`, `KL ≤ 1.100×` |
| | FAD-CLAP ↓ / PQ ↑ | `0.2145` / `7.6452` | `0.1744` / `7.6302` (`CI95_lo -0.0397`) | `FAD ≤ 1.100×`, `CI95_lo ≥ -0.05` |
| **Confirm** (`N=6.720`) | SSIM ↑ / KL×1000 ↓ | `0.6092` / `0.8839` | `0.6912` / `0.9711` (`1.099×`) | `KL ≤ 0.9723` (`1.100×`) |
| | FAD-CLAP ↓ / PQ ↑ *(`n=5.306`)* | `0.0756` / `7.7103` | `0.0466` / `7.7061` (`CI95_lo -0.0104`) | `0` vi phạm trên 19 Guards |
| **Confirm Non-Fallback** (`n=2.083`) | Single (`1.651`) SSIM / KL / PQ | `0.6440` / `0.5021` / `7.7810` | `0.9229` / `0.7118` / `7.7885` (`+0.0075`) | FAD: `0.0697 → 0.0163` (`-76.6%`) |
| | Double+Triple (`432`) SSIM / KL / PQ | `0.6359` / `0.6702` / `7.7994` | `0.8453` / `1.2254` / `7.7188` (`-0.0806`) | FAD: `0.0914 → 0.0484` (`-47.0%`) |

### 1.2. 3 Điểm yếu mục tiêu mà **Hybrid-v3** phải xử lý
1. **Confirm `KL×1000` đang quá sát trần (`1.099×` vs `1.100×`)** — cần hạ xuống vùng an toàn `≤ 1.05×`.
2. **Nhóm `Double+Triple` bị suy giảm (`KL 1.83×`, `PQ -0.0806`)** do `branch_p.py` tối ưu 4 khối tuần tự rời rạc (`EQ → Dynamics → Stereo → LUFS`) và [`match_dynamics`](file:///d:/SonicMaster_Reproducing/scripts/extension/branch_p.py#L118-L153) bỏ qua toàn bộ trường hợp cần giãn động học (`skipped_expansion`).
3. **Độ phủ bypass mới đạt `~31%` (`2.083/6.720`)** — còn 2 nhóm rất tiềm năng có thể mở khóa mà không dùng deverb rời:
   - **`mud` (`368` mẫu Confirm ≈ 5.5%):** 100% là lỗi EQ tuyến tính (`200–500 Hz`), chỉ bị fallback vì `match_eq_fir` 1/3-octave làm nhòe biên.
   - **`xband` (`818` mẫu Confirm ≈ 12.2%):** chỉ mất dải tần rìa, dải trung âm/vocal trong `x` vẫn sạch nguyên bản.

---

## 2. Cơ Sở Tài Liệu & Nguyên Tắc Thiết Kế

Ba bài báo tham khảo được chọn lọc có chủ đích cho từng module:
1. **Nercessian (2020) — *Neural Parametric Equalizer Matching Using Differentiable Biquads* (DAFx 2020):**
   - Dùng chuỗi cascaded biquads (RBJ Audio EQ Cookbook) và tối ưu trong **miền tần số (`freqz` / frequency sampling)** thay vì BPTT theo mẫu thời gian; tối ưu trên **log-magnitude (dB)** + $L_1$ sparsity trên gain $G_k$.
2. **BABE-2 (2024) — *A Diffusion-Based Generative Equalizer for Music Restoration* (`arXiv:2403.18636`, Sec. 3.1, 3.4):**
   - Dùng **Long-Term Average Spectrum (LTAS) với Gaussian smoothing 1/3-octave** để khởi tạo bộ lọc ổn định và phát hiện chính xác vùng cắt băng thông của `xband`.
3. **DiffVox (2025) — *A Differentiable Model for Capturing and Analysing Vocal Effects Distributions* (`arXiv:2504.14735`, Sec. 2.1, 2.2, 3.2, 3.3):**
   - Kết hợp **6+ band PEQ** + **Differentiable DRC đầy đủ Compressor & Expander**; tối ưu **đồng thời (joint fitting)** bằng Adam với loss **Multi-Scale STFT (MSS) trên cả L/R & M/S + Microdynamics MAE (MLDR)**.

---

## 3. Quy Tắc Bắt Buộc & Các Bẫy Kỹ Thuật Phải Tránh

Agent thực thi **PHẢI** tuân thủ toàn bộ danh sách sau (đúc rút từ lỗi thực tế khi làm PA1):

1. **Tuyệt đối không đọc `clean` khi suy luận (Zero Data Leakage):** Mọi ước lượng tham số DSP, mask, guard chỉ được đọc `x` (degraded) và `s` (SonicMaster `A0`).
2. **Tuyệt đối không tune trên Confirm (No Data Snooping):** Toàn bộ thiết kế, chọn siêu tham số, ablation (`V3-A`, `V3-B`, `V3-C`) **chỉ thực hiện trên 280 mẫu Dev**. Confirm (`6.720` mẫu) chỉ chạy **1 lần duy nhất** sau khi đã khóa `artifacts/10_extension/lock_v3.json`.
3. **Không động lại vào `deverb` rời và `clip`:** Giữ cố định `{clip, small, big, mix, real}` và guard mù `20·log10(rms(s)/rms(x)) > 24.0 dB` ở nhánh fallback `S` (`A0`).
4. **Truy xuất Google Drive an toàn (`GEMINI.md` Mục 8):**
   - Nhiều file trong `artifacts/14_run7000/` và `artifacts/10_extension/` đã di chuyển sang `G:\My Drive\SonicMaster_Reproducing\`.
   - Mọi script đọc file phải có helper `res(p)` kiểm tra `(ROOT / p).is_file()` trước, nếu không có thì fallback `G:\My Drive\SonicMaster_Reproducing / p`.
   - Đọc audio từ `G:\` phải dùng `sf.read(io.BytesIO(path.read_bytes()), ...)` kèm **vòng lặp retry 3 lần (sleep 1s)** để tránh lỗi stream `LibsndfileError` / `OSError: [Errno 22]`.
5. **Chống lỗi âm thầm ở `eval_tables123.py`:** Luôn xác minh `assert len(rows) > 0` và kiểm tra đủ 15 (Dev) / 19 (Confirm) thuộc tính được tính.
6. **Chống mount nhầm dataset cũ & lỗi `numpy` trên Kaggle:**
   - Khi chạy PQ/FAD trên Kaggle, **tạo dataset slug mới hoàn toàn** (vd `sonicmaster-ext-dev-pq-fad-v3`, `sonicmaster-ext-confirm-pq-fad-v3`), tạo sẵn thư mục `C:\Users\ADMIN\AppData\Local\Temp\.kaggle-uploads`, đợi `dataset_status == "ready"`, và **in `KAGGLE_SAMPLE_SHA256` của 1 file mẫu** trong kernel để so khớp 100% với `LOCAL_SHA256`.
   - Trong script chạy trên Kaggle, **chạy `pip install` trước khi `import numpy`** để tránh `ValueError: numpy.dtype size changed`.
   - Không bao giờ in nội dung `C:\Users\ADMIN\.kaggle\access_token.txt` (dòng 0 = `aajayagh`, dòng 5 = `nthkhanh`).
7. **Môi trường Python & Tái sử dụng tài nguyên đã có:**
   - `tools\extension-env\Scripts\python.exe`: PyTorch CUDA, `scipy`, `soundfile`, `pyloudnorm`, `librosa`, `skimage`, `pytest`.
   - `tools\eval-env\Scripts\python.exe`: `pyroomacoustics`, `pandas`, `librosa`, `skimage`, `scipy`.
   - System `python`: có `kaggle` API (không có `soundfile`/`pandas`).
   - **Tái sử dụng stem `vocal` có sẵn (KHÔNG cần chạy lại `bs_roformer_1297`!):**
     - Dev stems: `G:\My Drive\SonicMaster_Reproducing\artifacts\10_extension\kaggle_kernel_p1_out\out\dev_stems_flac\` (`<sid>_voc.flac`, `<sid>_acc.flac`).
     - Confirm stems (282 file cho 141 cặp `vocal`): `d:\SonicMaster_Reproducing\artifacts\10_extension\confirm\hybrid_v2\vocal_stems_flac\`.
   - **Tái sử dụng baseline `14_run7000` cho Confirm:**
     - `G:\My Drive\SonicMaster_Reproducing\artifacts\14_run7000\evaluation\kl_ssim\kaggle\per_sample_7000.csv`
     - `G:\My Drive\SonicMaster_Reproducing\artifacts\14_run7000\evaluation\attributes\per_sample_attributes_7000.csv`
     - `G:\My Drive\SonicMaster_Reproducing\artifacts\14_run7000\evaluation\fad_pq\s01..s16\clap_embeddings.npz` & `per_sample_pq_7000.csv`
   - **Quản lý dung lượng ổ `D:\`:** Lưu WAV dạng `PCM_16`; sau khi chấm điểm Confirm xong phải xóa `out_wav/` (~36 GB), chỉ giữ lại JSON/CSV/NPZ/logs/lock.

---

## 4. Đặc Tả Kỹ Thuật 3 Module Mới

### Module 1: `scripts/extension/biquad_eq.py` — Nercessian 8-Band Differentiable Biquad PEQ + BABE-2 LTAS Init

- **Đầu vào:** `(u, s, sr=44100, freq_weights=None)` với `u, s` shape `(T, 2)` float32.
- **Bước 1 — Tiền xử lý `volFirst`:** Cân bằng RMS `u` theo `s`: `xv = u * ((rs + 1e-9) / (rx + 1e-9))`.
- **Bước 2 — Tính phổ mục tiêu & Khởi tạo LTAS (BABE-2 Sec. 3.4):**
  - Tính STFT power trung bình theo thời gian và 2 kênh (`n_fft=4096, hop=1024`) để lấy $P_x(f)$ và $P_s(f)$.
  - Chiếu $P_x, P_s$ lên lưới **1/3-octave** (`20 Hz – 20 kHz`), tính log-ratio $d(b) = 10\log_{10}((E_s(b)+\epsilon)/(E_x(b)+\epsilon))$ và làm mượt bằng **Gaussian kernel ($\sigma = 1.0$ band)**.
  - Dùng $d_{\text{smooth}}(b)$ khởi tạo:
    - Low-Shelf (`LS`): $f_c = 120\text{ Hz}$, $G$ từ trung bình vùng `< 150 Hz`.
    - High-Shelf (`HS`): $f_c = 8000\text{ Hz}$, $G$ từ trung bình vùng `> 6000 Hz`.
    - 4 băng Peaking (`PK1..PK4`): chọn 4 cực trị (peak/dip) cách nhau ít nhất 2/3 octave trong vùng `[80 Hz, 10 kHz]` (mặc định chia đều `[180, 350, 1200, 3500] Hz` nếu phẳng; lưu ý có 1 băng mặc định quanh `350 Hz` để bắt nhanh `mud` `200–500 Hz`).
    - `HP` ($f_c \approx 25\text{ Hz}, Q = 0.707$) và `LP` ($f_c \approx 19500\text{ Hz}, Q = 0.707$).
- **Bước 3 — Tham số hóa ổn định & Frequency-Sampling Forward (Nercessian 2020):**
  - Dùng biến không ràng buộc qua `sigmoid`/`tanh` ép chắc chắn mọi biquad đều ổn định tuyệt đối:
    - $\log f_c \in [\log 20, \log(0.47 \cdot sr)]$, $G_{\text{dB}} \in [-18.0, +18.0]$, $Q \in [0.5, 8.0]$.
  - Viết công thức RBJ Audio EQ Cookbook khả vi trong PyTorch cho `HP, LP, LS, HS, PK` trả về hệ số chuẩn hóa $(b_0, b_1, b_2, a_1, a_2)$ (với $a_0 = 1$).
  - Đánh giá đáp ứng tần số tại lưới $M = 256$ điểm log-freq từ `20 Hz – 20 kHz` **kết hợp** với lưới `128` tâm dải Mel (`librosa.mel_frequencies(128, fmin=0, fmax=22050)`):
    $$H_k(e^{j\omega}) = \frac{b_{0,k} + b_{1,k}e^{-j\omega} + b_{2,k}e^{-j2\omega}}{1 + a_{1,k}e^{-j\omega} + a_{2,k}e^{-j2\omega}}, \quad |H(e^{j\omega})|^2 = \prod_{k=1}^{8} |H_k(e^{j\omega})|^2$$
- **Bước 4 — Hàm mất mát & Tối ưu (CPU, < 0.1s/clip):**
  - Tính phân bố Mel trung bình $M_{\text{mat}} \cdot (|H|^2 P_x)$ so với $M_{\text{mat}} \cdot P_s$ (dùng đúng ma trận filterbank `librosa.filters.mel(sr=44100, n_fft=2048, n_mels=128)`).
  - Hàm loss (có hỗ trợ trọng số `freq_weights` cho `xband` ở Module 3):
    $$\mathcal{L}_{\text{EQ}} = \mathcal{L}_1^{\text{weighted}}\!\left(10\log_{10}(|H|^2 P_x + \epsilon),\; 10\log_{10}(P_s + \epsilon)\right) + \lambda_{\text{KL}}\,\mathcal{L}_{\text{Mel-KL}} + \lambda_{\text{sp}}\sum_{k=1}^{8}|G_k|$$
    *(Khởi đầu đề xuất: $\lambda_{\text{KL}} = 10.0$, $\lambda_{\text{sp}} = 0.01$, Adam `lr = 0.05`, `250 steps`.)*
- **Bước 5 — Áp bộ lọc SOS lên sóng âm:**
  - Chuyển hệ số đã tối ưu sang mảng `sos` shape `(8, 6)` cho `scipy.signal`.
  - **CẢNH BÁO KỸ THUẬT QUAN TRỌNG:** Cần thử nghiệm A/B trên Dev giữa 2 chế độ áp lọc:
    - **Mode A (Minimum-phase `sosfilt`):** Áp trực tiếp `sos` với đúng $G_k$ đã fit.
    - **Mode B (Zero-phase `sosfiltfilt`):** Vì `sosfiltfilt` lọc xuôi + lọc ngược làm bình phương biên độ $|H|^2$ (gấp đôi dB!), nếu dùng `sosfiltfilt` **BẮT BUỘC phải fit với $|H|^4$ trong loss (hoặc tạo `sos` với gain $G_k / 2$ và 1-pole cho HP/LP)**, nếu không EQ sẽ bị khuếch đại gấp đôi!

---

### Module 2: `scripts/extension/diff_drc_joint.py` — DiffVox Differentiable DRC (Compressor + Expander) & Joint Fitting

- **Phạm vi kích hoạt:** Chỉ chạy cho các mẫu thuộc nhóm `full_mix_P` có chứa bất kỳ lỗi nào trong `{"comp", "punch", "stereo"}` **hoặc** `n_degradations >= 2`. Các mẫu EQ đơn thuần (`Single` của `bright, dark, airy, warm, boom, clarity, mic, mud, volume`) dùng thẳng Module 1.
- **Cấu trúc chuỗi DSP khả vi (PyTorch, chạy trên GPU local `tools\extension-env`):**
  1. **PEQ:** Khởi tạo (warm-start) từ tham số $(f_c, G_k, Q_k)$ của Module 1; áp lọc bằng phép nhân phổ STFT overlap-add (hoặc FIR/FFT-convolve từ impulse response cắt ngắn 4096 mẫu của `H(e^{j\omega})` — vừa **nhanh 100×** trên GPU vừa **100% khả vi**!).
  2. **Differentiable Feed-Forward DRC (Compressor + Expander — DiffVox Sec. 2.2):**
     - Tính RMS envelope theo khung ngắn (`frame = 512`, `hop = 128`, ~2.9 ms) trên tín hiệu stereo linked (`max(|L|, |R|)` hoặc RMS tổng hợp), làm mượt bằng bộ lọc attack/release dạng tích chập exponential kernel (hoặc `torchcomp` nếu cài được).
     - Đường cong gain tĩnh hai vùng (soft-knee độ rộng $W = 6\text{ dB}$):
       - **Downward Compression** khi $e_{\text{dB}} > T_c$: giảm theo tỷ số $R_c \in [1.0, 10.0]$.
       - **Upward/Downward Expansion** khi $e_{\text{dB}}$ ở vùng động học mục tiêu (ví dụ $e_{\text{dB}} > T_e$ tăng transient / hoặc dưới $T_e$ giãn noise-floor): cho phép **khôi phục độ biến thiên RMS (`std_rms`) và độ sắc transient (`punch`)** cho các ca bị nén bẹp!
     - Nội suy tuyến tính gain từ cấp khung (`hop=128`) lên cấp mẫu và nhân vào waveform.
  3. **Stereo Mid/Side Width & Output Gain:**
     - $M = 0.5(L+R),\; S = 0.5(L-R),\; (L', R') = g_{\text{out}} \cdot (M + k_{\text{ms}} S,\; M - k_{\text{ms}} S)$ với $k_{\text{ms}} \in [0.0, 4.0]$ (nếu `rms(S_x) < 1e-4` thì giữ nguyên luật mượn side từ `s` như `match_stereo`).
- **Hàm loss tối ưu đồng thời (DiffVox Sec. 3.2–3.3):**
  - Chạy **120–150 bước Adam (`lr = 0.01`)** (warm-start PEQ với `lr_peq = 0.002` để không phá đường EQ tốt đã fit):
    $$\mathcal{L}_{\text{joint}} = \mathcal{L}_{\text{MSS}}(y_{\text{L/R}}, s_{\text{L/R}}) + \mathcal{L}_{\text{MSS}}(y_{\text{M/S}}, s_{\text{M/S}}) + \lambda_{\text{dyn}}\,\mathcal{L}_{\text{MLDR}}(y, s) + \lambda_{\text{KL}}\,\mathcal{L}_{\text{Mel-KL}}(y, s)$$
    trong đó:
    - $\mathcal{L}_{\text{MSS}}$: Multi-Scale STFT L1 (linear + log magnitude) với `n_fft ∈ [512, 1024, 2048]`.
    - $\mathcal{L}_{\text{MLDR}}$: MAE của chuỗi frame-RMS dB (`frame=2048, hop=1024`) + MAE của hiệu năng lượng ngắn/dài (`short_rms - long_rms`) để bắt trực tiếp cả `comp` lẫn `punch`.
  - Kết thúc bằng bước chuẩn hóa `match_volume_lufs` và `np.clip(y, -1, 1)`.

---

### Module 3: `scripts/extension/xband_crossover.py` — `xband` Complementary Spectral Crossover

- **Phạm vi kích hoạt:** Các mẫu chứa `xband`, **không** chứa `{clip, small, big, mix, real}` và vượt qua `guard(rms_dB <= 24.0)`.
- **Cơ chế mù hoàn toàn (chỉ dùng `x` và `s`):**
  1. Cân RMS trên dải trung âm tham chiếu (`400–2000 Hz`, nơi cả `x` và `s` thường đều có năng lượng).
  2. Tính phổ công suất trung bình $P_x(f), P_s(f)$ (`n_fft=2048, hop=512`), làm mượt Gaussian theo dải 1/3-octave (BABE-2 Sec. 3.4) và tính tỷ số $r_{\text{dB}}(f) = 10\log_{10}((P_x(f)+\epsilon)/(P_s(f)+\epsilon))$.
  3. Tạo mặt nạ mềm theo tần số:
     $$M_{\text{band}}(f) = \sigma\!\left(\frac{r_{\text{dB}}(f) - \tau_{\text{dB}}}{\Delta_{\text{dB}}}\right), \quad \tau_{\text{dB}} = -18.0\text{ dB},\; \Delta_{\text{dB}} = 4.0\text{ dB}$$
     - Tại các tần số bị cắt sâu ($r_{\text{dB}} \ll -18\text{ dB}$): $M_{\text{band}}(f) \to 0$ (lấy 100% phần tái tạo từ `s`).
     - Tại dải tần còn nguyên vẹn ($r_{\text{dB}} \approx 0\text{ dB}$): $M_{\text{band}}(f) \to 1$ (lấy từ `x` đã chỉnh EQ).
  4. **QUAN TRỌNG:** Khi chạy Module 1 (`biquad_eq`) trên `x`, truyền `freq_weights = M_band(f)` để PEQ **chỉ khớp EQ trong dải còn tín hiệu**, tuyệt đối không cố khuếch đại +18 dB vào vùng tần số đã chết!
  5. Mở rộng $M_{\text{band}}(f)$ theo trục thời gian và trộn bù STFT bằng [`mix_complementary_stft`](file:///d:/SonicMaster_Reproducing/scripts/extension/mix.py) (`y = M_band * p_out + (1 - M_band) * s`).

---

## 5. Lộ Trình Thực Thi Từng Bước & Tiêu Chí Nghiệm Thu

Agent thực thi tuần tự theo **4 Phase** dưới đây. Không được nhảy cóc sang Confirm khi chưa chốt Dev Gate.

### Phase A: Cài đặt Unit Tests & Sanity Checks (`scripts/extension/tests/test_v3_dsp.py`)
Viết và chạy `tools\extension-env\Scripts\pytest.exe scripts/extension/tests/test_v3_dsp.py` đảm bảo pass 100%:
1. **Biquad exactness:** Hàm `freqz` trong PyTorch khớp `scipy.signal.sosfreqz` với sai số max `< 1e-5` cho cả 5 loại (`HP, LP, LS, HS, PK`).
2. **Identity test:** Khi `u == s`, `biquad_eq`, `diff_drc_joint`, và `xband_crossover` đều trả về tín hiệu gần đồng nhất (`SNR > 45 dB`, không có NaN/Inf).
3. **Synthetic `mud` recovery:** Tạo tín hiệu nhiễu hồng/nhạc mẫu bị boost `+10 dB` tại `350 Hz (Q=1.5)`; `biquad_eq` phải khôi phục với sai số băng `200–500 Hz` `< 0.01` và `SSIM > 0.99`.
4. **Zero-phase gain check:** Nếu test `sosfiltfilt`, kiểm tra đáp ứng biên độ thực đo của `sosfiltfilt(x)` khớp `|H|` (không bị nhân đôi dB).
5. **DRC compression + expansion test:** Kiểm tra `diff_drc_joint` giảm rõ sai số `std(frame_rms_dB)` cho cả 2 chiều: tín hiệu quá dynamic (cần nén) và tín hiệu bị nén bẹp (cần giãn).
6. **`xband` mask test:** Với tín hiệu bị low-pass ở `3 kHz`, `M_band(f)` phải `> 0.95` dưới `2.5 kHz` và `< 0.05` trên `4 kHz`.

### Phase B: Ablation Tuần Tự Trên 280 Mẫu Dev
Tạo script `scripts/extension/run_hybrid_v3_dev.py` hỗ trợ cờ `--mode {v3a, v3b, v3c}` và `--eq-phase {minphase, zerophase}`, xuất WAV ra `artifacts/10_extension/dev/hybrid_v3_<mode>/out_wav/`.

- **Step B1 — `V3-A` (Module 1 + mở khóa `mud`):**
  - Chạy trên 280 Dev (so `minphase` vs `zerophase`).
  - Chấm `eval_dev.py` và `eval_tables123.py`.
  - **Tiêu chí giữ lại:**
    - `Violations == []` (**bắt buộc `mud` pass `norm_det ≤ +0.05`**).
    - Dev `KL×1000 < 1.1215` (cải thiện so với PA1) và Dev `SSIM > 0.7120`.
- **Step B2 — `V3-B` (`V3-A` + Module 2 Joint DRC/Expander):**
  - Chạy trên 280 Dev (đo thời gian trung bình/clip để dự toán cho Confirm).
  - Chấm `eval_dev.py` và `eval_tables123.py`, so sánh riêng phân nhóm **`Double+Triple`** và 3 thuộc tính `comp`, `punch`, `stereo` so với `V3-A`.
  - **Tiêu chí giữ lại:**
    - Cải thiện rõ `KL×1000` (hoặc `comp`/`punch` guard) trên `Double+Triple`, giữ `Violations == []`, và tổng thời gian dự kiến cho Confirm `< 45 phút`.
    - Nếu `V3-B` không cải thiện đủ rõ hoặc quá chậm: loại bỏ Module 2 và giữ `V3-A` làm nền cho Step B3.
- **Step B3 — `V3-C` (Best of B1/B2 + Module 3 `xband` Crossover):**
  - Chạy trên 280 Dev.
  - Chấm `eval_dev.py` và `eval_tables123.py`.
  - **Tiêu chí giữ lại:**
    - `xband` Attribute Guard `≤ +0.05` (`Violations == []`), Dev `SSIM` tăng rõ (`≥ +0.03` thêm), Dev `KL×1000 ≤ 1.05× A0`.

### Phase C: Nghiệm Thu Đầy Đủ 5/5 Dev Gates & Khóa `lock_v3.json`
Với phiên bản thắng cuộc trên Phase B (`V3-best`):
1. Cập nhật `scripts/extension/kaggle_dev_pq_fad.py` (đổi `ds_slug = "sonicmaster-ext-dev-pq-fad-v3"`, trỏ tới `out_wav` của `V3-best`, giữ bước kiểm tra `KAGGLE_SAMPLE_SHA256 == LOCAL_SHA256`).
2. Xác nhận **PASS toàn bộ 5/5 Dev Gates** VÀ **vượt PA1** (đặc biệt phải có biên độ an toàn KL tốt hơn PA1 và cải thiện `Double+Triple` PQ hoặc độ phủ SSIM).
3. *(Khuyến nghị)* Xuất 10 clip Dev vocal (so `A0` vs `PA1` vs `V3-best`) cho người dùng nghe kiểm chứng nhanh nếu thay đổi lớn.
4. Ghi khóa cấu hình vào `artifacts/10_extension/lock_v3.json` (ghi rõ commit/hash code, siêu tham số từng module, và đủ 5 chỉ số Dev Gate).

### Phase D: Chạy Confirm Một Lần Duy Nhất (`6.720` Mẫu) & Báo Cáo Cuối
1. Viết `scripts/extension/run_hybrid_v3_confirm.py`:
   - Tái sử dụng trực tiếp 282 file stem tại `artifacts/10_extension/confirm/hybrid_v2/vocal_stems_flac/` (**0 phút chạy `bs_roformer`**).
   - Dùng `ProcessPoolExecutor` cho các nhánh CPU (`S`, `V3-A`, `vocal_mask_P`, `xband`) và GPU local cho nhánh Joint DRC (nếu `V3-B` được giữ).
2. Chạy đánh giá nhanh `eval_confirm_fast.py` (cập nhật `S_FALLBACK` theo `lock_v3.json` để chấm đúng toàn bộ các mẫu thay đổi và ghép với `14_run7000` cho phần `S` còn lại).
3. Chạy `kaggle_confirm_pq_fad.py` (đổi `ds_slug = "sonicmaster-ext-confirm-pq-fad-v3"`, upload **chỉ các mẫu thay đổi**, `pip install` trước `import numpy`, in `SHA-256` xác thực), rồi ghép với `s01..s16/clap_embeddings.npz` & `per_sample_pq_7000.csv`.
4. Tính bảng tổng hợp **Toàn tập Confirm (`6.720`)**, **Riêng non-fallback**, **Riêng `vocal_mask_P`**, **Riêng `mud`**, **Riêng `xband`**, đối chiếu trực tiếp 3 cột: **`A0` | `Hybrid-v2 PA1` | `Hybrid-v3`**.
5. **Dọn dẹp ổ `D:\`:** Xóa `artifacts/10_extension/confirm/hybrid_v3/out_wav/` (~36 GB) và các folder tạm sau khi đã kiểm tra đủ JSON/CSV/NPZ.
6. Xuất báo cáo cuối tại `d:\SonicMaster_Reproducing\HYBRID_V3_REPORT.md`.

### Phase E: Blind / Zero-Label Routing (`scripts/extension/blind_router.py`) — Không Dùng Nhãn Suy Hao Hay Prompt
Mục tiêu: Chứng minh **Hybrid-v3** vẫn hoạt động hiệu quả và an toàn ngay cả trong kịch bản **mù hoàn toàn** (chỉ có cặp `(x, s)`, **không đọc `degradation_types` lẫn `prompt`**).

Triển khai và đối chiếu 2 cơ chế mù trên **Dev (`280` mẫu)** trước khi chạy Confirm (tuyệt đối không tune ngưỡng trên Confirm):

1. **Cơ chế E1 — Blind Residual Fit Gate (Tự kiểm chứng qua phần dư `y_P` vs `s`):**
   - Với mọi mẫu vượt qua guard mù `20·log10(rms(s)/rms(x)) ≤ 24.0 dB`:
     1. Tính trước mặt nạ cắt băng thông mù `M_band(f)` (Module 3). Nếu tỷ lệ băng bị cắt sâu (`r_dB(f) < -18 dB`) vượt ngưỡng (vd `> 10%` băng tần hoạt động), kích hoạt nhánh `xband_crossover`.
     2. Chạy thử `y_P = Branch_P_v3(x, s)` (Module 1 + Module 2).
     3. Tính bộ chỉ số **phần dư tự kiểm chứng** giữa chính `y_P` và `s` (hoàn toàn mù):
        - $d_{\text{KL}} = \text{Mel-KL}(y_P, s) \times 1000$
        - $d_{\text{mod}} = \|\text{ModSpec}(y_P) - \text{ModSpec}(s)\|_2$ (bắt chính xác các ca **reverb** mà DSP không xóa được đuôi vang!)
        - $d_{\text{flat}} = |\text{SpecFlatness}(y_P) - \text{SpecFlatness}(s)|$ (bắt các ca **clip** bị méo phi tuyến!)
        - $d_{\text{rms}} = |\text{std}(\text{RMS}_{\text{dB}}(y_P)) - \text{std}(\text{RMS}_{\text{dB}}(s))|$
     4. Chọn ngưỡng $(\tau_{\text{KL}}, \tau_{\text{mod}}, \tau_{\text{flat}})$ **chỉ trên 280 mẫu Dev** (tối đa hóa recall trên nhóm tuyến tính với `0` vi phạm Attribute Guards): nếu vượt bất kỳ ngưỡng nào → **tự động fallback sang `s`** (hoặc dùng `vocal_mask_P` nếu chỉ lệch cục bộ ở dải giọng).
2. **Cơ chế E2 — Blind Time–Frequency Confidence Mask $M(t, f)$ (Trộn mềm không cần `if/else` cứng):**
   - Sau khi fit `y_P`, tính độ lệch biên độ log-STFT cục bộ giữa `Y_P(t, f)` và `S(t, f)`, làm mượt theo thời gian (`~50 ms`) và 1/3-octave:
     $$M(t, f) = \text{clip}\!\left(\sigma\!\left(\frac{\tau - \text{smooth}\!\left(\left|\log|Y_P(t,f)| - \log|S(t,f)|\right|\right)}{\Delta}\right),\; m_{\min},\; m_{\max}\right)$$
   - Trộn bù STFT: $Y(t, f) = M(t, f)\cdot Y_P(t, f) + (1 - M(t, f))\cdot S(t, f)$.
   - *Lưu ý kỹ thuật:* So sánh trực tiếp trên Dev để đảm bảo không sinh *musical noise* (tiếng "lọc nước") và kiểm tra xem E1 (hard residual gate) hay E2 (soft TF mask) cho `SSIM`, `KL`, `PQ`, `FAD` và `15 Guards` sạch hơn.
3. **Nghiệm thu Phase E:**
   - Báo cáo song song 2 hàng trong `HYBRID_V3_REPORT.md`:
     - **`Hybrid-v3 (Prompt/Label-routed)`** (có prompt đầu vào chính thức của SonicMaster).
     - **`Hybrid-v3-Blind (Zero-Label)`** (không dùng nhãn lẫn prompt — chỉ dùng `(x, s)`).
   - Tiêu chí thành công: **`Hybrid-v3-Blind`** phải **PASS đủ 5/5 cổng** trên cả Dev (`280`) và Confirm (`6.720`) với độ suy giảm SSIM nhỏ so với bản có prompt (`ΔSSIM` vẫn vượt xa `A0` và `A5`).

---

## 6. Checklist Kiểm Tra Nhanh Cho Agent Trước Khi Báo Cáo Hoàn Thành
- [ ] `pytest scripts/extension/tests/test_v3_dsp.py` pass 100%.
- [ ] Không có bất kỳ đoạn code suy luận nào đọc `path_clean`.
- [ ] Ablation `V3-A` → `V3-B` → `V3-C` trên Dev (`280`) có đầy đủ số liệu bảng so sánh với `A0` và `PA1`.
- [ ] Phase E (`Hybrid-v3-Blind`) hoàn toàn không đọc `degradation_types` và `prompt`, chỉ tune ngưỡng trên Dev (`280`), và được đối chiếu song song với bản có prompt.
- [ ] `KAGGLE_SAMPLE_SHA256 == LOCAL_SHA256` khớp tuyệt đối cho cả Dev lẫn Confirm PQ/FAD.
- [ ] `eval_tables123.py` và `eval_confirm_fast.py` báo đủ 15 (Dev) / 19 (Confirm) thuộc tính với `Violations == []`.
- [ ] `artifacts/10_extension/lock_v3.json` được khóa trước khi chạy Confirm.
- [ ] Đã dọn `out_wav/` dung lượng lớn trên `D:\` và lưu `HYBRID_V3_REPORT.md` ở thư mục gốc dự án.
