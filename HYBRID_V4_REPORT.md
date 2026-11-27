# Báo Cáo Chi Tiết Kiến Trúc & Thực Nghiệm Hybrid-v4 (`V4-0` → `V4-4`)

**Dự án:** Tái lập & Mở rộng *SonicMaster* (`Codec-Bypass Neural Music Restoration`)  
**Phạm vi báo cáo:** Toàn bộ chuỗi phiên bản **`V4-0`, `V4-1`, `V4-2`, `V4-3`, `V4-4`** (và lộ trình `V4-5`), kèm giải thích trực quan, thông số kỹ thuật, nguồn gốc bài báo (References), và đối chiếu trực tiếp với hệ thống **NPU-ASLP (CCF-AATC 2026)**.

![Sơ đồ kiến trúc Hybrid-v4](file:///d:/SonicMaster_Reproducing/artifacts/10_extension/figures/hybrid_v4_architecture.jpg)

---

## 1. Tóm Tắt Điều Hành & Bối Cảnh Bài Toán

### 1.1. Tại sao cần `Hybrid-v4`?
- **SonicMaster (`A0`, đầu ra gọi là `s`)** giải quyết tốt các suy hao phi tuyến và mất thông tin (khử reverb, sửa clipping, phục hồi dải tần `xband`, điều khiển qua prompt), đạt `Audiobox PQ = 7.645` trên Dev `280`. Tuy nhiên, vì `s` đi qua bộ mã hóa/giải mã **VAE của Stable Audio Open 1.0**, pha và kết cấu chi tiết của sóng gốc bị làm mờ (`Mel-SSIM` giảm xuống `~0.58–0.61`).
- **Âm thanh suy hao đầu vào (`x`)** tuy bị sai EQ, nén hoặc vang, nhưng **giữ nguyên 100% độ sắc nét và pha gốc** (không bị tổn hao codec).
- **`Hybrid-v4` (Codec-Bypass)** không thay thế SonicMaster, mà dùng một mạng neural nhỏ gọn (`0.99M – 1.95M` tham số) **lấy `x` làm nền chính, nhìn `s` làm bản mẫu gợi ý**, sửa trực tiếp trên `x` ở dạng STFT tuyến tính không nén và chỉ pha `s` hoặc bù phần dư `R` ở những vùng/kênh `x` thực sự mất thông tin.

### 1.2. Thiết lập Dữ liệu & Đánh giá (Chống Rò Rỉ Tuyệt Đối)
Theo [pilot_splits.json](file:///d:/SonicMaster_Reproducing/artifacts/10_extension/v4/pilot_splits.json) (khởi tạo bằng [v4_prepare_splits.py](file:///d:/SonicMaster_Reproducing/scripts/extension/v4_prepare_splits.py), `seed = 20261122`), toàn bộ trích từ bộ test 7.000 mẫu của SonicMaster với **tách nguồn tuyệt đối (`overlap = 0`)**:
- **Pilot Train:** `400` bài nguồn (`2,800` cặp clip 5.0s) — đóng gói trên Kaggle dataset `aajayagh/sonicmaster-v4-pilot-5s`.
- **Pilot Val:** `80` bài nguồn (`560` cặp clip) — dùng để chọn checkpoint `best.pt`.
- **Pilot Test (`Dev 280`):** `40` bài nguồn × 7 variants = `280` clip 30s — hoàn toàn độc lập với train/val, trùng khớp 1-1 với tập Dev của `A0` và `Hybrid-v3`.
- **Locked Confirm:** `280` bài nguồn (`1,960` cặp) — khóa kín, chỉ mở 1 lần cuối cùng sau khi chốt kiến trúc.

---

## 2. Nguồn Gốc Kỹ Thuật & Bài Báo Tham Chiếu (Method Provenance)

Mỗi thành phần trong `Hybrid-v4` đều được kế thừa từ các công trình đã công bố và điều chỉnh cho bài toán có teacher `s`:

| Thành phần trong `Hybrid-v4` | Bài báo / Nguồn gốc chính xác | Đóng góp cụ thể vào `V4` |
|---|---|---|
| **Bài toán, SonicMaster `s` (`A0`), Dataset & Bộ metric Bảng 1–3** | J. Melechovsky, A. Mehrish, A. Roy, D. Herremans, *"SonicMaster: Towards Controllable All-in-One Music Restoration and Mastering"*, ICML 2026 ([arXiv:2508.03448](https://arxiv.org/abs/2508.03448); [GitHub](https://github.com/AMAAI-Lab/SonicMaster)) | Cung cấp đầu ra dẫn đường `s`, dữ liệu `x/s/clean`, và bộ hàm chấm EQ, Dynamics, Reverb, Clip, Stereo, KL, SSIM. |
| **Scale-Aware Band Encoder, Complex Mapping (`V4-0/1`), `TF-MoE` (`V4-5`) & VGGish warm-up** | ASLP Group (**NPU-ASLP**), *"The NPU-ASLP System for Music Restoration in the CCF-AATC 2026 Challenge"*, Oct 2026 | Cung cấp ý tưởng chuẩn hóa theo dải tần giữ lại log-RMS (`Scale-Aware`), ánh xạ phổ phức trực tiếp (`DirectMap`), khung xương `TF-MoE Top-1`, và chiến lược warm-up perceptual loss. |
| **24-Band Split & Dual-Axis Time–Frequency Modeling** | Y. Luo & J. Yu, *"Music Source Separation with Band-split RNN"* (BSRNN), IEEE/ACM TASLP 2023 ([arXiv:2209.15174](https://arxiv.org/abs/2209.15174)); Z.-Q. Wang et al., *"TF-GridNet: Integrating Full- and Sub-Band Modeling for Speech Separation"*, IEEE/ACM TASLP 2023 ([arXiv:2211.12433](https://arxiv.org/abs/2211.12433)) | Thiết kế chia dải tần không đều theo thính giác (`24 bands`) và xử lý luân phiên theo 2 trục: xuyên dải tần (`Cross-Band Attention`) + theo thời gian (`Temporal BiGRU`). |
| **Complex Multi-Frame Filtering (`MF7` / `MF9`) & Psychoacoustic Sub-bands** | H. Schröter et al., *"DeepFilterNet: A Low Complexity Speech Enhancement Framework for Full-Band Audio based on Deep Filtering"*, IEEE ICASSP 2022 ([arXiv:2110.05588](https://arxiv.org/abs/2110.05588)) | Cơ sở cho bộ lọc phức đa khung `ΔW_k` tác động trực tiếp lên STFT tuyến tính thay cho mặt nạ điểm đơn. |
| **Delayed / Dilated Multi-Frame Taps (`V4-3`, ~220 ms)** | T. Nakatani et al., *"Speech Dereverberation Based on Variance-Normalized Delayed Linear Prediction"* (**WPE**), IEEE TASLP 2010 | Nguyên lý dùng các khung quá khứ giãn cách để dự đoán và triệt tiêu đuôi vang (`late reverberation`) mà không làm méo âm trực tiếp. |
| **Multi-Resolution STFT Loss** | R. Yamamoto, E. Song, J.-M. Kim, *"Parallel WaveGAN"*, IEEE ICASSP 2020 ([arXiv:1910.11480](https://arxiv.org/abs/1910.11480)) | Hàm mất mát đa phân giải (`512, 1024, 2048`), được mở rộng tính song song trên cả `(L, R)` và `(Mid, Side)`. |
| **Perceptual Quality (`PQ`)** | A. Tjandra et al. (Meta FAIR), *"Meta Audiobox Aesthetics: Unified Automatic Quality Assessment for Speech, Music, and Sound"*, 2025 ([arXiv:2502.05139](https://arxiv.org/abs/2502.05139)) | Thước đo chất lượng cảm nhận tự động `PQ`. |
| **FAD-CLAP & Mel-SSIM** | Y. Wu et al., *"LAION-CLAP"*, IEEE ICASSP 2023 ([arXiv:2211.06687](https://arxiv.org/abs/2211.06687)); A. Gui et al., *"Adapting Fréchet Audio Distance for Generative Music Evaluation"* (`fadtk`), IEEE ICASSP 2024 ([arXiv:2311.01616](https://arxiv.org/abs/2311.01616)); Z. Wang et al., IEEE TIP 2004 (`SSIM`) | Đánh giá khoảng cách phân phối ngữ nghĩa âm thanh (`FAD-CLAP`) và độ tương đồng cấu trúc phổ (`Mel-SSIM`). |

---

## 3. Giải Thích Chi Tiết Kỹ Thuật, Thông Số & Ý Nghĩa (Từ Dễ Hiểu Đến Chuyên Sâu)

### 3.1. Tiền xử lý & Chuẩn hóa (`Track Norm + Lag Align`)
| Thông số / Kỹ thuật | Giải thích dễ hiểu (cho người ngoài ngành) | Ý nghĩa kỹ thuật & Giá trị thực tế |
|---|---|---|
| **`estimate_lag_samples(x, s, max_lag=512)`** | Dịch chuyển nhẹ file `s` sang trái/phải vài phần nghìn giây để sóng của `s` và `x` trùng nhịp nhau. | VAE của SonicMaster có thể làm trễ/lệch pha tuyến tính vài chục mẫu; căn lag trước giúpEncoder và phép trộn `(1-g)F_W + g Z_s` không bị triệt tiêu pha (`comb filtering`). |
| **`compute_track_norm(x, s)`** | Đưa cả bài về cùng độ to chuẩn `RMS = 1` trước khi cắt khúc, tránh chỗ to chỗ nhỏ khi ghép lại. | Chuẩn hóa 1 lần cho toàn bài, kẹp hệ số căn `g_align ∈ [0.1, 16.0]` (`±24 dB`) để không khuếch đại nhiễu lượng tử ở bài quá nhỏ. |
| **`raw_stats` (3 chiều)** | 3 con số tóm tắt độ to gốc của bài (`log_rms_x`, `log_rms_s`, `gain_dB`) đưa cho mạng biết. | Giúp mạng không bị "mù âm lượng" sau khi chuẩn hóa, biết bài gốc bị nhỏ hay to bất thường so với `s`. |

### 3.2. Biểu diễn Âm thanh & Mã hóa (`PsychoacousticBandEncoder`)
| Thông số / Kỹ thuật | Giải thích dễ hiểu | Ý nghĩa kỹ thuật & Giá trị trong [v4_models.py](file:///d:/SonicMaster_Reproducing/scripts/extension/v4_models.py) |
|---|---|---|
| **`lr_to_ms` (`Mid / Side`)** | Tách âm thanh 2 tai `Trái/Phải` thành **`Mid`** (âm thanh ở giữa: giọng hát, bass, trống) và **`Side`** (âm thanh 2 bên tạo độ rộng Stereo). | `M = (L+R)/√2`, `S = (L-R)/√2`. Cho phép chỉnh độ rộng stereo trực tiếp mà không làm hỏng giọng hát ở giữa. |
| **`n_fft=2048, hop=512`** (`44.1 kHz`) | Cắt bài nhạc thành từng lát cắt ngắn (khung thời gian), mỗi khung cách nhau `~11.6 ms`, phân tích thành `1,025` tần số. | Cửa sổ Hann ~46.4 ms cho độ phân giải tần số ~21.5 Hz/bin, đủ mịn để chỉnh EQ bass/mid/treble và khử vang. |
| **Nén biên độ `|Z|^0.3`** *(chỉ ở Encoder)* | Làm các âm cao vốn rất nhỏ trở nên dễ nhìn hơn đối với bộ não mạng. | Tín hiệu dải bass mạnh hơn dải >10 kHz hàng chục lần; nén mũ `0.3` giúp Encoder thấy rõ lỗi treble. **Quan trọng:** tín hiệu đi xuống Decoder để lọc vẫn là STFT tuyến tính không nén! |
| **`BAND_WIDTHS` (`24 dải`)** | Chia `1,025` tần số thành **24 nhóm** theo cách tai người nghe (chia rất mịn ở âm trầm, gộp rộng ở âm rất cao). | `8` dải cho `0–1,012 Hz` (`5–6` bin/dải), `8` dải cho `1–4 kHz` (`17–18` bin/dải), `5` dải cho `4–12 kHz` (`74–75` bin/dải), `3` dải cho `12–22 kHz` (`155–156` bin/dải). Tổng `= 1,025` bins. |
| **Dual Scale-Aware** | Chuẩn hóa phổ từng dải về đơn vị nhưng kẹp thêm chỉ số độ to log-RMS và chênh lệch `M/S` của cả `x` và `s`. | Học từ NPU-ASLP nhưng mở rộng cho **cặp `(x, s)`**: mạng biết chính xác dải nào `s` đang to hơn `x` (ví dụ `x` bị tối hoặc mất dải cao). |
| **`embed_dim = 64`** | Mỗi dải tần tại 1 thời điểm được tóm tắt thành 1 vector `64` con số. | Cân bằng giữa sức biểu đạt và độ nhẹ (`< 2M` tham số). |

### 3.3. Bộ Não Xử Lý 2 Trục (`4 × DualAxisBlock`)
- Gồm **4 block** nối tiếp (lấy cảm hứng từ **BSRNN** và **TF-GridNet**), tổng `433,152` tham số:
  1. **`Cross-Band Multi-Head Self-Attention` (`4 heads`):** Tại cùng 1 thời điểm, 24 dải tần "trao đổi" với nhau (ví dụ: *bass đang ù thì dải trung/cao cần bù thế nào*).
  2. **`Temporal BiGRU` (2 chiều theo thời gian):** Tại cùng 1 dải tần, nhìn xuyên suốt quá khứ và tương lai để phân biệt tiếng gõ ngắn (punch) với đuôi tiếng vang kéo dài (reverb).
  3. **`Dense FFN` (`64 → 256 → 64`, `GELU`):** Tổng hợp phi tuyến ra đặc trưng điều khiển cho Decoder.

### 3.4. Bộ Giải Mã (`Decoder` — Điểm cốt lõi làm nên từng Version)
- **`ΔW_0` (Ma trận phức 2×2 trên `Mid/Side` tại `k=0`):**
  - *Dễ hiểu:* Tại đúng khung hiện tại, mạng tạo ra 4 số phức `[MM, MS; SM, SS]` (8 kênh thực/ảo) nhân vào `(Z_x_M, Z_x_S)`. Nó vừa chỉnh được độ to + pha của `Mid` và `Side`, vừa trộn chéo `Mid ↔ Side` khi cần.
  - *An toàn:* Giới hạn bằng `2.0 * tanh(...)` và **khởi tạo bằng `0`** (`zero-init`) để tại bước 0 mô hình giữ nguyên `Z_x` (`Identity-Centered`).
- **`MF7_OFFSETS` vs `MF9_DILATED_OFFSETS` (Các khung thời gian `k ≠ 0`):**
  - *Dễ hiểu:* Tiếng vang (`reverb`) là âm thanh từ quá khứ vọng lại đè lên hiện tại. Muốn khử vang, bộ lọc phải nhìn lại các khung quá khứ (`k < 0`) để trừ phần vọng đó đi (nguyên lý **WPE**).
  - `V4-2`: `7` khung liên tiếp `{-5, -4, -3, -2, -1, 0, +1}` → tầm nhìn `~70 ms` (chỉ đủ cho phòng nhỏ).
  - `V4-3 / V4-4`: `9` khung giãn cách `{-18, -12, -8, -5, -3, -2, -1, 0, +1}` → mở rộng tầm nhìn lên **`~220 ms`** (`-209 ms` quá khứ + `11.6 ms` tương lai) mà chỉ thêm `4,160` tham số ở `mf_head` (chia sẻ theo 96 sub-band = 4 sub-band × 24 dải).
- **Cổng trộn `g ∈ (0, 1)` (`g_M`, `g_S`):**
  - Công thức: `(1 - g) · F_W(Z_x) + g · Z_s`.
  - `g ≈ 0`: giữ nguyên sóng sắc nét từ `x` (không qua VAE). `g` lớn: mượn `s` ở các ô bị clipping vỡ hoặc reverb quá nặng.
  - **`gate_bias_init`:** Trong `V4-2` đặt `-3.5` (`sigmoid(-3.5) ≈ 0.029`, ép đóng rất chặt) cộng phạt `λ_g = 0.002`; sang `V4-3/V4-4` nới lên `-1.5` (`sigmoid(-1.5) ≈ 0.182`) và đặt **`λ_g = 0.0` (Free Gate)** để mạng tự quyết định khi nào cần mở.
- **Phần dư cộng trực tiếp `R_θ(Z_x, Z_s)` (riêng `V4-4`):**
  - Công thức: `Z_hat = (1 - g) · F_W + g · Z_s + R`.
  - *Dễ hiểu:* Phép nhân `ΔW · Z_x` không thể tạo ra tín hiệu từ số `0` (khi kênh `Side` bị sụp về 0 ở `stereo`, hoặc dải cao bị cắt cụt ở `bright`). Nhánh `R` cho phép mạng **tự vẽ thêm phần sóng phức còn thiếu rồi cộng thẳng vào đầu ra**.
  - *An toàn:* `R` được **zero-init** và nhân với **`band_rms`** (độ lớn trung bình của chính dải tần đó) để không cộng quá tay ở dải treble vốn có năng lượng nhỏ.

### 3.5. Hàm Mất Mát (`HybridV4Loss` trong [v4_losses.py](file:///d:/SonicMaster_Reproducing/scripts/extension/v4_losses.py))
| Thành phần Loss | Trọng số (`V4-2` / `V4-3` / `V4-4`) | Giải thích dễ hiểu & Vai trò |
|---|---|---|
| **`MultiResSTFTLoss`** (`512, 1024, 2048`) | `1.0` (mọi bản) | So độ lệch phổ ở 3 độ mịn thời gian/tần số khác nhau trên cả `(L, R)` và `(M, S)`. |
| **`CompComplex L1`** (`\|Z\|^0.3`) | `0.5` (mọi bản) | Ép cả phần thực và phần ảo (tức cả biên độ lẫn pha) khớp với bản sạch `Clean`. |
| **`Wav L1`** | `0.5` (mọi bản) | Giữ dạng sóng thời gian sát bản sạch. |
| **`MelKLLoss`** *(từ `V4-3`)* | `0.0` / `2.0` / `2.5` | So phân bố năng lượng trên 128 băng Mel (giống công thức `KL` lúc chấm thi) + log-mel L1 để cân bằng màu âm EQ. |
| **`l_ms`** (`RMS(Side)/RMS(Mid)`) *(Stage 2, từ Ep 13)* | `0.2` / `0.5` / `1.0` | Ép độ rộng Stereo (`tỷ lệ Side/Mid`) của đầu ra khớp với bản sạch. |
| **`l_g`** (`mean(g)`) | `0.002` / `0.0` / `0.0` | Phạt đóng gate ở `V4-2`; bỏ (`0.0`) ở `V4-3/V4-4` để không kìm cổng ở Reverb/Clip. |
| **`l_w`** (`dw0_rms^2 + dwk_rms^2`) | `0.005` (mọi bản) | Giữ bộ lọc `ΔW` nhỏ gọn, ổn định, tránh nhiễu. |

### 3.6. Ý Nghĩa Các Chỉ Số Đánh Giá (Evaluation Metrics)
- **`Mel-SSIM ↑`:** Độ giống nhau về cấu trúc hình ảnh phổ Mel so với bản sạch (`Clean`), thang `[0, 1]`. Càng cao nghĩa là sóng âm càng sắc nét, ít bị làm mờ bởi VAE.
- **`Mel-KL × 1000 ↓`:** Độ lệch phân bố năng lượng tổng thể qua 128 dải Mel so với `Clean`. Càng thấp nghĩa là cân bằng âm sắc (EQ) càng chuẩn.
- **`FAD-CLAP ↓`:** Khoảng cách phân phối ngữ nghĩa âm nhạc (dùng embedding LAION-CLAP tính một lần trên toàn tập). Càng thấp nghĩa là âm nhạc nghe càng tự nhiên, giống nhạc sạch thật.
- **`Audiobox PQ ↑`:** Điểm chất lượng cảm nhận tự động (Production Quality) của Meta Audiobox Aesthetics (`ΔPQ = PQ(model) - PQ(A0)`).
- **`15 Attribute Guards` (`norm_det` vs `A0`):** Kiểm tra từng loại lỗi cụ thể (`boom`, `bright`, `clip`, `comp`, `dark`, `mic`, `mud`, `punch`, `stereo`, `vocal`, `volume`, `xband`, và 3 nhóm `low-n`). `norm_det < 0` là **tốt hơn cả SonicMaster `A0`**; vượt `> +0.050` (khi `n >= 10`) bị đánh dấu **FAIL**.

---

## 4. Toàn Cảnh Tiến Hóa Các Phiên Bản `V4-0` → `V4-4` (Kết Quả Thực Tế Trên `Dev 280`)

![Biểu đồ tiến trình Hybrid-v4](file:///d:/SonicMaster_Reproducing/artifacts/10_extension/figures/v4_metrics_progression.png)

![Bảng tổng hợp kết quả Hybrid-v4](file:///d:/SonicMaster_Reproducing/artifacts/10_extension/figures/v4_results_table.png)

### Chi tiết từng bước chuyển đổi
1. **`V4-0` (`NoGuide`, `994,828` params):** Chỉ đưa `x` vào tự đoán STFT sạch (`DirectMap`). Kết quả `SSIM = 0.7495`, `KL = 5.1708` → chứng minh mạng ~1M không có `s` dẫn đường sẽ thất bại.
2. **`V4-1` (`DirectMap`, `1,270,180` params):** Đưa thêm `s` vào Encoder, vẫn tự vẽ lại toàn bộ STFT. Kết quả cải thiện (`SSIM = 0.8023`, `KL = 2.6238`) nhưng `FAD = 0.2507` (kém `A0 = 0.2083`), `PQ = 7.0820`, rớt **`10/12` Guard** do mất chi tiết sóng gốc khi vẽ lại từ đầu.
3. **`V4-2` (`MF7 + Gate`, `1,682,410` params):** Bỏ `DirectMap`, giữ đường tắt `Z_x` + lọc phức `2×2 M/S` 7 khung (`-5..+1`) + Gate `g`. Kết quả bứt phá mạnh: `SSIM = 0.8325`, `KL = 1.2963`, `FAD = 0.1848` (vượt `A0`), `PQ = 7.3530`, lỗi Guard giảm còn **`6/12`**. Lộ 3 điểm yếu: gate bị ép đóng (`g_reverb = 0.0952`), thiếu `Mel-KL`, và lệch `volume (+0.087)`.
4. **`V4-3` (`Dilated MF9 + FreeGate + MelKL + RMS`, `1,686,570` params):** Mở rộng tầm lọc lên 9 tap giãn cách (`~220 ms`), mở tự do gate (`λ_g=0`), thêm `MelKLLoss (λ_kl=2.0)`, `λ_ms=0.5`, và khớp RMS đầu ra theo `rms(s)`.
   - Sửa sạch `4/6` guard (`boom -0.011`, `mic +0.012`, `vocal -0.030`, `volume +0.005`), chỉ còn **`2/12` lỗi** (`bright +0.194`, `stereo +0.177`).
   - Đạt **`FAD` tốt nhất (`0.1775`)** và **`PQ` cao nhất (`7.3809`)**, `Reverb_106 ΔPQ` cải thiện từ `-0.7072 → -0.6439`.
5. **`V4-4` (`MF9 + Band-Scaled Additive Complex Residual R`, `1,953,070` params):** Thêm nhánh cộng trực tiếp `R` (+266.5k params), tăng `λ_ms = 1.0`, `λ_kl = 2.5`.
   - Đạt **`KL` thấp nhất (`1.0468`)**, `SSIM` lên **`0.8284`**, cắt gần nửa lỗi **`stereo` (`+0.177 → +0.094`)** và giảm 30% lỗi **`bright` (`+0.194 → +0.135`)**. **10/12 guard PASS** (6 thuộc tính vượt `A0`: `punch -0.151`, `comp -0.075`, `clip -0.029`, `boom -0.028`, `xband -0.026`, `vocal -0.007`).
   - *Đánh đổi:* `R` thay thế một phần vai trò của `s` khiến `g_reverb` thu về `0.0958` (vs `0.1392` ở `V4-3`), làm `PQ` (`7.3339`) và `FAD` (`0.1865`) nhỉnh kém nhẹ so với `V4-3` *(tái chấm matched CLAP ở `residual_sweep_dev` đạt `FAD = 0.1304` vs `A0 = 0.1674`)*.
6. **`V4-4bs` (`V4-4 + HighFreqBrightLoss + M/S từ Epoch 1`, `1,953,070` params):** Bật `l_ms` ngay từ Epoch 1 (`--stage2-epoch 1`, `--lambda-ms 1.5`), thêm `HighFreqBrightLoss` (`>= 6 kHz`, `--lambda-hf 1.5`), giữ `--lambda-kl 2.5`, Free Gate (`--lambda-g 0.0`, `--gate-bias-init -1.5`).
   - **Cải thiện `stereo` (`n=40`):** **Stereo cải thiện rõ (~16.4% về sai số thô so với `V4-4` cũ: `0.1131 → 0.0945`), còn cách ngưỡng khoảng `0.0018` theo giao thức chấm chung** (`+0.05184` theo [matched_dev_v4_4bs/summary.json](file:///d:/SonicMaster_Reproducing/artifacts/10_extension/v4/matched_dev_v4_4bs/summary.json) do mẫu số input là `0.40720`, vs `+0.04994` vừa đạt ở script `eval_tables123.py` có mẫu số input `0.42272`; trong khi `V4-4` cũ là `+0.09745` matched / `+0.09387` legacy). Theo giao thức chấm chung có `2/12` guard vượt `0.05` (`bright = +0.1286`, `stereo = +0.0518`); theo `eval_tables123.py` còn `1/12` (`bright = +0.1619`).
   - **Chỉ số toàn tập (`Dev 280`):** `SSIM = 0.8076`, `KL×1000 = 1.1229`, **matched `FAD-CLAP = 0.1315`** (tốt hơn `A0 = 0.1674` `-21.5%`), `Audiobox PQ = 7.3322` (`ΔPQ vs A0 = -0.3131`, CI95 `[-0.3966, -0.2364]`; `ΔPQ vs V4-4 = -0.0017`, CI95 `[-0.0173, +0.0135]`), `Reverb_106 ΔPQ = -0.7059`, `Clip_48 ΔPQ = -0.3982`. Xem bảng đầy đủ tại [v4_4bs_results_table.md](file:///d:/SonicMaster_Reproducing/artifacts/10_extension/v4/pilot_dev_eval/v4_4bs_results_table.md).
7. **`V4-5p` (`Decoupled Magnitude–Phase Fusion + ModSpecLoss + r=0.5`, `1,953,200` params):** Kết quả từ `Reverb Branch Audit` cho thấy phép trộn phức `(1-g)P + gS` triệt tiêu ~24% biên độ trên dải `1–22 kHz` do pha lệch giữa `P` và `S`. `V4-5p` tách trộn biên độ khỏi pha (`MP-SENet` style), thêm `phase_gate_head` mượt theo 24 dải (+130 params), thêm `ModSpecLoss` (`1–12 kHz` + toàn dải, `λ_mod=2.0`), hạ `w_comp=0.25`, `w_wav=0.15`, giữ `l_ms=1.0` từ Epoch 1 và `residual_scale=0.5`.
   - **Phá kỷ lục độ trung thực & đồng nhất (`Matched Common Protocol, N=280`):**
     - **`Mel-SSIM = 0.8396`** (cao nhất mọi phiên bản; `+0.0135` vs `V4-4` CI `[+0.0109, +0.0164]`, `+0.2332` vs `A0`).
     - **`Mel-KL×1000 = 0.9345`** (thấp nhất mọi phiên bản; **vượt cả `A0 = 1.1064` tới `-15.5%`** và `-0.1130` vs `V4-4`).
     - **`FAD-CLAP = 0.1154`** (giảm `-31.1%` vs `A0 = 0.1674`, tốt hơn `V4-4 = 0.1304` và `V3-Blind = 0.1159`).
     - **`Audiobox PQ = 7.3775`** (`+0.0436` có ý nghĩa thống kê vs `V4-4`, CI `[+0.0265, +0.0617]`); trên `Reverb (106)` đạt **`Official MSD = 0.3790`** (vượt cả `A0 = 0.3853`!) và `PQ = 6.9494` (`+0.0617` vs `V4-4`). Trên `Non-Reverb (174)` đạt `SSIM = 0.9280`, `KL = 0.6520`, `FAD = 0.0446`, `PQ = 7.6383` (`ΔPQ vs A0 = -0.0324`).
   - **Attribute Guards (`Matched Common Protocol`):** **`11/12` PASS** — **`stereo = +0.0431 [PASS]`** (cải thiện rõ từ `+0.0974` ở `V4-4` và `+0.0518` ở `V4-4bs`!), `boom = -0.0500`, `vocal = -0.0439`, `punch = -0.1261`, `volume = +0.0007`; chỉ còn duy nhất **`bright = +0.1710 [FAIL]`**. Xem bảng đầy đủ tại [v4_5p_results_table.md](file:///d:/SonicMaster_Reproducing/artifacts/10_extension/v4/matched_dev_v4_5p/v4_5p_results_table.md).
8. *(Dự kiến)* **Stage 3 Perceptual / `TF-MoE` (`V4-5`):** Fine-tune thêm loss cảm nhận bất biến pha (VGGish/CLAP) để mở rộng cổng `g` ở nhóm Reverb (`PQ = 6.95 → 7.5+`) + dải cao `bright`, sau đó mới đấu đối chứng `TF-MoE (Top-1)` vs `Dense FFN`.

---

## 5. So Sánh Trực Tiếp Pipeline `Hybrid-v4` Của Chúng Ta vs. `NPU-ASLP (CCF-AATC 2026)`

Dù `Hybrid-v4` kế thừa tư duy Scale-Aware, Complex Mapping và Dual-Axis/TF-MoE từ bài báo **NPU-ASLP**, **bài toán và thiết kế pipeline của hai hệ thống khác nhau căn bản ở 6 điểm then chốt**:

| Tiêu chí so sánh | **NPU-ASLP (CCF-AATC 2026)** | **Hybrid-v4 (Pipeline của chúng ta)** | Lý do chúng ta điều chỉnh |
|---|---|---|---|
| **1. Đầu vào & Chế độ hoạt động** | **Single-input (`x → y`):** Chỉ nhận âm thanh thu qua điện thoại bị suy hao, không có mô hình generative dẫn đường. | **Teacher-Guided (`(x, s) → y`):** Nhận cả `x` lẫn đầu ra `s` của SonicMaster (đã có prompt ngôn ngữ tự nhiên). | Giữ được khả năng điều khiển bằng câu lệnh của SonicMaster (đã chứng minh `V4-0` không có `s` thất bại nặng `KL=5.17`). |
| **2. Cách tái tạo tín hiệu (`Decoder`)** | **Direct Complex Spectral Mapping:** Ước lượng trực tiếp toàn bộ phổ phức sạch từ đầu (tương đương `V4-0/V4-1`). | **Identity-Centered Multi-Frame `ΔW` + Gate `g` + Additive Residual `R`:** Lấy `Z_x` làm gốc qua đường tắt không nén. | Thực nghiệm chứng minh `DirectMap` (`V4-1`) bị mờ sóng gốc (`10/12` guard FAIL), còn giữ đường tắt `Z_x` (`V4-2..4`) giữ `SSIM ~0.83` và vượt `A0` về `FAD`. |
| **3. Chia dải tần & Kênh Stereo** | `K = 80` subbands kiểu Mel; xử lý theo cấu hình mono/kênh của cuộc thi. | **`24` dải psychoacoustic** + tách **`Mid / Side`** + ma trận lọc chéo `2×2` trên `(M, S)`. | Tập test SonicMaster là **Stereo 44.1 kHz** có thuộc tính guard `stereo` riêng; tách `M/S` cho phép kiểm soát độ rộng stereo trực tiếp. |
| **4. Kiến trúc Backbone** | **`TF-MoE` (`Top-1`)** ngay trong hệ thống chính (`2.951M` params, `82.66 GMACs / 10s`). | **`4×DualAxisBlock` với `Dense FFN` trước** (`1.69M – 1.95M` params, `~2.0 GMACs / 4s` tại `V4-2`), để dành `TF-MoE` cho `V4-5`. | Cô lập biến số (ablation discipline): kiểm chứng Decoder `MF7/MF9/Residual` trước khi tăng độ phức tạp routing của MoE. |
| **5. Dữ liệu huấn luyện** | Tự xây simulator **Playback–Propagation–Recording** tạo **1,572.12 giờ** dữ liệu ghép cặp nhân tạo (`202` điều kiện phòng/loa). | Dùng cặp `(x, s, clean)` chuẩn hóa từ **SonicMasterDataset** (Pilot `400` nguồn = `2,800` clip 5s, tách nguồn tuyệt đối với Dev & Confirm). | Không cần giả lập lại chuỗi loa/mic mới vì mục tiêu là khôi phục đúng phân bố suy hao và benchmark chuẩn của SonicMaster ICML 2026. |
| **6. Hàm Loss & Chỉnh độ sáng/Stereo** | Tối ưu cho `LSD`, `FAD`, `ViSQOL` trên bản thu di động (phần lớn lỗi phi tuyến nặng). | **`MR-STFT (L/R + M/S) + CompComplex + Wav L1 + MelKL + M/S Ratio`** + Output RMS match. | Khớp trực tiếp với bộ `15 Attribute Guards` (`EQ`, `Stereo`, `Volume`, `Clip`, `Reverb`) và bảo toàn độ sắc nét gốc của `x`. |

### Tóm lược bài học đối chiếu với NPU-ASLP
- **Điểm `NPU-ASLP` mạnh hơn về mặt bài toán độc lập:** Tự xử lý được khi *không có* teacher `s` nhờ 1,572 giờ dữ liệu mô phỏng và `Direct Complex Mapping` + `TF-MoE`.
- **Điểm `Hybrid-v4` vượt trội trong bài toán *Codec-Bypass cho SonicMaster*:** Bằng cách giữ đường tắt `Z_x` (thay vì `DirectMap`), dùng lọc đa khung giãn cách `MF9` + phần dư `R` scale theo `band_rms`, `Hybrid-v4` nhẹ hơn (`1.95M` vs `2.95M`), ít phép tính hơn nhiều lần, đạt **`FAD-CLAP` tốt hơn cả SonicMaster gốc (`0.1775` vs `0.2083`)**, và bảo toàn cấu trúc phổ **`SSIM = 0.8284`** (vượt xa ngưỡng `0.609` của SonicMaster VAE).

---

## 6. Tài Liệu & Artifacts Liên Quan
- Mã nguồn mô hình & loss: [v4_models.py](file:///d:/SonicMaster_Reproducing/scripts/extension/v4_models.py), [v4_losses.py](file:///d:/SonicMaster_Reproducing/scripts/extension/v4_losses.py), [v4_infer_dev_gpu.py](file:///d:/SonicMaster_Reproducing/scripts/extension/v4_infer_dev_gpu.py), [v4_eval_pilot_dev.py](file:///d:/SonicMaster_Reproducing/scripts/extension/v4_eval_pilot_dev.py)
- Unit tests (`17/17 passed`): [test_v4_step0.py](file:///d:/SonicMaster_Reproducing/scripts/extension/tests/test_v4_step0.py)
- Kết quả JSON & Bảng Dev `280`: [v4_3_summary.json](file:///d:/SonicMaster_Reproducing/artifacts/10_extension/v4/pilot_dev_eval/v4_3_summary.json), [v4_4_summary.json](file:///d:/SonicMaster_Reproducing/artifacts/10_extension/v4/pilot_dev_eval/v4_4_summary.json), [v4_4bs_results_table.md](file:///d:/SonicMaster_Reproducing/artifacts/10_extension/v4/pilot_dev_eval/v4_4bs_results_table.md), [matched_dev_v4_4bs/summary.json](file:///d:/SonicMaster_Reproducing/artifacts/10_extension/v4/matched_dev_v4_4bs/summary.json), [matched_dev_v4_5p/summary.json](file:///d:/SonicMaster_Reproducing/artifacts/10_extension/v4/matched_dev_v4_5p/summary.json), [v4_5p_results_table.md](file:///d:/SonicMaster_Reproducing/artifacts/10_extension/v4/matched_dev_v4_5p/v4_5p_results_table.md)
- Kiểm toán & Thiết kế mở rộng: [reverb_branch_audit_table.md](file:///d:/SonicMaster_Reproducing/artifacts/10_extension/v4/reverb_branch_audit/reverb_branch_audit_table.md), [DESIGN_V4_5P_DECOUPLED_FUSION.md](file:///d:/SonicMaster_Reproducing/DESIGN_V4_5P_DECOUPLED_FUSION.md)
- Hướng dẫn ngắn gọn: [HYBRID_V4_GUIDE.md](file:///d:/SonicMaster_Reproducing/HYBRID_V4_GUIDE.md) | Nhật ký: [SONICMASTER_DIARY.md](file:///d:/SonicMaster_Reproducing/SONICMASTER_DIARY.md)
- GitHub Repo: https://github.com/aajayagh/sonicmaster-hybrid-extension
