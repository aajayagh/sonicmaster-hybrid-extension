# Bảng Danh Mục & Trạng Thái Thí Nghiệm (Experiment Inventory)

Tài liệu này theo dõi chi tiết toàn bộ các bảng, biểu đồ và thí nghiệm cần tái lập theo bài báo *SonicMaster: Towards Controllable All-in-One Music Restoration and Mastering* (ICML 2026).

## Trạng thái tổng quan

| Mục tiêu | Nhiệm vụ | Số lượng hạng mục | Trạng thái | Ghi chú & Rào cản |
|---|---|---|---|---|
| **M0** | Smoke & Baseline Verification | 1 setup (sample audio) | `measured` | Hoàn thành 100% trên Kaggle T4 (Mel-SSIM 0.9798, 7.89s/chunk) |
| **M1** | Benchmark Checkpoint Large (Bảng 1, 2, 3) | 31 chỉ số (EQ, Dynamics, Reverb, Amplitude, Stereo, FAD, KL, SSIM, PQ) | `planned` | Cần hoàn thành `03_stage` $\to$ `04_infer` $\to$ `05_score` |
| **M2** | So sánh mở rộng (Full-song, RemFX, Solo Piano, Ablation) | 8 hạng mục | `planned` | Cần dữ liệu RemFX, BABE-2, Full-song |
| **M3** | Huấn luyện từ đầu (Train from Scratch) | 1 setup (40 epoch backbone) | `planned` | Phụ thuộc quyết định cấp phát GPU sau M1 |
| **M4** | Nghiên cứu nghe (Listening Studies 1 & 2) | 2 nghiên cứu | `blocked` | Cần phê duyệt hội đồng tuyển chọn người nghe thật |

---

## Chi tiết từng bảng / hình vẽ

### Bảng 1: Thuộc Tính EQ & Microphone / X-Band (Energy Ratio & Cosine Distance)
*Tất cả giá trị: Sai số thấp hơn là tốt hơn (↓).*

| Thuộc tính / Metric | Định nghĩa đo lường | Paper (SonicMaster) | Measured | Delta | Trạng thái |
|---|---|---|---|---|---|
| Clarity | Energy ratio error (A.5) | *Xem Bảng 1* | - | - | `planned` |
| Boom | Energy ratio error (A.5) | *Xem Bảng 1* | - | - | `planned` |
| Airy | Energy ratio error (A.5) | *Xem Bảng 1* | - | - | `planned` |
| Bright | Energy ratio error (A.5) | *Xem Bảng 1* | - | - | `planned` |
| Dark | Energy ratio error (A.5) | *Xem Bảng 1* | - | - | `planned` |
| Muddy | Energy ratio error (A.5) | *Xem Bảng 1* | - | - | `planned` |
| Warm | Energy ratio error (A.5) | *Xem Bảng 1* | - | - | `planned` |
| Vocals | Energy ratio error (A.5) | *Xem Bảng 1* | - | - | `planned` |
| Mic EQ | Cosine distance 9 dải | **0.0088** | - | - | `planned` |
| X-band EQ | Cosine distance 9 dải | **0.0358** | - | - | `planned` |

---

### Bảng 2: Dynamics, Reverb, Amplitude & Stereo
*Tất cả giá trị: Sai số thấp hơn là tốt hơn (↓).*

| Thuộc tính / Metric | Định nghĩa đo lường | Paper (SonicMaster) | Measured | Delta | Trạng thái |
|---|---|---|---|---|---|
| Compression | RMS std error (2048/1024) | *Xem Bảng 2* | - | - | `planned` |
| Punch | Onset strength error | *Xem Bảng 2* | - | - | `planned` |
| Reverb (Small) | Modulation spectrum distance | *Xem Bảng 2* | - | - | `planned` |
| Reverb (Big) | Modulation spectrum distance | *Xem Bảng 2* | - | - | `planned` |
| Reverb (Mixed) | Modulation spectrum distance | *Xem Bảng 2* | - | - | `planned` |
| Reverb (Real) | Modulation spectrum distance | *Xem Bảng 2* | - | - | `planned` |
| Clipping | Spectral flatness error (×1000) | **1.506** | - | - | `planned` |
| Volume | RMS error | *Xem Bảng 2* | - | - | `planned` |
| Stereo Image | Side/Mid RMS ratio error | **0.1058** | - | - | `planned` |

---

### Bảng 3: Đánh Giá Toàn Diện Trên Test Set (Snippets 30s)
*FAD ↓, KL×1000 ↓, SSIM ↑, PQ ↑.*

| Phương pháp | FAD (All) | KL×1000 (All) | SSIM (All) | PQ (All) | Trạng thái |
|---|---|---|---|---|---|
| **Clean (Tham chiếu)** | - | - | - | **7.886** | `planned` |
| **Input (Degraded)** | **0.106** | **5.131** | **0.777** | **7.026** | `planned` |
| **Reconstructed (VAE)** | **0.196** | **5.273** | **0.546** | **6.885** | `planned` |
| **SonicMaster (Large)** | **0.073** | **0.888** | **0.609** | **7.705** | `planned` |

---

### Bảng 4, Hình 2 & Ablations (M2)

| Hạng mục | Nhiệm vụ | Paper Target | Trạng thái | Ghi chú |
|---|---|---|---|---|
| **Hình 2 (RemFX)** | SI-SDR Dynamics / Reverb | **47.11 dB / 45.76 dB** | `planned` | Cần tải tập test RemFX |
| **Bảng 4 (Solo Piano)** | Đánh giá zero-shot lịch sử piano | BABE / BABE-2 so sánh | `planned` | Cần audio BABE-2 |
| **Bảng 8 (Euler Steps)** | 1 vs 10 vs 100 bước & RK4 | So sánh chất lượng vs runtime | `planned` | Chạy sau khi có checkpoint Large |
| **Bảng 9 (Prompt Ablation)** | Normal vs Empty vs Shuffled | Phân tích mức độ phụ thuộc prompt | `planned` | Chạy trên cùng checkpoint |
| **Bảng 10 (Model Sizes)** | Small (2/6), Medium (4/12, 6/6), Large (6/18) | Kiểm tra scaling law | `blocked` | Checkpoints Small/Medium chưa được phát hành công khai |

---

### Nghiên Cứu Nghe (M4)

| Nghiên cứu | Quy mô | Tiêu chí | Trạng thái | Lý do chặn |
|---|---|---|---|---|
| **Nghiên cứu 1** | 43 cặp, 12 người nghe | MOS thang 1–7 (text relevance, overall quality, preference) | `blocked` | Cần hội đồng người nghe thật, không dùng LLM/AI |
| **Nghiên cứu 2** | 20 mẫu, 20 người nghe | So sánh X-band & Reverb với baseline | `blocked` | Cần hội đồng người nghe thật, không dùng LLM/AI |
