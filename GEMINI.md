# GEMINI.md — Nguyên Tắc & Quy Chuẩn Thực Hiện Dự Án Tái Lập SonicMaster

Tài liệu này định nghĩa toàn bộ nguyên tắc cốt lõi, giới hạn an toàn (red lines), tiêu chuẩn kỹ thuật và quy trình vận hành bắt buộc đối với AI Agent và các cộng tác viên khi thực hiện dự án **tái lập bài báo *SonicMaster: Towards Controllable All-in-One Music Restoration and Mastering* (ICML 2026) trên môi trường Kaggle**.

---

## 1. Mục Tiêu Dự Án & Thứ Tự Ưu Tiên Nghiệm Thu

Dự án nhằm tái lập độc lập, khách quan, có thể kiểm chứng từ đầu (from scratch) các kết quả thực nghiệm trong bài báo của tác giả.

### Các Mốc Nghiệm Thu (Milestones)

| Mốc | Tên nhiệm vụ | Sản phẩm nghiệm thu | Tiêu chuẩn kết luận |
|---|---|---|---|
| **M0** | Smoke & Baseline Verification | Chạy checkpoint trên các mẫu WAV chính thức (`samples/`), xác nhận tham số nạp đúng, tạo ra âm thanh nghe được. | Xác nhận tái lập thành công thao tác suy luận cơ bản trên mẫu kiểm chứng. |
| **M1** *(Ưu tiên cao nhất)* | Benchmark Checkpoint Large trên Test Set | Chạy suy luận model Large trên toàn bộ test snippet chính thức; tái lập hàng **input**, **VAE roundtrip**, và **SonicMaster** trong Bảng 1, 2, 3; đối chiếu số lượng mẫu và bảng chênh lệch với paper. | Đo đạc lại đầy đủ các hàng tương ứng trên đoạn ngắn (snippets), xác định độ lệch chuẩn xác. |
| **M2** | Thí Nghiệm Mở Rộng & Đối Chứng | Chạy các phương pháp đối chứng (baselines), bài dài (full-song), RemFX, solo piano (Bảng 4), và ablation studies (Bảng 8–10) khi có đủ đầu vào. | Tái lập các so sánh được đánh dấu hoàn tất với đầy đủ bằng chứng kiểm toán. |
| **M3** | Huấn Luyện Lại (Train from Scratch) | Huấn luyện backbone mới từ dữ liệu phát hành, sau đó đánh giá trên tập test đã khóa cố định. | Đánh giá mức độ tái lập quá trình huấn luyện trong giới hạn tài nguyên thực tế. |
| **M4** | Nghiên Cứu Nghe (Listening Studies) | Chuẩn bị và thực hiện 2 nghiên cứu nghe độc lập (Subjective Evaluation) nếu đủ điều kiện tuyển người tham gia. | Thu thập bằng chứng cảm nhận khách quan từ hội đồng người nghe mới. |

> [!IMPORTANT]
> **M1 hoàn toàn độc lập với M3.** Bắt buộc hoàn thành đánh giá checkpoint chính thức (M1) trước khi xem xét cấp phát tài nguyên cho huấn luyện lại (M3). Không được trì hoãn M1 để chờ M3. Khi một nhánh thiếu tài nguyên, tiếp tục thực hiện các nhánh độc lập khác.

---

## 2. Các Điều Cấm Tuyệt Đối (Strict Prohibitions / Red Lines)

Vi phạm bất kỳ điều nào dưới đây sẽ làm mất giá trị khoa học của kết quả tái lập:

1. **Tuyệt đối không sử dụng tài nguyên không rõ nguồn gốc:**
   - Không mở, sửa, chạy, gắn làm input, hay trích xuất kết quả từ bất kỳ notebook hoặc dataset thực nghiệm không chính thức nào của các nhóm trước đây.
   - Chỉ sử dụng mã nguồn, trọng số và dữ liệu chính thức đã được khóa định danh (pin commit/hash).
2. **Tuyệt đối không ngụy tạo hay sao chép kết quả:**
   - Không được điền số liệu của paper vào các ô kết quả tự chạy.
   - Mỗi báo cáo phải phân định rạch ròi 3 cột: **Số tác giả công bố (Paper)**, **Số tự đo (Measured)**, và **Chênh lệch (Delta)**. Các mục chưa hoàn thành phải ghi rõ `planned`, `running`, `partial`, hoặc `blocked`.
3. **Không can thiệp để làm đẹp điểm số (No Cherry-Picking / Data Snooping):**
   - Không thêm module lọc hậu kỳ (refiner), không can thiệp tinh chỉnh mô hình ngầm.
   - Không thử nghiệm nhiều prompt rồi chọn prompt cho điểm test cao nhất (prompt cherry-picking). Bắt buộc dùng prompt chính thức và áp dụng quy tắc chọn cố định đã công bố.
   - Không tự ý tạo split ngẫu nhiên rồi gán nhãn là test split của tác giả.
   - Không căn chỉnh dạng sóng hoặc volume tùy tiện theo bản sạch (clean ground-truth) để tăng điểm metric.
4. **Không chạy mô hình với trọng số rác:**
   - Nếu việc nạp checkpoint thiếu các khối trọng số cốt lõi (`missing_keys` trong backbone), phải dừng lại ngay lập tức; không được để mô hình chạy ngầm với trọng số ngẫu nhiên.
5. **Không im lặng bỏ qua dữ liệu lỗi:**
   - Không được tự ý loại bỏ các mẫu bị lỗi hoặc silent drops mà không ghi nhận số lượng và nguyên nhân cụ thể trong log kiểm toán.
6. **Không ngụy tạo đánh giá người nghe:**
   - Tuyệt đối không thay thế phiếu khảo sát người nghe thật bằng điểm số của mô hình đánh giá (PQ) hoặc suy đoán từ LLM.

---

## 3. Quản Lý Nguồn Lực & Khóa Phiên Bản (Provenance & Pinning)

Mọi phụ thuộc phải được ghi lại trong tệp `source-lock.json` với URL, commit hash và mã băm SHA-256 cụ thể:

* **Bài báo gốc chuẩn:** *SonicMaster: Towards Controllable All-in-One Music Restoration and Mastering* (ICML 2026), [arXiv:2508.0338](https://arxiv.org/abs/2508.0338).
* **Mã nguồn chính thức:** [AMAAI-Lab/SonicMaster](https://github.com/AMAAI-Lab/SonicMaster) — Commit bắt buộc: `ab32b2d456e461b6bbc188d666caf2eabefa2f31`.
* **Trọng số SonicMaster (Large):** [amaai-lab/SonicMaster](https://huggingface.co/amaai-lab/SonicMaster) — Revision: `93c7fa73c4dd446c9e656af34524b5f6a416afda` (`model.safetensors`).
* **Dữ liệu chính thức:** [amaai-lab/SonicMasterDataset](https://huggingface.co/datasets/amaai-lab/SonicMasterDataset) — Revision: `04cc0dd51ff748b41373308d9ee152796c840233`.
* **Mô hình VAE phụ trợ:** [stabilityai/stable-audio-open-1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0) (sử dụng thư mục `vae`).
* **Bộ mã hóa văn bản:** [google/flan-t5-large](https://huggingface.co/google/flan-t5-large).
* **Bảo mật Kaggle:**
  - Mọi API token (Hugging Face, v.v.) phải được cấu hình qua **Kaggle Secrets**.
  - Không in token ra log màn hình hoặc file text.
  - Không sao chép các trọng số/tài nguyên có bản quyền hạn chế sang các dataset public.

---

## 4. Kiến Trúc Pipeline Trên Kaggle (Step-by-Step Jobs)

Quy trình tái lập được tổ chức thành chuỗi job độc lập có kiểm toán trong namespace riêng (ví dụ: `sonicmaster-paper-repro-v1`).
Mỗi job đọc dữ liệu đầu vào từ **output đã commit** của job trước hoặc từ dataset có version xác định:

```
[00_inventory] -> [01_manifest] -> [02_smoke] -> [03_stage_audio]
                                        |               |
                                        v               v
                                  [04_infer] <----------+
                                        |
                                        v
                                  [05_score] ---------> [09_report]
                                        |                     ^
                     [06_comparisons] --+                     |
                     [07_train] ------------------------------+
                     [08_listening] --------------------------+
```

### Bảng Phân Công Nhiệm Vụ Từng Job

1. **`00_inventory` (CPU):**
   - Lập danh mục nguồn chính thức, xác thực quyền truy cập, kiểm tra giấy phép.
   - Xuất: `source-lock.json`, bảng theo dõi trạng thái các hàng/cột của paper (`planned`/`running`/`measured`/`partial`/`blocked`).
2. **`01_manifest` (CPU):**
   - Tải và phân tích metadata trước khi tải audio. Đọc `meta.split` thực tế từ Parquet/JSONL.
   - Kiểm tra tính rời rạc của `source_id` giữa các split (không giao nhau).
   - Thống kê variant: Đảm bảo 1.000 nguồn test × 7 variants = 7.000 cặp (4.000 single, 3.000 double+triple).
   - Tách trường `hidden_clipping` khỏi danh sách lỗi có prompt theo logic tác giả.
   - Xuất: Manifests chi tiết, báo cáo phân bổ shard và kiểm tra trùng ID.
3. **`02_smoke` (GPU ngắn):**
   - Khóa môi trường thực thi (Python, PyTorch, CUDA, torchaudio, diffusers, transformers).
   - Nạp `model.safetensors` với `configs/tangoflux_config.yaml`.
   - Ghi log và kiểm tra `missing_keys` / `unexpected_keys`. Dừng nếu thiếu khối backbone.
   - Chạy thử `infer_single.py` trên các mẫu trong `samples/`. So sánh với output chính thức của tác giả (lưu ý phát hiện file pointer Git LFS nếu có).
   - Đo đạc thực tế: VRAM tiêu thụ, thời gian thực thi (giây/clip).
4. **`03_stage_audio` (CPU/Network):**
   - Tải các shard audio phục vụ test set theo manifest của `01`.
   - Kiểm tra checksum SHA-256 sau khi tải; sử dụng cơ chế ghi nguyên tử (`.part` $\to$ rename); retry với exponential backoff và jitter khi gặp lỗi mạng.
   - Xuất: Bảng ánh xạ clean/degraded, báo cáo xác minh checksum.
5. **`04_infer` (GPU theo shard):**
   - Tính toán ngân sách GPU dự kiến dựa trên số liệu từ `02`.
   - Chia batch theo `source_id`, giữ nguyên toàn bộ 7 variants của cùng một source trong cùng một shard.
   - Chạy inference cho các nhánh:
     - Input gốc.
     - VAE Reconstructed (degraded $\to$ VAE encode $\to$ VAE decode).
     - SonicMaster restored (với prompt chuẩn).
   - Giữ nguyên kênh Stereo cho mọi xử lý yêu cầu stereo.
   - Xuất: Tệp WAV suy luận có version, bảng kê hash và log hoàn tất.
6. **`05_score` (CPU + GPU):**
   - Thực thi mã đo đạc từ thư mục `evaluation/` của tác giả.
   - Đối chiếu chéo giữa mã nguồn và công thức tại Phụ lục A.5.
   - Tính toán toàn bộ chỉ số cho Bảng 1, Bảng 2, Bảng 3.
   - Xuất: Bảng điểm chi tiết từng mẫu (per-sample CSV), embeddings, và thống kê tổng hợp cấp tập.
7. **`06_comparisons` (Tùy nhánh):**
   - Chạy các nhánh so sánh khi có đủ đầu vào: Baselines (Text2FX, WPE, HPSS, Mel2Mel), Full-song, RemFX, Solo Piano, và Ablation (Euler steps, prompt ablation, v.v.).
8. **`07_train` (GPU - chỉ khi có quyết định cấp phát tài nguyên):**
   - Kiểm tra tính source-disjoint tuyệt đối của bộ train/val/test.
   - Tiền mã hóa latents (`preencode_latents_acce2.py`).
   - Khớp nối các tham số cấu hình: đối chiếu giữa paper (5 L40S, 40 epoch, batch 80) và YAML repo (201 epoch, batch 16, per-device).
   - Chạy pilot train để đo loss, gradient, khả năng checkpoint/resume và dự toán giờ GPU trước khi chạy full 40 epochs.
9. **`08_listening` (Không bắt buộc GPU):**
   - Chuẩn bị giao diện đánh giá mù/ngẫu nhiên theo protocol Nghiên cứu 1 (43 cặp, thang 1–7) và Nghiên cứu 2 (20 mẫu, X-band & Reverb).
   - Lưu trữ dữ liệu ẩn danh, phân tích paired t-test và MOS với khoảng tin cậy 95% CI khi có dữ liệu người nghe thật.
10. **`09_report` (CPU):**
    - Tổng hợp toàn bộ bảng kết quả, biểu đồ và trích xuất artifact bàn giao cuối cùng.

---

## 5. Quy Chuẩn Đo Đạc Metric (Bảng 1–3)

Mọi phép đo phải tuân thủ nghiêm ngặt các quy tắc sau:

### Sanity Checks Bắt Buộc Trước Khi Chấm Full Test
Trước khi tính điểm toàn bộ tập test, bắt buộc phải vượt qua các kiểm tra kiểm soát:
- **Tín hiệu sạch so với chính nó:** Ground Truth vs Ground Truth phải cho sai số $\approx 0$ và SSIM $\approx 1.0$.
- **Tín hiệu kiểm soát nhân tạo:** Kiểm tra trên tín hiệu mono/stereo chuẩn, giảm volume biết trước, và kiểm tra phát hiện lỗi khi tráo đổi ID.
- **Phát hiện NaN / Silent:** Mọi giá trị im lặng hoặc NaN phải được ghi nhận và thống kê rõ ràng.

### Quy Cách Từng Chỉ Số
* **EQ (Bảng 1):**
  - 8 thuộc tính (clarity, boom, airy, bright, dark, muddy, warm, vocals): Tính sai số tỷ lệ năng lượng dải mục tiêu so với Ground Truth, ranh giới dải tần theo Phụ lục A.5.
  - Mic & X-band: Cosine distance giữa phân bố năng lượng 9 dải tần.
* **Dynamics, Reverb, Amplitude, Stereo (Bảng 2):**
  - Compression: Sai số độ lệch chuẩn RMS theo khung 2048 / bước nhảy 1024.
  - Punch: Sai số trung bình onset strength.
  - Reverb (Small, Big, Mixed, Real): Khoảng cách Euclid của phổ điều chế (modulation spectrum) chuẩn hóa.
  - Clip: Sai số spectral flatness (lưu ý: nhân 1000 khi trình bày bảng).
  - Volume: Sai số RMS.
  - Stereo: Sai số tỷ lệ RMS(side) / RMS(mid), giữ độc lập L/R và epsilon theo đúng mã nguồn.
* **KL Divergence & SSIM (Bảng 3):**
  - Dùng `extract_kl_ssim_mass.py`, đảm bảo đúng thứ tự truyền GT và Output.
  - Audio mono 44.1 kHz, mel 128, FFT 2048, hop 512, chuyển đổi dB theo giá trị cực đại.
  - KL nhân 1000 khi trình bày.
* **FAD-CLAP (Bảng 3):**
  - Trích xuất embedding theo `extract_clap_gt.py` và `extract_fad_mass.py`.
  - Khóa cố định CLAP checkpoint, chiến lược fusion và resample.
  - **Quy tắc vàng:** Tính toán FAD một lần duy nhất trên toàn bộ embedding đã gộp của cả tập so sánh. **Tuyệt đối không lấy trung bình cộng FAD của từng clip hoặc từng shard.**
* **Perceptual Quality - PQ (Bảng 3):**
  - Sử dụng checkpoint chính thức của **Audiobox Aesthetics**.
  - Luôn chấm điểm cả bản Clean gốc làm mốc tham chiếu chuẩn.

### Mốc Đối Chiếu Paper (Snippet / All)
Dùng để đối chiếu và phát hiện bất thường, **không dùng để điền sẵn vào báo cáo**:
- **Bảng 3 (FAD ↓ / KL×1000 ↓ / SSIM ↑ / PQ ↑):**
  - Clean (PQ): `7.886`
  - Input: `0.106 / 5.131 / 0.777 / 7.026`
  - Reconstructed (VAE): `0.196 / 5.273 / 0.546 / 6.885`
  - SonicMaster: `0.073 / 0.888 / 0.609 / 7.705`
- **Bảng 1 (SonicMaster):** Mic = `0.0088`, X-band = `0.0358`.
- **Bảng 2 (SonicMaster):** Clip×1000 = `1.506`, Stereo = `0.1058`.

### Quy Trình Điều Tra Khi Số Liệu Bị Lệch (Troubleshooting Hierarchy)
Khi phát hiện kết quả đo đạc bị lệch so với bài báo, bắt buộc thực hiện kiểm tra theo đúng thứ tự ưu tiên:
1. **Kiểm tra hàng Input và Ground Truth trước:** Vì 2 hàng này hoàn toàn không phụ thuộc vào checkpoint SonicMaster. Nếu hàng Input bị lệch: kiểm tra lại split, bộ giải mã audio, sample rate, xử lý mono/stereo, chuẩn hóa âm lượng hoặc hệ số nhân (×1000).
2. **Kiểm tra hàng VAE Reconstructed:** Kiểm tra version VAE, hệ số scale latents, sampling mode (mean/sample).
3. **Kiểm tra hàng SonicMaster Output:** Khi 2 bước trên đã khớp mà output mô hình vẫn lệch, kiểm tra các khóa trọng số (keys), cấu hình Large, prompt được cấp, solver (Euler), số bước inference (10 steps), seed và kiểu dữ liệu (precision fp16/fp32).

---

## 6. Quy Chuẩn Báo Cáo & Bàn Giao Sản Phẩm (Deliverables)

Mọi báo cáo kết thúc hoặc bàn giao trung gian phải tuân thủ cấu trúc chuẩn:

1. **Bảng So Sánh Chuẩn:**
   Tất cả các bảng kết quả tái lập (Bảng 1, 2, 3, v.v.) phải có đủ các cột:
   - `Metric Name` (Tên chỉ số & Đơn vị)
   - `Paper` (Số công bố trong bài báo)
   - `Measured` (Số thực tế tự đo được)
   - `Delta` (Độ chênh lệch tuyệt đối và tỷ lệ %)
   - `N` (Số lượng mẫu thực tế đưa vào tính toán)
   - `Protocol & Evidence` (Đường dẫn log/artifact chứng minh)
   - Phân tách rõ ràng giữa các tập: **Single Variant**, **Double + Triple Variants**, và **All Variants**.
2. **Hồ Sơ Chứng Minh Kèm Theo (Audit Artifacts):**
   - File `source-lock.json` và `environment-lock.txt`.
   - File danh sách mẫu chi tiết kèm kết quả đo (per-sample CSV).
   - Log thực thi đầy đủ của các job trên Kaggle (bao gồm cả log nạp weights).
   - Bộ mẫu âm thanh minh chứng gồm 3 file tương ứng theo từng sample ID: `Clean (Orig)`, `Degraded (Input)`, và `Restored (SonicMaster)`.
3. **Minh Bạch Trạng Thái Dự Án:**
   - Không được tuyên bố "Đã tái lập hoàn toàn bài báo" khi mới chỉ hoàn thành mốc M1.
   - Nếu có hạng mục chưa thực hiện hoặc bị chặn (do thiếu quota GPU, thiếu dataset full-song, v.v.), phải ghi rõ: Tên hạng mục, Tài sản bị thiếu, Các nguồn đã kiểm tra, Mức độ ảnh hưởng và Giải pháp tiếp theo.

---

## 7. Quy Tắc Hành Động Của Agent

1. **Khởi đầu tuần tự:** Bắt đầu từ Job `00_inventory` và `01_manifest`, hoàn thành `source-lock.json` và kiểm toán manifest trước khi kích hoạt bất kỳ GPU nào.
2. **Đo đạc trước khi mở rộng (Pilot First):** Luôn chạy thử trên 1 mẫu $\to$ 1 shard pilot $\to$ đo đạc thời gian, VRAM, dung lượng lưu trữ $\to$ tính toán tổng quota cần thiết trước khi chạy toàn bộ test set.
3. **Bảo toàn dữ liệu:** Lưu trữ checkpoint trung gian và output theo cơ chế versioned artifact; không ghi đè dữ liệu chưa được sao lưu.
4. **Trung thực khoa học tuyệt đối:** Báo cáo chính xác mọi điểm sai lệch, lỗi tensor, hoặc bất nhất giữa mã nguồn và bài báo. Không tự ý sửa code tác giả nếu không có tài liệu giải trình đi kèm.

---

## 8. Lưu Trữ & Truy Xuất Artifacts Trên Google Drive

Để tiết kiệm dung lượng ổ `D:\`, các thư mục dữ liệu/artifact lớn trong `d:\SonicMaster_Reproducing\artifacts\` được di chuyển dần lên Google Drive (tài khoản `ngothaoanh305@gmail.com`).

- **Thư mục gốc dự án trên Drive:** `SonicMaster_Reproducing` (Folder ID: `1Alfx4-n-hmf2xtVCNiF3msihUOnHEVU8`)
- **Thư mục `artifacts` trên Drive:** `SonicMaster_Reproducing/artifacts` (Folder ID: `17OzccpbogBtB84QNvACJIS9IyzWrsglh`)
- **Đường dẫn local qua Google Drive for Desktop (chế độ Stream files):** `G:\My Drive\SonicMaster_Reproducing\artifacts\`

### Cách Agent Truy Xuất File Không Còn Trên Ổ `D:\`
1. **Trong sandbox (ưu tiên dùng `gdrive` CLI qua plugin Google Drive):**
   - Liệt kê thư mục: `gdrive readonly ls 17OzccpbogBtB84QNvACJIS9IyzWrsglh`
   - Tìm file theo tên: `gdrive readonly search --name-contains "<tên_file>" --parent-id 17OzccpbogBtB84QNvACJIS9IyzWrsglh`
   - Tải đúng file cần xử lý về `tmp/`: `gdrive readonly download <FILE_ID> --out d:\SonicMaster_Reproducing\tmp\<tên_file>` *(xóa file tạm trong `tmp/` sau khi xử lý xong)*.
2. **Ngoài sandbox (hoặc khi được cấp quyền đọc `G:\`):**
   - Đọc trực tiếp từ `G:\My Drive\SonicMaster_Reproducing\artifacts\...` (Drive for Desktop tự stream đúng file được mở mà không tải toàn bộ kho).
