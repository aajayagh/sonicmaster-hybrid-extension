# Kiểm chứng bổ sung cho pilot VRG và blend

Cập nhật 08/11/2026 16:13. Hoàn tất các phép kiểm bằng máy trong protocol bổ sung. Nghe thật và các giới hạn dữ liệu vẫn được ghi riêng.

Đây là phân tích thăm dò sau khi đã xem confirmation. Giữ blend0,1; không chọn lại hệ số, không đổi ngưỡng hoặc thay quyết định pilot. Các đối chứng mới không phải một tập test độc lập mới.

## Kết quả chính

- SSIM vẫn tăng trước xuất WAV: +0.011528, CI95% [+0.007945; +0.015579]. Sau PCM16, mức tăng +0.012463. Bước xuất làm thay đổi delta +0.000935; không giải thích toàn bộ lợi ích SSIM.
- CI của delta FAD toàn nhóm chứa0 ở cả validation và confirmation. Chưa có bằng chứng ổn định về lợi ích FAD tổng thể; bootstrap chỉ xét thành phần nguồn khi crop/embedding đã cố định.
- KL×1000 trên confirmation tăng +0.054499, CI95% [+0.007538; +0.109565]: xấu hơn theo metric này.
- Sau cân loudness, delta PQ blend−SM = -0.016597, CI95% [-0.032911; -0.002653]. Delta gốc = -0.013567. Chi tiết và giới hạn bên dưới.

## 1. FAD và độ nhạy theo nguồn

Mỗi replicate lấy lại nguồn có hoàn lại, giữ toàn bộ biến thể/cặp clean–SM–blend, tính FAD toàn nhóm. 2.000 replicate/nhóm, seed20261108. Kiểm weighted low-rank khớp expanded SVD và FAD gốc; không lấy trung bình FAD theo clip/shard. Giá trị âm nghĩa blend tốt hơn theo FAD.

| Tập | Nhóm | ΔFAD | CI95% nguồn | Khoảng leave-one-source-out |
|---|---|---:|---|---|
| selection | single | -0.002762 | [-0.007820; +0.000587] | [-0.003455; -0.001745] |
| selection | double_triple | +0.007297 | [+0.000909; +0.018446] | [+0.005327; +0.008102] |
| selection | all | +0.001172 | [-0.002858; +0.006168] | [+0.000379; +0.001778] |
| confirmation | single | -0.004237 | [-0.008944; -0.000619] | [-0.005024; -0.002449] |
| confirmation | double_triple | +0.004768 | [-0.002863; +0.013094] | [+0.002333; +0.006760] |
| confirmation | all | -0.000570 | [-0.004550; +0.003656] | [-0.001654; +0.000366] |

Bỏ lần lượt một nguồn chỉ là sensitivity diagnostic, không loại nguồn trong kết quả gốc. CI thăm dò không khắc phục bias FAD ít mẫu, không đánh giá crop randomness hoặc xác nhận audibility. Các nhóm được báo cùng nhau; không chọn riêng nhóm đẹp để nhận phương pháp.

## 2. SSIM trước và sau xuất WAV

| Tập | ΔSSIM float trước xuất | ΔSSIM PCM16 | Phần thay đổi delta do đường xuất |
|---|---:|---:|---:|
| selection | +0.012538 | +0.012990 | +0.000452 |
| confirmation | +0.011528 | +0.012463 | +0.000935 |

Điểm đã có được gộp lại theo nguồn; không chạy hoặc sửa evaluator để làm đẹp điểm. Đường xuất gồm cả clamp và lượng tử hóa. SSIM tăng không chứng minh chi tiết hữu ích được khôi phục hoặc nghe tốt hơn.

## 3. PQ sau cân cùng loudness

Tái dựng đúng PCM16 gốc rồi áp dụng một gain cố định cho cả đoạn. Hai nhánh cùng target tối đa−18LUFS; nếu cần giảm thêm thì giảm target chung theo peak để không tạo hard clipping mới. Không chỉnh theo clean, không limiter và không cân từng cửa sổ. Cắt đỉnh đã có trong PCM gốc được giữ.

Đủ 140 cặp/280 file, 3 cặp cần hạ target dưới−18. Chênh loudness sau PCM lớn nhất 0.000013 LU. FFmpeg EBU R128 kiểm chéo 12 mẫu đã chọn trước.

| Điều kiện | SonicMaster PQ | Blend PQ | Δblend−SM | CI95% ΔPQ |
|---|---:|---:|---:|---|
| PCM gốc | 7.743102 | 7.729534 | -0.013567 | [-0.030014; +0.000863] |
| Cùng loudness | 7.770015 | 7.753418 | -0.016597 | [-0.032911; -0.002653] |

Chênh giữa hai delta (cùng loudness trừ gốc): -0.003030, CI95% [-0.005959; -0.000347]. Mỗi nguồn được trung bình 7 biến thể rồi gộp đều 20 nguồn. Không lấy tỷ lệ thành phần trăm giải thích và không dùng ngưỡng pilot cũ để tuyển một candidate mới.

Sau cân loudness, PQ blend vẫn thấp hơn và CI nằm dưới0. Chênh loudness toàn đoạn chưa giải thích hết khoảng cách PQ trong đối chứng này; chưa quy phần còn lại cho một cơ chế cụ thể.

Đối chứng có một lần lượng tử hóa PCM16 bổ sung; mức−18 chỉ là một mức đã chốt. Cùng loudness toàn đoạn không loại hết khác biệt loudness ngắn hạn. PQ vẫn là dự đoán mô hình, không là phiếu người nghe.

Kaggle job `143393527` v1: https://www.kaggle.com/code/aajayagh/sonicmaster-vrg-level-matched-check. Thời gian script 383.19s. Đã đối chiếu đủ280 hash mới và280 hash PCM gốc, pin Audiobox0.0.4/revision/weight/precision như trước. Không inference SonicMaster lại.

## 4. Những gì đã khép lại và những gì còn thiếu

| Hạng mục | Trạng thái và giới hạn |
|---|---|
| Công thức clipping, dữ liệu và export | Audit đã xong; chưa có evaluator/environment tạo số paper nên chênh còn chưa giải thích hết. |
| Full-song/joins12 và đối chứng xuất WAV | Metric đã kiểm chứng; cohort synthetic amplitude-only, không mở rộng theo endpoint đã khóa. |
| Residual VRG / blend | Validation và confirmation đã hoàn tất. Residual không qua validation; blend chỉ có sự đánh đổi, không xác nhận chất lượng tổng thể. |
| FAD, SSIM-export, PQ-loudness bổ sung | Đã chạy và kiểm chứng trong báo cáo này; phân tích sau quan sát, không test độc lập mới. |
| Nhận xét nghe | Chưa có phiếu thật; không thể tự điền bằng metric. |
| 11 thuộc tính confirmation ít mẫu | Không được coi là qua guard riêng; đặc biệt mud vẫn là quan sát cần thận trọng. Muốn kết luận thêm cần thiết kế cohort mới. |
| Exposure/near-duplicates ngoài các kiểm đã có | Chưa xác minh đầy đủ; chạy lại cùng cache không giải quyết được. |

Bộ nghe bổ sung: `artifacts/12_vrg_followup_checks/listening_level/LISTEN.html`. Giữ cùng12 mẫu và mapping A/B, chỉ cân loudness cho A/B. Clean/input/VAE tham chiếu giữ gain gốc. Phiếu bổ sung tách khỏi phiếu gốc, không coi hai bộ là người nghe độc lập. File/hash/thời lượng và cú pháp JS được kiểm; giao diện/phát âm thanh chưa kiểm vì trình duyệt công cụ chặn file cục bộ.

Bằng chứng: PROTOCOL.md, statistics.json, level_lock.json, level_receipts/, pq/{pq_lock,submission,budget}.json, pq_analysis.json và listening_level/verification.json, đều trong artifacts/12_vrg_followup_checks.
