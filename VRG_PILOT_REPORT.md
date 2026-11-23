# Pilot residual VRG và trộn input: validation và confirmation

Cập nhật 08/11/2026 15:38, giờ Việt Nam. Đã hoàn tất cache và chấm CPU trên **280 mẫu, từ 40 nguồn nhạc**, mỗi nguồn 7 dạng suy giảm. Đây là thí nghiệm mở rộng sau reproduction, dùng tập validation để chọn hệ số.

## Đọc nhanh

- Residual VRG: không có hệ số nào qua đủ điều kiện validation trên 40 nguồn. Tăng độ giống phổ có thể làm giảm tác dụng sửa một số loại lỗi.
- Blend cố định 90% SonicMaster + 10% input: trên 20 nguồn xác nhận, ΔSSIM = +0.012463, ΔPQ = -0.013567. SSIM tăng; PQ trung bình thấp hơn, chưa có bằng chứng nghe tốt hơn.
- Hệ số đã khóa trước confirmation. Đây là pilot thăm dò, không phải một mô hình mới đã được huấn luyện hoặc chứng minh tốt hơn SonicMaster toàn diện.
- Phần tính toán và phần nhận xét nghe được tách riêng. Bộ nghe 12 mẫu đã chuẩn bị; hiện chưa có đánh giá của người nghe.

## Đang kiểm chứng điều gì?

VRG lấy phần chênh giữa input và bản VAE, rồi cộng một phần nhỏ vào SonicMaster. Giả thuyết là phần chênh có thể giúp giữ chi tiết. Nhưng nó cũng có thể chứa lỗi input hoặc khác biệt pha, nên phải đo khả năng sửa lỗi và chất lượng, không chỉ độ giống bản sạch.

- Input: x; VAE: r; SonicMaster: s.
- VRG residual: s + λ × (x − r).
- Đối chứng blend: (1 − α) × s + α × x.
- Grid đã khóa: 0; 0,1; 0,25; 0,5; 0,75; 1. Hệ số 0 dùng chung baseline. Chưa huấn luyện mô hình.

## Đã kiểm chứng dữ liệu

- Đủ 280 cache, 1120 hash WAV và 280 hash float. Cấu hình/nguồn khớp bản khóa, hệ số 0 tái tạo baseline và blend=1 tái tạo input.
- Kiểm 7560 cửa sổ input–VAE: 7180 đủ tin cậy; lag của các cửa sổ này nằm trong [-4, 4] sample.
- Có 3 mẫu không đạt coverage tương quan gốc: 1406227_deg5, 1406227_deg6, 1582649_deg1.
- Giữ nguyên gate tương quan gốc và mọi mẫu khó. Quyết định kỹ thuật cho phép chấm công thức không dịch thời gian, dựa trên mapping code, clock probe và audit; không coi VAE đồng nhất pha hoặc khẳng định căn chỉnh tuyệt đối từng sample.
- 3.640 dòng điểm = 280 mẫu × 13 nhánh. Mỗi nguồn được tính trung bình 7 variants trước khi gộp. Không dùng 280 variants như 280 nguồn độc lập.

## Kết quả

SSIM cao hơn nghĩa là cấu trúc phổ gần clean hơn; không tự chứng minh nghe hay hơn. KL thấp hơn tốt hơn theo phép đo của paper. “Vượt biên” là tỷ lệ sample-channel float có |y| > 1 trước khi xuất, khác metric clip của paper.

| Nhánh | Hệ số | SSIM ↑ | ΔSSIM so SM | KL ×1000 ↓ | Vượt biên (%) | PQ ↑ |
|---|---:|---:|---:|---:|---:|---:|
| input | — | 0.780772 | +0.182547 | 4.849925 | 0.00000 | 6.963843 |
| vae | — | 0.537950 | -0.060275 | 4.975617 | 1.68272 | 6.756538 |
| sonicmaster | 0 | 0.598225 | +0.000000 | 0.934439 | 0.03279 | 7.621606 |
| blend | 0.1 | 0.611215 | +0.012990 | 0.949363 | 0.00955 | 7.604751 |
| blend | 0.25 | 0.639809 | +0.041585 | 1.039431 | 0.00284 | chưa đo |
| blend | 0.5 | 0.694983 | +0.096759 | 1.392621 | 0.00124 | chưa đo |
| blend | 0.75 | 0.748815 | +0.150590 | 2.290906 | 0.00106 | chưa đo |
| blend | 1 | 0.780772 | +0.182547 | 4.849925 | 0.00000 | chưa đo |
| residual | 0.1 | 0.607984 | +0.009759 | 0.974762 | 0.02935 | chưa đo |
| residual | 0.25 | 0.624890 | +0.026665 | 1.054861 | 0.02801 | chưa đo |
| residual | 0.5 | 0.647293 | +0.049068 | 1.215494 | 0.04029 | chưa đo |
| residual | 0.75 | 0.663424 | +0.065199 | 1.463132 | 0.13957 | chưa đo |
| residual | 1 | 0.670851 | +0.072626 | 1.765940 | 0.34174 | chưa đo |

## Kiểm tra điều kiện chọn hệ số

Ngưỡng đã khóa: ΔSSIM ≥ 0,01; suy giảm target chuẩn hóa trung bình ≤ 0,02; từng thuộc tính đủ mẫu ≤ 0,05; PQ giảm không quá 0,05; tỷ lệ vượt biên không tăng quá 0,1 điểm phần trăm.

Mẫu số chuẩn hóa là lỗi input trung bình của thuộc tính đó trên validation. Chỉ lấy phần xấu đi dương của từng thuộc tính rồi trung bình; cải thiện ở thuộc tính khác không bù trừ phần xấu đi. Loại metric clip chưa giải quyết khỏi guard này. Thuộc tính phải có ≥10 mẫu và ≥5 nguồn để áp dụng guard riêng; nhóm nhỏ vẫn được báo.

| Nhánh | Hệ số | Suy giảm target trung bình | Guard CPU | Lý do chưa qua CPU | CI95% của ΔSSIM |
|---|---:|---:|---|---|---|
| blend | 0.1 | 0.009061 | qua | — | [+0.010165; +0.016220] |
| blend | 0.25 | 0.038586 | chưa qua | average_target, represented_attributes: airy, mud, mic, stereo | [+0.036179; +0.047545] |
| blend | 0.5 | 0.155548 | chưa qua | average_target, represented_attributes: clarity, airy, bright, dark, mud, vocal, mic, xband, comp, small, mix, real, volume, stereo | [+0.087715; +0.106171] |
| blend | 0.75 | 0.333922 | chưa qua | average_target, represented_attributes: clarity, airy, bright, dark, mud, warm, vocal, mic, xband, comp, small, mix, real, volume, stereo | [+0.136930; +0.163862] |
| blend | 1 | 0.605567 | chưa qua | average_target, represented_attributes: clarity, airy, bright, dark, mud, warm, vocal, mic, xband, comp, punch, small, big, mix, real, volume, stereo | [+0.162700; +0.202358] |
| residual | 0.1 | 0.002672 | chưa qua | ssim | [+0.006868; +0.012904] |
| residual | 0.25 | 0.011223 | chưa qua | represented_attributes: dark | [+0.022037; +0.031250] |
| residual | 0.5 | 0.039867 | chưa qua | average_target, represented_attributes: bright, dark, mic | [+0.041680; +0.056158] |
| residual | 0.75 | 0.067931 | chưa qua | out_of_range, average_target, represented_attributes: bright, dark, warm, mic | [+0.055661; +0.074695] |
| residual | 1 | 0.097522 | chưa qua | out_of_range, average_target, represented_attributes: airy, bright, dark, warm, mic, real | [+0.061454; +0.083558] |

Thuộc tính vắng mặt: không. Thuộc tính có nhưng chưa đủ subgroup: boom. Không suy rộng thành bảo đảm trên toàn bộ 19 thuộc tính.
CI95% dùng 2.000 bootstrap theo nguồn, không thay thế các ngưỡng point-estimate đã khóa. Các mẫu low-correlation vẫn nằm trong toàn bộ bảng; điểm subgroup lưu riêng trong timing_subgroup_scores.json và không được dùng chọn hệ số khác.

Có 1 hệ số qua guard CPU. Qua CPU chỉ đủ điều kiện đo PQ, chưa phải được chấp nhận.

## Các quan sát dễ hiểu

- VRG residual hệ số 0,1 có tăng SSIM, nhưng mức tăng 0.009759 thấp hơn mốc 0,01 đã đặt trước. Vì vậy nó chưa qua tiêu chí vận hành của pilot; không diễn giải thành hoàn toàn không có lợi ích.
- Tăng residual lên 0,25 giúp SSIM tăng rõ hơn, nhưng lỗi ở thuộc tính dark xấu đi chuẩn hóa 0.122998, vượt trần 0,05. Nhóm này có 17 mẫu/15 nguồn. Chỉ xem điểm trung bình mọi thuộc tính sẽ bỏ sót đánh đổi này.
- Blend 0,1 là trộn 90% SonicMaster với 10% input. Nó tăng SSIM 0.012990 và qua các guard CPU. Đây là đối chứng khác với VRG residual; quyết định sau PQ được ghi riêng dưới đây.
- Kết quả trên đủ 40 nguồn khác mức tăng quan sát ở 4 nguồn smoke. Smoke được dùng kiểm quy trình, không dùng quyết định hệ số.
- Input vẫn có SSIM cao hơn SonicMaster trong khi input mang các suy giảm đã tạo. Vì vậy tối đa hóa riêng SSIM có thể kéo output về input chưa được sửa.

### So sánh PQ theo nguồn

Đã kiểm 1400 dòng PQ; mọi hash WAV dùng chấm khớp bản dựng tại máy. Cả năm nhánh dùng cùng support thực tế, xuất clamp → PCM16, không chỉnh gain hoặc pad bên ngoài. PQ clean trung bình: 7.865489.

CI95% dưới đây dùng 2.000 bootstrap theo 40 nguồn, seed 20261108; là khoảng bất định thăm dò trên validation. PQ là dự đoán của mô hình, chưa phải đánh giá nghe của người dùng.

| Nhánh | Hệ số | ΔPQ so SM | CI95% nguồn |
|---|---:|---:|---|
| clean | — | +0.243883 | [+0.155971; +0.344806] |
| input | — | -0.657763 | [-0.798739; -0.531564] |
| vae | — | -0.865068 | [-1.014698; -0.738326] |
| sonicmaster | 0 | +0.000000 | [+0.000000; +0.000000] |
| blend | 0.1 | -0.016855 | [-0.029684; -0.005919] |

Blend 0,1 đổi PQ -0.016855 điểm so với SonicMaster. Điều kiện đã khóa là ΔPQ ≥ −0,05: **đạt**. Ngưỡng này là tiêu chí vận hành pilot, không phải mức khác biệt tai người chắc chắn nghe được.

## Kết luận hiện tại

Các hệ số đạt mọi guard validation và đã khóa để xác nhận: **blend hệ số 0.1**. Các bảng trên chỉ là vòng validation; hệ số không được điều chỉnh lại theo confirmation.
Confirmation trên 20 nguồn/140 mẫu đã hoàn tất: **qua các tiêu chí pilot đã khóa**. Bảng confirmation và giới hạn ở cuối báo cáo. Residual VRG không có hệ số qua validation; kết quả blend không phải bằng chứng residual VRG thành công.

Confirmation 20 nguồn/140 mẫu đã được mở sau khi khóa hệ số; trạng thái cache: complete_verified. Tính độc lập với lịch sử bên ngoài workspace vẫn chưa được xác minh đầy đủ. Chưa có đánh giá nghe cho VRG; không dùng nhận xét nghe của thí nghiệm joins làm kết quả cho VRG.

## Dữ liệu và khả năng kiểm tra lại

- selection_analysis/cache_audit.json, timing_resolution.json, timing_forensics.json: integrity và giới hạn timing.
- selection_analysis/scoring_lock.json, grid_scores.json, per_source_scores.json: recipe và điểm.
- selection_analysis/target_guards.json, cpu_shortlist.json: mẫu số, coverage, từng guard và shortlist.
- selection_analysis/pq_verification.json, pq_summary.json, validation_decision.json: có khi bước PQ hoàn tất.
- SCORING_PROTOCOL_V1.md, SELECTION_IMPLEMENTATION_LOCK.md và các AMENDMENT_*: kế hoạch và thay đổi.
- Báo cáo smoke cũ: REPORT_SMOKE28.md. Nhật ký: SONICMASTER_DIARY.md ở thư mục dự án.

Đường dẫn tương đối trong mục này nằm tại artifacts/10_vrg_pilot. Reproduction 1.400 mẫu và điểm/audio gốc được giữ riêng.

## FAD bổ sung sau khi khóa hệ số

Đã kiểm toàn bộ 1.400 hash PCM và 1.160 embedding. FAD dùng LAION-CLAP 1.1.7, cùng checkpoint, cùng support và cùng vị trí crop 10 giây cho các nhánh của một nguồn. Seed crop cố định theo source ID. Đây là phép đo mở rộng có kiểm soát crop; không phải đo toàn bộ 30 giây/toàn bài và không thay số FAD reproduction cũ.

| Nhóm | N mẫu/nguồn | Input ↓ | VAE ↓ | SonicMaster ↓ | Blend 0,1 ↓ | Δ blend−SM |
|---|---:|---:|---:|---:|---:|---:|
| single | 160/40 | 0.150912 | 0.281024 | 0.168270 | 0.165508 | -0.002762 |
| double_triple | 120/40 | 0.365588 | 0.503362 | 0.201598 | 0.208895 | +0.007297 |
| all | 280/40 | 0.231369 | 0.362655 | 0.178528 | 0.179700 | +0.001172 |

FAD là khoảng cách phân phối trên toàn nhóm, không trung bình các shard. FAD trên cohort nhỏ có giới hạn; số ở đây chỉ mô tả, không dùng chọn lại hệ số. SSIM tăng không đồng nghĩa mọi metric đều tốt hơn.

---

## Confirmation: 20 nguồn mới / 140 mẫu

Cập nhật 08/11/2026 15:38, giờ Việt Nam. **Blend 0,1 lặp lại mức đánh đổi nằm trong các ngưỡng pilot đã chốt trên 20 nguồn xác nhận. Đây vẫn là kết quả thăm dò; chưa đủ để gọi chất lượng nghe tốt hơn.**

Vòng này chỉ kiểm một bản đã khóa: 90% SonicMaster + 10% input. Không chạy lại grid và không chọn hệ số theo kết quả confirmation. Residual VRG đã không qua validation, nên không được gọi là phương pháp đã cải thiện.

| Nhánh | SSIM ↑ | KL ×1000 ↓ | PQ ↑ | Vượt biên trước clamp (%) |
|---|---:|---:|---:|---:|
| input | 0.772038 | 5.696032 | 7.143221 | 0.00000 |
| vae | 0.554382 | 5.705174 | 7.049340 | 1.23722 |
| sonicmaster | 0.625201 | 0.915931 | 7.743102 | 0.04466 |
| blend | 0.637663 | 0.970430 | 7.729534 | 0.01151 |

### Mức thay đổi của blend so với SonicMaster

| Phép đo | Validation: 40 nguồn | Confirmation: 20 nguồn | Quy tắc đã chốt |
|---|---:|---:|---|
| ΔSSIM | +0.012990 | +0.012463 | ≥ +0,01 |
| CI95% ΔSSIM | [+0.010165; +0.016220] | [+0.009064; +0.016280] | Báo bất định, không đổi ngưỡng |
| ΔPQ | -0.016855 | -0.013567 | ≥ −0,05 |
| CI95% ΔPQ | [-0.029684; -0.005919] | [-0.030014; +0.000863] | PQ là dự đoán, chưa phải nghe |
| Suy giảm target chuẩn hóa trung bình | 0.009061 | 0.012443 | ≤ 0,02 |
| Tăng vượt biên (điểm phần trăm) | -0.02324 | -0.03315 | ≤ +0,1 |

SSIM mô tả độ giống cấu trúc phổ với bản sạch. PQ dự đoán chất lượng âm thanh. Khi SSIM tăng nhưng PQ giảm hoặc FAD tăng, cần trình bày là sự đánh đổi; không gộp thành một kết luận “tốt hơn” chung. FAD càng thấp càng tốt.

Quy tắc confirmation giữ nguyên: điểm trung bình phải qua mọi guard; CI95% của ΔSSIM phải nằm trên 0 và cận dưới CI95% của ΔPQ phải ≥ −0,05. Không yêu cầu toàn bộ CI của ΔSSIM nằm trên +0,01. Vì vậy việc qua tiêu chí pilot không chứng minh mức tăng tối thiểu +0,01 với độ tin cậy 95%.

Mỗi nguồn được trung bình 7 biến thể trước khi gộp; CI95% dùng 2.000 lần bootstrap theo 20 nguồn, seed 20261108. Không coi 140 mẫu là 140 nguồn độc lập.

### Khả năng giữ tác dụng sửa lỗi

Mẫu số chuẩn hóa giữ nguyên từ validation. Các giá trị dưới đây tính trên các mẫu mang nhãn thuộc tính đó; phần cải thiện ở thuộc tính này không bù phần xấu đi của thuộc tính khác. Metric clipping chưa giải quyết được loại khỏi guard chính.

| Thuộc tính | Mẫu/nguồn | Suy giảm chuẩn hóa | Đủ điều kiện guard riêng? |
|---|---:|---:|---|
| clarity | 12/10 | 0.000000 | Đạt |
| boom | 5/5 | 0.023103 | Không đủ mẫu |
| airy | 9/8 | 0.000000 | Không đủ mẫu |
| bright | 8/8 | 0.023694 | Không đủ mẫu |
| dark | 5/4 | 0.000000 | Không đủ mẫu |
| mud | 5/5 | 0.114022 | Không đủ mẫu |
| warm | 9/8 | 0.002808 | Không đủ mẫu |
| vocal | 5/5 | 0.000000 | Không đủ mẫu |
| mic | 10/9 | 0.017786 | Đạt |
| xband | 21/13 | 0.006997 | Đạt |
| comp | 25/14 | 0.000000 | Đạt |
| punch | 6/6 | 0.009222 | Không đủ mẫu |
| small | 8/8 | 0.000000 | Không đủ mẫu |
| big | 7/5 | 0.001200 | Không đủ mẫu |
| mix | 13/11 | 0.007425 | Đạt |
| real | 18/15 | 0.008383 | Đạt |
| volume | 5/5 | 0.000000 | Không đủ mẫu |
| stereo | 23/13 | 0.009340 | Đạt |

Guard riêng chỉ áp dụng khi có ít nhất 10 mẫu và 5 nguồn. Các nhóm nhỏ/vắng mặt không chứng minh được khả năng giữ chất lượng của toàn bộ 19 thuộc tính.

Điều kiện CPU theo điểm trung bình: **đạt**. Điều kiện PQ: **đạt**.

### Kiểm dữ liệu và giới hạn

- Đủ 140 cache; đã đối chiếu 560 hash WAV và 140 hash float. 560 dòng CPU và 700 dòng PQ đã kiểm coverage/tính hữu hạn/hash PCM.
- Timing input–VAE: 3657/3780 cửa sổ đủ tương quan; lag đáng tin cậy [-4, 4] sample. Mẫu không đạt coverage gốc: 1539576_deg7, 311386_deg5.
- Giữ mọi mẫu và mọi gate gốc, không dịch thời gian hoặc cân gain theo clean. Bằng chứng clock không đồng nghĩa VAE giữ nguyên pha.
- Không trùng source ID hoặc clean hash chính xác với validation/reproduction đã kiểm. Chưa kiểm hết bản nhạc gần trùng hoặc lịch sử tiếp xúc ngoài workspace, nên tính độc lập vẫn có điều kiện.
- Cohort nhỏ, PQ là mô hình dự đoán và nghe còn chưa có điểm. Không coi kết quả này là tái lập thêm các baseline của paper.

Quan sát cần giữ lại: nhóm mud xấu đi chuẩn hóa 0.114022, lớn hơn mức 0,05, nhưng chỉ có 5 mẫu/5 nguồn nên không đủ điều kiện guard riêng đã khóa. Không bỏ nhóm này khỏi báo cáo và không diễn giải kết quả tổng thể thành bảo đảm cho nhóm này.

CI95% của ΔPQ chứa 0: chưa xác định chắc chiều thay đổi PQ trên tập xác nhận. Cận dưới vẫn cao hơn −0,05, nên qua ngưỡng giảm tối đa của pilot. Đây không phải bằng chứng PQ tăng hoặc hai bản nghe tương đương.

### FAD confirmation sau khóa

Cùng recipe crop 10 giây có seed theo nguồn như validation; FAD tính toàn nhóm, không trung bình shard. Cohort nhỏ và số nguồn khác validation nên không dùng chênh FAD giữa hai tập làm mức cải thiện.

| Nhóm | Input ↓ | VAE ↓ | SonicMaster ↓ | Blend 0,1 ↓ | Δ blend−SM |
|---|---:|---:|---:|---:|---:|
| single | 0.164453 | 0.337237 | 0.199995 | 0.195758 | -0.004237 |
| double_triple | 0.389020 | 0.549084 | 0.244089 | 0.248857 | +0.004768 |
| all | 0.248178 | 0.413788 | 0.215378 | 0.214809 | -0.000570 |

Ở cả validation và confirmation, blend có FAD thấp hơn SonicMaster trong nhóm một suy giảm, nhưng cao hơn trong nhóm hai/ba suy giảm. FAD gộp tất cả hơi tăng ở validation và hơi giảm ở confirmation. Đây là quan sát mô tả trên hai cohort nhỏ; chưa có CI FAD và không kết luận cải thiện FAD tổng quát.

Đã kiểm 700 hash PCM và 580 embedding; tái tính 12 FAD toàn nhóm bằng công thức SVD độc lập. Crop 10 giây không đánh giá hết đoạn 30 giây hoặc toàn bài.

### Nghe và bước còn lại

Bộ nghe 12 mẫu đã chọn trước, 60 WAV, đã sẵn sàng tại artifacts/10_vrg_pilot/listening_confirmation/LISTEN.html. A/B giấu danh tính; clean/input/VAE là tham chiếu. File nghe giữ đúng PCM đã chấm, không normalize riêng từng bản.

Chưa có nhận xét nghe cho các mẫu này. Không chuyển nhận xét “chỉ khác âm lượng” từ pilot nối đoạn sang đây. Người nghe có thể trả lời không nhận ra khác biệt hoặc không đánh giá được.

Đã kiểm đủ 60 file/hash, thời lượng WAV và cú pháp JavaScript. Trình duyệt của công cụ chặn file://, nên chưa kiểm trực tiếp giao diện/phát âm thanh; người dùng cần mở LISTEN.html trên máy và nghe xác nhận.

Không tự huấn luyện backbone/mask, không chạy SC203 hoặc mở rộng 7.000 mẫu. Nếu muốn một phương pháp mới, cần giả thuyết và thí nghiệm mới; không dùng confirmation đã mở để tinh chỉnh rồi tiếp tục gọi là test độc lập.

Bằng chứng: confirmation_cache/confirmation_lock.json; confirmation_analysis/{cache_audit,scores,summary,target_guards,pq_verification,pq_summary,confirmation_result}.json; confirmation_fad/verification.json. Các đường dẫn này thuộc artifacts/10_vrg_pilot.


## Kiểm chứng bổ sung sau pilot

<!-- vrg-followup-report-link -->

Đã hoàn tất bootstrap FAD, kiểm SSIM trước/sau xuất WAV và PQ cùng loudness. Xem `D:/SonicMaster_Reproducing/VERIFICATION_FOLLOWUP_REPORT.md`. Đây là phân tích thăm dò sau quan sát; không thay điểm hoặc quyết định pilot gốc. Nghe thật còn pending.
