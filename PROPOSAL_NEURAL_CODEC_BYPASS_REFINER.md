# Phục hồi nhạc với SonicMaster và giữ chi tiết giọng từ bản thu đầu vào

**Đề xuất nghiên cứu và kế hoạch pilot · 20/11/2026.** Trạng thái: thiết kế. Các số ở phần bằng chứng thuộc các thử nghiệm trước. Chưa train hoặc đánh giá mô hình đề xuất.

## 1. Câu hỏi nghiên cứu

Với một bản nhạc suy hao và một yêu cầu phục hồi, liệu hệ thống có tự chọn được khi nào nên giữ tín hiệu từ bản thu để giọng nghe tự nhiên hơn, trong khi vẫn thực hiện được yêu cầu phục hồi? Điểm đáng thử nhất là những bài mà quy tắc PA1 hiện chọn trả lại nguyên output SonicMaster.

**Input khi triển khai:** mix stereo suy hao x và prompt. SonicMaster đóng băng sinh s. **Output:** một mix stereo 44,1 kHz. Bản sạch y chỉ dùng để train và đánh giá. Nhãn suy hao chỉ dùng để phân tích kết quả và chạy baseline PA1. Bản triển khai không cần tách stem.

## 2. Những gì đã biết

- Người nghe nhận thấy giọng kim loại cả trên clean sau một vòng VAE. VAE có thể tự gây lỗi. Vai trò riêng của pha, độ nén, decoder và flow còn mở. Pilot fine-tune phần cuối decoder (6 bài train, khoảng 24 giây âm thanh, 48 bước) chưa đem lại khác biệt nghe được.
- EQ chuyển từ SonicMaster sang waveform input cải thiện cân bằng phổ so với input trên 16/16 bài bổ sung, thắng SonicMaster ở 4/16 bài. Đây là phép đo sửa EQ trên mix. Nó chưa trả lời câu hỏi về chất giọng.
- Trên 280 mẫu Dev, PA1 sửa 100 mẫu và trả lại s ở 180 mẫu theo quy tắc đã khóa. So với s, SSIM tăng 0,1056, FAD giảm 18,7%, KL tăng 1,4%, PQ giảm khoảng 0,015. Có 12 thuộc tính đủ mẫu để áp ngưỡng và cả 12 đều đạt. Ba thuộc tính còn lại có dưới 10 mẫu. Mud và X-band là EQ tuyến tính, dù PA1 xếp chúng vào nhóm fallback.
- Kết quả nghe được báo cáo cho **Hybrid-v2 pilot** là 6 thắng, 4 hòa trên 10 đoạn Dev ở câu hỏi về tiếng kim loại. PA1 là phiên bản khác. Cần thu hồi phiếu gốc trước khi dùng số này làm bằng chứng xác nhận.

Nguồn: [chẩn đoán VAE](outputs/vocal_clean_vae_diagnostic/listening_feedback.json), [pilot decoder](outputs/decoder_mixed_latent_pilot/listening_feedback.json), [16 bài EQ](outputs/eq_transfer_pilot/additional_tracks.json), [cấu hình và kết quả PA1](artifacts/10_extension/lock_v2.json), [paper SonicMaster](https://arxiv.org/abs/2508.03448).

## 3. Pilot tối thiểu của proposal codec-bypass

**Quan hệ với thiết kế ban đầu.** Kiến trúc gốc có nhánh T học biến đổi input, cổng g và nhánh R học tinh chỉnh output SonicMaster. Pilot lấy biến thể 1B làm điểm xuất phát, cố định T bằng DSP và R bằng output SonicMaster. Kết quả pilot chỉ kiểm tra giá trị của cổng giữa hai nhánh cố định. Nhánh T học được vẫn là giả thuyết riêng có thể so trực tiếp với DSP.

**Quyết định thực nghiệm đã có.** Theo nhóm, post-filter đã chạy thử, không đạt ngưỡng tiếp tục và đã dừng. Kế hoạch này giữ quyết định đó. Cấu hình, điểm số và ngưỡng cụ thể cần dẫn từ log gốc khi báo cáo kết quả.

Dùng u = apply_branch_p(x, s) làm đường giữ bản thu. Hàm hiện có khớp RMS, EQ, dynamics, stereo và loudness. Tính u một lần trên cả đoạn nguồn trước khi cắt crop. Giữ nguyên s làm đường dự phòng. **Pilot chỉ học một cổng**. Phương trình cho từng kênh là Y = (1−g)U + gS trong miền STFT. g=0 trả về u, g=1 trả về s. Dùng cùng g cho hai kênh. Cách này cô lập giá trị của việc lựa chọn giữa hai đường xử lý.

- Dùng cùng STFT/ISTFT, độ dài, padding và sample clock cho hai đường. Điểm xuất phát: Hann 2048, hop 512. Đặc trưng gồm tỷ lệ biên độ log, chênh RMS trước chuẩn hóa và coherence cục bộ giữa U và S. Tính coherence bằng |mean(U·conj(S))|² / (mean(|U|²)·mean(|S|²)+ε) trên cửa sổ ban đầu 9 frame × 5 bin, từng kênh rồi lấy trung bình kênh. Một ô riêng lẻ không đủ để ước lượng coherence. Đánh dấu vùng năng lượng thấp; giới hạn gain trước phép chia ở input gần im lặng.
- Dùng CNN nhỏ với đầu ra sigmoid và làm mượt cổng theo thời gian/tần số. Train trên cặp thực (x, s, y), loss phổ đa độ phân giải trên mix. Đo tỷ lệ g gần 0/1, triệt tiêu năng lượng ở vùng chuyển nhánh và độ lệch stereo. Coherence cao vẫn có thể đi cùng pha ngược. Kiểm tra thêm cos của chênh pha trước khi trộn.
- Bốn đối chứng: s; PA1 theo nhãn; u luôn bật; và một hệ số trộn cố định chọn trên validation. So sánh có/không dùng coherence với cùng kiến trúc, dữ liệu và ngân sách. Đo một đường tham chiếu dùng clean để chọn giữa **hai bản toàn đoạn** trên train-validation nhằm ước lượng lợi ích khả dĩ của chọn nhánh ở mức đoạn. Đường tham chiếu này không phải trần của cổng theo thời gian–tần số.
- Chưa đưa discriminator, Apollo, decoder mới hay proxy clean qua VAE vào pilot đầu. Đường dự phòng dùng s cố định trong kế hoạch hiện tại.

## 4. Plan thực hiện và điểm quyết định

| Bước | Công việc và sản phẩm | Quyết định |
|---|---|---|
| 0 · Khóa dữ liệu | Lưu manifest theo source_id, hash clean và lịch sử sử dụng. Chọn 100 cặp train có s thật làm pilot khả thi, chia 80/20 theo nguồn cho train/validation; báo số nguồn thực tế và độ phủ từng nhóm. Kiểm tra giao với 1.000 nguồn test. Khôi phục manifest Dev/Confirm/Listen; ghi rõ Listen có là tập con Confirm hay không và tập nào đã được xem. | Chỉ dùng phần chưa tham gia chọn mô hình để xác nhận cuối. Nếu không chứng minh được lịch sử holdout, công bố kết quả là thăm dò. |
| 1 · Kiểm tra đường tín hiệu | So hash/cấu hình baseline; thử identity, trễ 1 sample, ngược pha, im lặng, stereo rộng, gain lớn. Đo sai lệch giữa hai đầu cổng và từng baseline, kiểm tra NaN, clipping, triệt tiêu và biên overlap-add. Đo runtime/VRAM một batch trước khi dự toán GPU. | Hai đầu cổng phải tái tạo baseline tương ứng (sai số RMS tương đối < 1e−5 trên tín hiệu khác 0). Nếu vùng chuyển làm mất tiếng, sửa phép ghép trước khi train. |
| 2 · Train và đối chứng | Cố định seed, split và ngân sách update. So cổng học với bốn đối chứng trên cùng source; chọn checkpoint bằng validation từ train. Dùng 280 Dev cho quyết định nghiên cứu. Báo riêng 100 mẫu PA1 xử lý và 180 fallback, sau đó phân theo EQ, clipping, reverb, gain sâu và suy hao phối hợp. | Nếu cổng không vượt trộn cố định hoặc u luôn bật trên kết quả mục tiêu, giữ phương pháp đơn giản hơn. Nếu bỏ coherence vẫn cho kết quả tương đương, không quy lợi ích cho coherence. |
| 3 · Kiểm tra giọng và lỗi gốc | Dùng đúng các đoạn vocal đã ghi nhận để chẩn đoán. Căn thời gian, kiểm tra độ tin cậy căn chỉnh, dùng cùng separator cho clean và các điều kiện. Đo MAE đường bao và cấu trúc phổ mịn ở frame có giọng, tách riêng sai số phục hồi thuộc tính trên mix. Kiểm tra gain-only như đối chứng cho phép đo. Báo cả bài xấu đi. | Trên Dev fallback có vocal, tín hiệu để đi tiếp là giảm sai số cấu trúc mịn trung bình so với s, đồng thời đạt các ngưỡng bảo vệ lỗi gốc. Nếu chỉ tăng SSIM, chỉ báo cáo cải thiện SSIM. |
| 4 · Khóa và xác nhận | Khóa checkpoint, preprocessing, ngưỡng, bảng so sánh và quy tắc bỏ qua frame trước khi chấm holdout. Tính CI theo source_id với 2.000 bootstrap, báo Single/Double+Triple/All. Dành tập nghe độc lập cho ứng viên cuối. | Đạt metric là kết quả định lượng. Chỉ kết luận giảm tiếng kim loại khi có kết quả nghe mù phù hợp. |

**Ngưỡng xác nhận đề xuất.** Kế thừa các ngưỡng so với s: ΔSSIM ≥ +0,01 với cận dưới CI95% > 0; cận dưới CI95% của ΔPQ ≥ −0,05; FAD và KL ≤ 1,10 lần baseline; không có thuộc tính đủ n≥10 xấu đi trên 5% theo cách chuẩn hóa của project. Trên nhóm PA1 xử lý, áp lại guard thuộc tính so với PA1 và báo riêng nhóm thiếu mẫu. Trên nhóm fallback có vocal, so sánh chính là ứng viên với s vì PA1 trả nguyên s. Báo hiệu ứng ghép cặp theo bài và CI, cùng số nguồn đóng góp.

**Nghe xác nhận.** Khóa trước 20 nguồn có vocal, đoạn 8–10 giây, ≥3 người nghe, thứ tự ngẫu nhiên, cân loudness, có lựa chọn hòa. Câu chính là bản nào ít kim loại hơn ở giọng, câu phụ là chất lượng chung. Mỗi phiếu quy thành thắng=1, hòa=0,5, thua=0; lấy trung bình theo từng nguồn rồi tính CI theo nguồn, báo thêm theo người nghe. Ngưỡng đề xuất cho câu chính: điểm trung bình ≥0,65 và cận dưới CI95% >0,5. Câu phụ: cận dưới ≥0,45 (biên suy giảm 5 điểm phần trăm). Chốt biên trước khi nghe. Mẫu nhỏ có thể cho kết quả chưa phân định. Đây là điều chỉnh thiết kế nghe cho nghiên cứu mới; báo riêng khi so với giao thức PA1.

## 5. Các hướng mở sau pilot

Nếu cổng thắng ở EQ nhưng thua reverb/clip, giới hạn kết luận vào EQ. Nếu lỗi còn lại đến từ phép chỉnh DSP trên input, có thể thử T học được của proposal gốc: giữ cổng và fallback cố định, chỉ thay bộ biến đổi trên input, rồi so với DSP trên cùng bài. Nếu cả hai đường đều kém và chưa có bằng chứng input còn chi tiết hữu ích, dừng mở rộng pilot. Nếu thiếu dữ liệu s thật, so train chỉ với s thật và train thêm D(E(clean)) với cùng tập bài thật và số update. Chỉ dùng s thật khi đánh giá. Chưa có cơ sở gán hệ số tiết kiệm GPU hay kết luận latent đã mất thông tin.

## Phụ lục. Các mốc định lượng lịch sử

### Tái lập SonicMaster trên 7.000 mẫu

| Nhóm | Nhánh | FAD-CLAP ↓ (Paper / Measured) | KL×1000 ↓ (Paper / Measured) | Mel-SSIM ↑ (Paper / Measured) | Audiobox PQ ↑ (Paper / Measured) |
|---|---|---:|---:|---:|---:|
| **Single** | Input | 0,061 / 0,0626 | 3,859 / 3,8580 | 0,838 / 0,8376 | 7,321 / 7,3216 |
| **Single** | VAE Roundtrip | 0,139 / 0,1397 | 3,990 / 3,9907 | 0,574 / 0,5747 | 7,172 / 7,1724 |
| **Single** | SonicMaster (`A0`) | 0,069 / 0,0685 | 0,696 / 0,6975 | 0,624 / 0,6242 | 7,743 / 7,7431 |
| **Double+Triple** | Input | 0,184 / 0,1866 | 6,827 / 6,8256 | 0,696 / 0,6962 | 6,632 / 6,6326 |
| **Double+Triple** | VAE Roundtrip | 0,290 / 0,2911 | 6,984 / 6,9879 | 0,507 / 0,5070 | 6,501 / 6,5017 |
| **Double+Triple** | SonicMaster (`A0`) | 0,082 / 0,0810 | 1,145 / 1,1531 | 0,589 / 0,5889 | 7,654 / 7,6543 |
| **All (7.000)** | Input | 0,106 / 0,1082 | 5,131 / 5,1299 | 0,777 / 0,7770 | 7,026 / 7,0263 |
| **All (7.000)** | VAE Roundtrip | 0,196 / 0,1957 | 5,273 / 5,2752 | 0,546 / 0,5457 | 6,885 / 6,8850 |
| **All (7.000)** | SonicMaster (`A0`) | **0,073 / 0,0726** | **0,888 / 0,8928** | **0,609 / 0,6091** | **7,705 / 7,7051** |

### Hybrid-v2 PA1 trên 280 mẫu Dev

| Chỉ số (All 280 Dev) | `A0` (SonicMaster) | `A5` (Hybrid v1) | **Hybrid-v2 PA1** | Chênh lệch PA1 vs `A0` | Ngưỡng cổng | Trạng thái |
|---|---:|---:|---:|---|---|---|
| **Mel-SSIM ↑** | 0,6064 | 0,6279 | **0,7120** | **`+0,1056`** *(Single: 0,7789; D+T: 0,6227)* | `Δ ≥ +0,01` | **PASS** |
| **KL×1000 ↓** | 1,1064 | 1,1490 | **1,1215** | **`1,014×` (+1,4%)** *(Single: 0,8433; D+T: 1,4924)* | `≤ 1,10×` (1,2171) | **PASS** |
| **FAD-CLAP ↓** | 0,2145 | 0,2174 | **0,1744** | **`0,813×` (-18,7%)** *(Single: 0,1554 vs 0,2125; D+T: 0,2170 vs 0,2305)* | `≤ 1,10×` (0,2360) | **PASS** |
| **Audiobox PQ ↑** | 7,6452 | 7,6388 | **7,6302** | **`-0,0151`** *(CI95%: `[-0,0397, +0,0096]`)* | `CI95_lo ≥ -0,05` | **PASS** |
| **Attribute guards (12/15 đủ mẫu)** | — | 0 vi phạm | **0 vi phạm** | `comp -6,4%`, `airy -10,8%`, `warm -1,3%`; max xấu đi trên nhóm `n≥10`: `stereo +2,5%` | `0` vi phạm (`n≥10`, `>5%`) | **PASS** |

| Thuộc tính | `n` | Input err | `A0` err | PA1 err | Norm. Deterioration | Đủ mẫu (`n≥10`) | Guard |
|---|---:|---:|---:|---:|---:|---|---|
| `comp` | 51 | 0,04720 | 0,02488 | 0,02186 | **-6,39%** *(tốt hơn)* | Có | PASS |
| `clip` | 48 | 2,51e-6 | 1,66e-7 | 1,66e-7 | 0,00% *(nhánh S)* | Có | PASS |
| `stereo` | 40 | 0,42272 | 0,07342 | 0,08405 | +2,52% | Có | PASS |
| `xband` | 32 | 0,21378 | 0,07293 | 0,07293 | 0,00% *(nhánh S)* | Có | PASS |
| `mic` | 26 | 0,33270 | 0,01100 | 0,01199 | +0,30% | Có | PASS |
| `volume` | 22 | 0,17381 | 0,04656 | 0,04656 | 0,00% | Có | PASS |
| `punch` | 20 | 0,10654 | 0,08826 | 0,09054 | +2,14% | Có | PASS |
| `bright` | 19 | 0,00998 | 0,00613 | 0,00611 | **-0,13%** | Có | PASS |
| `dark` | 14 | 0,08722 | 0,00725 | 0,00709 | **-0,19%** | Có | PASS |
| `mud` | 14 | 0,42637 | 0,05385 | 0,05385 | 0,00% *(nhánh S)* | Có | PASS |
| `boom` | 13 | 0,40559 | 0,07259 | 0,07925 | +1,64% | Có | PASS |
| `vocal` | 12 | 0,24624 | 0,04919 | 0,05277 | +1,45% | Có | PASS |
| `warm` | 9 | 0,44724 | 0,03974 | 0,03409 | **-1,26%** | Không | Ít mẫu |
| `clarity` | 8 | 0,03623 | 0,01230 | 0,01467 | +6,52% | Không | Ít mẫu |
| `airy` | 6 | 0,00399 | 0,00167 | 0,00124 | **-10,77%** | Không | Ít mẫu |
