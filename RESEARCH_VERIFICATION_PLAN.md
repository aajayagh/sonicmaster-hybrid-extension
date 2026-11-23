# Kế hoạch kiểm chứng trước khi chọn hướng nghiên cứu

Ngày khóa: 09/11/2026. Phiên bản 1. Phạm vi: sàng lọc bằng chứng và kiểm tra tính khả thi trước khi phát triển phương pháp mới. Thực hiện theo thứ tự và điều kiện dưới đây; không mặc định chạy toàn bộ nhánh hoặc huấn luyện mô hình.

## 1. Mục tiêu và những điều giữ cố định

Trả lời ba câu hỏi: (1) Có vấn đề cụ thể đủ lớn để nghiên cứu không? (2) Có cách đơn giản đã giải quyết được không? (3) Nhóm có dữ liệu, phép đo và tài nguyên để kiểm chứng độc lập không?

- Kết quả reproduce 1.400 mẫu/200 nguồn là dữ liệu thăm dò đã xem nhiều lần. Không gọi lại là test độc lập dù chia lại thành các fold.
- Không ghi đè audio, metric, báo cáo reproduce, VRG hoặc báo cáo chính. Chỉ đọc kết quả cũ và tạo phân tích mới trong `artifacts/13_research_checks/`.
- Không chạy lại inference của 1.400 mẫu. Không điền metric hoặc nhận xét nghe còn thiếu bằng suy đoán.
- Hạn chế của tác giả, quan sát của nhóm, giả thuyết và tính mới là bốn phần khác nhau. Chưa xác nhận gap mới nào.
- Mỗi lần đổi mục tiêu, metric, dữ liệu hoặc ngưỡng sau khi xem kết quả phải có amendment. Không sửa âm thầm protocol này.
- Nhật ký ở `SONICMASTER_DIARY.md`; trạng thái máy đọc được ở `artifacts/13_research_checks/status.json`.

## 2. Thứ tự thực hiện

| ID | Kiểm chứng | Dữ liệu / đối chứng | Sản phẩm | Điều kiện chuyển bước |
|---|---|---|---|---|
| V00 | Nguồn, nhiệm vụ và giả định có đúng không? | Paper chính thức, code đã pin, manifest và audit cũ | `READINESS_REVIEW.md`, nguồn và giới hạn | Không nhầm VAE, nhiệm vụ, metric hay nguồn độc lập |
| V01 | Chọn input hoặc SonicMaster còn dư địa cải thiện không? | Metric đã có của toàn bộ 1.400 mẫu; cả hai chính sách cố định và oracle theo từng metric | Bảng oracle, thay đổi metric chéo, khoảng tin cậy theo nguồn | Tách lợi ích thật theo metric khỏi đánh đổi; không dùng SSIM một mình để chọn hướng |
| V02 | Một quy tắc đơn giản không nhìn clean có tận dụng được không? | PQ của input, quy tắc ngưỡng; chia theo 200 nguồn | Kết quả ngoài fold, ngưỡng từng fold, đối chứng cố định | Không đưa metric so với clean vào đặc trưng; không tối ưu trên fold đánh giá |
| V03 | Hiện tượng có thể nhận ra khi nghe và còn sau kiểm soát âm lượng không? | Tối đa 12 nguồn từ ba nhóm kết quả; clean + A/B giấu nhãn, bản giữ tương quan âm lượng và bản cùng loudness | Bộ nghe và phiếu xuất JSON, chưa tự điền | Cần phản hồi người nghe; phản hồi một người là thăm dò, không phải MOS |
| V04 | Pilot mới G01: sạch / lỗi nhẹ / nặng / cục bộ | Một tác vụ đã xác định, nguồn mới, input/VAE/SM/đối chứng đơn giản/oracle | Protocol bổ sung khóa trước inference, smoke rồi pilot | Chỉ chạy khi V00–V03 xác định được lỗi và mục tiêu đủ rõ; chưa đủ thì không huấn luyện gate |
| V05 | G02: có đối chứng đúng nhiệm vụ cho codec chain không? | Phạm vi train của SM/Apollo/MPASCNet; codec, checkpoint, dữ liệu | Bảng sẵn sàng; kế hoạch MP3/AAC/MP3→AAC | Chạy pilot khi dữ liệu và baseline chạy được, mức hỏng được kiểm soát; SM ngoài nhiệm vụ chỉ là đối chứng phụ |
| V06 | G03/G04: giả định và phép đánh giá có thực hiện được không? | Stable Audio Open VAE + mean shift; bài giữ âm vị + dữ liệu hát có lời | Mục đã biết / còn thiếu / phép kiểm tối thiểu | Không chạy latent correction khi chưa có protocol tái lập mean shift; không kết luận lời sai chỉ từ ASR |

V00–V02 và kiểm tra điều kiện V05–V06 thực hiện bằng CPU/đọc nguồn trước. V03 là điểm cần người nghe thật. Có thể hoàn tất các việc độc lập trong lúc chờ nghe. Nếu bằng chứng chưa đủ thì trạng thái là `needs_listening` hoặc `needs_protocol`, không tự chuyển sang train.

## 3. V01 — Phân tích oracle đã khóa trước lần chạy này

Đây là phân tích mới sau quan sát, không phải preregistration của thí nghiệm cũ. Đọc nguyên giá trị per-sample có sẵn, không chấm lại WAV. Kiểm đủ ID, mỗi nguồn 7 variants, không trùng khóa, không NaN; đối chiếu manifest và hash file nguồn trước/sau.

- Phân tích riêng KL×1000 (thấp hơn tốt), SSIM (cao hơn tốt), Audiobox PQ (cao hơn tốt). FAD không dùng cho từng mẫu.
- Với mỗi metric, oracle chọn input hoặc SM có giá trị tốt hơn; hòa chọn SM. Oracle chỉ là trần theo metric đó trong hai lựa chọn có sẵn, không phải chất lượng nghe tối ưu.
- Báo hiệu quả oracle trên cả ba metric, tỷ lệ bypass và mức hơn cả hai chính sách cố định. Báo All, Single, Double+Triple; không chỉ chọn nhóm đẹp.
- Báo các target-property error riêng theo thuộc tính, không cộng trực tiếp các đại lượng khác đơn vị. Clipping giữ nhãn `diagnostic_only` vì giới hạn evaluator đã biết.
- Đếm trường hợp input/SM trội trên cả ba metric và trường hợp đánh đổi. Sai khác <=1e-9 chỉ là hòa số học.
- Kiểm thêm oracle KL có ràng buộc: chỉ đổi từ SM sang input nếu KL giảm >=5% tương đối so với SM, PQ không giảm quá 0,05 và SSIM không giảm quá 0,01. Đây là ngưỡng sàng lọc kỹ thuật, không phải ngưỡng nghe được đã được xác lập.
- Gộp đều theo nguồn; khi một nguồn có nhiều biến thể thì lấy trung bình trong nguồn trước. CI95% bằng 2.000 bootstrap nguồn, seed 20261109. CI không sửa được bias do dữ liệu đã được xem hoặc metric chưa tương ứng nghe.
- Sàng lọc dư địa theo KL: mức giảm oracle có ràng buộc so với SM >=5% trung bình KL, CI của delta KL nằm dưới 0, có ít nhất 20 nguồn nhận bypass. Nếu không đạt, không mở huấn luyện gate toàn đoạn theo mục tiêu này. Điều đó không bác bỏ mọi dạng lựa chọn cục bộ hoặc mục tiêu cảm nhận khác.

## 4. V02 — Đối chứng ngưỡng PQ đơn giản

Không cần backbone mới. PQ của input là đặc trưng không cần clean, nhưng chi phí tính PQ không bằng 0.

- 5 fold theo source_id: sắp theo SHA256(`20261109|source_id`) rồi chia vòng tròn; giữ đủ 7 variants cùng fold.
- Ứng viên: luôn SM, luôn input, và giữ input khi PQ cao/thấp hơn ngưỡng. Ngưỡng chỉ lấy từ các phân vị 10–90% của input PQ trong phần fit, cả hai chiều để không áp đặt quan hệ.
- Chọn quy tắc tối thiểu hóa mean KL theo nguồn trên phần fit, đồng thời mean PQ không giảm quá 0,05 và SSIM không giảm quá 0,01 so với luôn SM. Hòa chọn quy tắc bypass ít hơn. Cả hai chính sách cố định vẫn được báo dù không qua ràng buộc.
- Không dùng KL(input,clean), SSIM(input,clean), target error, điểm clean, ID hoặc nhãn oracle làm đặc trưng inference. Chỉ dùng clean-derived metric làm mục tiêu đánh giá/fit trong phần được phép.
- Ghi từng ngưỡng trước khi dự đoán fold còn lại, không sửa quy tắc sau khi đọc kết quả ngoài fold. Báo out-of-fold trên 200 nguồn và so với baseline cố định được chọn trên fold fit.
- Kết quả tốt chỉ chứng minh baseline đơn giản đáng kiểm tiếp. Kết quả không tốt chỉ bác bỏ baseline này trong thiết lập này. Không kết luận tất cả predictor đều không thể hoạt động.

## 5. V03 — Kiểm tra nghe cần hoàn tất trước khi nhận lỗi chất lượng

Chọn tối đa 4 nguồn/nhóm: input trội cả ba metric, hai nhánh đánh đổi, SM trội cả ba. Trong mỗi nhóm sắp hash theo seed; tối đa một mẫu mỗi nguồn; tránh nguồn đã có phiếu nghe cũ. Nhóm không đủ mẫu ghi thiếu, không đổi ngưỡng để lấp chỗ.

Mỗi mẫu có clean và A/B giấu tên. Mục tiêu hỏi riêng: bản nào gần clean hơn, bản nào nghe dễ chịu hơn, khác biệt ở đâu, có nghe được khác biệt không. Có lựa chọn hòa/không chắc. Không hiển thị metric hay nhóm lựa chọn trước khi nghe.

- Giữ WAV gốc, hash lại những file dùng nghe. Cắt cùng prefix theo support ngắn nhất, không fit lag/gain theo clean, không co giãn thời gian.
- Bộ gốc dùng một hệ số chung nếu cần để tránh peak; giữ tỷ lệ mức giữa clean/A/B.
- Bộ cùng loudness dùng một gain hằng mỗi bản, mục tiêu chung đủ headroom cho cả ba; không limiter, không cân từng cửa sổ. Ghi LUFS, gain, peak và hash. Phép này bổ sung, không thay kết quả gốc.
- Một người nghe đủ để tìm ví dụ cần nghiên cứu, không đủ xác nhận chất lượng tổng quát. Không coi đây là test mù tuyệt đối nếu người nghe đã quen bản nhạc.
- Khi có phiếu: đối chiếu hai chế độ âm lượng và metric, không loại phản hồi trái kỳ vọng. Chỉ mở V04 khi có hiện tượng được mô tả rõ hoặc lý do định lượng đủ cụ thể để đặt pilot mới.

## 6. V04 — Khóa pilot G01 mới sau sàng lọc

Chưa chọn một loại lỗi bằng kết quả đẹp. Ưu tiên tác vụ trong phạm vi SM có đối chứng xử lý tín hiệu đơn giản; nếu lỗi chỉ là gain thì gain-only là baseline bắt buộc. Tránh clipping làm endpoint chính khi evaluator còn chưa được giải thích đầy đủ.

Protocol bổ sung phải khóa: loại lỗi, mức mild/strong, vị trí vùng lỗi và vùng chuyển, phép xử lý tạo lỗi, prompt hoặc chế độ không prompt, mục tiêu sạch hay mastering, endpoint chính, vùng đánh giá và mức chênh tối thiểu có ý nghĩa. Không coi yêu cầu đổi âm sắc là yêu cầu giữ nguyên input.

Quy mô dự kiến: 2 nguồn smoke kỹ thuật rồi 24 nguồn pilot mới, chia theo bài và cân vocal/instrumental trong phạm vi dữ liệu. Đây là screening, không bảo đảm statistical power. Không lấy lại nguồn reproduce/VRG/demo đã mở. Kiểm cả ID và hash audio; công khai giới hạn near-duplicate/training exposure nếu không xác minh được.

Các điều kiện: clean, mild, strong, local. Baseline: input, VAE roundtrip, SM, xử lý đơn giản, oracle trong các lựa chọn cố định. Local có đối chứng mask/ghép giả; dùng cùng output SM để kiểm tác động lựa chọn thay vì đổi nhiều biến cùng lúc. Tách vùng hỏng, vùng tốt, vùng chuyển. Loại lợi ích do gain/timing/export trước khi quy cho mô hình.

Ngân sách vòng mới tối đa 2 GPU-giờ ban đầu, chỉ một job GPU tại một thời điểm; đo thực tế sau 2 nguồn rồi mới chia batch. Không cancel notebook của người khác. Không vượt ngân sách khi chưa có căn cứ/trao đổi mới. Không upload tài nguyên gated thành public.

Sau pilot, chỉ phát triển predictor nếu còn dư địa hơn baseline đơn giản và mục tiêu đáng đo. Sau khi fit phải có tập xác nhận mới; không gọi 24 nguồn đã xem là test cuối.

## 7. V05/V06 — Điều kiện mở các nhánh khác

### Codec robustness (G02)

Xác minh chức năng degrader chính thức, định dạng training và decoder. Chưa thấy MP3/AAC trong code không chứng minh audio gốc chưa từng bị nén. Phân biệt codec ngoài nhiệm vụ với distribution shift trong nhiệm vụ. Pin repo/checkpoint Apollo hoặc MPASCNet, xác minh quyền tải và smoke trước benchmark. Cùng bitrate không có nghĩa cùng mức hỏng; đo mức hỏng đầu vào và kiểm soát độ trễ/padding của codec. Pilot chỉ 2–3 cấu hình khóa trước, không quét để tìm thất bại. Dừng nếu baseline đúng tác vụ đã xử lý tốt và không còn câu hỏi cụ thể.

### Latent correction (G03)

Sửa tiền đề: SM dùng Stable Audio Open VAE, và bài Latent Geometry cũng khảo sát VAE này. Kiểm đúng revision/config. Trước adaptive correction phải tái lập fixed mean shift, có input, VAE roundtrip và restorer chuyên biệt. Dữ liệu fit/test tách theo bài và cutoff. Không chuyển kết quả từ BWE sang mọi suy giảm. Code/latent layout hoặc trọng số chưa xác minh thì chỉ báo readiness, chưa train adapter.

### Lời hát/âm vị (G04)

Đọc BABE-2, UNIVERSE++, LOUDAR và ProCraft để phân biệt nội dung lời, đặc trưng ca sĩ và chất lượng âm thanh. Xác minh dữ liệu có lời đúng, giấy phép, clean/reference và vocal đủ rõ. Không dùng transcript ASR làm ground truth. Pilot nhỏ cần người kiểm lời; nếu sai khác chỉ ở ASR hoặc lỗi không lặp lại thì dừng. Không tự gán phoneme alignment.

## 8. Cách kết luận

- `measured`: có artifact và số liệu đã kiểm; chỉ nói đúng endpoint.
- `needs_listening`: phần máy hoàn tất, cần phiếu thật.
- `needs_protocol`: chưa định nghĩa đủ phép kiểm; không khởi động GPU.
- `stopped_no_headroom`: không đạt sàng lọc cụ thể, không đồng nghĩa mọi hướng bị bác bỏ.
- `candidate_gap`: có bằng chứng sơ bộ, vẫn phải review nghiên cứu gần nhất.
- Không đánh dấu `confirmed_gap` hoặc `improved_method` từ riêng các kiểm chứng này.

Nguồn chính: [SonicMaster](https://arxiv.org/html/2508.03448v5), [RemFX](https://arxiv.org/html/2308.16177), [Latent Geometry](https://arxiv.org/html/2608.03721v1), [UNIVERSE++](https://www.isca-archive.org/interspeech_2024/scheibler24_interspeech.html), [ProCraft](https://arxiv.org/abs/2611.07423), [Apollo](https://github.com/JusperLee/Apollo). Tài liệu hướng dẫn người dùng gửi được dùng làm đầu vào, không tự coi các đề xuất trong tài liệu là lệnh chạy.
