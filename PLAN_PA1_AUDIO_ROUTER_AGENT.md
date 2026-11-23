# Bàn giao · PA1 tự chọn nhánh từ audio

## 1. Câu hỏi và phạm vi
Với một WAV stereo, có thể giữ bao nhiêu lợi ích phục hồi của Hybrid-v2 PA1 khi phải tự chọn giữa S (SonicMaster), full_mix_P và vocal_mask_P? Đóng băng SonicMaster, separator và cả ba phép tạo waveform. Chỉ học quyết định chọn nhánh cho cả bài. Công việc này độc lập với pilot refiner/post-filter. Viết mã mới dưới scripts/pa1_audio_router/, ghi kết quả dưới artifacts/pa1_audio_router/<run_id>/. Bảo toàn mọi artifact PA1/M1.

## 2. Kiểm kê và khóa điều kiện suy luận
1. Đọc artifacts/10_extension/lock_v2.json, scripts/extension/run_hybrid_v2_confirm.py, branch_p.py, mask.py, mix.py và evaluator Confirm. Tái dùng đúng thứ tự DSP, gain guard >24 dB, quy tắc PCM, checkpoint và nguồn stem của baseline. Nhãn vocal EQ chỉ một loại suy hao, khác với việc bài có người hát.
2. Kiểm tra prompt thực tế sinh S. Điều kiện triển khai chính nhận WAV x và sinh S bằng một prompt phục hồi chung cố định, nếu SonicMaster cần prompt. Ghi nguyên văn prompt và cấu hình trước khi chấm. Kiểm tra prompt đó chạy được. Mọi đối chứng trong điều kiện này dùng cùng S. Chỉ tái dùng cache có cùng hash audio, prompt và cấu hình. Nếu S lịch sử dùng prompt có nhãn, báo thí nghiệm dùng lại S ấy là điều kiện riêng với router mù nhãn, vì toàn pipeline vẫn nhận thông tin qua S. Tính lại PA1 có nhãn trên S chung để có so sánh công bằng.
3. Kiểm kê manifest, split, WAV, log, checkpoint và evaluator. Đối chiếu bảng lịch sử do nhóm cung cấp: Dev xử lý 100/280, Confirm xử lý 2.083/6.720, SSIM 0,6423 → 0,9068. FAD/PQ nhóm xử lý có 2.077 mẫu. Ghi ID thiếu theo từng bảng và điều kiện prompt. Giữ riêng kết quả lịch sử với kết quả chạy mới.
4. Tại inference, chỉ truyền waveform, sample rate và các đặc trưng suy từ x, S. Kiểm tra đổi tên file hoặc xáo metadata vẫn cho cùng quyết định. Clean, nhãn suy hao, ID bài, nhãn nghe và tên file bị loại khỏi tensor đặc trưng. Bản chạy một WAV phải tự sinh stem khi router chọn vocal_mask_P, kể cả với bài chưa có stem trong cache.

## 3. Dữ liệu, chi phí và smoke test
- Tách train, validation và holdout theo bài nguồn trước khi cắt đoạn hoặc tạo suy hao. Kiểm tra giao source_id, hash clean, các biến thể và đoạn cùng bản ghi. Dev/Confirm đã xem dùng để đối chiếu lịch sử. Dành nguồn mới độc lập cho đánh giá cuối. Ghi rõ nếu chỉ có thể kết luận thăm dò.
- Lập manifest theo ID với split, hash/path x, S, clean, prompt, suy hao đơn/phối hợp và nhánh PA1. Giữ cùng phân phối test mục tiêu, báo số nguồn từng nhóm. Nếu lấy mẫu cân bằng khi train, báo kết quả chính trên phân phối tự nhiên. Khóa seed, manifest, evaluator, tiêu chí và chính sách thiếu file trong protocol.json.
- Smoke test 3–5 bài gồm cả ba nhánh, một ca gain sâu và một ca suy hao phối hợp nếu có. Đối chiếu output router nhãn thật với baseline khi dùng đúng cache lịch sử. Kiểm tra stereo, sample rate, số frame, peak và khả năng tái chạy. Đo riêng chi phí sinh S, DSP, separator, feature và chấm metric. Dự toán giờ GPU, dung lượng và thời gian trước khi scale. Ghi lỗi theo ID, thử lại, đếm cả ca lỗi trong thống kê độ phủ.

## 4. Đối chứng triển khai theo thứ tự

| Hệ thống | Đầu vào của router | Quyết định |
|---|---|---|
| A0 | Không cần router | Luôn xuất S. |
| Full P | x, S | Luôn chạy full_mix_P để đo giá trị định tuyến. Báo đây là đối chứng luôn xử lý nếu bỏ guard. |
| PA1 có nhãn | Nhãn thật, x, S | Đúng luật PA1 đã khóa. Đây là baseline tham chiếu. |
| Router bắt chước PA1 | Đặc trưng audio | Học trực tiếp nhãn ba nhánh của PA1. |
| Router dự đoán suy hao | Cùng đặc trưng | Dự đoán đa nhãn rồi áp luật PA1, cố định ngưỡng trên validation. |
| Router dự đoán lợi ích | Cùng đặc trưng | Chạy cả ba ứng viên trên train, dùng clean để tạo target chênh SSIM, KL và lỗi thuộc tính của từng nhánh so với S. Học hồi quy lợi ích và rủi ro. |
| Oracle chẩn đoán | Clean khi phân tích | Chọn từ chính ba waveform theo quy tắc lợi ích dùng target thật. Báo mức dư địa của bộ ứng viên theo tiêu chí đó. |

Dùng chung guard >24 dB cho ba router học và báo tỷ lệ guard riêng. Với router lợi ích, xếp nhánh theo mức tăng SSIM dự đoán sau khi loại nhánh dự đoán vượt ngân sách KL hoặc làm xấu thuộc tính. Cho phép chọn S. Với nhiều suy hao, theo dõi từng thuộc tính có mặt. Để dùng khi suy luận, mô hình phải tự dự đoán rủi ro từ audio. Công thức target, chuẩn hóa theo train, lưới ngưỡng và cách phá hòa phải ghi trước khi xem holdout. FAD là metric theo tập.

## 5. Thứ tự thí nghiệm và quyết định
1. Trên train/validation, trích bộ đặc trưng nhỏ gồm mức tín hiệu, dải động, cân bằng năng lượng theo dải, mid/side và chênh lệch tương ứng giữa x và S. Bắt đầu bằng mô hình tuyến tính hoặc cây nông. Giữ cùng họ mô hình, split và ngân sách tìm tham số giữa các router. Chỉ sinh ba waveform ứng viên trên dữ liệu cần target hoặc chấm điểm. Inference bình thường chỉ chạy nhánh đã chọn sau khi có S.
2. So với luôn S và luôn P. Đo confusion matrix với route PA1, nhưng chọn theo chất lượng audio. Trên validation, lập đường cong mức can thiệp so với SSIM và KL. Khóa một điểm vận hành trước khi mở holdout. Nếu không có ứng viên đạt guard dự án và tăng SSIM so với S, dừng trước holdout, báo nguyên nhân và oracle.
3. Tiêu chí đề xuất riêng cho câu hỏi giữ được lợi ích: giữ ít nhất 80% mức tăng SSIM của PA1 trên toàn tập cùng điều kiện prompt, với KL trung bình/router chia KL trung bình/A0 ≤1,10. Tính retention = (SSIM_router − SSIM_A0)/(SSIM_PA1 − SSIM_A0). Nếu mẫu số không dương, báo chênh tuyệt đối và kết luận tiêu chí retention không đánh giá được. Mốc 80% là mục tiêu nghiên cứu mới; ghi vào protocol trước khi fit. Tra và sao chép đầy đủ các guard FAD, PQ, thuộc tính từ giao thức gốc kèm nguồn. Cần kiểm tra tất cả trước khi gọi kết quả đạt. Chọn mô hình đơn giản nhất khi các ứng viên cùng đạt và khác biệt chưa rõ.
4. Chạy holdout một lần với checkpoint đã khóa. Báo n mẫu, n nguồn, coverage từng nhánh, SSIM, KL×1000, FAD-CLAP, PQ và lỗi gốc theo đúng evaluator. Bootstrap ghép cặp theo bài nguồn cho chênh SSIM, KL và PQ. Tính FAD trên cùng tập ID cho tất cả hệ thống. Báo toàn tập, single/multiple, nhóm PA1 vốn xử lý và vốn fallback; ghi riêng nhóm hai router bất đồng. Trên từng nhóm, báo tỷ lệ can thiệp và tỷ lệ ca SSIM giảm hoặc sai số thuộc tính tăng so với S. Không gộp hai mẫu số khác nhau.
5. Xuất ca thất bại với ID, route hai bên, sai khác metric và WAV đồng bộ. Dùng một lượt nghe mù ngẫu nhiên trên cả ca đồng thuận và bất đồng để đánh giá giọng kim loại và lỗi gốc. Ghi phiếu chưa chấm là pending. Với WAV thực không có clean, kiểm thử chạy từ file đến file và đánh giá nghe riêng; không suy ra chất lượng phục hồi nhạc thực chỉ từ benchmark tổng hợp.

## 6. Bàn giao bắt buộc
- README có lệnh thực tế đã kiểm tra từ thư mục gốc cho inventory, audit, cache, train, validation, chấm holdout, tái lập bảng và chạy một WAV tùy ý. CLI một WAV nhận đường dẫn input/output và checkpoint, không yêu cầu nhãn.
- protocol.json, manifest/split audit, hash checkpoint, phiên bản môi trường, tài nguyên thực đo, bảng từng đối chứng và danh sách file thiếu. Ghi riêng chi phí offline và latency end-to-end.
- Kết luận tiếp tục/dừng theo từng gate. Nếu bắt chước PA1 đã đạt mục tiêu và router dự đoán lợi ích chưa tạo thêm lợi ích tái lập, chọn bản bắt chước. Nếu độ phủ hoặc chất lượng suy giảm do prompt chung, báo riêng ảnh hưởng từ SonicMaster và từ router bằng đối chứng cùng prompt. Chưa có holdout độc lập hoặc thiếu metric bắt buộc thì ghi rõ phần chưa xác nhận.
