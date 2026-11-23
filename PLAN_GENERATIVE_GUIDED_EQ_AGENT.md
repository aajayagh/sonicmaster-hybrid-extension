# Bàn giao nghiên cứu · Dùng SonicMaster để ước lượng EQ cho waveform gốc

## 1. Câu hỏi và phạm vi

Với bản stereo bị suy hao EQ x, tạo s = SonicMaster(x, "Make it sound better!"), ước lượng bộ lọc tĩnh hai kênh dùng chung từ (x,s), rồi xuất y = EQ(x). Kiểm tra y có sửa suy hao gốc và gần bản sạch hơn chính s trên các bài độc lập hay không. Clean dùng để chấm, với một đối chứng chẩn đoán được đánh dấu riêng. X-band là giả thuyết chính nhờ pilot. Mud, Mic, Clarity là ba câu hỏi về giới hạn. Các shelf là nhóm kiểm tra bổ sung.

SonicMaster là đối chứng chính. Input đo mức phục hồi so với tín hiệu ban đầu. FIR và các biến thể biquad giải thích cơ chế. PA1 là tiền đề lịch sử. Với Mud và X-band, router PA1 trả s, nên một hàng SonicMaster đã đại diện cho output PA1 ở các mẫu fallback này. Một phép ép FIR chạy trên Mud phải ghi là FIR thực nghiệm. Thành công được quyết định theo so sánh với SonicMaster. Việc vượt PA1, huấn luyện router, phân loại tuyến tính/phi tuyến và mở lại post-filter nằm ngoài mục tiêu đợt này.

## 2. Thứ tự triển khai và bằng chứng phải thu

### Lượt chạy ưu tiên sau kiểm tra tái lập

Trên cùng 9 mẫu X-band Dev và cùng cache prompt chung, so A1 (SonicMaster) với hai chuỗi DSP đầy đủ: P1 (B3 trong proposal, tâm/Q cố định) và P2 (B4, tối ưu thêm tâm/Q). Khởi tạo P2 từ P1, khóa cùng số băng, loss, giới hạn gain và ngân sách tối ưu; tính cả chi phí khởi tạo. A1 là đối chứng chính; P2–P1 đo đóng góp của tâm/Q. Với X-band và Mud, PA1 gốc fallback về A1, còn P1/P2 là các nhánh được chủ động chạy trong thí nghiệm. Tái dùng số liệu pilot phù hợp để kiểm tra từng khối. Ưu tiên kết quả P1/P2 đầy đủ trước khi mở rộng sang FIR lưới mịn, loss mới, toán tử nghịch hoặc routing.

### A. Kiểm kê và tái lập trước khi phát triển

- Đọc scripts/extension/branch_p.py, run_hybrid_v2_confirm.py, artifacts/10_extension/lock_v2.json, scripts/research_pilots/xband_dev_replay.py, xband_generic_infer.py, xband_generic_score.py, artifacts/xband_generic/README.md và paired_metrics.csv. Tạo mã thử riêng và run mới dưới artifacts/generative_guided_eq/<run_id>/. Bảo toàn checkpoint, routing, WAV và bảng lịch sử.
- Tái chấm từ cache đủ 9 mẫu X-band thuộc 8 bài Dev. Mốc trung bình theo mẫu với prompt chung: SonicMaster SSIM 0,6346, KL×1000 2,0158, cosine 0,04313; FIR 0,9537 / 1,8442 / 0,04465; biquad SciPy 0,9571 / 1,4645 / 0,03477. Dùng số đầy đủ trong JSON để đối chiếu từng mẫu và giải thích mọi lệch vượt 1e-5. Kiểm tra 9 receipt và hash, số frame, stereo, gain, clipping, crop, cách tính metric. Dữ liệu cũ đã xem thuộc Dev.
- Ghi chính xác pilot dùng least_squares, thử 8–12 số băng, cố định lưới tâm và Q theo mỗi cấu hình, fit gain từng băng cùng gain tổng. Pilot chấm riêng EQ trên waveform float. Phân biệt nhánh EQ này với cả chuỗi RMS → EQ → dynamics → stereo → LUFS của PA1.
- Lập bảng provenance cho prompt, model, VAE, solver, seed, precision, phiên bản evaluator. Cache s chỉ tái dùng nếu khớp hash input, prompt và cấu hình. Cache có prompt tiết lộ suy hao thuộc điều kiện lịch sử riêng. Lưu biên lai theo clip để resume.

### B. Chọn dữ liệu và khóa protocol

- Manifest từng clip gồm source_id, sample_id, split, khoảng thời gian, suy hao đơn/phối hợp, tham số sinh nếu có, hidden_clipping, vocal/instrumental, hash và đường dẫn x, s, clean. Nhãn chỉ dùng lấy mẫu, phân tích nhóm và chấm thuộc tính. Cùng một thuật toán chạy cho mọi WAV. Bất kỳ kiến trúc nào nhận số băng thật, hệ số thật hoặc tên suy hao của clip phải ghi là chẩn đoán có thông tin bổ sung.
- Tách theo bài trước khi lấy crop. Rà ID, hash nội dung clean PCM, biến thể cùng bản ghi, giao với các pilot, lịch sử dùng Dev/Confirm/Listen. Lấy nguồn đánh giá cuối độc lập với chọn biến thể và ngưỡng. Với Confirm đã xem, ghi rõ vai trò thăm dò. Nếu thiếu holdout đã xác minh, vẫn hoàn tất ablation Dev và bàn giao kết quả ở mức thăm dò.
- Ưu tiên suy hao đơn ở phép chính. Giữ danh sách trước khi infer và ghi đủ ID lỗi/thiếu. Tách clipping ẩn và suy hao phối hợp thành nhóm stress test. Lấy bài có/không vocal. Đếm số nguồn, tránh xem nhiều biến thể cùng bài là nhiều quan sát độc lập.
- Đề xuất lấy ít nhất 30 nguồn X-band mới cho pilot xác nhận nếu tài nguyên cho phép. Trước khi chốt kích thước, ước lượng độ phân tán chênh lệch ghép cặp từ Dev và báo độ rộng CI dự kiến. Mốc 30 là dự toán, không bảo đảm đủ lực thống kê. Nếu thiếu nguồn ở Mud/Mic/Clarity thì ghi n thực tế và giới hạn kết luận cho từng nhóm.
- Ghi protocol.json trước khi mở holdout: danh sách và hash split, endpoint chính, biên an toàn, seed, phiên bản metric, cách xử lý NaN/tie/missing, ngân sách thử, chọn checkpoint/cấu hình, ngân sách runtime. Chỉ clean Dev được dùng để chọn siêu tham số toàn cục.

### C. Đối chứng và implementation

| Mã | Biến thể | Điều được kiểm tra |
|---|---|---|
| A0 / A1 | Input gốc / SonicMaster prompt chung | Mức suy hao ban đầu và đối chứng chính |
| F0 | FIR PA1 chạy riêng theo pilot | Chuyển EQ từ s lên x bằng phương pháp hiện có |
| F1 / F2 | F1 chỉ giảm smoothing; F2 đổi mật độ lưới từ F1 | Tách tác dụng smoothing và độ phân giải, cùng chiều dài FIR |
| P0 | Biquad cố định theo code SciPy đã có | Mốc đã hoạt động, có thể là phương pháp cuối nếu các mở rộng không giúp |
| P1 | PyTorch fit gain với cùng lưới/Q, loss và giới hạn như P0 | Xác minh phép tính đáp ứng khả vi; báo sai lệch response, output và cost |
| P2 | Cho phép học cả tâm tần số và Q, khởi tạo từ P1 | Tác dụng của việc di chuyển băng. Khóa cùng số băng và ngân sách đánh giá loss cho P1/P2 |
| P3 | Từ P2 thêm riêng Mel-KL, riêng MR-STFT, sau đó phối hợp nếu có ích | Độ nhạy với cách khớp pseudo-reference |
| P4 tùy theo kết quả | Fit H(s) ≈ x rồi đảo có regularization và giới hạn boost | So ước lượng chiều thuận/nghịch, cùng ngân sách gain |

- Chạy A0/A1/F0/P0 trước, rồi F1/F2/P1/P2. Chỉ mở P3/P4 khi có giả thuyết cụ thể từ ca thất bại và còn ngân sách Dev. Giữ số băng như nhau khi so một thay đổi. Nếu dùng số băng khác cho phiên bản cuối thì chấm thêm chi phí và ghi cách chọn.
- Dùng (x,s) stereo căn theo quy ước evaluator, cùng cửa sổ hợp lệ. Khóa chính sách chỉnh mức trước EQ cho cả FIR và biquad. Nếu thử RMS-match thì cho cả hai cùng dùng và ghi riêng với pilot gốc. Phép thử chính chấm đầu ra cuối của chuỗi cân RMS → EQ → dynamics → stereo → LUFS. Chấm thêm ngay sau EQ trên cùng clip để đo tác động riêng của bộ lọc.
- Ước lượng tỉ lệ công suất theo tần số với trọng số thấp ở vùng input thiếu năng lượng. Tính đáp ứng cascade bằng tổng log magnitude, tối ưu tham số trên miền log-frequency. Ràng buộc tâm có thứ tự, Q dương và hữu hạn, gain từng băng và gain tổng. Kiểm tra cực SOS nằm trong vòng tròn đơn vị, response trên lưới dày, DC/Nyquist, tràn, NaN, yên lặng, near-silent và stereo đối pha. Render cùng bộ lọc lên hai kênh. Với clipping hoặc limiter ở xuất file, định nghĩa cùng chính sách cho tất cả nhánh và báo riêng điểm trước/sau xuất PCM.
- MR-STFT thật chấm trên waveform render. Nếu dùng phép xấp xỉ magnitude theo từng frame để tối ưu, gọi đúng tên và so với render thật. Với Mel-KL trong loss, định nghĩa phân bố năng lượng không âm, epsilon và chiều KL. Giữ đúng metric KL lịch sử khi báo kết quả chính. Nếu hiệu ứng pha có thể giải thích thắng/thua, render một FIR linear-phase với cùng đường biên độ của P2.
- Smoke test trên bypass 0 dB, một EQ tổng hợp biết trước, một clip thật và một ca cắt sâu. Kiểm tra gradient so với sai phân hữu hạn trên vài tham số, impulse response PyTorch/SciPy, độ dài, kênh và reproducibility. Đo riêng tạo s, fit và render cho clip 30 giây, gồm cold start và p50/p95 chạy nóng. 0,3–0,5 giây là mục tiêu kỹ thuật để đối chiếu.

### D. Tách giới hạn của bộ lọc và sai số của SonicMaster

- Tạo bảng 2×2 theo từng suy hao: FIR/biquad × fit theo s/fit theo clean. Cùng gain cap, target biểu diễn và ngân sách. Cột clean là đối chứng chẩn đoán thuận lợi; kết quả optimizer hữu hạn là mức đạt được, không chứng minh trần lý thuyết.
- X-band: dùng tham số thật chỉ để tái tạo forward từ clean, trừ DC theo kênh trước/sau lọc, cho phép một gain tổng khi kiểm chứng đến độ chính xác PCM. Nếu báo sai số đường cong, tách sai số gain tổng với sai số hình dạng đường cong và khai báo dải/weight.
- Mud: bộ sinh trả bandpass Chebyshev II 200–500 Hz. Clarity: low-pass Butterworth. Mic: đáp ứng xung micro. Kiểm tra suy hao ngoài dải và năng lượng còn lại theo bài. Nếu fit theo clean tốt nhưng fit theo s kém, xem lại tín hiệu hướng dẫn. Nếu cả hai khó phục hồi trong cùng gain cap, báo giới hạn quan sát của cấu hình này. Tránh quy nguyên nhân cho smoothing khi F1/F2 và đối chứng clean chưa ủng hộ.
- So bản clean, x, s, y ở các ca thắng/thua, lưu đường đáp ứng, gain bão hòa, peak/clipping và sai số thuộc tính. Với Mud/Mic/Clarity, ghi ca mà SSIM tăng trong khi lỗi gốc xấu đi.

### E. Chọn ứng viên, đánh giá và dừng

1. **Endpoint đề xuất** cho giả thuyết chính X-band: giảm cosine distance của phổ đa dải so với SonicMaster. Tính chênh ghép cặp y − s, gộp biến thể theo nguồn rồi bootstrap 2.000 lần theo source_id. Ưu thế xác nhận nếu cận trên CI95% < 0. Báo mức tuyệt đối, phần trăm, số nguồn thắng/hòa/thua, cùng chênh y − x để xác minh lợi ích so với input. Các nhóm khác có endpoint thuộc tính riêng theo evaluator gốc; công bố dưới dạng mở rộng thăm dò hoặc khóa quy tắc đa kiểm định nếu muốn kết luận xác nhận.
2. **Ràng buộc chất lượng đề xuất**: KL trung bình ≤1,10× SonicMaster, FAD-CLAP ≤1,10×; cận dưới CI95% của ΔPQ ≥−0,05; sai số chuẩn hóa từng thuộc tính có ≥10 mẫu xấu đi tối đa 5%. Báo ΔSSIM và CI; nếu claim đồng thời cải thiện SSIM theo chuẩn dự án, kiểm tra Δ≥0,01 và cận dưới CI >0. Ghi rõ tập dùng cho mỗi guard trước khi chạy. FAD tính trên cùng tập ID và báo n; dữ liệu nhỏ/thiếu metric thì kết luận đúng phần đo được. Có thể báo hiệu quả theo nguồn và thêm bảng có trọng số theo mẫu để đối chiếu pilot.
3. **Dev**: giữa các ứng viên đạt guard, chọn cấu hình có endpoint chính tốt và chi phí hợp lý; ưu tiên cấu hình đơn giản khi hiệu ứng tương đương theo biên đã khóa. Nếu tất cả đều kém SonicMaster theo mục tiêu hoặc liên tục vi phạm guard, dừng trước xác nhận và bàn giao kết quả âm cùng đối chứng clean. Nếu P0 đạt mà P2 không thêm lợi ích, tiếp tục bằng P0.
4. **Holdout**: khóa một ứng viên và lệnh chạy trước khi chấm. Báo nhóm X-band chính, các nhóm EQ còn lại và stress test riêng. Nếu chỉ đạt X-band, kết luận giới hạn ở X-band. Hiệu quả trên tập suy hao đã biết và khả năng tự chọn khi gặp WAV bất kỳ là hai mức tuyên bố khác nhau.
5. **Nghe mù** nếu muốn kết luận âm sắc hoặc giọng robot: lấy clip theo bài độc lập và quy tắc lấy vị trí đặt trước, cân loudness chỉ cho bản nghe, ẩn tên, đảo thứ tự y/s. Hỏi riêng tự nhiên của vocal và độ phục hồi lỗi ban đầu; có lựa chọn hòa. Ghi số bài, người, phiếu và CI theo bài. Nếu chưa có người nghe, bàn giao bộ nghe và ghi trạng thái pending.

## 3. Deliverables để nghiệm thu

- README có lệnh đã chạy từ input WAV → SonicMaster prompt chung → EQ → WAV output. CLI chính nhận input/output/config, không yêu cầu nhãn. Ghi rõ hỗ trợ độ dài, sample rate, crop và hành vi ở ngoài phạm vi.
- Manifest, audit giao nguồn, protocol đã khóa, hash code/config/cache, chi phí thực đo và lệnh resume; script tái tạo toàn bộ bảng.
- Bảng điểm từng clip, theo nguồn, theo suy hao và theo phương pháp; bảng riêng fit theo clean và fit theo s. Lưu các ca thắng/thua có ID, waveform ví dụ, thông số bộ lọc và lý do vi phạm guard.
- Kết luận ngắn theo ba câu: phương pháp có tốt hơn SonicMaster ở đâu, phần nào của thuật toán giải thích lợi ích, còn giới hạn nào do input hoặc pseudo-reference. Tách rõ kết quả đã đo, kết quả Dev và kết quả xác nhận.

## 4. Tài liệu kỹ thuật chính

- Tối ưu đáp ứng biquad trong miền phổ: [Nercessian, DAFx 2020](https://www.dafx.de/paper-archive/details/lpNI_8szg2jo_eJO1RnKxw).
- Fit DSP theo instance và multi-scale STFT: [DiffVox](https://arxiv.org/abs/2504.14735), [code tác giả](https://github.com/SonyResearch/diffvox).
- Ước lượng forward degradation phục vụ đối chứng P4: [BABE-2](https://arxiv.org/abs/2403.18636).
- Nhận dạng hiệu ứng phi tuyến nếu mở rộng sau này: [UnNAFx](https://arxiv.org/abs/2504.04751).

