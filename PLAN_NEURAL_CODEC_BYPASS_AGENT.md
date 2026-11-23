# Bàn giao triển khai · Refiner phục hồi nhạc từ hai nguồn tín hiệu

Ngày 20/11/2026 · Trạng thái planned · Repo D:\SonicMaster_Reproducing

## 1. Nhiệm vụ cho agent nhận việc

Triển khai một thí nghiệm mở rộng độc lập. Với x là mix suy hao, s = SonicMaster(x, prompt) và y là mix sạch, kiểm tra liệu một mạng dùng cả x và s có giữ giọng tự nhiên hơn trong khi vẫn sửa được lỗi gốc. Sonically, x có thể giữ chi tiết giọng còn s có thể đã phục hồi được phần mà x mất. Khóa SonicMaster. Khi suy luận, chỉ được dùng x, prompt và s dẫn xuất từ chúng; clean, stem clean, nhãn suy hao, điểm metric và nhãn người nghe chỉ dùng đúng vai trò train/eval. Output là mix stereo 44,1 kHz cùng độ dài và mức tín hiệu theo quy ước của evaluator.

**Thứ tự ưu tiên của người dùng:** refiner hai đầu vào là hướng chính; PA1 đã tối ưu là đối chứng mạnh; cổng chỉ chọn giữa hai nhánh là ablation. File proposal hiện tại mô tả pilot gate-only; khi khác nhau về thứ tự triển khai, dùng kế hoạch này. Post-filter cũ đã không qua ngưỡng và dừng. Không giao lại công việc đó như một hướng mới. Không thay đổi checkpoint, bảng, split, WAV hoặc kết luận của M1/PA1.

## 2. Bước đầu tiên · Kiểm kê trước khi dùng GPU

1. Đọc GEMINI.md, PROPOSAL_NEURAL_CODEC_BYPASS_REFINER.md, PLAN_EXTENSION_CODEC_BYPASS.md; ghi rõ đây là extension. Xem scripts/extension/{routing,branch_p,mask,mix,eval_dev,eval_tables123,eval_pq_fad}.py, artifacts/10_extension/lock_v2.json và kết quả Dev trong artifacts/10_extension/dev/hybrid_v2. Đối chiếu commit/checkpoint thực tế và các hàm chấm trước khi tái sử dụng.
2. Tìm post-filter cũ trong nhật ký, manifest và run đã lưu. Ghi input thực tế, target, loss, split, ngưỡng dừng và kết quả, mỗi thông tin kèm đường dẫn. Nếu chưa tìm được thì ghi unknown. Nếu nó đã nhận x+s, xác định khác biệt kiểm chứng được trước khi train lại cấu hình tương đương. Vẫn tiến hành các bước độc lập như kiểm kê dữ liệu, kiểm thử và dựng baseline.
3. PA1 Dev để kiểm tra đối chiếu: 280 mẫu/40 nguồn, 100 xử lý và 180 fallback; SSIM 0,7120, KL×1000 1,1215, FAD 0,1744, PQ 7,6302. So với A0 cùng evaluator. Kết quả nghe 6 thắng/4 hòa thuộc pilot Hybrid-v2. Không gán cho PA1 cuối. Đối chiếu hash các WAV tái dùng. File WAV trong Confirm chỉ chứng minh đã có output, không chứng minh tập còn mù hay đã chấm xong.
4. Tạo thư mục riêng artifacts/neural_codec_bypass/<run_id>/ và mã mới dưới scripts/neural_codec_bypass/. Ghi inventory.json, baseline_provenance.json, split_audit.json, status.json. Tách số liệu lịch sử, phép đo mới và dự toán. Nếu thiếu tệp local, thử đường dẫn kho dự án đã cấu hình. Không ghi token vào log hoặc tự xuất bản dữ liệu.

## 3. Dữ liệu và khóa giao thức

- Khôi phục danh sách Dev/Confirm/Listen từ manifest thực dùng, không đoán danh tính theo số lượng. Lưu SHA-256, giao theo source_id và hash clean, đặc biệt ghi Listen có nằm trong Confirm không. Phần đã dùng chọn kiến trúc/ngưỡng phải gắn nhãn thăm dò. Khi thiếu nguồn độc lập, đánh dấu kết quả partial.
- Dùng train chính thức để tạo cặp (x, s_real, y). Đề xuất pilot 100 nguồn train riêng, một biến thể mỗi nguồn, chia 80/20 train/validation theo nguồn, seed 20261120. Phân tầng trên nhãn nguồn có vocal và loại suy hao, giữ cả EQ, reverb, clip, dynamics/stereo, tổ hợp và ca gain sâu nếu có. Ghi trước quy tắc lấy mẫu, số có sẵn và thiếu ở mỗi tầng; không chọn theo chất lượng sau infer. Pilot nhỏ dùng để chọn hướng; CI ở nhóm hiếm chỉ mang tính mô tả.
- Manifest từng mẫu: sample_id, source_id, split, prompt, degradation, vocal_tag, đường dẫn/hash x,s,y, sample rate, số frame, route PA1 theo bản khóa. Lưu quyết định train/val trước khi sinh crop. Dùng cùng crop stereo đồng bộ cho mọi biến thể; loại padding khỏi loss. Không fit gain, độ trễ hoặc filter theo clean để cải thiện điểm mix.
- Chỉ infer s thật cho cache còn thiếu với cấu hình tái lập đã khóa. Tách thời gian setup, inference, train, separator và evaluator. Log lịch sử khoảng 7,73–8,33 s/mẫu là mốc inference từ worker Kaggle trước, không phải dự toán end-to-end. Đo 3–5 mẫu và một batch train trên máy thực; báo GPU-hours và dung lượng theo từng stage trước khi mở rộng. Cache theo hash input + prompt + config. Resume theo manifest và ghi mọi lỗi.
- Khóa protocol.json trước khi chạy lựa chọn: seed, tập, kiểu metric, ngưỡng, cách xử lý thiếu mẫu, ties và đơn vị bootstrap. Nhóm khó chính đề xuất: reverb có vocal thuộc PA1 fallback; phân tích EQ, clip và mixed riêng. Nếu không đủ số nguồn, báo thiếu lực thống kê hoặc bổ sung nguồn train/validation trước khi khóa; không chọn lại nhóm chính sau khi thấy điểm.

## 4. Ma trận thí nghiệm

| ID | Hệ thống | Vai trò |
|---|---|---|
| A0 / PA1 | s gốc / PA1 theo lock | Mốc chất lượng và baseline mạnh dùng nhãn định tuyến; báo chi phí và quyền truy cập nhãn. |
| DSP / Gate | u = apply_branch_p(x,s); gate học trộn u,s, hai nhánh cố định | Phân biệt lợi ích chọn đường với lợi ích thực sự sửa tín hiệu. So thêm trộn scalar chọn trên validation. |
| X / S / XS | Cùng backbone phục hồi, lần lượt chỉ nhận x, chỉ nhận s, hoặc nhận cả hai phổ phức | Phép thử chính về giá trị thông tin bổ sung. S là control matched-budget cho nghiên cứu này. Tái dùng post-filter cũ làm mốc bổ sung nếu khác điều kiện. |
| XM | Cùng XS, đầu vào dẫn đường của đường chính chỉ giữ biên độ s | Kiểm tra vai trò của pha dẫn đường. |
| XM+R / XM+R+G | Thêm nhánh dự phòng nhận x,s; sau đó mở đặc trưng coherence của gate | Kiểm tra kiến trúc bất đối xứng của proposal gốc. Chỉ chạy sau khi XS/XM có tín hiệu; thêm từng thay đổi. |

**Kiểm soát so sánh.** Ở X/S/XS dùng một tensor input cố định và zero các kênh bị che; residual/output skip chỉ được lấy từ nguồn khả dụng (cách đơn giản nhất là cả ba dự đoán trực tiếp cùng dạng output). Kiểm tra bằng thay đổi ngẫu nhiên nguồn bị che: prediction phải y hệt. Chuẩn hóa của X chỉ được dùng x, của S chỉ dùng s. Không cho X nhìn rms(s), DSP(u) hay đặc trưng coherence. Với XM, thay Re/Im(S) bằng |S| và kênh zero phù hợp; mọi skip đến nhánh chính đều phải tuân thủ ràng buộc. Phép so XM+R có/không coherence giữ nguyên gate, số kênh và tham số, zero riêng đặc trưng coherence ở control. Cả hai vẫn có thể dùng R. Lưu số tham số, số bước và FLOPs/latency gần đúng cho từng cấu hình.

## 5. Xây dựng, smoke, train

1. Tạo entry point prepare, train, infer, evaluate và config dùng chung. Không tạo pipeline chấm mới nếu evaluator dự án đã đáp ứng. DSP chạy trên đoạn đầy đủ trước khi cắt. Ghi quy ước tensor, đơn vị, thứ tự kênh và mức tín hiệu; không peak-normalize riêng từng bản trước khi chấm.
2. Cấu hình khởi đầu đề xuất: STFT Hann 2048/hop 512, stereo; backbone CNN miền STFT có ngữ cảnh thời gian, dùng chung cho ablation. Đầu ra phổ phức hai kênh, ISTFT với length gốc. Train crop 4 s, infer overlap-add có padding và chuẩn hóa tổng cửa sổ. Luôn kiểm tra trên đầu/đuôi và chỗ nối. Pilot loss chung: trung bình L1 trên log(1+|STFT|) ở FFT 512/1024/2048/4096, hop = FFT/4, cộng 0,2 × waveform L1 ở scale gốc. Thử loss khác trong run riêng sau khi xong đối chứng.
3. Nếu thử gate: coherence = |mean(U·conj(S))|²/(mean(|U|²)mean(|S|²)+ε), cửa sổ ban đầu 9 frame × 5 bin; cùng gate cho stereo, đánh dấu bin năng lượng thấp. Coherence cao vẫn có thể ngược pha. Kiểm tra triệt tiêu khi trộn, thống kê g theo vùng vocal, tỷ lệ g gần 0/1. Ghi rõ xử lý gần im lặng và gain rất lớn.
4. Smoke: STFT→ISTFT identity (RMS tương đối <1e−5 trên tín hiệu khác 0), silence hữu hạn, mono giả stereo, stereo rộng, dịch 1 sample, ngược pha, clipping, crop cuối. Gate=0/1 khôi phục đúng hai endpoints. Chạy 1 batch backward, kiểm gradient/NaN; overfit một crop để bắt lỗi wiring. Checkpoint resume tái tạo cùng đoạn infer. Chạy thử file ngắn và clip 30 giây qua entry point thật.
5. Pilot đề xuất AdamW lr 1e−4, effective batch 8, 2.000 optimizer updates/arm, validation mỗi 200 bước; đo thời gian thực rồi khóa ngân sách chung trước so sánh. Chọn checkpoint theo loss validation cố định; log train/val theo source. Nếu ngân sách phải giảm sau smoke, điều chỉnh đồng loạt trước khi xem kết quả so sánh. Dùng cùng khởi tạo/crop order cho các arm matched; xác minh lại ứng viên và control mạnh nhất với ít nhất một seed bổ sung trước mở rộng. Proxy D(E(y)), Apollo và discriminator là ablation về sau, mỗi lần thêm một yếu tố.

## 6. Các phép đo và cách đọc kết quả

- Chấm raw mix bằng đúng evaluator dự án: SSIM, KL×1000, PQ, FAD theo tập; sai số thuộc tính Tables 1–2. Báo all/single/multiple, PA1 xử lý/fallback và nhóm khó chính. Tách nhóm mixed theo quy tắc đã khóa để không cộng trùng. Báo n mẫu và n nguồn, Δ ghép cặp với A0 và PA1, 10 mẫu tăng KL nhiều nhất kèm ID. FAD tính theo tập trên cùng danh sách, không chấm/trung bình FAD từng file.
- Đo vocal trên các vị trí 6 giây cố định cho mọi arm, chọn trước theo ghi nhận sẵn có hoặc detector khóa. Cùng separator và phiên bản cho clean/input/output; cache stem. Tách log phổ FFT 4096/hop 1024 thành đường bao cepstral (K=60) và phần dư phổ mịn. Đo MAE dB từng phần với clean trên cùng frame vocal đủ năng lượng; khóa floor, ngưỡng năng lượng và mask từ reference trước khi so arm. Kiểm tra delay qua ca dịch đã biết, báo log độ trễ và độ tin cậy. Chấm mix chính trên audio xuất; mọi phép căn chỉnh cho chẩn đoán báo riêng. Dùng clean/clean, gain-only, clean qua VAE làm control độ nhạy. Điểm phổ mịn là proxy, không phải nhãn “robot”.
- Lấy trung bình theo nguồn cho phép so proxy, bootstrap theo source_id 2.000 lần. Kiểm tra việc giảm lỗi vocal có xảy ra đồng thời với sửa đúng suy hao trên cùng bài. Bảng kết quả tối thiểu: method × stratum × n_sources × metric × delta_vs_A0 × delta_vs_PA1 × CI. Khi tính FAD và các nhóm nhỏ cần ghi rõ cỡ tập và hạn chế. Không điền số liệu ước tính vào bảng measured.

## 7. Quyết định đã định trước

- **Sau pilot:** chọn XS/XM để đi tiếp nếu trên validation s thật nó giảm trung bình sai số phổ mịn vocal so với cả X và S, đồng thời không xuất hiện vi phạm guard thuộc tính đủ n so với A0. Nếu seed đảo chiều hoặc CI rộng, ghi inconclusive. Nếu cổng đơn giản đạt cùng lợi ích và hệ thống phức tạp không có lợi ích lặp lại, giữ cổng làm ứng viên hoặc dừng kiến trúc phức tạp.
- **Trên Dev:** ứng viên phải đạt các guard dự án so với A0: ΔSSIM ≥ +0,01 (CI95% cận dưới >0), cận dưới CI95% của ΔPQ ≥ −0,05, FAD và KL ≤1,10×, không thuộc tính n≥10 nào xấu quá 5% theo chuẩn hóa trong repo. So riêng với PA1 trên 100 mẫu PA1 xử lý (guard thuộc tính cùng ngưỡng, ghi n thực tế), và trên nhóm khó chính so proxy vocal với PA1/A0, yêu cầu ΔMAE âm và không hy sinh chức năng sửa suy hao. Số 100/180 chỉ là phân nhóm Dev, không áp sang split khác. Nếu KL nổ trong reverb/EQ, lưu âm thanh và phân tích case theo ID dù trung bình pass.
- **Confirm:** khóa config/checkpoint, tập nguồn, tiêu chí và hash trước khi chạy. Dùng split độc lập đã kiểm chứng. Kết luận giá trị hai nguồn dựa trên đối chứng X/S; kết luận giá trị thực tế dựa trên PA1. Chỉ tuyên bố cải thiện nhóm nào có đủ mẫu, hiệu ứng ghép cặp và guard phù hợp. Để xác nhận nhóm khó chính, cận trên CI95% của ΔMAE vocal phải <0 so với PA1. Giới hạn phát biểu theo đúng phạm vi được kiểm chứng.
- **Nghe mù:** chuẩn bị 20 nguồn vocal độc lập, ít nhất 3 người nghe nếu tuyển được. Cùng câu hát, cùng loudness, random thứ tự và khóa đáp án. So ứng viên–PA1 là cặp chính; cặp với A0 là phụ (bỏ cặp trùng khi PA1=A0). Hai câu riêng: ít kim loại hơn, và chất lượng chung. Thắng/hòa/thua = 1/0,5/0, tổng hợp theo nguồn và báo theo người. Mốc đề xuất: câu chính ≥0,65 với cận dưới CI95% >0,5; câu tổng thể cận dưới ≥0,45. Phân tích mang tính điều kiện trên nhóm người nghe đã tuyển. Thiếu phiếu thật thì kết luận perceptual pending. Dừng ở bàn giao bộ nghe, không tạo phiếu giả hay tự gửi mời bên ngoài.

## 8. Gói kết quả cần giao

Trong thư mục run lưu manifest/splits có hash, protocol và config đã khóa, lệnh tái chạy thực tế, environment, training logs/checkpoints, run time/VRAM, outputs_manifest có hash, failures, per_sample_scores, summary và bảng ablation. Lưu danh sách file nghe mù và key riêng. Báo cáo cuối trả lời lần lượt: hai nguồn có bổ sung nhau không; cấm pha ở đường chính có giúp không; coherence có giá trị ngoài gate thường không; ứng viên giữ và vượt PA1 trên nhóm nào; còn thiếu dữ liệu nghe nào. Ghi cả thí nghiệm fail và lý do dừng. Nếu quota hoặc dữ liệu thiếu, bàn giao lệnh resume, chi phí còn lại và phần có thể hoàn tất độc lập.

**Điểm bắt đầu cho agent:** làm mục 2–3, xuất manifest pilot có kiểm tra rò rỉ và bảng đối chứng sẵn có, sau đó smoke mục 5. Chưa dùng kết quả Confirm để chọn kiến trúc. Không khởi chạy hàng loạt biến thể khi chưa kiểm tra đường tín hiệu và đo chi phí thật.
