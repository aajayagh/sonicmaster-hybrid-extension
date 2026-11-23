# Kế hoạch bàn giao agent · Reproduce SonicMaster và phân tích failure cases

## 1. Nhiệm vụ và phạm vi

Thực hiện dự án tại D:\SonicMaster_Reproducing, suy luận GPU trên Kaggle. Tiếp tục từ metadata, manifest và pilot của dự án này. Đọc GEMINI.md, AGENTS.md nếu có và kế hoạch đang lưu trong dự án trước khi sửa code. Bỏ qua notebook, checkpoint, refiner và dataset thực nghiệm thuộc dự án SC203 trước đây.

Mục tiêu chính gồm hai phần liên kết với nhau:

1. Dùng checkpoint SonicMaster chính thức để đo lại kết quả trên tập test chính thức, tạo các hàng Input, VAE roundtrip và SonicMaster của Bảng 1–3.
2. Tìm, xác nhận và phân tích các trường hợp đầu ra phục hồi kém, làm mất chi tiết, sửa sai mục tiêu hoặc thiếu nhất quán theo thời gian.

Phạm vi mặc định là tái lập đánh giá checkpoint. Huấn luyện từ đầu, fine-tune, thêm refiner và tối ưu mô hình nằm ngoài lượt triển khai này. Các thay đổi solver/prompt phục vụ chẩn đoán được ghi thành thí nghiệm bổ sung. Duy trì một đường benchmark chuẩn theo tác giả xuyên suốt.

## 2. Trạng thái đầu vào cần xác minh

- source-lock.json và repo SonicMaster đã có. Commit tham chiếu ab32b2d456e461b6bbc188d666caf2eabefa2f31.
- Trọng số chính thức amaai-lab/SonicMaster, revision 93c7fa73c4dd446c9e656af34524b5f6a416afda.
- Dữ liệu amaai-lab/SonicMasterDataset, revision 04cc0dd51ff748b41373308d9ee152796c840233.
- Manifest đã kiểm tra có 7.000 ID test thuộc 1.000 source_id, gồm 4.000 single, 2.000 double, 1.000 triple. Tự kiểm lại trước khi dùng.
- artifacts/04_pilot có CSV và 28 bộ audio từ 4 nguồn, tổng 112 WAV gồm clean/input/VAE/SonicMaster. Dùng chúng để kiểm toán và chấm lại khi hợp lệ.
- Báo cáo pilot có các kết luận cần sửa: tương quan cao chưa xác nhận hai pipeline giống hoàn toàn; peak=1 chưa xác nhận tác giả peak-normalize; EQ distance gộp chưa tương ứng cột Mic trong paper; dung lượng báo cáo đã đếm 4 loại audio nhưng ghi 3; 0,0024/0,2243 khoảng 1,07%.
- Các quan sát này là điểm bắt đầu kiểm toán. Nếu artifact hiện tại đã đổi, ghi phiên bản mới và xác minh lại thay vì áp kết luận cũ một cách máy móc.

## 3. Sản phẩm cuối cùng

| Sản phẩm | Nội dung tối thiểu |
|---|---|
| reproduction_report.md | Protocol, nguồn, môi trường, số liệu tự đo và đối chiếu paper, phần chưa hoàn thành |
| table1_eq.csv / table2_other.csv / table3_global.csv | Paper, measured, delta, N mẫu/N nguồn, quy tắc lọc và provenance cho từng ô |
| per_sample_metrics.parquet hoặc CSV | Điểm từng sample × method, ID clean/input/output và metadata lỗi |
| failure_catalog.csv | Ứng viên, xác nhận nghe, loại lỗi, timestamp, mức chắc chắn và chẩn đoán |
| failure_case_cards/ | Hồ sơ từng ca tiêu biểu, biểu đồ, prompt, audio và kết luận có giới hạn |
| failure_analysis.md | Tần suất theo metric, xu hướng theo nhóm, ví dụ và kết quả kiểm chứng giả thuyết |
| evidence_index.json | Kaggle notebook/version, dataset version, hash file và script tạo từng artifact |
| runbook.md | Lệnh tái chạy, resume, số GPU-giờ và dung lượng đã dùng |

Không đặt tỷ lệ phần trăm hoàn thành tùy ý. Dùng trạng thái measured, partial, blocked hoặc planned cho từng thí nghiệm, kèm lý do.

## 4. Giai đoạn A · Kiểm toán pilot và khóa giao thức

### A1. Kiểm tra tính toàn vẹn

Đối chiếu source-lock, notebook đã submit thực tế, log và code local. Lưu snapshot phiên bản cũ, patch/diff và hash phiên bản mới. Xác nhận 28 sample_id, 4 nguồn, số kênh, sample rate, độ dài thực, prompt và nguồn Parquet. So checksum các clean lặp lại theo nguồn trước khi deduplicate.

Kiểm tra load checkpoint: missing/unexpected keys, kích thước backbone, VAE/text encoder revision và tokenizer. Với khóa embed_tokens bị thiếu, kiểm tra tied weights trên object sau load (storage hoặc identity và giá trị). Ghi chính xác phép kiểm thay cho suy luận chỉ từ tên khóa.

### A2. Xác minh hai mẫu smoke có chênh lệch âm lượng

Với 1115233_deg2 và 1464612_deg1, ghi vùng audio được đo, độ dài, hash, preprocessing để giải thích RMS khác nhau giữa các báo cáo. Kiểm tra prompt metadata và prompt trang demo. Chạy một thí nghiệm nhỏ trên Kaggle với hai prompt/cùng input, weights, seed, precision và solver. Đo output thô, so với bản demo và clean. Nếu thử chuẩn hóa peak, đặt nhãn diagnostic và giữ file thô.

Đo RMS/peak từng kênh, số sample chạm biên PCM, sai số RMS dB, tương quan, mel-SSIM và độ trễ. Báo cả trước/sau căn chỉnh nếu dùng căn chỉnh trong chẩn đoán; benchmark chuẩn giữ preprocessing của protocol. Không suy ra nguyên nhân chỉ từ peak=1. Chấp nhận kết luận nguyên nhân chưa xác định nếu bằng chứng chưa đủ và ghi nó có ảnh hưởng đến đường benchmark test 30 s hay không.

### A3. Kiểm toán metric trên audio đã lưu

Đọc mã upstream evaluation/ và Phụ lục A.5. Viết metric_registry.yaml ghi tên hàm, input, sample rate, stereo handling, tham số, hướng tốt, đơn vị, hệ số hiển thị, filter tập mẫu và aggregation. Chỉ dùng câu “khớp protocol” cho phần đã đối chiếu.

- EQ: tỷ lệ năng lượng theo dải cho 8 thuộc tính; cosine distance 9 dải cho Mic/X-band. Giữ riêng từng cột.
- Dynamics: compression theo độ lệch chuẩn RMS frame 2048/hop 1024; punch theo onset envelope upstream.
- Reverb: modulation spectrum distance, thông số theo code/phụ lục; báo Small/Big/Mixed/Real riêng.
- Clip: sai số spectral flatness, hệ số hiển thị ×1000. Volume: sai số RMS. Stereo: sai số RMS(side)/RMS(mid).
- Mel-SSIM/KL: code upstream dùng mono 44,1 kHz, mel 128, FFT 2048/hop 512, log-mel; xác minh thứ tự đầu vào, crop, chuẩn hóa, KL theo mean log-mel, SSIM data_range=1. Ghi raw KL và cột ×1000 riêng.
- FAD: dùng CLAP đúng checkpoint/config, giữ embeddings; tính trên toàn tập/subset cần báo. Không trung bình FAD từng shard.
- PQ: pin Audiobox Aesthetics và preprocessing; giữ điểm từng clip và clean reference.

Tách điều kiện tính điểm có ý nghĩa cho từng thuộc tính khỏi metric chẩn đoán áp lên mọi mẫu. Không lấy EQ gộp đối chiếu riêng cột Mic. Kiểm tra caller upstream để biết mẫu đơn/multiple nào tạo từng hàng.

Test identity clean–clean; kiểm tra một biến đổi gain biết trước, mono/stereo biết trước, ghép nhầm ID và trường hợp im lặng/NaN. Kiểm tra per-sample output với hàm upstream trên vài mẫu. Mỗi ô N=0 phải ghi missing; bỏ mẫu cần lý do và danh sách.

### A4. Đo tài nguyên và kiểm tra resume

Đo riêng setup, tải/giải mã, warm-up, VAE, inference, chấm điểm và export. VRAM đo đúng tiến trình CUDA và synchronize khi đo thời gian. Tính storage theo từng loại file thực tế, cả đầu vào Parquet tạm và output đồng thời. Kiểm quota/giới hạn hiện hành trên tài khoản Kaggle; nếu không đọc được, ghi chưa xác minh.

Khởi đầu một shard nhỏ, kiểm tra notebook commit thành công và đọc lại output từ phiên khác. Thử resume bỏ qua đúng file đã có hash/config tương ứng. Chỉ chuyển sang full benchmark sau khi A1–A4 đạt hoặc mọi khác biệt có phạm vi ảnh hưởng được ghi rõ.

## 5. Giai đoạn B · Benchmark chính thức trên Kaggle

### B1. Chốt danh sách và quy tắc trước chạy

Giữ đúng meta.split của dataset, tách theo source_id và kiểm không trùng train/val/test. Không dùng tên container Hugging Face train để suy ra split thí nghiệm. Xác minh đủ 7 variant/nguồn, prompt và hidden_clipping. Chốt manifest, protocol hash và seed trước khi chấm toàn tập. Pilot đã xem được đánh dấu trong manifest.

### B2. Chia shard và lưu output

Chọn kích thước shard từ pilot và peak disk, giữ các variant cùng source trong một shard. Đọc weights/dataset từ version cố định. Chuẩn bị tải bằng CPU nếu thuận tiện, bật GPU cho suy luận. Mỗi job lưu structured progress theo sample_id, retry có backoff giới hạn, checkpoint danh sách đã xong và output manifest. Không silent skip hoặc tự thay file thiếu bằng nguồn khác.

Chạy input, VAE roundtrip và SonicMaster. Với inference snippet mặc định, đối chiếu upstream, 44,1 kHz stereo, 30 s, Euler 10 bước, guidance theo code, prompt chính thức. Xác minh latent sampling/scaling và precision; dùng eval/inference mode. Không đưa GT sạch vào conditioning test ngoài giao thức tác giả.

Giữ output đủ precision để tránh clipping/quantization ngoài ý muốn. Nếu benchmark chính thức yêu cầu PCM, lưu định dạng tương ứng, đồng thời ghi peak và ảnh hưởng lượng tử; dùng bản float diagnostic cho ca cần điều tra nếu ngân sách cho phép. Không tự peak-normalize toàn bộ đầu ra để làm đẹp điểm. GT deduplicate chỉ sau xác minh hash.

Mỗi shard gồm WAV hoặc archive có index, per-sample scores sơ bộ, log, environment, manifest, config và checksum. Trước khi dừng phiên, xác nhận Kaggle đã lưu outputs. Khi hỏng, chỉ chạy lại ID chưa hoàn tất hoặc artifact không hợp lệ.

### B3. Gộp và tái lập bảng

Kiểm đủ ID theo manifest, không trùng và đúng N. Ghép embeddings rồi tính FAD ở cấp tập, kiểm cách lặp hoặc giữ unique clean theo upstream. Chấm PQ và các metric thuộc tính. Lập Bảng 1–3 theo cột Single / Double+Triple / All và đúng filters. Báo thời gian metric riêng để tránh nhầm ngân sách inference với toàn quy trình.

Các mốc snippet/All của Bảng 3 để đối chiếu, theo thứ tự FAD, KL×1000, SSIM, PQ:

- Input: 0.106, 5.131, 0.777, 7.026.
- VAE reconstructed: 0.196, 5.273, 0.546, 6.885.
- SonicMaster: 0.073, 0.888, 0.609, 7.705; PQ clean 7.886.

Các mốc chỉ xuất hiện ở cột paper. Khi khác, kiểm input trước, rồi VAE, rồi model để khoanh dữ liệu/metric/weights. Chốt dung sai dựa trên repeatability và cách làm tròn trước khi xem kết quả full. Giữ báo cáo sai khác ngay cả khi chưa giải thích được. Không tune theo test để ép gần số tác giả.

## 6. Giai đoạn C · Phát hiện failure cases

### C1. Định nghĩa và tính điểm từng mẫu

| Loại failure | Dấu hiệu sàng lọc | Bằng chứng xác nhận |
|---|---|---|
| Target regression | Sai số thuộc tính mục tiêu tăng sau xử lý | Metric đúng protocol + nghe/đồ thị liên quan |
| Collateral damage | Mục tiêu cải thiện nhưng một phần khác biến dạng | Timestamp, clean/input/VAE/output, nghe có mô tả cụ thể |
| Instruction mismatch | Hành vi khác yêu cầu prompt | Kiểm metadata, so prompt trên cùng input, đánh giá nghe |
| Clean preservation | Clean qua mô hình bị thay đổi có hại với yêu cầu giữ nguyên | Thử clean/no-op riêng, prompt lưu đầy đủ, đối chứng VAE |
| Temporal inconsistency | Âm lượng/chất âm thay đổi gần biên chunk | Đồ thị theo cửa sổ, vị trí nối và kiểm soát transient tự nhiên |
| Execution failure | OOM, NaN, file hỏng hoặc thiếu output | Log và trạng thái kỹ thuật riêng với lỗi chất lượng |

Với metric error thấp tốt, tính delta_error=error_output-error_input. Với SSIM/PQ cao tốt, tính delta_score=score_output-score_input. Positive delta_error hoặc negative delta_score là ứng viên; đặt epsilon trước để loại nhiễu số dựa vào độ lặp lại của phép đo. Báo cả tỷ lệ delta>0 và tỷ lệ vượt epsilon; không chọn epsilon sau để thay đổi tỷ lệ failure. Không dùng FAD gán điểm từng clip.

Lưu ID/source, variant, prompt, attribute, genre/vocal tags nếu có, điểm input/VAE/output, delta, file/time range và lý do được chọn. Không gộp thẳng các error khác đơn vị thành một tổng điểm. Khi cần xếp hạng nhiều nhóm, rank riêng trong từng metric.

### C2. Tách khám phá và kiểm chứng

Benchmark Bảng 1–3 vẫn dùng toàn test theo paper. Cho phân tích failure, định trước một phân chia nguồn discovery/confirmation bằng seed đã lưu, đề xuất 50/50; đưa 4 nguồn pilot đã xem vào discovery và ghi cách phân chia. Dùng discovery để mô tả các giả thuyết lỗi và thiết kế thí nghiệm. Khóa danh sách giả thuyết, script và ngưỡng trước khi chạy xác nhận trên confirmation. Nguồn confirmation đã lỡ được dùng thiết kế phải đánh dấu exposed và không tuyên bố độc lập.

Các thống kê toàn test theo quy tắc sàng lọc cố định vẫn được báo, kèm nhãn mô tả. Mọi kết luận phát triển sau khi xem dữ liệu phải được phân biệt với kiểm chứng đã định trước.

### C3. Chọn mẫu nghe có hệ thống

Trong discovery, lấy tối đa 2 ca xấu nhất mỗi thuộc tính mục tiêu, gộp trùng ID; lấy thêm 10 ca metric mâu thuẫn, 10 ca chọn ngẫu nhiên và 5 ca cải thiện rõ. Giảm số ca nếu thiếu, ghi mẫu số. Cố gắng phân bố nhiều source; ghi ca chỉ có một source. Mục tiêu khoảng 40–65 case cards, có thể gộp cùng mẫu bị nhiều lỗi. Việc tuyển mẫu extreme được công khai.

Tính tỷ lệ xấu đi trên toàn tập/nhóm riêng với tỷ lệ xác nhận nghe trên mẫu được chọn. Mẫu top-worst cung cấp ví dụ; muốn ước lượng tỷ lệ lỗi người nghe cảm nhận cần mẫu ngẫu nhiên và độ bất định tương ứng. Không ngoại suy tỷ lệ nghe trên top-worst ra toàn bộ dữ liệu.

## 7. Giai đoạn D · Xác nhận bằng nghe và biểu đồ

Chuẩn bị trình phát local hoặc bộ file tĩnh, ID ẩn tên phương pháp, thứ tự phát cố định ngẫu nhiên có log; có thể cung cấp clean reference riêng để nhận diện nội dung. Chấm target repair, content preservation và preference bằng rubric rõ. Ghi rater_id ẩn danh, ngày, tai nghe/thiết bị, thứ tự, lựa chọn và mức chắc chắn. Nếu chỉ có một người nghe, báo phân tích cá nhân. Agent tự nghe khi công cụ thực sự hỗ trợ, không viết đánh giá cảm nhận dựa vào metric.

Mỗi case card cần:

1. ID, source, degradation, prompt chính xác, selection rule.
2. Audio clean/degraded/VAE/SonicMaster, hash, sample rate và đoạn thời gian đáng chú ý.
3. Metric liên quan trước/sau, N/A khi không áp dụng.
4. Hình waveform/spectrogram cùng trục và thang màu; RMS theo cửa sổ cho ca âm lượng.
5. Mô tả nghe cụ thể như phụ âm bị nhòe ở giây 12–15; tránh kết luận chung “nghe tệ”.
6. Đối chứng và giải thích đang ở mức quan sát, phù hợp giả thuyết hay có thí nghiệm hỗ trợ.

Audio phát nguyên mức âm cho bài toán volume; nếu có thêm loudness-matched để chấm màu âm, lưu bản phụ và ghi phép biến đổi. Không để bản đã xử lý phục vụ nghe quay lại metric benchmark.

Chờ người dùng cung cấp đánh giá nếu cần. Agent có thể hoàn tất phân tích khách quan và chuẩn bị case cards trước; không tạo phiếu nghe giả hoặc gán nhãn “human-confirmed” từ PQ/LLM. Không tự gửi lời mời cho người khác.

## 8. Giai đoạn E · Kiểm tra nguyên nhân bằng các run nhỏ

Chạy trên 2–3 ca thuộc các nguồn khác nhau cho mỗi giả thuyết ưu tiên, kèm ít nhất một ca đối chứng tốt. Chọn ba giả thuyết đầu dựa trên discovery; khóa rồi thử nguồn confirmation. Mỗi run thay một yếu tố, lưu cấu hình và giới hạn GPU trước khi submit.

| Giả thuyết | Đối chứng cần có | Kết luận cho phép |
|---|---|---|
| VAE gây mất chi tiết | Clean roundtrip, degraded roundtrip, SonicMaster, raw clean/input | Lỗi đã xuất hiện ở bước nào; hạn chế kết luận nhân quả khi chưa tách được tương tác |
| Prompt chi phối lỗi | Prompt chính thức, empty, paraphrase viết trước trên cùng input/seed | Độ nhạy trên các câu đã thử; không chọn prompt tốt nhất làm benchmark mới |
| Solver/steps ảnh hưởng | Euler mặc định với một biến thể có trong upstream, giữ mọi thứ khác | Thay đổi cụ thể metric và audio trên ca đã thử |
| Ghép chunk gây nhảy âm lượng | Cùng vùng audio chạy full-song và standalone; ghi biên overlap/crossfade | Lỗi có liên hệ với biên ghép và đường xử lý nào |
| Rè do xuất file | Tensor hoặc WAV float trước ghi và PCM sau ghi, đếm clipping | Lỗi do model hay bước serialization được khoanh vùng |
| Bất ổn ngẫu nhiên | Lặp cùng seed và thêm seed nếu code có sampling thực sự | Độ lặp lại/độ nhạy; nếu inference deterministic, báo điều đó |

Không dựa vào hai variant khác hiệu ứng/ngẫu nhiên để khẳng định tác động nhân quả của “nhiều lỗi hơn”. Muốn thí nghiệm có kiểm soát, cần cùng nguồn và cùng phép biến đổi nền, thêm đúng một yếu tố và mô tả đó là thí nghiệm bổ sung.

## 9. Giai đoạn F · Full-song và clean/no-op bổ sung

Ưu tiên full-song sau benchmark snippet vì paper ghi nhận giọng robot, mất chi tiết và âm lượng thiếu nhất quán; Bảng 2 full-song có reverb error tăng ở Small, Big và Real.

- Tìm manifest full-song và input/GT phù hợp từ nguồn tác giả. Nếu thiếu, lập bộ thử bổ sung có quyền dùng và nguồn rõ, giữ nhãn exploratory. Không gọi bộ tự chọn là full-song benchmark chính thức.
- Pilot vài bài dài theo cấu hình 30 s/overlap 10 s, lưu timestamp từng chunk/crossfade. So RMS/LR balance quanh biên với vùng trong chunk và clean để loại biến thiên tự nhiên của bài. Sau pilot tính lại ngân sách trước mở rộng.
- Chọn clean/no-op subset theo source bằng seed trước khi suy luận, khai báo prompt giữ nguyên và empty prompt riêng nếu dùng cả hai. So clean, clean→VAE, clean→SonicMaster. Không suy ra mức bảo toàn hoàn hảo chỉ từ SSIM. Đây là kiểm tra bổ sung với giới hạn về ngôn ngữ lệnh.
- Các phiên GPU chẩn đoán/nhánh bổ sung có ngân sách đo từ pilot và được theo dõi tách với inference 7.000 mẫu.

## 10. Phân tích thống kê và kết luận

Báo count, denominator, mean, median, phân vị và tỷ lệ regression theo nhóm. Vì 7 variant cùng một bài phụ thuộc nhau, bootstrap theo source_id, giữ các variant cùng nguồn trong mỗi lần resample. Nếu so nhiều giả thuyết, xác định nhóm phép kiểm và hiệu chỉnh đa kiểm định hoặc ghi exploratory. Không dùng số 7.000 như 7.000 bài nguồn độc lập.

Khoảng tin cậy và mức thay đổi thực tế được báo cùng nhau. Những nhóm ít mẫu/nguồn có nhãn thiếu bằng chứng. Phân tích genre/vocals chỉ dùng metadata có sẵn, kèm N; không suy ra quan hệ nhân quả từ liên hệ nhóm.

Mỗi kết luận failure phải trỏ tới số liệu hoặc case card. Phân biệt metric regression, human-confirmed degradation và lỗi kỹ thuật. Một metric xấu đi có thể mâu thuẫn với cảm nhận nghe; ghi mâu thuẫn như một kết quả.

## 11. Notebook và thứ tự triển khai

| Job | Việc cần chạy | Điều kiện hoàn tất |
|---|---|---|
| 04b_pilot_audit | Chấm lại, kiểm weights/prompt/gain, resource audit | Báo cáo sửa và nguyên nhân hoặc giới hạn của chênh lệch |
| 05_protocol_and_stage | Khóa registry/manifest, phân chia failure analysis, tải shard | IDs, data integrity, chi phí và quota được xác minh |
| 06_infer_shard_N | Input/VAE/SonicMaster theo shard | Output commit, đủ IDs, checksum, đọc lại được |
| 07_score_and_merge | Metric toàn test và Bảng 1–3 | Đủ mẫu hoặc báo thiếu, FAD gộp đúng cấp tập |
| 08_failure_discovery | Delta, ranking, thống kê, case cards | Lý do chọn mẫu và bằng chứng từng ca |
| 09_diagnostics | Run nhỏ kiểm tra giả thuyết | Controlled comparisons + xác nhận trên nguồn mới |
| 10_longform_noop | Nhánh bổ sung nếu tài sản/quota cho phép | Protocol và phạm vi công bố rõ |
| 11_final_report | Tổng hợp reproduce và failures | Mỗi số có provenance, mỗi kết luận có bằng chứng |

## 12. Cách agent làm việc và xử lý blocker

Được phép sửa code liên quan, tạo notebook/dataset riêng tư mới và chạy Kaggle trong quota khả dụng. Trước batch lớn, kiểm điều kiện A, dung lượng và dự toán; chạy một shard hoàn chỉnh rồi kiểm output trước khi tăng số job. Khi điều kiện đạt, tiếp tục các shard và scoring đến hết phạm vi chính mà không hỏi lại các thao tác thông thường. Không tự mua tài nguyên trả phí. Khi quota không đủ, lưu mọi tiến độ, báo phần có thể chạy bằng CPU và thời điểm/điều kiện tiếp tục.

Không âm thầm bỏ mẫu, đổi test hoặc thay weights để vượt blocker. Báo tài sản thiếu cụ thể và tiếp tục nhánh độc lập. Khi cần người nghe hoặc quyền truy cập từ chủ tài khoản, chuẩn bị đầy đủ giao diện/tài sản cần duyệt rồi hỏi đúng việc đó.

Cập nhật tiến độ bằng số sample/source/shard thực tế, GPU-giờ, artifact đã commit và bước tiếp theo. Kết thúc với báo cáo đo được, failure catalog và phần còn thiếu. Nhận định “reproduce thành công” chỉ áp dụng cho từng bảng/hàng đã chạy và kiểm chứng.

## Nguồn kỹ thuật

- Repo pin: https://github.com/AMAAI-Lab/SonicMaster/tree/ab32b2d456e461b6bbc188d666caf2eabefa2f31
- Weights pin: https://huggingface.co/amaai-lab/SonicMaster/tree/93c7fa73c4dd446c9e656af34524b5f6a416afda
- Dataset pin: https://huggingface.co/datasets/amaai-lab/SonicMasterDataset/tree/04cc0dd51ff748b41373308d9ee152796c840233
- PDF ICML 2026 người dùng đã cung cấp: C:\Users\ADMIN\Downloads\melechovsky26a.pdf. Dùng bản này để đối chiếu bảng và phụ lục, kiểm version nếu lấy bản online.
