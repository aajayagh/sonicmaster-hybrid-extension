# Kế hoạch: kiểm tra clipping → pilot nối đoạn → thử VRG

Ngày: 08/11/2026 15:36 (giờ Việt Nam). Trạng thái: QUANTITATIVE_COMPLETE_LISTENING_PENDING. A audit/supplement hoàn tất. B full12 và đối chứng âm lượng/xuất WAV hoàn tất; nghe chi tiết còn pending, không mở rộng joins. C đã kiểm cache280/40 nguồn (1120WAV+280float) và 3640 dòng điểm CPU; giữ nguyên3 mẫu low-correlation và gate gốc=false. Validation40 và confirmation20 của blend0,1 đã hoàn tất metric/PQ/FAD; nghe còn pending. Residual không qua validation. Kế hoạch gốc dưới đây được giữ; thay đổi ghi trong amendments và SELECTION_IMPLEMENTATION_LOCK.md. Xem artifacts/10_vrg_pilot/status.json và SONICMASTER_DIARY.md.

## 0. Phạm vi và nguyên tắc

Thứ tự bắt buộc: A — audit clipping; B — pilot full-song và vùng nối; C — VRG không huấn luyện. Huấn luyện mặt nạ chỉ là nhánh mở rộng có điều kiện, không nằm trong lượt thử ban đầu.

Toàn bộ code, manifest, báo cáo, notebook và kết quả mới lưu trong D:/SonicMaster_Reproducing. Inference nặng chạy Kaggle; tận dụng CPU/audio đã tải cho audit. Không chạy SC203, không tự mở rộng lên 7.000 mẫu, không tái huấn luyện backbone. Không yêu cầu phân loại failure cases; dùng observation, giới hạn và bằng chứng nghe. Plan này cập nhật điều kiện bắt đầu trong kế hoạch VRG cũ: không bắt buộc phải có failure catalog hoặc chứng minh nguyên nhân VAE trước thử nghiệm kiểm chứng giả thuyết.

Không thay output hoặc điểm reproduction gốc. Mọi phép đo diagnostic có nhãn riêng. Một giai đoạn được đóng khi có kết luận dựa trên bằng chứng, kể cả kết luận chưa giải thích hết hoặc không phát hiện hiệu ứng. Clipping chưa khớp paper không chặn B/C, miễn metric được giữ cố định và giới hạn được ghi rõ.

Đã có: 1.400 mẫu/200 nguồn, 5.600 WAV kiểm hash; FAD/PQ hoàn tất; diagnostic clipping211; 7 nhận xét nghe; 13 demo full-song tác giả; người dùng chưa nhận thấy bất thường ở các vùng nối đã nghe trên 2 demo. Các demo không có log inference để xác nhận mốc nối và có output dài hơn clean 10,25–27,98 giây. Không dùng chúng làm bằng chứng paired reproduction.

### Cấu trúc output

- artifacts/08_clip_audit_v2/
- artifacts/09_join_pilot/
- artifacts/10_vrg_pilot/
- scripts/research_pilots/ và kaggle_pipeline/research_pilots/
- PLAN_CLIP_JOINS_VRG.md (tài liệu này), SONICMASTER_DIARY.md.

Mỗi phase có protocol.json, source_lock.json, manifest.csv, status.json, per_sample.csv, summary.json, REPORT_VI.md. status: planned/running/partial/complete/blocked; không dùng complete chỉ vì kernel kết thúc. Receipt ghi commit, hash nguồn/weights, config, phiên bản gói, artifact ID, thời gian GPU/CPU, dữ liệu thiếu và lý do.

## A. Kiểm tra lại clipping

### A1. Câu hỏi cần trả lời

1. Code phát hành và mô tả A.5 có cùng quy ước không? Có tìm được phiên bản evaluator thực sự tạo bảng paper không?
2. Phần chênh nào do công thức, cách tổng hợp, kênh, floor số học hoặc lựa chọn tập?
3. Metric phục hồi waveform thông dụng cho kết luận gì trên cùng audio?

Không tối ưu công thức để khớp số paper. Nếu không truy được environment/receipt của tác giả, kết luận đúng là chưa xác định hết, không sửa số bảng.

### A2. Dữ liệu và khóa baseline

- Dùng đúng 211 mẫu clip đã có; cả clean/input/VAE/SonicMaster, receipts và hash cũ.
- Đọc manifest để tách explicit clip đơn, clip phối hợp; kiểm hidden clipping ở nhóm khác và quy tắc tác giả có đưa vào mẫu số không. Không tự thêm vào N=211.
- Khóa FFT2048, hop512, sample rate44.1kHz, mono policy, window/center/pad/floor và thứ tự chuẩn hóa theo evaluator. Ghi chính xác chiều dài từng nhánh.
- Lấy 12 mẫu smoke có phân tầng explicit đơn/phối hợp và mức năng lượng, chọn bằng hash ID với seed clip-audit-20261108; không chọn theo điểm output.
- Không cần chạy lại inference211 nếu hash và provenance hợp lệ.

### A3. Kiểm code, số học và dữ liệu

- Truy commit/tag/history evaluator trong repository chính thức; đối chiếu A.5 trang18 và Table2 trang7 của đúng PDF đã dùng. Ghi khác biệt phiên bản nếu có. Không tự liên hệ tác giả bằng email.
- Tái lập upstream: abs(mean(flatness(output)) − mean(flatness(clean))) ×1000.
- Đường U: S=abs(STFT)^2, power=2 (upstream).
- Đường P: S=abs(STFT)^2, power=1 (power-spectrum convention của librosa).
- Đường M: S=abs(STFT), power=2. M phải khớp P trong tolerance số học phù hợp. Lưu max error; kiểm cả silence/gần silence.
- Lưu flatness thô từng frame của clean/input/VAE/output, không chỉ điểm cuối, để xem vùng silence/floor chi phối ra sao.
- Phân biệt abs(mean difference) với mean(abs(frame difference)). Chỉ so theo frame trên cùng timeline/common support đã xác minh; báo riêng vì đây là phép đo khác.
- Tách ba thí nghiệm đơn yếu tố: mono vs từng kênh rồi trung bình; toàn support vs common-prefix; toàn frame vs loại silence bằng mask từ clean được khóa trước. Lấy amin mặc định làm chính, các amin diagnostic cố định {1e-10,1e-8,1e-6}. Không mở full Cartesian search.
- Báo độ nhạy tại 12 mẫu trước, sau đó chỉ mở rộng những đối chiếu đã định nghĩa có ý nghĩa lên211. Không chọn biến thể vì số gần paper.
- Kiểm clipping/quantization khi xuất PCM16, gain trước xuất, silence/tail, file gốc FLAC nếu có. So roundtrip I/O với fixture đã biết để tách lỗi xuất file khỏi mô hình.

### A4. Metric bổ sung

- Waveform SDR = 10 log10(sum(c^2)/sum((c-y)^2)); định nghĩa rõ zero-mean policy và cách gộp stereo (sum năng lượng hai kênh). Delta SDR = SDR(output,clean) − SDR(input,clean).
- SI-SDR tính scale projection theo công thức khóa; báo thêm, không dùng thay volume. Không dùng BSS-Eval có bộ lọc cho phép biến đổi tín hiệu mà không ghi rõ.
- Kiểm sample rate, độ dài, kênh và lag trước. Bảng chính không alignment bằng clean để làm đẹp điểm. Nếu có lag cố định từ pipeline, sửa nguyên nhân và ghi biến thể; diagnostic alignment tách riêng.
- Nếu có clipping mask và threshold thật từ quá trình tạo input, báo lỗi vùng clipped và vùng reliable. Không suy mask từ |x|≈1 sau peak normalization.
- Nếu metadata không đủ tái dựng mask, chạy thêm phép thử synthetic clipping trên 12 clean cố định, threshold cố định, chỉ để kiểm phản ứng metric; không gọi đó là benchmark gốc.
- MR-STFT/LSD có thể là bổ sung sau; PEAQ/PEMO-Q chỉ chạy nếu có implementation hợp lệ được xác minh, không là điều kiện bắt buộc đóng A.

### A5. Kiểm thử và đầu ra

Fixture: clean=self cho error0; M/P tương đương; sine/hard-clipped sine và silence không sinh NaN; sai số gain/timing được phát hiện. Không bắt metric flatness phải tăng đơn điệu theo mức clipping nếu công thức không bảo đảm điều đó; ghi quan sát.

Xuất formula_comparison.csv, frame_diagnostics.npz, subset_counts.json, sdr_scores.csv, environment.json, REPORT_VI.md. Bảng giải thích có 3 mức: đã xác minh / giả thuyết / chưa đủ dữ liệu. Kế thừa kết quả211 cũ: upstream SM0.001394, P0.228177, paper1.506 (×1000); không tái công nhận chúng là số mới nếu chưa chạy lại.

**Điều kiện đóng A:** tái lập số upstream; công thức P/M được kiểm; có phân tích độ nhạy và dataset; metric bổ sung có kết quả hoặc lý do unavailable; còn chênh được ghi rõ. Không đòi phải khớp paper bằng mọi giá.

## B. Pilot full-song và vùng nối

### B1. Giả thuyết

Sau khi kiểm soát timeline, sai lệch loudness/phổ tại vùng overlap có lớn hơn vùng ngoài overlap không? Hiệu ứng có đổi khi dịch lịch chia đoạn không và có nghe thấy không?

Paper đã đánh giá full-song. Đây là kiểm tra tính nhất quán theo thời gian, chưa khẳng định research novelty. Việc người dùng chưa nghe thấy khác biệt trên2 demo là observation hợp lệ; không chọn bài cố tìm lỗi.

### B2. Cohort

- Audit13 demo chỉ để hiểu duration/tail và kiểm công cụ, không đưa vào kết quả inference do nhóm thực hiện.
- Pilot mới:12 nguồn validation có clean và degraded full-song khớp timeline, 2–5 phút nếu nguồn cho phép; 1 variant/nguồn.
- Ưu tiên đủ EQ/reverb/dynamics/amplitude và lỗi phối hợp nếu có; ghi actual coverage, không ép quota bằng cách đổi nhãn.
- Chọn cố định bằng hash source ID với seed join-pilot-20261108 trước xem output. Hai bài smoke là tập con của12.
- Nếu dữ liệu full-song chính thức không có clean/input paired hợp lệ: dùng clean chính thức có quyền sử dụng, tạo degradation bằng code chính thức có tham số/seed/mask lưu lại. Ghi cohort synthetic extension, không gọi là tái lập test full-song paper. Nếu thiếu clean, báo thiếu và không giả paired metrics.
- Các nguồn đã dùng B được đánh dấu exposure; không dùng làm confirmation VRG sau này.

### B3. Sửa vấn đề timeline trước đo

1. Ghi source hash, waveform length, sample rate, channels và duration. Xác nhận sạch/suy giảm có cùng nội dung và timeline.
2. Đọc lại code gốc:30s, overlap10s, stride20s; lưu từng chunk: input_start/end, pad_frames, encoded/decoded_frames, previous-condition span, overlap interval và vị trí output thực.
3. Giữ raw upstream output. Bản duration-controlled chỉ bỏ padding khi chứng minh mapping; không cắt phần nhạc hợp lệ hoặc mặc định dư chỉ là silence.
4. Kiểm sai lệch độ dài từng decoded chunk và drift cộng dồn. Nếu có drift, xuất mapping; không dùng dynamic time warping để ép metric tốt hơn. Nếu cần bù độ dài chunk, gắn nhãn pipeline correction và chấm riêng.
5. Kiểm lag ở đầu/giữa/cuối. Không đủ căn chỉnh thì chặn paired waveform/SSIM; vẫn có thể báo duration hoặc metric không reference với giới hạn.

### B4. Các nhánh

- I: input, không xử lý.
- V: VAE roundtrip theo cùng lịch chunk/crossfade, để xem ảnh hưởng codec/ghép.
- S0: SonicMaster lịch gốc30/10, conditioning từ đoạn trước, Euler10 và cấu hình checkpoint baseline đã khóa.
- Sshift: cùng cấu hình nhưng dịch lịch chia5s; prepend5s được ghi rõ, loại tiền tố sau khi xác minh mapping; chấm vùng nội bộ chung, loại vùng đầu/cuối chịu padding theo rule khóa trước. Khác biệt có thể đến từ ngữ cảnh/conditioning và nối, không quy riêng cho crossfade.
- Chưa quét overlap, solver, guidance hoặc nhiều phương pháp ghép trong pilot.

Mọi nhánh chạy trên cùng12 bài; giữ chunk output trước ghép để phân tích overlap. Không thay dtype để tiết kiệm mà không ghi amendment. Kiểm quota hiện tại, đo smoke thực trước ước lượng batch.

### B5. Metric và cách lấy vùng

**Toàn bài:** metric thuộc tính paper phù hợp nhãn; KL/SSIM nếu timeline hợp lệ; integrated loudness, LRA, true peak. PQ dùng cửa sổ30s liên tiếp bao phủ toàn bài, cửa sổ cuối có policy padding/gating ghi rõ; tổng hợp theo bài, không gắn nhãn trực tiếp là PQ paper full-song nếu protocol khác. FAD trên12 bài chỉ là exploratory, không primary endpoint; nếu chạy phải xác nhận CLAP không chỉ crop10s rồi gọi là full-song và không gộp trung bình FAD shard.

**Theo thời gian:** EBU short-term loudness3s, hop0.1s; momentary400ms bổ sung; error e(t)=L_output(t)−L_clean(t). Mask low-energy/không đo được lấy từ clean, dùng chung tất cả nhánh; ghi coverage, không cho NaN thành0.

**Phổ:** log-magnitude STFT error trên cửa sổ3s cùng hop; FFT2048/hop512/Hann, log floor cố định, trung bình hai kênh; metric diagnostic riêng, không gọi là metric upstream nếu khác recipe. Theo dõi thêm side/mid chỉ ở vùng có đủ năng lượng.

**Vùng nối:** lấy chính xác toàn overlap theo log, cộng vùng lân cận2s hai phía để nghe/biểu đồ. Control window ngoài overlap lấy cùng bài, ghép tương đồng về loudness và onset density dựa vào clean, không dựa vào output; thiếu control thì báo thiếu. Không chỉ xét một sample ở chính giữa crossfade.

Primary endpoint: D_song = mean(|e(t)| trong overlap) − mean(|e(t)| ở control matched), tính một số mỗi bài. Endpoint phụ: chênh lỗi phổ, percentile95 của |e(t)|, mismatch giữa hai chunk trên cùng timeline trước crossfade. Báo tác động input/VAE và lịch dịch để tránh nhầm chuyển đoạn tự nhiên của bài.

### B6. Nghe và quyết định

- Bộ nghe pilot: ít nhất1 vùng overlap +1 control mỗi bài; cắt ngữ cảnh20s, order ẩn danh A/B và random hóa. Lưu mapping riêng. Người nghe cho điểm thay đổi âm lượng/màu âm/độ mượt và mức chắc chắn, cho phép không nhận thấy.
- Bộ chính giữ loudness gốc, cố định volume thiết bị. Bộ normalize chỉ bổ sung nghe chi tiết, không thay bộ chính. Nếu chỉ có một người nghe thì gọi exploratory, không gọi formal MOS study.
- Tổng hợp theo bài; bootstrap2.000 lần theo source, không coi cửa sổ hay các lịch của cùng bài là mẫu độc lập. Với n12, CI chỉ thăm dò.
- Tiêu chí định hướng trước khi xem output: đáng mở rộng nếu ít nhất8/12 bài có D_song>0, mean D_song≥0.5 LU và CI95% không qua0, kèm chênh phổ nhất quán hoặc nghe mù có dấu hiệu liên quan. 0.5 LU là ngưỡng thực tiễn do nhóm đề xuất, KHÔNG phải ngưỡng nghe phổ quát. Nếu đổi ngưỡng sau pilot, khóa lại trên cohort xác nhận mới.
- Nếu kết quả yếu/không ổn định, không nghe thấy hoặc không đủ căn chỉnh: báo chưa có bằng chứng, dừng mở rộng joins; vẫn sang C. Không cần chứng minh có lỗi nối để được thử VRG.

Output: chunk_map.jsonl, fullsong_scores.csv, window_scores.parquet, boundary_vs_control.csv, plots/loudness_by_song.*, LISTEN.html, listening_responses.csv, REPORT_VI.md. Tất cả mốc nghe từ log của run mới, không ước lượng như demo.

## C. Thử VRG: residual không học trước

### C1. Giả thuyết và đối chứng

Ký hiệu x=input, r=VAE(x), s=SonicMaster(x), c=clean. Residual x−r có thể chứa thông tin hữu ích và cả degradation/phase error. Không đồng nhất residual với chi tiết sạch.

Các nhánh: copy-input x; VAE r; SonicMaster s; blend (1−a)s+a*x; residual scalar s+l*(x−r). a,l cố định grid {0,0.1,0.25,0.5,0.75,1}. Đây là hai phương pháp khác nhau, cần giữ cả hai để xem residual có ích hơn trộn input đơn giản không. l=0 phải tái tạo baseline.

Chưa dùng clean/prompt label/metric output để chọn hệ số riêng cho mỗi test sample. Clean chỉ dùng chấm validation/confirmation. Chưa fine-tune VAE/backbone hoặc học mask.

### C2. Chia dữ liệu

- 40 nguồn official validation, 7 variants/nguồn =280 mẫu để chọn hệ số; 20 nguồn official test chưa tiếp xúc, 7 variants=140 mẫu để xác nhận một lần.
- Smoke4 nguồn là tập con của40. Kiểm coverage19 thuộc tính; cohort pilot nhỏ không bảo đảm đủ mọi nhóm, báo actual N.
- exposure_registry gồm nguồn đã xem trong benchmark1400, SC203 (chỉ metadata exposure), full-song demo/pilot và các phiên trước. Không gọi tập đã dùng thiết kế là unseen. Kiểm source ID và duplicate audio; mọi variant cùng nguồn cùng split.
- Nếu không truy được lịch sử exposure, confirmation chỉ là exploratory và ghi rõ. Không tái dùng1400 đã xem để chọn l/a rồi công bố như cải thiện trên test độc lập.

### C3. I/O và metric

- Cache x,r,s một lần từ checkpoint/config baseline. Dùng full precision/config như reproduction đã khóa; ghi actual precision. Không chạy inference lại cho từng hệ số.
- Trước cộng residual: xác minh timing, sample rate, kênh, support; lấy common support có quy tắc cho tất cả nhánh. Không shift/gain-fit bằng clean cho từng mẫu. Nếu residual lệch pha/timeline, sửa pipeline trước.
- Tính float32; lưu peak/out-of-range trước clamp, bản float diagnostic. Xuất PCM16 theo cùng policy baseline, không peak-normalize ngầm. Nếu clamp ảnh hưởng kết luận thì báo.
- Chấm cùng evaluator thuộc tính, KL/SSIM, SDR/SI-SDR diagnostic từ A. Clipping upstream/P luôn hai cột riêng; không dùng clipping unresolved làm primary endpoint.
- PQ trên các hệ số shortlisted sau vòng CPU; FAD chỉ sau khóa candidate, dùng toàn cohort với protocol giống nhau. Không chọn candidate bằng FAD nhiễu trên cohort nhỏ.
- Primary benefit: SSIM tăng so với s nhưng phải thỏa ràng buộc giữ khả năng sửa mục tiêu; thêm PQ và nghe để kiểm cảm nhận. Không coi tăng SSIM nhờ gần x hơn là đủ.

### C4. Chọn hệ số và điều kiện sang confirmation

Trên validation, chuẩn hóa mức xấu đi từng thuộc tính theo mean input error của chính validation; floor và thuộc tính năng lượng gần0 phải xử lý theo rule từ smoke, không tùy chỉnh theo candidate.

Ngưỡng pilot đề xuất, khóa trước xem candidate: mean SSIM tăng≥0.01 so với s; trung bình suy giảm target error chuẩn hóa≤0.02; không thuộc tính đủ N tăng error>0.05 đơn vị chuẩn hóa; PQ giảm không quá0.05 điểm; không tăng tỷ lệ out-of-range trước clamp quá0.1 điểm phần trăm. Đây là ngưỡng vận hành thử nghiệm, không phải ngưỡng cảm nhận đã được xác nhận. Báo cả giá trị tuyệt đối, CI và N; không che subgroup ít mẫu.

Chọn một a và một l toàn cục theo ràng buộc trên; tie-break ưu tiên hệ số nhỏ hơn. Nếu không candidate đáp ứng, kết luận scalar residual chưa có lợi ích phù hợp; không tự động huấn luyện.

Trước confirmation: khóa hệ số, recipe, cohort và hash output. Chạy20 nguồn mới một lần, báo toàn bộ nhánh và delta; bootstrap theo source. Nếu CI chưa rõ, gọi inconclusive, không quay lại chỉnh rồi đo lại cùng test dưới nhãn xác nhận.

Nghe12 mẫu chọn trước từ confirmation, có cả các nhóm tác động, so sánh ẩn danh baseline/blend/residual. Nếu người dùng chưa nghe thì artifact chỉ listening pending, không bịa kết luận. Kết quả có thể là residual hữu ích, blend cũng tốt tương tự, hoặc không có cải thiện — cả ba đều phải báo.

### C5. Khi nào mới cân nhắc learned gating?

Chỉ sau khi dữ liệu train/validation cho thấy residual có ích cục bộ và không làm giảm khả năng sửa lỗi quá mức. Khi đó lập amendment mới: source-disjoint train/val/test, mask nhỏ, baseline scalar và blend giữ lại, kiểm literature novelty. Không dùng confirmation đã mở để đánh giá model mask mới. Chưa mở học mask trong plan thử nghiệm này.

## D. Thứ tự job và tài nguyên

1. A00 source/formula audit CPU → A01 smoke12 CPU → A02 paired211 CPU → A03 report.
2. B00 cohort/timeline CPU → B01 smoke2 Kaggle GPU → kiểm output length/map → B02 inference12 (tính cả smoke, không chạy trùng) → B03 CPU scoring/PQ GPU nếu cần → B04 nghe và report.
3. C00 exposure/manifest → C01 smoke4/cache → C02 cache phần còn lại của val40 → C03 scalar/blend CPU → C04 shortlisted PQ → khóa hệ số → C05 confirmation20/cache/scoring → C06 nghe/report.

Quota lấy trực tiếp Kaggle tại lúc thực thi, không dùng số30h tuần trước. Tính RTF trên smoke = GPU seconds/audio seconds, dự toán theo tổng thời lượng, số nhánh và overhead1.3; ghi thành bảng trước batch. Chi phí CPU và GPU tách riêng; không hứa thời gian khi chưa có smoke.

Ngân sách khởi điểm đề xuất: A dùng CPU/audio có sẵn; B smoke tối đa1 GPU-hour rồi đặt trần pilot từ RTF; C smoke tối đa1 GPU-hour rồi đặt trần từ RTF. Dùng tối đa70% quota còn tại lúc mở chiến dịch, giữ30% sửa lỗi; đây là giới hạn vận hành đề xuất, không phải cam kết tài nguyên. Nếu batch vượt trần, giảm cohort theo quy tắc lấy mẫu trước xem output và ghi amendment, không bỏ âm thầm các job lỗi.

Mỗi job checkpoint theo sample/chunk; fail thì lấy log, sửa đúng lỗi, resume; không restart toàn bộ nếu cache hợp lệ. Không upload token vào notebook hoặc artifact. Sau mỗi thay đổi đáng kể cập nhật diary.

## E. Bàn giao cuối

- CLIP_AUDIT_REPORT_V2.md: cái gì giải thích được và cái gì chưa.
- JOIN_PILOT_REPORT.md: duration/timeline, đường loudness/phổ, overlap vs control, quan sát nghe hoặc pending.
- VRG_PILOT_REPORT.md: baseline/blend/residual, ràng buộc và kết quả confirmation/negative/inconclusive.
- OBSERVATIONS_REPRODUCE_SONICMASTER.md giữ quan sát reproduction; thêm link sang extension, không thay bảng gốc.
- Một bảng status rõ: đã chạy / chưa chạy / đã xác minh / còn giới hạn. Không viết cả ba phase thành công khi chỉ notebook complete.

## F. Nguồn phương pháp

- Paper SonicMaster do người dùng cung cấp: Tables1–3, A.5 và A.10; code chính thức trong SonicMaster/.
- Librosa spectral flatness: https://librosa.org/doc/0.11.0/generated/librosa.feature.spectral_flatness.html
- Survey declipping: https://arxiv.org/abs/2007.07663 và https://rajmic.github.io/declipping2020/
- Loudness: https://tech.ebu.ch/loudness và https://www.itu.int/rec/r-rec-bs.1770/_page.print
- Long-form metric reference: https://github.com/Stability-AI/stable-audio-metrics (khác recipe SonicMaster; không trộn điểm).
- CLIPPING_FULLSONG_INVESTIGATION.md, BAO_CAO_REPRODUCE_TABLES_1_2_3.md và các receipts hiện có.

Đây là kế hoạch đã chuẩn bị để thực thi; chưa phát sinh job Kaggle mới trong lượt lập kế hoạch.


## Kiểm chứng bổ sung sau pilot

<!-- vrg-followup-report-link -->

Đã hoàn tất bootstrap FAD, kiểm SSIM trước/sau xuất WAV và PQ cùng loudness. Xem `D:/SonicMaster_Reproducing/VERIFICATION_FOLLOWUP_REPORT.md`. Đây là phân tích thăm dò sau quan sát; không thay điểm hoặc quyết định pilot gốc. Nghe thật còn pending.
