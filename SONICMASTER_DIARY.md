# Nhật ký dự án SonicMaster

Ngày tạo: 06/11/2026 (Asia/Saigon). Tài liệu dùng để bàn giao và tiếp tục dự án.

## Quy tắc cập nhật từ nay

Sau mỗi đợt công việc có thay đổi đáng kể, thêm mục ngày/giờ nếu biết, yêu cầu hoặc quyết định, việc thực sự đã làm, bằng chứng, kết quả và việc còn thiếu. Không ghi token/mật khẩu. Không đánh dấu hoàn tất chỉ vì notebook complete. Không bịa thời điểm cho các công việc lịch sử; các mục hồi cứu dưới đây ghi theo bằng chứng đã có, không phải nhật ký thời gian chạy chính xác. Ghi sửa đổi mới thay vì xóa kết quả phủ định.

## Mục tiêu và quyết định hiện hành

- Reproduce và tìm failure cases của SonicMaster trên Kaggle.
- Quy mô hiện tại: 1.400 mẫu / 200 bài, không tự mở rộng lên 7.000 mẫu.
- **Người dùng yêu cầu hoãn thử VRG. Chưa train mask, chưa fine-tune backbone hoặc chạy job nâng cấp.**
- Hoàn tất kiểm chứng reproduction và failure analysis trước khi xét kế hoạch nâng cấp.

## 06/11/2026 — Hồi cứu tài sản đã kế thừa

- Đã có audit pilot 28 mẫu, pipeline 70 mẫu và diagnostic clipping. Xem `outputs/sonicmaster_audit_v2/CONTINUATION.md`; đây là trạng thái lịch sử, không thay báo cáo 1.400 mẫu hiện tại.
- Đã lập kế hoạch VRG, gồm gates, ablation scalar trước mask, dữ liệu tách nguồn và kiểm tra exposure: `outputs/SonicMaster-Residual-Upgrade-Plan.md`. Đây chỉ là kế hoạch, chưa thực thi.
- Bộ điều phối chạy cục bộ gửi các notebook Kaggle. Notebook đã gửi chạy trên Kaggle; việc tự gửi bước tiếp phụ thuộc bộ điều phối. Không mặc định toàn bộ pipeline tiếp tục khi máy/ứng dụng đóng.

## 06/11/2026 — Xác nhận inference và thu kết quả

- Kết nối API Kaggle thành công sau khi cấp quyền mạng. Cả `sonicmaster-1400-infer-1` đến `-4` báo complete, không có failureMessage. Bằng chứng: `outputs/sonicmaster_audit_v2/run1400/live_status.json`, `state.json`.
- Đã duyệt hết pagination output của cả 4 notebook, mỗi notebook 1.766 file được liệt kê. Tải manifest, progress, run_lock, provenance và bảng điểm; chưa tải toàn bộ WAV.
- Script tái chạy: `outputs/sonicmaster_audit_v2/run1400/collect_results.py`. API output dùng query userName/kernelSlug; pageSize=100. Thử endpoint dạng path trả 404 và pageSize=1000 trả 400; đã sửa.
- Kiểm đủ 1.400 sample ID / 200 source ID, 7 variants/nguồn, không thiếu/trùng; manifest prompt/nhãn khớp. Có đủ tên 5.600 WAV. Hash progress khớp audio inventory, chưa kiểm độc lập byte của toàn bộ 5.600 file.
- Hash evaluator và cấu hình cốt lõi inference khớp bốn shard; điểm hữu hạn. Bằng chứng: `run1400/evaluation/validation.json`, `evidence_lock.json` dưới thư mục audit.

## 06/11/2026 — Tổng hợp objective và failure candidates

- Script: `outputs/sonicmaster_audit_v2/run1400/aggregate_results.py`.
- Gộp điểm từng mẫu, không lấy trung bình đơn giản các bảng shard. Có All / Single / Double+Triple và CI 95% bằng 2.000 bootstrap theo source.
- Sai số trung bình 19 thuộc tính mục tiêu đều giảm so với input. Có 256/2.200 cặp mẫu–thuộc tính tăng sai số (>0, chưa áp tolerance), thuộc 234 mẫu. Đây là ứng viên theo metric, không phải tỷ lệ audible failure.
- SSIM: input 0,779314; VAE 0,547511; SonicMaster 0,611359. KL ×1000 của SonicMaster: 0,886757. Không suy diễn chất lượng nghe chỉ từ SSIM.
- Đã thêm bảng đối chiếu mốc paper kế thừa từ evaluator; chưa xác minh lại các mốc trực tiếp PDF.
- Đã tải và kiểm SHA256 32 WAV thuộc 8 mẫu discovery: 6 ứng viên và 2 ca đối chứng chọn theo SSIM. Bộ nghe có nhãn, không ngẫu nhiên đại diện toàn tập, chưa có người nghe xác nhận.
- Bộ nghe: `outputs/sonicmaster_audit_v2/run1400/evaluation/LISTEN.html`; script chuẩn bị: `run1400/prepare_listening.py`.
- Đã xem tổng hợp có chứa confirmation. Phải ghi nhận exposure này khi thiết kế/chọn phương pháp mới; không tùy tiện gọi dữ liệu đã dùng để phát triển là unseen.

## 06/11/2026 — Điều tra chênh clipping và cập nhật báo cáo

- Người dùng hỏi vì sao clip lệch lớn, yêu cầu ghi giải thích vào report và lập nhật ký này.
- Điểm hiện tại: 0,0013937855 / 211 mẫu; mốc paper kế thừa 1,506. Không tuyên bố tốt hơn paper khoảng 1.080 lần.
- Xác nhận upstream truyền abs(STFT)**2 vào spectral_flatness mặc định power=2. Tài liệu librosa hướng dẫn power=1 cho phổ đã bình phương.
- Diagnostic pilot đã có: upstream ×1000 = 0,0003874561; power=1 ×1000 = 0,2911719944. Không phải kết quả diagnostic của toàn bộ 211 mẫu.
- Chưa chứng minh paper dùng power=1 hoặc đây là nguyên nhân duy nhất. Giữ benchmark upstream; diagnostic tách riêng.
- Đã sửa hàng clipping trong báo cáo và thêm mục cảnh báo. Báo cáo: `outputs/sonicmaster_audit_v2/run1400/evaluation/REPORT_VI.md`.

## Công việc còn lại

1. Đối chiếu trực tiếp PDF, đặc biệt mốc và thang clipping.
2. Tính diagnostic clipping song song trên cùng 211 mẫu, không thay benchmark upstream.
3. Xem xét FAD-CLAP và Audiobox PQ; hiện chưa tính, không coi benchmark đầy đủ.
4. Nghe thật và ghi phiếu, tách lỗi pipeline khỏi lỗi mô hình. Không tự tạo nhận xét nghe.
5. Hoàn thiện failure analysis và đóng báo cáo reproduction/G0 với phần thiếu được khai báo.
6. VRG tiếp tục hoãn theo yêu cầu; file plan không kích hoạt chạy tự động.

## Lưu ý khi tiếp tục

`aggregate_results.py` hiện tạo lại REPORT_VI.md; chạy lại có thể mất phần ghi chú thủ công và bảng tham chiếu bổ sung. Trước khi chạy lại, bảo toàn hoặc tích hợp các phần này vào script. Nhật ký này phải được cập nhật cùng các thay đổi đáng kể tiếp theo.


## 06/11/2026 — Đối chiếu PDF và bắt đầu diagnostic 211 mẫu

- Đã đọc PDF gốc, trích xuất các trang liên quan và xem ảnh render trang 7. Table 2 xác nhận Clip ×1000: input 5,122; reconstructed/VAE 3,871; SonicMaster 1,506. Phụ lục A.5 trang 18 ghi phổ công suất, FFT 2048/hop 512, absolute error, nhưng không chỉ rõ power=1 hay power=2.
- Bằng chứng PDF/hash: `outputs/sonicmaster_audit_v2/run1400/evaluation/clipping211/paper_evidence.json`; ảnh: `work/clipping-table-page7.png`.
- Đang chạy `run1400/diagnose_clipping211.py` bằng môi trường sonic-audit-env/librosa 0.11.0: tải và kiểm SHA256 cả clean/input/VAE/SonicMaster cho đúng 211 mẫu; so upstream với điểm đã lưu; tính power=1 riêng. Có receipt từng mẫu để resume, chỉ xuất summary sau đủ 211. Chưa có kết quả toàn tập tại thời điểm ghi mục này.
- Đã sửa aggregate_results.py xuất `GENERATED_METRIC_REPORT_VI.md` riêng; không ghi đè báo cáo đã biên tập. Lưu ý nguy cơ ghi đè ở mục cũ được giải quyết.
- VRG không chạy.


### Sự cố tải diagnostic và xử lý

- Tải riêng từng file qua API gặp HTTP 429 sau 24 mẫu hoàn tất. Kaggle trả Retry-After=155 giây; dừng lượt tải cũ và chờ đúng giới hạn.
- Đã đổi script sang duyệt danh sách output rồi dùng URL tải mà Kaggle cấp; không gửi bearer token sang URL dữ liệu, không gọi endpoint API riêng cho từng WAV. Đang chờ để resume; chưa tuyên bố hoàn tất 211 mẫu.


## 06/11/2026 — Hoàn tất diagnostic clipping 211 mẫu

- Đã resume sau 429, chấm đủ 211 mẫu × 4 nhánh, kiểm hash; upstream khớp điểm lưu.
- input: upstream ×1000=0.010770756; power=1 ×1000=3.463702933; paper=5.122.
- vae: upstream ×1000=0.010298032; power=1 ×1000=3.162535069; paper=3.871.
- sonicmaster: upstream ×1000=0.001393785; power=1 ×1000=0.228176709; paper=1.506.
- Report đã bổ sung bảng riêng, benchmark không đổi. Không khẳng định đã tìm được công thức paper. VRG chưa chạy.
- Việc tiếp theo: xác minh thêm nguyên nhân chênh còn lại, FAD/PQ và nghe xác nhận failure cases.

- Kết luận: power=1 tăng điểm SonicMaster khoảng 164 lần nhưng vẫn cách paper khoảng 6,6 lần. Chỉ giải thích một phần chênh lệch; không tuyên bố đã reproduce clipping paper hoặc tốt hơn paper. Các mục còn thiếu về chấm diagnostic và kiểm mốc clip trước đó đã hoàn tất; nguyên nhân phần lệch còn lại chưa xác định.


## 06/11/2026 — Sửa hiển thị 0s trong bộ nghe

- Người dùng báo toàn bộ audio hiển thị 0s. Kiểm tra 32 WAV đều đọc được, dài 29,95374–30 giây, không phải file rỗng.
- HTML cũ dùng preload=none nên chưa yêu cầu metadata trước khi phát. Đổi preload=metadata; thêm thời lượng lấy trực tiếp từ WAV, link mở WAV và thông báo nếu tải lỗi; tự dừng audio khác khi phát.
- Đã kiểm 32 đường dẫn tồn tại. Chưa kiểm phát thực tế trong trình duyệt người dùng; preview có thể hạn chế truy cập file tương đối.


## 06/11/2026 — Hướng dẫn nghe cho người không chuyên

- Người dùng không biết input đã bị thay đổi gì; bổ sung nhãn degradation và hướng dẫn nghe bằng từ thông thường cho cả 8 mẫu, từ manifest thật; hiển thị prompt gốc và clipping ẩn ở 1115233_deg5. Không bịa tham số tạo degradation hoặc kết luận nghe.
- Sắp xếp volume/clipping trước; hướng dẫn clean → input → output cùng đoạn, cho phép không nghe ra/chưa chắc. VAE là bước chẩn đoán sau. Đây là nghe khám phá có hướng dẫn, không dùng như thử nghiệm nghe mù.


## 07/11/2026 — Báo cáo tổng hợp nộp giảng viên

- Đọc SC203_v5_bao_cao.pdf (6 trang), giữ mốc 05h38 ngày 07/11/2026: B0/B1 theo nguồn, B2 2.850/3.264 clip đã kiểm; không tự công nhận 414 clip hoặc B3 hoàn tất.
- Tổng hợp với kết quả benchmark 1.400 mẫu, failure candidates, diagnostic clipping 211 mẫu và công việc còn thiếu. Phân biệt FAD/PQ có ở SC203 B1 với phần chưa chạy trên 1.400 mẫu; giữ khác biệt cohort/precision/giao thức, VRG chưa chạy.
- Đối chiếu thêm trực tiếp Table 3 trang 8 paper gốc, xác nhận KL All 0,888 và SSIM All 0,609.
- Đã xuất PDF 8 trang: output/pdf/Bao_cao_tien_do_SonicMaster_2026-11-07.pdf; bản nội dung .md cùng tên và report_sources.json lưu hash nguồn.
- Render và kiểm trực quan 8 trang, sửa ký tự số mũ không hiển thị, kiểm lại trang clipping. Không bịa thông tin sinh viên hoặc nhận xét nghe. Script tạo báo cáo: work/build_submission_report.py.

## 07/11/2026 — Tiếp tục hoàn thiện reproduce 1.400 mẫu (bỏ qua SC203 v5)

- Gửi notebook Kaggle chấm global FAD-CLAP và Audiobox PQ: `aajayagh/sonicmaster-1400-fad-clap-and-audiobox-pq`, version 1, kernel ID 143165640. Script xác minh nguồn 5.600 WAV và hash trước khi chấm; không gộp shard để lấy trung bình FAD.
- Lúc kiểm tra cuối trong phiên này, Kaggle vẫn báo `running` sau khoảng 25 phút GPU; API chưa công bố log hoặc file kết quả. Do chưa có artifact nên chưa chốt FAD/PQ, chưa cập nhật số metric. Quota còn nhiều. Cần kiểm tra kernel trước khi kết luận hoặc chạy lại.
- Các việc đã xong: benchmark metric hiện tại và hash inventory; failure candidates được khoanh vùng bằng metric (không phải xác nhận nghe); diagnostic clipping 211 mẫu; bộ nghe hướng dẫn 8 mẫu/32 WAV tại `outputs/sonicmaster_audit_v2/run1400/evaluation/LISTEN.html`.
- Cần người dùng nghe/ghi xác nhận trên bộ nghe. VRG chưa chạy; SC203 v5 được bỏ qua theo yêu cầu.


07/11 — FAD/PQ v1 failed on SciPy import (atleast_nd), after verifying 5600 WAV hashes. Fixed installation before imports with fresh Python exec; fixed clean PQ key and scalar JSON serialization. Submitted retry; metrics still pending.


## FAD-CLAP và Audiobox PQ đã hoàn tất — version 2

Kaggle hoàn tất sau khoảng 79 phút. Đã kiểm hash 5.600 WAV; có 4.400 bản ghi PQ duy nhất (200 clean và 1.400 mỗi nhánh input/VAE/SonicMaster), tất cả điểm hữu hạn. Clean được dùng chung theo source. FAD tính toàn bộ từng nhóm, không lấy trung bình shard. Các ghi chú FAD/PQ pending trước đây là trạng thái lịch sử.

| Nhóm | Nhánh | FAD-CLAP (thấp tốt) | PQ (cao tốt) |
|---|---|---:|---:|
| single | input | 0.100944 | 7.339644 |
| single | vae | 0.186470 | 7.179219 |
| single | sonicmaster | 0.104824 | 7.731483 |
| double_triple | input | 0.242771 | 6.597549 |
| double_triple | vae | 0.354734 | 6.465913 |
| double_triple | sonicmaster | 0.123558 | 7.643610 |
| all | input | 0.150533 | 7.021603 |
| all | vae | 0.246570 | 6.873516 |
| all | sonicmaster | 0.109114 | 7.693823 |

Toàn tập, SonicMaster cải thiện FAD và PQ so với input. Tuy nhiên nhóm single có FAD tăng từ 0.100944 lên 0.104824, dù PQ tăng; không kết luận mọi mẫu đều cải thiện. FAD là điểm phân phối, không xác định failure từng mẫu. Cần nghe xác nhận. So sánh paper phải giữ khác biệt tập 1.400/7.000 và giao thức; clipping vẫn chưa giải quyết hoàn toàn. VRG tiếp tục hoãn.

Dữ liệu gốc: fad_pq/fad_clap.json, audiobox_pq_summary.json, pq_records.jsonl, status.json, audio_verification.json.


## Nhận xét nghe do người dùng cung cấp — 7 mẫu

Nghe khám phá có hướng dẫn, một người nghe; chưa nghe mù hoặc cân bằng âm lượng. Đây là quan sát cảm nhận, không phải tỷ lệ failure toàn tập. Chưa xác định các lỗi nghe thấy là lỗi mới do mô hình hay lỗi input chưa sửa hết.

| Mẫu | Nhận xét người nghe | Diễn giải và giới hạn |
|---|---|---|
| 1283374_deg2 | Đã tăng âm lượng lên gần clean; giọng hát gãy, vỡ, không mượt như clean. VAE rất nhỏ, hầu như không nghe thấy. | Cải thiện âm lượng nhưng còn hiện tượng giọng hát gãy/vỡ; chưa xác định là lỗi mới hay lỗi còn sót. VAE khó đánh giá do âm lượng thấp. |
| 792131_deg2 | SonicMaster nhỏ hơn input và VAE nên khó nghe rõ; phần nhạc ngân như kèn mượt hơn input khá nhiều. | Cải thiện độ mượt ở một số đoạn; đánh giá tổng thể chưa chắc. Chênh lệch âm lượng ảnh hưởng khả năng so sánh. |
| 1577531_deg1 | SonicMaster rất gần clean, nhưng tiếng gõ ở clean mạnh và rõ hơn. | Gần clean; tiếng gõ còn yếu/kém rõ hơn.  |
| 1909745_deg6 | SonicMaster cảm giác vẫn khá vang so với clean. | Người nghe ghi nhận độ vang còn dư so với clean.  |
| 1640646_deg6 | Âm gõ ở clean thanh và mượt hơn; SonicMaster còn rè và sạn. | Người nghe ghi nhận âm gõ chưa mượt, còn rè/sạn.  |
| 1811880_deg5 | Khá sát clean. | Người nghe đánh giá gần clean. Không suy ra đồng nhất waveform hoặc hết mọi lỗi. |
| 1115233_deg5 | Khá giống clean, không nghe thấy sự khác biệt. | Người nghe chưa nhận thấy khác biệt với clean. Không suy ra hai bản giống nhau tuyệt đối. |

Đã nhận phản hồi 7/8 mẫu trong bộ nghe. Cần đối chiếu nhãn/prompt và input cùng đoạn trước khi chốt failure theo nhiệm vụ; đặc biệt 792131_deg2 và VAE của 1283374_deg2 bị hạn chế bởi âm lượng. Giữ bản gốc để đánh giá volume; bản cân bằng âm lượng nếu tạo chỉ dùng để nghe chi tiết, không thay metric.


Đã đối chiếu 7 phản hồi nghe với nhãn, prompt và điểm từng mẫu. Phân tích: artifacts/07_run1400/evaluation/FAILURE_ANALYSIS_VI.md; bằng chứng: failure_listening_evidence.json. Có cải thiện kèm hạn chế về giọng hát, reverb và âm gõ. Không xác định tỷ lệ failure nghe được, chưa quy nguyên nhân cho VAE. VRG hoãn.


Đã tạo BAO_CAO_REPRODUCE_TABLES_1_2_3.md: đối chiếu paper/reproduction của Input, VAE, SonicMaster cho Tables 1–2 và toàn bộ 3 nhóm Table 3, ghép phân tích 7 ca nghe; ghi rõ baseline/full-song chưa tái lập.


Theo yêu cầu người dùng, tổng hợp observation vào OBSERVATIONS_REPRODUCE_SONICMASTER.md: ba bảng, xu hướng metric, VAE, clipping, biến thiên từng mẫu, 7 nhận xét nghe và giới hạn; không phân loại failure cases.


Đã nghiên cứu metric declipping từ survey, đối chiếu librosa power; tìm thấy code full-song pad đoạn cuối mà không trim trước lưu. Bắt đầu audit13 demo full-song (không phải inference mới). Chi tiết CLIPPING_FULLSONG_INVESTIGATION.md.

Hoàn tất audit13 full-song demo: output dài hơn clean từ 10.253 đến 27.981s; chưa xác minh alignment, không coi là kết quả paired reproduction.


Chuẩn bị artifacts/fullsong_listening_2/LISTEN_JOINS.html: 2 bài demo1115233_deg2 và1464612_deg1, audio cả bài và12 clip20s quanh2 vùng nối mỗi bài. Mốc suy từ code/độ dài, chưa log xác nhận; không normalize hoặc khẳng định alignment.


## 08/11/2026 — Kế hoạch clipping → joins → VRG

Đã lập PLAN_CLIP_JOINS_VRG.md theo yêu cầu: A audit211 clipping dùng12 smoke; B pilot12 full-song với timeline/log chính xác và overlap-control; C scalar residual/blend trên40 nguồn val và20 nguồn confirmation mới. Không yêu cầu failure catalog; chưa chạy job mới, chưa học mask. Ghi rõ negative/inconclusive cũng là kết luận hợp lệ, không trì hoãn C vô hạn vì clip chưa khớp paper.


08/11 — Audit clip211v2 completed raw measurements: upstream matches; power1 frameMAE SM0.305249 still below paper1.506; nonsilent power1 SM0.228229. Waveform SDR diagnostic awaiting lag. Prepared and submitted joins smoke2 with synthetic amplitude on official full-song demo clean cohort; source/protocol amendment explicit.

## 08/11/2026 — Tiếp tục pilot nối đoạn và chuẩn bị VRG

Kết nối lại Kaggle, xác nhận join smoke v1 và VRG stage smoke v1 complete. Tải đủ 59 output join smoke, lưu receipt SHA256. Hai bài, hai lịch chia, 42 chunk. Xác minh decoded chunk thiếu 2.040 frame/30s; ghép gốc trôi 0,416–0,463s ở chunk cuối và còn pad dư 22,218–27,912s (lịch0). 114 kiểm VAE/input có lag ±2 samples. Tạo bản anchored có nhãn riêng; không sửa baseline.

Matched diagnostic: SonicMaster D âm ở cả hai bài/hai lịch, VAE D dương. Chưa có bằng chứng SM sai hơn tại nối; chưa có phản hồi nghe. Báo cáo JOIN_PILOT_REPORT.md. Quy tắc matching khóa sau khi đã xem unmatched nên chỉ gọi exploratory.

Gửi sonicmaster-join-pilot-remaining10 v1, kernel143346359; đang chạy, dự toán65phút, trần2giờ, không chạy trùng smoke. Metadata selection VRG đủ280 mẫu; gửi CPU sonicmaster-vrg-stage-selection v1 kernel143346726 để tái dùng28 và chuẩn bị252 mẫu còn lại. Chưa chạy VRG inference/hệ số. Code cache smoke đã chuẩn bị để dùng sau quyết định pilot.

Đã bổ sung audit clip12:36 synthetic clipping có mask thật,108 cửa sổ lag, common support cả4 nhánh, frame diagnostics và environment. Roundtrip các WAV PCM16 hiện có không đổi; chưa loại trừ quantization/clamp trước lưu. Lưu supplement, không thay điểm211.

Đã xuất đồ thị và bộ nghe join smoke4 nhóm×4 bản20s; gain chung0,5490 để tránh phát sinh clipping, giữ chênh âm lượng, mapping riêng, chưa nhận phản hồi. Thấy loudness nhấp nhô tại overlap; diagnostic năng lượng bổ sung cho thấy linear mix giảm khoảng0,24/0,68dB trên SM giữa overlap của2 bài. Đây là observation hậu kiểm, không đổi endpoint hay khẳng định audibility/failure. VRG stage smoke đã tải và kiểm đủ56 hash audio.

### Phản hồi nghe tiếp theo — 08/11/2026

Người dùng: “mình chỉ thấy khác nhau âm lượng giữa các bản thôi”. Lưu nguyên văn trong artifacts/09_join_pilot/listening_smoke/human_observations.json và cập nhật JOIN_PILOT_REPORT.md. Đây là phản hồi tổng quát của một người, chưa có mã trial, mức chắc chắn hoặc xác nhận đã nghe đủ4 nhóm. Không gán cho từng trial và không hiểu chênh âm lượng giữa các bản thành nhảy âm lượng tại vùng nối. Chưa xác nhận lỗi nối nghe được; không kết luận các bản tương đương hoặc hoàn toàn không có lỗi. Giữ nguyên endpoint và các số đo.

### Tận dụng slot GPU trống — 08/11/2026

Người dùng cho biết còn một slot và mời chạy thêm công việc hữu ích. Kiểm API lúc04:14:59UTC: joins10 còn running, VRG stage selection complete, quota còn97.439s (~27,1h). Ghi amendment artifacts/10_vrg_pilot/AMENDMENT_01_CACHE_SCHEDULING.md trước khi gửi: chạy sớm cache độc lập trên28 mẫu val (4 nguồn), còn thử/chọn hệ số vẫn chờ quyết định joins và kiểm alignment.

Đã gửi aajayagh/sonicmaster-vrg-cache-smoke v1, kernel143350836. VAE+SonicMaster cấu hình gốc, lưu float trước clamp và PCM16 baseline; không huấn luyện hoặc chạy confirmation. Dự toán proxy945,8s (~16phút), script có trần wall-clock1giờ qua cả re-exec và checkpoint từng mẫu. Không mở rộng280 inference trước khi kiểm smoke. Lưu payload, budget và receipt; cập nhật status bằng API.

### Sau khi các notebook hoàn tất — 08/11/2026

Xác nhận joins remaining10, VRG stage selection và VRG cache smoke đều complete. Tải323 file joins10 và173 file VRG cache. Toàn cohort joins đủ12 nguồn,24 lịch,300 chunk. Ghép metadata và giữ nguyên output gốc; chạy phân tích đủ12 nguồn ở analysis_full12, tách khỏi smoke.

VRG cache kiểm112 WAV+28 float hash khớp và hệ số0 tái tạo PCM baseline chính xác. 76/84 cửa sổ lag tin cậy nằm[-1,+1] sample; chỉ26/28 mẫu có cửa sổ đạt. Hai mẫu1406227_deg5/deg6 tương quan thấp dù peak lag gần0; kiểm hai dải tần cho clock gần0 nhưng chưa nâng quy tắc gốc thành pass. Không loại mẫu, chưa chạy grid/chọn hệ số/infer252. Báo cáo VRG_PILOT_REPORT.md.

Gửi đo PQ phủ toàn bộ12 bài trên cửa sổ30s, kernel aajayagh/sonicmaster-join-full12-pq v1 ID143354504; dùng chunk đã infer, không chạy lại SonicMaster. Xác minh324 hash input/chunk; tail zero-pad30s và tổng hợp theo thời lượng hữu ích mỗi bài, PCM16 clamp có báo riêng. Đây là secondary metric của extension, không gọi trực tiếp là protocol full-song paper. Trần script1giờ.


### Hoàn tất phân tích joins12 và PQ toàn bài — 08/11/2026

- Kaggle xác nhận remaining10, stage selection, cache smoke và full12 PQ complete. Tải đủ6 output PQ; không chạy thêm SonicMaster inference. PQ dùng Tesla T4, thời gian script532,53s.
- Hoàn tất phân tích12 nguồn/24 lịch/300 chunk. Tạo48 WAV FLOAT anchored đúng support input. Timeline gốc thiếu2040 frame/chunk; lịch0 trôi0,231–0,740s ở chunk cuối, dư cuối10,253–27,981s. Audit full12 có834/835 cửa sổ tin cậy, lag[-2,393]sample; giữ sample-exact metrics chưa được phép.
- Endpoint SonicMaster lịch0: meanD=-0,448194LU, CI95% nguồn[-0,559623;-0,350955],0/12 bài D>0; lịch5 meanD=-0,485715LU. Không đạt tiêu chí mở rộng joins. Không tuyên bố không có biến thiên hoặc không có lỗi nghe được.
- Đo FFmpeg EBU R128 trên72 stream: SonicMaster lịch0 lớn hơn clean khoảng4,90LU; true peak trung bình+5,425dBTP, cao nhất+7,2dBTP. Tỷ lệ sample-channel float vượt biên trung bình0,8214%; không đồng nhất với metric clipping paper.
- Xác minh toàn bộ612 cửa sổ PQ: tái dựng từ chunk, xuất PCM16 và đối chiếu612 SHA256 với Kaggle đều khớp;72 stream phủ đủ bài, trọng số và48 audio FLOAT khớp. PQ trung bình theo bài: clean7,773931; input7,790588; VAE lịch0 7,725506; SonicMaster lịch0 7,568076. ChênhSM-input=-0,222511, CI95% nguồn[-0,304419;-0,146713];11/12 bài giảm. Đây là extension amplitude-only, chưa quy nguyên nhân cho joins hoặc clamp, không so trực tiếp với paper full-song.
- Xuất12 đồ thị, overview và bộ nghe24 trial/96 WAV tại listening_full12. Gain chung0,439979, không normalize từng file. Giữ nguyên phản hồi smoke của người dùng, không tự điền điểm cho bộ mới.
- Cập nhật JOIN_PILOT_REPORT.md và artifacts/09_join_pilot/REPORT_VI.md; giữ báo cáo2 bài cũ trong JOIN_PILOT_REPORT_SMOKE_2.md. Không thay benchmark1.400 mẫu.
- VRG: amendment02 mở rộng timing trên756 cửa sổ, giữ ngưỡng.686 cửa sổ tin cậy lag[-2,+3], nhưng2 mẫu1406227_deg5/deg6 vẫn không có coverage; gate vẫn false. Chưa chấm grid/chọn hệ số/infer252 hoặc mở confirmation20. Không bỏ mẫu hay sửa ngưỡng để pass. Báo cáoVRG và plan cập nhật.
- Sửa update_research_status.py để trạng thái remote complete không ghi đè cache local complete_verified; bổ sung trạng thái PQ, phân tích và nghe. Phần tính toánB xong, phản hồi nghe chi tiết vẫn pending. Các job trên đều đã kết thúc; không có job mới được gửi trong lượt này.


### Tiếp tục: đối chứng âm lượng/xuất WAV và giải quyết gate VRG — 08/11/2026

Theo yêu cầu “rồi làm tiếp đi”, khóa protocol tại artifacts/11_level_export_controls/PROTOCOL.md trước khi xem điểm mới. Hai khối: g=0,25 chung để tách tác động hard clipping; mục tiêu -18LUFS cố định theo loudness của từng bản, không fit với clean, để so gain-only và SonicMaster. Cả48 source stream đạt mục tiêu bằng scalar mà không phải chạm cap peak0,98. Gửi notebook private sonicmaster-level-export-controls v1 ID143360498:96 stream/816 cửa sổ PQ, tái dùng12 bài, không inference SonicMaster. Dự toán1223s, cap1giờ; quota khả dụng lúc gửi95704s.

Phân tích waveform VAE/input trên28 cache: hai mẫu1406227_deg5/deg6 ít tương đồng ở cả hai kênh, không phải chỉ lỗi mono hoặc đổi kênh. Envelope10ms vẫn tương quan cao, năng lượng chủ yếu ở1–8kHz và coherence thấp; chưa gọi phần chênh là chi tiết sạch.

Ghi amendment03 rồi gửi VAE-only clock probe143360916 trên deg1/4/5/6 cùng nguồn:16 decode, cap20phút. Job hoàn tất152,37s. Tải7 output; kiểm lại hash4 probe và array. Repeat gần cache cũ (relativeRMS tối đa1,23e-6); TF32-off trùng repeat; dịch input ±1encoder hop cho output trùng sau bỏ đúng dịch đã chèn và loại biên. Không thay cache gốc. Giữ hai gate correlation gốc=false; known-hop test không tự chứng minh absolute latency/phase identity.

Ghi timing_resolution.json trước candidate scores: cho phép smoke diagnostic không căn chỉnh trên đủ28 mẫu dưới bằng chứng clock/code/provenance, ghi riêng subgroup2 mẫu. Chưa cho phép mở rộng252 hoặc confirmation. Bắt đầu score_vrg_smoke.py so residual/blend theo grid đã khóa; chưa chọn hệ số, chưa có PQ candidate. Đã tạo bộ nghe đối chứng mới24trial/48WAV; phản hồi vẫn pending.


### Tiếp tục VRG sau xác nhận của người dùng — 08/11/2026

Người dùng xác nhận “xác nhận chạy tiếp vrg”, thay thế hạn chế “chỉ bàn giao, không thực thi” trong nội dung bàn giao. Mọi công việc tiếp tục trong D:/SonicMaster_Reproducing; không inference lại 1.400 mẫu gốc.

Đối chứng âm lượng/xuất WAV đã hoàn tất và kiểm độc lập đủ 816 hash PCM/96 stream. Tránh hard clipping ở cùng gain tăng PQ +0,021579 (CI95% nguồn +0,011961 đến +0,031775). Khi cùng mục tiêu −18 LUFS, SM−input gain-only = −0,189945 (CI −0,269836 đến −0,116531), 12/12 bài âm. Chênh âm lượng/cắt đỉnh chưa giải thích hết khoảng cách PQ trên cohort 12 bài amplitude-only; không suy ra mức đóng góp nhân quả, audibility hoặc kết luận cho các loại degradation khác. Báo cáo LEVEL_EXPORT_CONTROLS_REPORT.md; bộ nghe chưa có điểm.

Smoke VRG đã hoàn tất 364 điểm (28 mẫu × 13 nhánh); không chọn hệ số từ smoke. Hệ số 0,1 cho ΔSSIM residual +0,016253, blend +0,018708; diagnostic suy giảm target tương ứng 0,004888 và 0,021613. Chưa có PQ candidate, chưa có candidate được chấp nhận.

Validation cache gồm ba batch 84 mẫu, tái dùng 28 cache smoke. B1/B2 đã complete trên Kaggle, B3 ID143363769 đã được gửi khi có slot và đang chạy tại lần kiểm 13:27. Bộ theo dõi cũ dừng do mất quyền ghi D: lúc tải B1, không phải lỗi inference. Đã giữ log cũ, ghi tiếp các lần tải, cấp lại quyền mạng/ghi dự án và tiếp tục tải file còn thiếu. Không gửi lại B1/B2/B3.

Chuẩn bị scorer đủ 280 mẫu/40 nguồn, có kiểm recipe và hash khi resume, 4 CPU worker tối đa; sửa lỗi resume của smoke scorer khi toàn bộ receipt đã tồn tại. Giữ function bodies metric upstream. Khóa SELECTION_IMPLEMENTATION_LOCK.md trước full-validation scores: tổng hợp theo nguồn, guard thuộc tính đúng mẫu số validation, source bootstrap 2.000 lần, chấm PQ cho mọi hệ số qua đủ guard CPU. Chưa chạy selection grid vì chờ kiểm cache/timing; confirmation 20 nguồn chưa mở.

Sửa update_research_status.py để ghi tiến độ validation và controls, không lùi về trạng thái chỉ có smoke. Thời điểm ghi nhật ký: 2026-11-08T13:37:19.559211+07:00.


### Đủ cache validation, bắt đầu chấm 280 mẫu — 08/11/2026

Ba batch Kaggle complete và tải đủ 509 artifact mỗi batch. Tái dùng cache28; toàn tập đủ280 mẫu/40 nguồn. Kiểm1120 hashWAV+280 hashfloat, support, cấu hình, endpointλ0 vàblend1 đều khớp; không có clean hash trùng giữa các nguồn. Audit7560 cửa sổ có7180 đủ tin cậy, lag[-4,+4]sample, không có reliable lag ngoài±10.

Gate coverage tương quan gốc vẫn false ở1406227_deg5,1406227_deg6,1582649_deg1. Forensics đủ tập cho thấy mẫu mới có waveform correlation khoảng0,336/0,411 theoL/R và envelope10ms khoảng0,979; không bỏ hoặc dịch mẫu. Ghi AMENDMENT_05_SELECTION_TIMING.md và timing_resolution.json trước full-selection scores, chỉ cho phép đánh giá thăm dò công thức không alignment dựa trên clock/code/provenance và probe cũ. Không khẳng định absolute latency hay phase identity.

Scorer mới tái hiện đúng26 dòng điểm của2 smoke cố định (sai khác0 trên các metric kiểm), identity19 thuộc tính và guard boundary controls đều pass. Lần chạy kiểm đầu bị lệch KL rất nhỏ do importNumPy trước cấu hình thread; đã sửa thứ tự import, không nới tolerance hay đổi metric. Bắt đầu score_vrg_selection.py lúc13:55,4worker,13nhánh×280=3640điểm; mỗi receipt khóa code/version/cache. Chưa chọn hệ số và chưa mở confirmation.

Chuẩn bị pipelinePQ cho shortlist, dùng Audiobox0.0.4 cùng revision model khóa trước đo; sẽ đối chiếu SHA256 của WAV giữa máy và Kaggle. Đây là mã chuẩn bị, chưa có jobPQ validation được gửi hoặc điểmPQ mới. Giữ toàn bộ báo cáo/điểm reproduction và smoke gốc.


### Hoàn tất CPU selection; chỉ blend0,1 vào vòng PQ — 08/11/2026

14:15 hoàn tất3640 dòng điểm/280 mẫu/40 nguồn. Kiểm độc lập coverage, tính hữu hạn, endpointblend1=input, tổng hợp nguồn và guard;364 dòng smoke tái chấm trùng kết quả cũ (maxdifference0). Tập selection có đủ19 nhãn; boom chỉ5 mẫu/5 nguồn, không đủ điều kiện subgroup≥10 mẫu. Không có trùngsourceID hoặc exactcleanWAVhash với200 nguồn reproduction cũ; không đồng nhất kiểm hash với near-duplicate detection hoặc lịch sử exposure ngoài workspace.

SSIM input0,780772; VAE0,537950; SM0,598225. Residualλ0,1 đạt0,607984 (Δ0,009759), dưới ngưỡngΔ≥0,01 đã khóa. Residualλ0,25 đạt0,624890 (Δ0,026665), mean task deterioration0,011223 nhưng nhómdark xấu đi chuẩn hóa0,122998 (>0,05), nên không qua. Các hệ sốresidual còn lại cũng không qua đủguard. Không nới ngưỡng hoặc đổi grid sau khi xem kết quả.

Chỉ blendα0,1 quaCPU: SSIM0,611215 (Δ+0,012990), mean task deterioration0,009061, các guardCPU còn lại đạt. Chưa được chấp nhận vì PQ chưa đo. Đây là đối chứng90%SM+10%input, khác residuals+λ(x−VAE).

Đã kiểm tính nhất quán xuấtPCM giữa NumPy2.4 của scorer và NumPy1.26/SoundFile0.14 củaPQ:26/26 hash của2 smoke×13nhánh khớp. Gửi notebookprivate aajayagh/sonicmaster-vrg-selection-pq v1 ID143375848 lúc14:16. Gồmclean/input/VAE/SM/blend0,1 trên280mẫu:1400dòngPQ, clean được tái dùng khi PCMidentical; không inferenceSonicMaster lại. ModelAudiobox revision9b1dd8e5df9af7216e836a98974fe3b82c56ded6 vàSHAweight được khóa. Dự toán1472s, scriptcap3600s; confirmation vẫn chưa mở.

Báo cáoVRG_PILOT_REPORT.md cập nhật sốCPU, bảngguard/CI, giới hạntiming và PQpending; lưu báo cáosmoke riêngREPORT_SMOKE28.md, snapshotCPU ởREPORT_CPU_SELECTION.md. Đồ thịvalidation_tradeoffs.png đã kiểm hình. JobPQ đang được theo dõi trong lượt làm việc.


### Hoàn tất PQ validation và quyết định scalar pilot — 08/11/2026 14:30

<!-- vrg-validation-decision:2504a523606ade09f0c0c326a17c94ea1faebd113643d9065b4fcc9b14d7adc1 -->

Đã tải và kiểm 1400 dòng PQ trên280mẫu/40nguồn, đối chiếu toàn bộ hashPCM, recipe và modelweight. Các nhánh clean/input/VAE/SonicMaster/blend0,1 dùng cùng support; không inferenceSonicMaster lại. blend 0.1: ΔPQ=-0.016855, guardPQ=đạt. Đã khóa hệ số qua validation; bước tiếp theo là xác nhận một lần trên cohort đã khóa, không chỉnh lại hệ số.

Residual0,1 vẫn dưới ngưỡngΔSSIM0,01; residual0,25 vẫn vi phạm guarddark. Không sửa grid/ngưỡng sau kết quả. Nguồn được gộp theo7variants, CI bootstrap2000 theo40nguồn. PQ là dự đoán mô hình; ngheVRG chưa có điểm. Ba gatecorrelation gốc vẫnfalse, boom thiếu mẫu cho guardriêng, independence bên ngoài chưa xác minh đầy đủ. Đã cập nhật VRG_PILOT_REPORT.md, status, headerplan và linkobservations; không đổi audio/điểm reproduction gốc.

### Khóa blend 0,1, mở confirmation và hoàn tất FAD validation — 08/11/2026 14:55

PQ validation blend−SM = −0,016855, CI95% [−0,029684; −0,005919]: giảm nhẹ nhưng vẫn trong trần −0,05 đã đặt. Đã khóa alpha 0,1; residual không có hệ số qua đủ điều kiện. Ghi AMENDMENT_06_CONFIRMATION.md và confirmation_lock.json trước khi mở audio của 140 mẫu/20 nguồn. Giữ mẫu số chuẩn hóa từ validation và không quét lại hệ số. Chọn trước 12 mẫu nghe bằng nhãn tác động và hash, tối đa một mẫu mỗi nguồn.

Stage confirmation ID143378930 đã complete, metadata 280 WAV/140 mẫu khớp. Có 20 clean hash khác nhau, không trùng selection hoặc reproduction; đây chỉ là kiểm file trùng chính xác, chưa loại trừ các bản nhạc gần trùng hoặc exposure ngoài workspace. Cache B1 ID143379772 và B2 ID143381434 đang chạy trên hai slot GPU. Bộ chấm điểm đã tái hiện đúng 8 dòng của 2 mẫu validation cũ, sai khác 0. Chưa có kết quả confirmation.

FAD selection ID143379319 đã hoàn tất và được kiểm độc lập 1.400 hash PCM/1.160 embedding. FAD toàn nhóm: input 0,231369; VAE 0,362655; SonicMaster 0,178528; blend 0,179700. Blend−SM = +0,001172 (FAD thấp hơn tốt hơn); không dùng kết quả này để chọn lại hệ số. Dùng CLAP 1.1.7 nonfusion và checkpoint 630k-audioset đã khóa hash, crop 10 giây với seed theo nguồn để mọi nhánh dùng cùng vị trí. Không gọi đây là phép đo toàn 30 giây/toàn bài hoặc so trực tiếp với số reproduction cũ. Tính lại FAD từ embedding bằng công thức tương đương SVD khớp với scipy; không trung bình FAD các shard.


### Kết quả confirmation một lần — 08/11/2026 15:36

<!-- vrg-confirmation-result:6e603bd76236f27f1f71fb5765c240c25b6fff98e8f476539f4d380cc100d60b -->

Blend 0,1 lặp lại mức đánh đổi nằm trong các ngưỡng pilot đã chốt trên 20 nguồn xác nhận. Đây vẫn là kết quả thăm dò; chưa đủ để gọi chất lượng nghe tốt hơn. ΔSSIM=+0.012463, CI95%[+0.009064; +0.016280]; ΔPQ=-0.013567, CI95%[-0.030014; +0.000863]. Suy giảm target chuẩn hóa trung bình=0.012443; thuộc tính đủ mẫu không đạt: không.

Đã kiểm 140 mẫu/20 nguồn, 560 WAV + 140 float hash, 560 điểm CPU + 700 dòng PQ. Giữ nguyên hệ số/mẫu số/ngưỡng; không loại mẫu và không căn chỉnh theo clean. Nghe còn pending; không khẳng định chất lượng cảm nhận tốt hơn. Chi tiết và giới hạn ở VRG_PILOT_REPORT.md.

### Hoàn tất kiểm chứng PQ/FAD và bàn giao bộ nghe — 08/11/2026 15:36

<!-- vrg-confirmation-measurements-143386665-143386698 -->

Hai cache confirmation B1/B2 đã complete, mỗi batch tải 425 artifact. Kiểm đủ 140 mẫu/20 nguồn, 560 WAV và 140 float hash. Audit timing có 3.657/3.780 cửa sổ đủ tin cậy, lag −4 đến +4 sample; không có cửa sổ tin cậy lệch ngoài ±10. Hai mẫu 1539576_deg7 và 311386_deg5 vẫn không qua coverage tương quan gốc; giữ nguyên gate=false, không loại hoặc căn lại mẫu. Đối chiếu độc lập 560 dòng CPU, mẫu số validation, tổng hợp theo nguồn, CI và guard đều khớp (sai khác tối đa 0).

PQ `aajayagh/sonicmaster-vrg-confirmation-pq` v1 ID143386665 và FAD `aajayagh/sonicmaster-vrg-confirmation-fad` v1 ID143386698 được gửi lúc 15:24 sau audit CPU, chạy song song theo amendment đã ghi trước kết quả. Cả hai complete; thời gian script lần lượt 489,41s và 579,35s, không tính toàn bộ thời gian khởi tạo/chờ của nền tảng. Không gửi lại notebook. Pipeline kết thúc với `measurements_complete`.

PQ kiểm đủ 700 dòng/hash PCM, tái dựng 20 PCM clean tại máy, đúng model đã pin. SonicMaster 7,743102; blend 7,729534. ΔPQ −0,013567, CI95% [−0,030014; +0,000863] chứa 0: chưa xác định chắc chiều thay đổi trên confirmation, nhưng vẫn qua ngưỡng giảm tối đa −0,05. Không diễn giải thành hai bản nghe tương đương.

FAD kiểm 700 hash PCM, 580 embedding, vị trí crop theo nguồn và tái tính 12 FAD toàn nhóm bằng SVD. FAD all: input 0,248178; VAE 0,413788; SonicMaster 0,215378; blend 0,214809. Δblend−SM = −0,000570; nhóm single −0,004237, double/triple +0,004768. Hai cohort đều có FAD tốt hơn ở single và xấu hơn ở double/triple; hướng FAD all khác nhau giữa validation và confirmation. Chưa có CI FAD; chỉ báo mô tả. Giữ recipe crop 10s, không gọi đây là đo toàn bài hoặc dùng nó chọn lại hệ số.

Confirmation có 7 thuộc tính ngoài clipping đủ điều kiện guard riêng và đều đạt; 11 thuộc tính ít mẫu. Nhóm mud xấu đi chuẩn hóa 0,114022 nhưng chỉ 5 mẫu/5 nguồn nên không đủ ngưỡng ≥10 mẫu. Ghi rõ quan sát này, không coi quyết định pilot là bảo đảm toàn bộ thuộc tính. KL×1000 của blend tăng 0,054499 so với SonicMaster. Residual VRG không có hệ số qua validation; chỉ blend 0,1 đã được xác nhận theo tiêu chí pilot.

Bộ nghe tại `artifacts/10_vrg_pilot/listening_confirmation/LISTEN.html` có 12 mẫu chọn trước, 60 WAV dài bằng nhau 29,9537415s, giữ PCM/gain đã đo, A/B giấu danh tính. Đã kiểm link, hash, thời lượng và cú pháp JavaScript. Chính sách trình duyệt công cụ chặn file:// nên chưa kiểm giao diện/phát âm thanh; không dùng đường vòng. Chưa có điểm nghe, không chuyển phản hồi cũ từ joins sang đây.

Cập nhật `VRG_PILOT_REPORT.md`, báo cáo confirmation, đồ thị so sánh và plan/status thành `complete_quantitative_listening_pending`. Reproduction 1.400 mẫu gốc giữ nguyên. Bước tiếp theo: người dùng nghe A/B, lưu nhận xét JSON; sau đó tổng hợp nhận xét với metric mà không chỉnh lại hệ số từ tập confirmation đã mở.

### Chạy nốt kiểm chứng bổ sung theo yêu cầu — 08/11/2026

Khóa `artifacts/12_vrg_followup_checks/PROTOCOL.md` trước các kết quả bổ sung. Đây là phân tích sau quan sát, giữ nguyên blend0,1/cohort/quyết định pilot. Hoàn tất bootstrap FAD2.000 theo nguồn: CI delta FAD all chứa0 ở cả validation và confirmation. SSIM confirmation trước xuất tăng0,011528, sau PCM16 tăng0,012463; KL×1000 tăng0,054499. Không đổi kết quả gốc.

Chuẩn bị140 cặp/280 file từ đúng PCM gốc, dùng scalar gain để hai bản cùng loudness, không fit clean hoặc limiter. Chênh loudness lớn nhất0,0000132LU;3 cặp cần hạ target chung dưới−18 để tránh clipping mới. Kiểm chéo24 stream của12 mẫu nghe bằng FFmpeg. Job private ID143393527 v1 được Kaggle nhận, receipt trả canonical slug `aajayagh/sonicmaster-vrg-level-matched-check`. Watcher ban đầu dùng slug yêu cầu `...-level-check` nên bị403; sửa để luôn đọc slug server trả, không gửi lại job.

Bộ nghe bổ sung `artifacts/12_vrg_followup_checks/listening_level/LISTEN.html` có24 WAV A/B đã kiểm hash,36 link tham chiếu gốc, storage/export phiếu riêng; JavaScript syntax pass. Chưa có đánh giá người nghe, chưa kiểm phát âm thanh bằng browser vì giới hạn file:// đã ghi trước.


### Hoàn tất kiểm chứng bổ sung sau pilot — 08/11/2026 16:13

<!-- vrg-followup-complete:9c80a115cd294adb834145afe298a6f7a837124866648f36bc3013a231e1f15d -->

Theo yêu cầu chạy nốt phần cần kiểm chứng, khóa protocol bổ sung trước các phép đo mới, giữ quyết định/hệ số/cohort gốc. Bootstrap FAD2.000 theo nguồn trên cả hai cohort, kiểm weighted SVD với expanded SVD; CI FAD all chứa0 ở cả hai. SSIM confirmation trước xuất tăng +0.011528, sau PCM +0.012463; KL×1000 tăng +0.054499.

Job PQ cùng loudness ID143393527 đã complete; kiểm280 hash file mới và PCM gốc. Delta PQ sau cân loudness -0.016597, CI[-0.032911; -0.002653]; delta gốc -0.013567. Không fit clean, không limiter, giữ clipping cũ, không inferenceSM mới. Lần chuẩn bị sửa lỗi cú pháp/chuyển NumPy bool sang JSON rồi resume receipts, không đổi phép đo.

Bộ nghe bổ sung12 mẫu có24 WAV A/B cùng loudness, giữ mapping/36 link tham chiếu gốc, phiếu riêng và chưa có nhận xét. Báo cáo VERIFICATION_FOLLOWUP_REPORT.md; đây là phân tích sau quan sát, không confirmation mới. Các giới hạn về người nghe, nhóm ít mẫu, exposure bên ngoài và evaluator của tác giả vẫn được giữ. Không mở rộng grid/cohort hoặc học mask.

### Bảng review music restoration và gap ứng viên — 09/11/2026

<!-- literature-review-2026-11-09-14papers -->

Theo yêu cầu lập bảng để mở rộng khảo sát ngoài SonicMaster, đã tạo MUSIC_RESTORATION_REVIEW.xlsx (hai sheet) và MUSIC_RESTORATION_REVIEW.md tại outputs/literature_review_2026-11-09/. Bảng gồm 14 công trình, ghi bài toán, phương pháp, cách đánh giá, hạn chế tác giả nêu, nhận xét riêng của nhóm, nguồn và phần đã đọc. Đây là review sàng lọc ban đầu, chưa phải tổng quan hệ thống đầy đủ; mức đọc và phần chưa xác minh được ghi từng bài.

Đã đưa vào bốn gap ứng viên: quyết định sửa theo lợi ích khi lỗi thay đổi trong đoạn nhạc; chuỗi codec/lỗi biến thiên; hiệu chỉnh nhẹ phụ thuộc đầu vào; giữ đúng phụ âm/lời hát. Mỗi hướng có phản chứng từ nghiên cứu trước, bằng chứng còn thiếu, pilot đề xuất, cách đánh giá và điều kiện dừng. Hai ý tưởng chung về ghép tách nguồn với phục hồi và xử lý toàn bài chưa đủ để tuyên bố gap. Chưa xác nhận tính mới của hướng nào, chưa khóa hoặc chạy pilot mới.

Đối chiếu quan trọng: RemFX đã thử trường hợp không có hiệu ứng và bỏ qua xử lý; AudioSR đã giữ dải tần thấp từ đầu vào; A2SB đã nghiên cứu âm thanh dài; LOUDAR đã đánh giá đặc trưng ca sĩ; BABE-2 báo cáo phụ âm bị tạo sai. Không đồng nhất giữ đặc trưng ca sĩ với giữ đúng lời. Không coi sử dụng SonicMasterDataset là sử dụng backbone SonicMaster.

Kiểm tra file: xuất XLSX thành công, đủ 14 ID paper và 6 ID hướng, hai bảng có bộ lọc và cố định hai cột đầu/năm hàng đầu; không thấy lỗi Excel trong kiểm tra cấu trúc/nội dung. Công cụ render Artifact Tool treo ở hai lần thử và đã dừng đúng tiến trình. Đã kiểm tra ảnh dự phòng từ OOXML và kích thước chữ Arial của cả hai sheet, không thấy tràn chiều cao trong phép đo. Ảnh này là xấp xỉ; chưa xác nhận giao diện bằng Microsoft Excel. File hỗ trợ/QA lưu trong support/.

Không chạy inference, không dùng GPU Kaggle, không tính lại metric, không sửa kết quả reproduce hay Report.tex/Report.pdf.

### Đọc và kiểm tra tài liệu định hướng nghiên cứu — 09/11/2026

<!-- review-research-directions-explained-2026-11-09 -->

Đã đọc toàn bộ C:/Users/ADMIN/Downloads/Music_Restoration_Research_Directions_Explained.md theo yêu cầu; đối chiếu báo cáo observations, joins, VRG/follow-up và các nguồn chính thức mới được tài liệu dẫn. Không sửa tài liệu nguồn hoặc bảng review, không mở pilot hay inference.

Xác nhận con số 234/1.400 mẫu là số mẫu có ít nhất một target-property error tăng với ngưỡng delta > 0; không phải số thất bại được xác nhận qua nghe. Pilot joins là 12 bài demo với suy giảm amplitude, chưa đại diện toàn bộ tác vụ hoặc nguồn độc lập với training. Blend có sự đánh đổi; cần giữ phân biệt PQ gốc, kiểm tra cùng loudness và độ bất định.

Phát hiện sai ở mục 5.4: SonicMaster sử dụng VAE Stable Audio Open (paper §3.2), và paper Latent Geometry cũng khảo sát chính dòng VAE này (§3.1). Không đúng khi khẳng định SonicMaster có VAE riêng không nằm trong paper đó. Chưa kiểm hash/config checkpoint giữa hai công trình; kết quả mean shift không tự chuyển thành cải thiện SonicMaster.

G01 cần bổ sung: oracle là lựa chọn dùng clean theo metric đã định, luôn không kém từng lựa chọn dưới metric ấy. Clean input tạo ưu thế bypass theo khoảng cách với clean một cách hiển nhiên. Cần xem kết quả trên nhóm có lỗi, so với hai chính sách cố định và quy tắc đơn giản; khoảng cách oracle không chứng minh predictability hoặc novelty. Phân biệt gate dùng input trước inference với selector dùng output sau inference: loại sau không tiết kiệm lần chạy SonicMaster. Cần khóa prompt/mục tiêu restoration thay vì lẫn với mastering.

G02 cần audit phạm vi training/tác vụ và mức hỏng trước khi gọi unseen. Cân loudness là đối chứng bổ sung, giữ đo lường bản gốc, nhất là tác vụ amplitude. Pilot 20–30 nguồn chỉ sàng lọc; đánh giá theo nguồn, tập mới và ngưỡng có ý nghĩa thực tế.

Đã mở nguồn chính thức xác nhận UNIVERSE++ có LoRA với phoneme fidelity loss trên speech; ProCraft là preprint 05/11/2026, liên quan chỉnh sửa theo nhạc cụ và giữ phần ngoài chỉnh sửa; AudioEditSurvey có tài nguyên tác giả. Đây là tài liệu gần cần đọc, chưa chứng minh các gap đã được giải quyết hay còn mới.

Nguồn: https://arxiv.org/html/2508.03448v5 ; https://arxiv.org/html/2608.03721v1 ; https://arxiv.org/html/2308.16177 ; https://www.isca-archive.org/interspeech_2024/scheibler24_interspeech.html ; https://arxiv.org/abs/2611.07423 ; https://github.com/MM-Speech/AudioEditSurvey .


### Kế hoạch và vòng kiểm chứng nghiên cứu ban đầu — 09/11/2026

<!-- research-initial-checks-20261109-v1 -->

Theo yêu cầu lập plan và đi theo plan, đã khóa RESEARCH_VERIFICATION_PLAN.md và artifacts/13_research_checks/PROTOCOL_V1.md (SHA256 f965659b53945f2aee4fccea9ed2136a42708d13619084721f59f2c34e22ab36). V00–V06 có điều kiện mở/dừng; không mặc định train. Chưa xác nhận gap mới.

V01/V02 đọc nguyên metric 1.400 mẫu/200 nguồn. Oracle guard từng mẫu giảm KL 2,73705%, không đạt 5%; 100 bypass/81 nguồn. Quy tắc input-PQ ở 5 fold đều chọn luôn SM. Input trội ba metric 80, SM 210, đánh đổi 1.110; không gọi là lỗi nghe. Kiểm độc lập 72 tổng hợp, sai khác tối đa 7,1e-15, hash input không đổi. Ghi rõ guard từng mẫu của V01 khác guard trung bình V02: oracle KL không guard giảm 6,05531%, deltaPQ −0,032386 và deltaSSIM +0,040999. Không dùng 2,74% làm trần cho mọi bộ chọn hoặc thay protocol sau kết quả.

V03 tải 36 WAV gốc khớp SHA, xuất 72 WAV của12nguồn ở hai chế độ, cùng độ dài; chênh loudness matched tối đa0,03LU. Nhận file thật của người dùng với5phiếu T01–T05 matched, SHA 4c3be64eac5f6d9c5fa6f7ad49136b06196bedd991ba39927bad0e36cd153918; giữ nguyên trong responses. Người dùng xác nhận bằng chat nhận xét original tương tự và T04 B nhỏ đi ở khoảng10s rồi tăng dần. Không tự điền original form/mẫu còn thiếu. Đo RMS sáu WAV T04 theo protocol hậu quan sát: clean tăng8,226dB, A14,711dB, B giảm1,672dB giữa6–9s và10–13s. Hình dạng sau cân loudness khác dưới0,001dB khi bỏ offset hằng. Không quy hiện tượng B là lỗi mới do SonicMaster; input đã có. Phần tăng dần chưa được khoảng đo muộn xác nhận rõ. Chưa có nghiên cứu nhiều người nghe.

V05 pin Apollo repo/model/SHA và kiểm MP3/AAC synthetic encode/decode. Tạo môi trường inference CPU riêng, không sửa môi trường global. Lần load đầu thiếu kiểu cấu hình OmegaConf; ghi amendment trước forward, giữ weights_only và strict load. Qua3/3 tín hiệu tổng hợp2s, giữ88200frame stereo44100Hz, output finite và FLOAT roundtrip exact; tổng forward26.23s. Không diễn giải thành phục hồi nhạc tốt. V06 audit79bản ghi JamendoLyrics/25000nguồn:13ID trùng train,66không thấy theo ID; chưa kiểm gần trùng/phoneme/ground truth. VAE cùng dòng với Latent Geometry, exact checkpoint/config còn thiếu.

Cập nhật RESEARCH_CHECKS_PROGRESS.md, READINESS_REVIEW.md, V01_INTERPRETATION_NOTE.md, V03_LISTENING_REPORT.md và status.json. Giữ snapshot readiness trước smoke. Chưa có job GPU/Kaggle hoặc inference SonicMaster mới; không đổi kết quả reproduce/VRG/Report.tex/Report.pdf. Bước tiếp theo cần khóa bài toán/dữ liệu mới cho dynamics hoặc codec; chưa train gate/adapter.


### Pilot G01/G02 mới: khóa dữ liệu, staging và sửa môi trường — 09/11/2026

<!-- research-fresh-pilots-stage-and-env-20261109 -->

Theo yêu cầu tiếp tục toàn bộ plan có điều kiện, khóa PILOT_PROTOCOL_V2.md/cohort/recipe trước inference. Sửa đề xuất nhiễu nền sang nén dải động vì additive noise không nằm trong 19 degrader đã công bố của SM. Loại 269 nguồn đã dùng, chọn theo SHA 2 nguồn smoke và 24 nguồn pilot; metadata 12 vocal/12 instrumental trong pilot chưa được xác nhận bằng nghe. Không chạy lại 1.400 mẫu.

CPU staging aajayagh/sonicmaster-g01-g02-stage-26 complete: 26 nguồn, 182 case, 401 file đã tải, bao gồm 286 asset FLOAT. 234 cửa sổ lag codec đều 0 sample; correlation tối thiểu 0,91517. AAC có dư8frame được trim cuối và ghi receipt. Hash/cohort/protocol metadata qua audit; nguồn gốc codec của bản ghi gốc và near-duplicate vẫn chưa xác minh.

GPU smoke v1 (ID143579132) lỗi import AutoImageProcessor trước inference; không có WAV/metric mô hình. Quota287,457s được tính vào giới hạn7200s. Ghi ENVIRONMENT_AMENDMENT_03.md; chuyển NumPy về2.1.3 đã xuất hiện trong lần reproduce thành công, tắt import optional TF/Flax và bỏ optional torchvision/peft/torchao trong kernel. Gửi CPU preflight trước khi thử GPU lại. Không đổi dữ liệu, prompt, model, seed hoặc ngưỡng vì kết quả chất lượng.

Hoàn tất V06_READINESS_FOLLOWUP.md: companion Latent Geometry truy cập được, chưa thấy link implementation trong các link tĩnh đã kiểm; exact checkpoint/latent/resampling recipe còn thiếu. Phân biệt BABE-2 phụ âm, UNIVERSE++ speech fidelity, LOUDAR Singer-ID và ProCraft preservation. Không train latent adapter hoặc tự tạo ground truth lời.


<!-- research-smoke-v2-started-20261109 -->
CPU environment preflight complete8/8: model imports, official Audiobox loader, Python audio stack passed; no inference. Local audit passed286FLOAT assets/26original PCM hashes/26local masks. Compressor source differs only141CRLF→LF, normalized text exact; both hashes recorded. Archived smokev1 intact in execution/attempts/smoke_v1. Smokev2 submitted15:00:52, same scientific settings; quota budget includes failed attempt. Locked12listening comparisons without consulting output quality.


### Tiếp tục smoke bằng cache riêng tư — 09/11/2026

<!-- research-smoke-resume-cache-20261109 -->

Smoke v2 hoàn tất 14 ca SM/VAE và 30 WAV, sau đó Apollo stereo 30s hết VRAM trước output đầu tiên. Đã giữ nguyên toàn bộ output; Amendment05 chuyển Apollo sang từng kênh với full30s, kiểm tương đương CPU đạt sai số tối đa 1,58325e-6. Note04 sửa baseline gain-only cho các input local có peak dưới0,8; không ghi đè kết quả raw. Người dùng duyệt upload cache 290879489 byte lên dataset riêng tư aajayagh/sonicmaster-research-smoke-v2-cache; đã upload và xác minh 41 file/30 WAV. Kaggle loại failed kernel source khỏi smoke-finish v1 dù hasError=false. Người dùng dừng phiên cũ; API xác nhận cancelAcknowledged. Lưu hồ sơ v1, sửa controller bắt mọi invalid source, gửi smoke-finish v2 với dataset cache và stage kernel hợp lệ. Không chạy lại SM/VAE đã có hoặc inference 1.400 mẫu. Theo dõi tổng quota trong trần7200s.


### Hoàn tất phần máy pilot 24 nguồn — 2026-11-09T16:52:00.108829+07:00

<!-- research-pilot-machine-complete-20261109 -->

Đã audit 168 ca mới, 1224 dòng metric và 960 bản chấm Audiobox. Gate oracle G01: True; gate mask biết trước: False. Đây là kết quả sàng lọc, chưa phải phương pháp triển khai hoặc kết luận chất lượng nghe. Báo cáo: artifacts/13_research_checks/execution/analysis/REPORT.md. Đã tạo 12 so sánh A/B/clean theo lựa chọn trước kết quả; phản hồi người nghe còn missing. Tổng GPU theo quota 3920.975/7200s. Không chạy lại inference 1.400 mẫu; các file báo cáo được bảo vệ giữ nguyên hash.


<!-- research-pilot-final-verification-20261109 -->

Xác minh độc lập 1.314 giá trị tổng hợp: sai khác lớn nhất 7,10543e-15; metric raw không đổi. G01 oracle giảm E 12,79%, PQ trung bình +0,0430; chỉ là đối chứng biết clean. Ghép vùng lỗi chưa đạt: 23/24 nguồn có E vùng hỏng tăng. Apollo giảm LSD ở 24/24 nguồn của từng codec; E còn đánh đổi ở AAC và chuỗi codec. Đã lưu diễn giải và bước tiếp theo tại execution/analysis/INTERPRETATION.md. Bộ nghe đủ 72 WAV, cùng độ dài 29,954s và sai khác mức matched tối đa 0,09 LU. Trình duyệt nhúng chặn file:// nên chưa xác nhận phát bằng trình duyệt; không thử cách vòng qua chính sách. Phiếu nghe vẫn missing. Không gửi thêm GPU job.


### Tiếp nhận10phiếu nghe và kiểm mức âm — 2026-11-09T17:34:40.483571+07:00

<!-- fresh-pilot-10-listening-imported-20261109 -->

Nhận10phiếu matched/7nguồn từ1người. Người dùng chọn tạm dùng10mẫu; U09–U10 và toàn bộ phiếu original giữ missing. U01–U04 tương đương khi cân độ lớn; U05–U08 nhận ra tụt mức khoảng10–12s, tăng lại20s; U11–U12 có nhận xét sạn/trầm/vocal mờ. Đã lưu raw snapshot SHA256 841bbd9767fc5f23cbe21b735beeb39efb702842809a8a1c6e9870323083c3c7. Follow-up07 đọc audio có sẵn trên2nguồn: input đã tụt mức ở vùng lỗi, output chưa khắc phục; export gain không tạo biến thiên cục bộ (RMS residual tối đa 0.001005dB). U11 người nghe/E ưu tiênB, LSD/PQ ưu tiênA; không suy rộng độ chính xác metric từ một ví dụ hoặc từ rawmetric vs matchedlistening. Báo cáo execution/listening/LISTENING_REPORT.md, có hình RMS; không đổi threshold, audio, metric, báo cáo gốc hoặc chạy GPU.


<!-- fresh-pilot-listening-metadata-fixed-20261109 -->

Đã đồng bộ phiếu trạng thái tiến độ với status.json: đã tổng hợp 10 phiếu nghe trên 7 nguồn. Sửa khoảng trắng trong phần diễn giải báo cáo nghe; giữ nguyên nhận xét nguyên văn, số đo và hồ sơ hoàn tất trước đó. U09–U10 và phiếu âm lượng gốc vẫn missing; không chạy lại phân tích hoặc gửi GPU job.


### G01/G02: tách mức âm, phổ và hồ sơ failure — 2026-11-09T18:07:35.054201+07:00

<!-- g01-g02-followup08-complete-20261109 -->

Đã khóa phép chẩn đoán sau quan sát trong followup08/FOLLOWUP_08.md rồi đọc WAV sẵn có của đủ 24 nguồn, tạo 744 dòng chẩn đoán mới. Không inference hoặc tạo audio mới. E tính kiểm khớp metric gốc tối đa 1,33e-15; kiểm 279 mean, hash input/metric/báo cáo gốc giữ nguyên. G01 local: input giữa đoạn thấp hơn clean trung bình 14,865 dB; SM thấp hơn 19,002 dB, nhỏ thêm ở 23/24 nguồn. Sau bù mức từng frame bằng clean, input có E thấp hơn SM ở 70/72 ca hỏng; dư địa oracle theo endpoint mới là 0,519% so giữ input, không thay gate E gốc 12,792% so luôn SM. Sai lệch phổ đã xuất hiện ở VAE, chưa quy nguyên nhân duy nhất hoặc khẳng định nghe được. Lập FAILURE_CASES.md cho U05/U08, giữ nguyên lời nhận xét và nhãn A/B. G02: Apollo giảm LSD mọi nguồn; E sau bù từng frame giảm 24/24 MP3, tăng 16/24 AAC và 11/24 chuỗi. PQ giảm trên ít nhất hai codec ở 9/24 nguồn, cả ba ở một nguồn; chưa phải kết luận chất lượng nghe. Báo cáo: artifacts/13_research_checks/execution/followup08/REPORT.md. Hình gain_diagnostic.png đã kiểm trực quan. Không train selector/adapter, không đổi audio, metric cũ, protocol hoặc thêm phiếu nghe.


### Đính chính diễn giải E_frame — 2026-11-09T18:26:33.058917+07:00

E_frame do trợ lý định nghĩa cho follow-up 08, không lấy từ paper SonicMaster hoặc một nguồn đã dẫn để xác thực cảm nhận. Phép bù từng khung nhìn clean có thể bỏ qua chính sai lệch mức âm của task G01. 70/72 ca input thắng dưới phép đo này không chứng minh nên bypass hoặc SM làm mất chi tiết nghe được; 0,52% không bác bỏ 12,79% theo endpoint gốc. Giữ phép đo ở phần chẩn đoán phụ, trở lại câu hỏi dự đoán lợi ích phục hồi và xác thực tiêu chí lựa chọn. Chuyển đường âm lượng vẫn chỉ là đề xuất chưa thử. Lưu execution/followup08/METRIC_SCOPE_CORRECTION.md, giữ nguyên báo cáo/số liệu đã lưu và hash. Không tính metric, chạy inference hoặc mở GPU job mới.


### Tổng kết và thu gọn G01/G02 — 2026-11-09T18:33:50.672422+07:00

Lưu artifacts/13_research_checks/G01_G02_TONG_KET_2026-11-09.md. Tách rõ mục tiêu, bằng chứng, phần chưa chứng minh và giới hạn E_frame. Giữ dư địa oracle E gốc 12,79%, failure mức âm U05/U08 và kết quả Apollo; chưa có selector thực tế, xác nhận lợi ích nghe rộng hơn hoặc research gap. Không dùng E_frame để kết luận input hay hơn hoặc chuyển mục tiêu; đề xuất chuyển đường âm lượng chưa chạy. Câu hỏi cần xác nhận tiếp là tiêu chí lợi ích phục hồi có phù hợp mục tiêu và lựa chọn nghe hay không. Không mở metric, model hoặc GPU mới trong lượt tổng kết.


### Audit nén cục bộ: bộ tạo lỗi, các nhánh và tài liệu — 2026-11-09T20:00:51.036322+07:00

<!-- local-compression-audit01-complete-20261109 -->

Hoàn tất ba khảo sát đầu, không chạy model/GPU mới. Xác minh 28 phép thử hằng và 4 cặp sóng sin: detector RMS của compressor phát hành trả năng lượng bình phương cho ngưỡng/công thức gain theo biên độ. Với ratio20, tăng input6,021dB làm mức ra ổn định giảm5,419dB; đối chứng lấy căn tăng0,301dB. Bốn bản code có AST giống nhau; không sửa source hoặc dữ liệu. Đây là kết luận về implementation/recipe, chưa xác minh lịch sử training checkpoint hay nguyên nhân duy nhất của phản ứng SM.

24 local khớp bitwise recipe; peak strong nằm giây đầu19/24 nguồn so clean4/24. Đọc lại số đo: input giữa đoạn−14,865dB so clean; VAE−14,273dB; SM−19,002dB. VAE lớn hơn input0,592dB ở24/24 nguồn; SM nhỏ hơn input4,137dB ở23/24. Gain-only trùng input22/24; expander làm nhỏ thêm6,819dB, chưa phải baseline inversion đủ mạnh. Kiểm độc lập210mean, sai khác tối đa3,55e−15;170audio và các file được bảo vệ giữ nguyên hash.

Đã rà4nghiên cứu DRC inversion/de-limiting và2repo. Có tiền lệ dự đoán gain theo thời gian và đảo compressor; chưa xác nhận gap cục bộ. Dừng mở rộng trên recipe cũ; bước tiếp là xác thực generator độc lập và baseline tương thích trước pilot GPU. E_frame vẫn chỉ là diagnostic; G01 selector tạm dừng. Báo cáo artifacts/13_research_checks/execution/local_compression_audit01/REPORT.md; tài liệu LITERATURE_REVIEW.md. Reproduce1400, VRG, protocol, audio/metric và phiếu nghe cũ giữ nguyên.


### Kiểm chứng bộ nén độc lập và smoke mới — 2026-11-09T20:57:03.292559+07:00

<!-- conventional-drc02-cpu-complete-smoke-v2-submitted -->

Đã kiểm tra FFmpeg acompressor (RMS, stereo maximum link, threshold -24 dB, ratio4, attack20 ms, release150 ms, không makeup/peak normalize). Tăng input6,0206dB làm output tăng1,50515dB, đúng slope4:1; DC/sine/silence/identity/timing/stereo/ramp đều đạt. Đã tạo32WAV FLOAT từ8nguồn đã quan sát: clean, giảm mức6dB, nén không makeup, cùng nén nhưng RMS giữa đoạn bù bằng clean. Ngoài10–20s giữ nguyên. Mean8nguồn: gain−6dB; DRC raw−10,5324dB; DRC matched≈0dB. Hai nhánh DRC giữ cùng sai khác dạng đường RMS≈0,82743dB, tách được yếu tố gain với thay đổi dynamics. Đây là kiểm tạo lỗi, chưa phải kết quả phục hồi.

Baseline DRC_Inversion pin aea5a0b, checkpoint musdb18hq-compression-mee.pt4,2MB khớp digest release nhưng torch.load không mở được (thiếu ZIP central directory). API mono/một bộ tham số5s giữa, khác stereo-linked/dry-wet của bộ thử; probe DSP own-model và3profile phát hành cũng chưa đạt ở môi trường hiện tại. Không kết luận paper baseline sai, không sửa source/trọng số.

User xác nhận gửi code/metadata≈28KB tới notebook riêng tư, không upload WAV/token. Kernel143629300 được Kaggle cấp slug sonicmaster-drc02-diagnostic-smoke. V1 xếp hàng lâu, user chủ động cancel rồi yêu cầu gửi lại; API xác nhận cancelAcknowledged, output0, quota chưa tăng. Đã gửi V2 cùng script/cấu hình, cap900s cộng margin600s,2nguồn/6case, trạng thái lần đọc cuối queued. GPU toàn kế hoạch đã dùng3920.975s/7200s theo quota snapshot. Sáu nguồn còn lại chưa inference. Giữ nguyên reproduce1400/VRG/pilot/protocol/report cũ; G01 selector tiếp tục tạm dừng. Báo cáo artifacts/13_research_checks/execution/conventional_drc02/CPU_REPORT.md.


### Chuyển hướng: toàn bộ 7.000 test samples — 2026-11-09T22:20:02.161235+07:00

<!-- run7000-scope-and-runtime-benchmark-prepared -->

Theo buổi họp với thầy, ưu tiên hoàn thành 7.000 mẫu, tối ưu thực thi và nghe/xem hình để tìm lỗi. Khóa 7.000 ID/1.000 nguồn; tái sử dụng 1.400 ID/200 nguồn, còn 5.600 ID/800 nguồn, chia 16 shard × 350. Đối chiếu hash inventory/progress cũ, đủ tên WAV trên 4 kernel complete; API không có kích thước, đã sửa lỗi ghi missing thành 0 byte. Hash live vẫn chờ tải. Không sửa audio/metric/model cũ. Ngoại suy 12,52 giờ inference còn lại, chưa gồm setup/queue/evaluation.

Chuẩn bị 14 mẫu mới/2 nguồn cho benchmark baseline, cache text cold/warm và inference_mode; giữ fp32/Euler10/guidance1/seed0, yêu cầu so raw tensor và PCM16 hash. CPU stage và GPU benchmark tách riêng. User đã cho phép hai payload riêng tư 20 KB + 28 KB, không WAV/token/trọng số; job CPU stage đã được Kaggle nhận. Chưa có kết quả tối ưu hay job production mới tại mốc này. Xem artifacts/14_run7000/PLAN.md và status.json.

Dừng hướng DRC02 theo yêu cầu; đã tạo STOP_REQUESTED và chặn submit/watch cũ. API sau đó báo complete, output status xác nhận 6 case/2 nguồn, script336,655s. Chỉ lưu hồ sơ trạng thái, chưa phân tích thêm hay mở rộng sáu nguồn còn lại. Quota mới thấp hơn snapshot cũ nên không suy ra GPU bill từ delta; giữ số accounting lịch sử và thêm chú thích.


### Chuẩn bị nhiều tài khoản Kaggle — 2026-11-09T22:48:03.813382+07:00

<!-- run7000-six-accounts-prepared -->

Đã xác thực 6 tài khoản khác nhau, tổng quota snapshot172,33giờ GPU. Token bổ sung được mã hóa bằng Windows DPAPI và lưu ngoài project; đã quét project, không có token plaintext trong script/payload/report. User xác nhận thêm quyền Reader cho tietietie, lquangnhtt, sirvesta128, mnhcutephoomaique; giữ VAE riêng tư, không upload hay sửa trọng số. API xác nhận 5 tài khoản đọc được VAE. vientaifjs dự phòng, chưa cấp quyền VAE. Phân việc 16 shard: tietietie1400, bốn tài khoản còn lại1050mẫu/tài khoản; chưa gửi job sản xuất. Khóa chống gửi trùng trong dispatch_ledger.json.

Đã bắt đầu tải 1.400 WAV SM cũ (~7,397GB) vào artifacts/14_run7000/reused1400; SHA256 so inventory/progress gốc, không inference lại. Trình theo dõi benchmark chỉ chạy tối đa60phút, chỉ gửi đúng benchmark đã được cho phép khi stage complete; không tự mở production. CPU stage private đang queued ở lần đọc hiện tại; 20 job gần nhất không có job nonterminal khác. Chưa xác định nguyên nhân queue, không khẳng định hết GPU.


### Xác minh lại đủ 1.400 WAV gốc — 2026-11-09T23:01:35.289435+07:00

<!-- run7000-original-1400-wavs-live-verified -->

Đã tải đủ 1.400 WAV SonicMaster từ 4 kernel gốc về artifacts/14_run7000/reused1400/sonicmaster_wav, tổng 7397437600 byte. SHA256 tất cả file khớp inventory/progress gốc; kiểm PCM16/stereo/44,1kHz/số frame đều đạt, không inference lại hoặc sửa WAV. Receipt: artifacts/14_run7000/reuse1400_verification.json. 5.600 mẫu còn thiếu chưa chạy; CPU benchmark stage vẫn queued, GPU benchmark chưa được gửi tại mốc này.


### Chuyển chuẩn bị benchmark sang CPU local — 2026-11-09T23:26:22.898302+07:00

<!-- run7000-local-cpu-stage-and-benchmark-submitted -->

Theo yêu cầu người dùng, bỏ phụ thuộc CPU notebook đang queue. Đã sửa lỗi cú pháp script local, tải đúng hai parquet của revision khóa và chuẩn bị 14 mẫu mới/2 nguồn, 28 WAV input/clean trong 38,938 giây (chưa gồm ZIP/kiểm độc lập). Kiểm đủ ID, 44.100 Hz/stereo/PCM16/1.323.000 frame, hash FLAC/WAV/ZIP; dữ liệu PCM khớp nguồn, chỉ nhân đôi mono khi cần. Không chạy model hoặc tính lại metric. Các file kết quả cũ được bảo vệ vẫn khớp hash.

ZIP 137.243.096 byte, 28 WAV + sáu JSON. Sau xác nhận rõ của người dùng, đã upload gói riêng tư aajayagh/sonicmaster-7000-benchmark-inputs. Kaggle tự giải nén; version 1 ready, private, đủ 35 file kể cả README, tên/kích thước khớp. Runner được sửa để đọc thư mục giải nén và kiểm 34 hash trước khi tải model. Khối đọc input đã kiểm local với UTF-8 giống Linux; không chạy model.

Đã gửi đúng một GPU runtime benchmark: https://www.kaggle.com/code/aajayagh/sonicmaster-7000-runtime-benchmark, kernel 144010286, version 1. Giữ seed0/Euler10/guidance1/fp32/T4, cap1.500 giây script +600 giây dự phòng. Lần đọc hiện tại queued; chưa có số đo tối ưu hoặc output mới. CPU stage cũ do người dùng cancel, API còn cancelRequested; không gửi v2, watcher cũ dừng. Banner Kaggle người dùng cung cấp báo lỗi khởi động/kết nối, phù hợp triệu chứng nhưng chưa chứng minh nguyên nhân duy nhất. Chưa gửi production; đủ 1.400 WAV cũ đã xác minh, 5.600 mẫu mới vẫn chờ. Hồ sơ tại artifacts/14_run7000.


### Chuẩn bị script tự chạy độc lập — 2026-11-10T00:24:06.382168+07:00

<!-- run7000-autopilot-prepared-awaiting-production-approval -->

Theo yêu cầu tiếp tục khi Codex ngắt kết nối, đã triển khai artifacts/14_run7000/autopilot: START/STOP/SUPERVISOR, checkpoint, catalog private, stage CPU local, private upload, bounded GPU dispatch và kiểm output. Khóa 5.586 mẫu mới/16 job, cộng 14 baseline benchmark hợp lệ và 1.400 output cũ; kiểm benchmark rồi shard đầu trước khi scale. Giữ baseline seed0/Euler10/guidance1/fp32/T4, không tự áp tối ưu. Max5 tài khoản, 1 job/account, max16 submission, cap7.200s/job +600s margin, giữ quota evaluation; không retry job lỗi hoặc POST không rõ kết quả, cửa sổ gửi72h. 11 test offline đạt, gồm stage thực trên2row nguồn cached, PCM exact, resume, shared clean, ZIP và khối đọc input GPU; catalog live xác nhận đích shard đầu chưa tồn tại.

Bộ duyệt tự động đã chặn lệnh khởi động vì cần xác nhận cụ thể các đích/payload production; đã hỏi người dùng, chưa bật coordinator hoặc upload/gửi production. Chỉ khởi động stage_only CPU local của shard1 PID31428, không gọi Kaggle hoặc inference, có khóa tránh hai worker ghi cùng shard. Tắt Codex được khi coordinator đã được bật; máy vẫn phải bật/không sleep/có Internet. Không sửa WAV, metric, protocol hoặc report cũ.


### Bật bộ điều phối production độc lập — 2026-11-10T00:55:04.227355+07:00

<!-- run7000-autopilot-production-authorized-and-started -->

Người dùng xác nhận rõ: “Cho phép bật tự chạy production”. Ghi PRODUCTION_AUTHORIZATION.json cho hash plan a1cecf0f5284bb6bb8a701adbfb9e74444281b86d8b8b22380e890146b50d30a: 16 đích private, 5.586 mẫu, 5 tài khoản, tối đa 32h script GPU/34,67h kể cả margin. Yêu cầu xác nhận trước đó đã được giải quyết. Kiểm 33 file triển khai khớp receipt rồi khởi động supervisor ẩn PID6864, coordinator PID30488. Heartbeat cập nhật qua nhiều vòng, xác nhận tiến trình còn chạy sau khi lệnh khởi động đã kết thúc. Windows nhận yêu cầu giữ máy thức.

CPU local shard1 hoàn tất 336 mẫu/384 WAV input+clean; gói 1.788.587.465 byte. Coordinator đã ghi upload_intent tới dataset private tietietie/sonicmaster-7000-auto-s01-inputs, upload đang được xử lý; chưa coi là ready khi chưa có receipt. Benchmark vẫn queued, chưa gửi production lúc ghi nhật ký. Sau benchmark đủ14baseline và đối chứng ổn định, script mới gửi shardđầu; sau xác minh shardđầu mới mở rộng. Không tự đổi cấu hình/áp tối ưu, không infer lại1.400 cũ. Trạng thái tiếp theo cập nhật tự động trong autopilot/STATUS.md và state.json. Đóng Codex được nhưng PC phải bật/có mạng; restart hoặc tắt máy cần chạy START.ps1 để tiếp tục.


### Xác minh upload đầu tiên và chạy lại bộ điều phối — 2026-11-10T01:03:12.995374+07:00

<!-- run7000-autopilot-metadata-visibility-repaired -->

Ở 00:55:46, bước đọc metadata ngay sau upload trả HTTP403 nên coordinator dừng gửi mới. Kiểm tra lại bằng SDK và REST lúc sau đều thành công: dataset tietietie/sonicmaster-7000-auto-s01-inputs version1 đã ready và private. Không coi 403 này là bằng chứng mất quyền; chưa xác định chính xác nguyên nhân phía Kaggle. Đã sửa riêng bước kiểm metadata: dùng SDK, xác minh owner/slug/private; chỉ chờ đọc lại403/404 trong30phút sau receipt tạo thành công. Không lặp upload hoặc POST inference. 15test đạt, gồm5test mới cho lỗi tạm thời, quyền riêng tư, sai owner và không ghi lại khi lỗi mạng.

Kiểm đủ387file remote/tên/kích thước khớp gói336mẫu/384WAV. Hash WAV sẽ được GPU runner kiểm trước model. upload_intent, upload_receipt, plan, quyền và gpu_template giữ nguyên hash; chưa có production submission. Lưu bản trước sửa và receipt ở autopilot/repairs/metadata_visibility_20261110. Gỡ đúng pause do kiểm metadata; khởi động lại supervisor ẩn PID15068, coordinator PID14956. Trạng thái lúc ghi: running, benchmark queued, shard1 input_ready, sleep_prevention_active=True. Giữ nguyên cửa sổ72giờ tính từ lần bật đầu. Tiến độ sống ở autopilot/STATUS.md; không thay đổi audio, metric hoặc cấu hình reproduce.


### Thử benchmark bằng API khác — 2026-11-10T08:20:04.700013+07:00

<!-- run7000-alternate-benchmark-tietietie-running -->

Theo yêu cầu “thử gửi notebook qua api khác đi”, đã gửi đúng1benchmark private tới tietietie/sonicmaster-7000-runtime-benchmark, kernel144105430, version1, lúc08:17:53 ngày10/11. Kaggle báo Running lúc08:17:56; bản gốc aajayagh vẫn Queued khi kiểm lúc08:19. Đây là khác biệt quan sát giữa hai phiên, chưa đủ xác định nguyên nhân hàng đợi. Script SHA256 giống hệt bản gốc (ef0a5d19b6f6ae5ffce3cd1379c1f068704fef709b3b050ce79bdf265b4f95f5), cùng14mẫu/T4/seed0/Euler10/guidance1/fp32/VAE mode; cap1500giây script+600giây margin. Tietietie còn30giờ trước gửi. Không inference lại1400mẫu cũ, không upload lại WAV/trọng số.

Cấp duy nhất Reader cho tietietie vào dataset input benchmark đã có (28WAV input/clean và metadata); giữ private và bảo toàn metadata khác. API trả201; bộ duyệt chặn chạy lại nhánh mutation. Đã dùng GET-only xác nhận quyền và quyền đọc thực tế, không gửi lại POST. Hồ sơ đầy đủ tại artifacts/14_run7000/alternate_benchmark.

Tạm dừng coordinator trước sửa, bổ sung benchmark_route.json ràng buộc request/plan/hashscript/hashreceipt và tài khoản được duyệt. Bộ điều phối chỉ theo benchmark mới để kích hoạt production một lần; bản cũ không còn là nguồn kích hoạt, vẫn chưa bị cancel. 19test đạt (thêm kiểm route, sai alias/source và receipt đổi). Đã khởi động lại supervisor PID29932, coordinator PID6880. Giữ nguyên hash plan production/quyền/scriptGPU, scope5586mẫu/16job, chặn mở rộng tới khi benchmark và shardđầu đạt. Trạng thái lúc ghi: benchmark running, verified1400; chưa có output mới được xác minh. Receipt bản vá ở autopilot/repairs/alternate_benchmark_20261110. Không thay đổi WAV/metric/protocol cũ.

### Benchmark tài khoản thay thế hoàn tất — 10/11/2026

<!-- run7000-alternate-benchmark-completed-58 -->

Benchmark tietietie/sonicmaster-7000-runtime-benchmark v1 đã Complete, đủ14mẫu/58lượt đo/28WAV baseline. Script1021,199giây (~17,02phút), trong đó setup575,774giây; không inference trên1400mẫu cũ. Đối chứng baseline_repeat ổn định; cache_cold/cache_warm/inference_mode và baseline_repeat khớp raw tensor/PCM theo summary. Chưa áp dụng tối ưu. Mean forward+read: baseline6,957s; cache_cold7,499s; cache_warm7,454s; inference_mode7,479s; baseline_repeat7,491s. Các số này chưa gồm toàn bộ setup/xuất file; baseline đầu và baseline lặp khác tốc độ nên chưa thể kết luận lợi ích tối ưu từ chênh lệch nhỏ.

Coordinator đã tải metadata và đang tải/kiểm hash/header28WAV trước khi xác nhận14mẫu mới rồi gửi shard336. Mốc ghi này chưa có benchmark/verified.json và chưa có submission của shard1; tổng đã xác minh vẫn1400/7000. Kết quả chi tiết ở artifacts/14_run7000/autopilot/benchmark. Trạng thái tiếp tục cập nhật tự động trong autopilot/STATUS.md.


### Benchmark verified và gửi shard đầu — 2026-11-10T08:39:03.399441+07:00

<!-- run7000-first-production-shard-submitted -->

Đã tải và xác minh đủ28WAV baseline của14mẫu benchmark; tổng inventory tăng lên1414/7000. Bộ điều phối tự gửi shard1 gồm336mẫu tới tietietie/sonicmaster-7000-auto-s01 lúc08:38:34, kernel144107497/version1/private/T4. Trạng thái API mới nhất: running. Giữ cấu hình reproduce; chỉ mở15shard còn lại sau khi shardđầu complete và đủ output hợp lệ.


### Mở các shard khác sau benchmark theo yêu cầu — 2026-11-10T08:43:11.371557+07:00

<!-- run7000-user-release-parallel-after-benchmark -->

Người dùng yêu cầu chạy các shard khác trong khi shard336 đang Running. Đã thay riêng điều kiện mở rộng: sau benchmark14mẫu đã verified, không cần đợi toàn bộ shard1 hoàn tất. Giữ tối đa5job production/1job mỗi tài khoản,16đích/5586mẫu, giới hạn32giờ script/34,67giờ gồm margin, cửa sổ72giờ và dự phòng evaluation. Plan/hash/model config/output cũ không đổi. Một worker CPU tiếp tục chuẩn bị+upload dữ liệu lần lượt, nên shard khác chỉ được gửi khi gói riêng đã ready. Lỗi thật ở job bất kỳ vẫn dừng gửi thêm; không retry inference tự động. 23test đạt, gồm kiểm benchmark/owner/cap/pause vẫn chặn đúng. Hồ sơ thay đổi ở autopilot/parallel_release.json và repairs/parallel_release_20261110.


### Tăng chuẩn bị dữ liệu để cấp job cho tài khoản rảnh — 2026-11-10T08:51:01.685209+07:00

<!-- run7000-fill-idle-accounts-two-preparations -->

Người dùng yêu cầu thêmjob ở tài khoản cònslot. Snapshotquota: aajayagh22,7125h; tietietie29,5923h; lquangnhtt29,6215h; sirvesta128 và mnhcutephoomaique30h. Quotareserved0 không được dùng riêng để kết luận không cójob vì shard1 đangRunning. Nútthắt là stage/upload local; đã nâng tối đa2worker chuẩn bị/upload độc lập, mỗi job cókhóa/checkpoint riêng và SDK upload táchtiếntrình theoaccount. Chỉ mởworkerthứ2 khiRAMkhảdụng>=2,5GiB, nếu không dùng1. Máy16GB/Ryzen5500U/12luồng; kiểmRAMrảnh~3,17GiB trước sửa. Giữ5job production/1mỗiaccount vàscope/cấuhình/quota gốc; không tựcancel/gửitrùng. 27test đạt, gồmbackoff/pause/lowmemory/khôngduplicateworker. Shard1 vẫnchạy trênKaggle; cácgói khác tựgửi khiđủdata vàquota.


### Shard1 hoàn tất inference — 2026-11-10T10:06:09.542067+07:00

<!-- run7000-s01-complete-336 -->

Kaggle báo complete336mẫu/672WAV. Script2894,62giây (~48,24phút), setup253,73giây, meaninference7,73giây/mẫu. Coordinator đang tải/kiểmhash vàheader, chưa cộng336vào inventoryverified. Stage2/3 gặp lỗi kết nối (ChunkedEncodingError/ConnectionError); cơ chế retryđọc đã hoạt động, stage2 tiếp tục từcheckpoint và stage4 cũng đang chuẩn bị. Không inferlại hoặc đổi cấu hình.


### Cap nhat cac shard — 2026-11-10T10:11:19.522824+07:00

[{"shard": 1, "account": "tietietie", "samples": 336, "state": "collecting", "input_ready_count": 336, "sonicmaster_wavs_verified": 259}, {"shard": 2, "account": "sirvesta128", "samples": 350, "state": "preparing", "input_ready_count": 217, "sonicmaster_wavs_verified": 0}, {"shard": 3, "account": "mnhcutephoomaique", "samples": 350, "state": "waiting_input", "input_ready_count": 77, "sonicmaster_wavs_verified": 0}, {"shard": 4, "account": "lquangnhtt", "samples": 350, "state": "preparing", "input_ready_count": 105, "sonicmaster_wavs_verified": 0}]

Shard5-16 chua chuan bi. Coordinator van running, pause_new=False. Haiworker dang xu ly shard2/4; shard3 cho retry sau loi ketnoi, khong mat77mau checkpoint. Snapshot daydu: artifacts/14_run7000/SHARD_PROGRESS.md.

## 2026-11-10 — Reroute shard 2 và 7
- Người dùng cho phép upload riêng tư hai gói input/clean đã chuẩn bị cho shard 2 (aajayagh) và shard 7 (lquangnhtt), giữ nguyên 350 sample ID/shard, mã nguồn inference và cấu hình.
- Hai dataset riêng tư đã upload và xác minh đủ 403 tệp mỗi bộ. Shard 2 đã gửi notebook recovery lên aajayagh; Kaggle xác nhận version 1, mã nguồn/dataset/GPU đúng, trạng thái running lúc 12:48. Kaggle tự đổi slug notebook từ recover thành recovery; điều phối đã theo biên nhận thực tế.
- Shard 7 chờ lquangnhtt hoàn tất shard 9 để không đè lượt GPU; recovery scheduler chạy local và tự gửi khi trống.
- Coordinator chính được giữ lượt trên hai owner này. Shard 10 có upload intent nhưng không có receipt và dataset không xuất hiện trong hai lần catalog; đã lưu review và thử upload lại đúng gói đầu vào một lần. Cần xác minh input_ready rồi bỏ pause_new để tiếp tục các shard khác.
- Không chạy lại 1.400 mẫu cũ, không tính lại metric, không đổi plan khóa.

- 12:58: Shard 10 dataset đã ready, riêng tư, đủ 403 tệp; lỗi sau cùng chỉ do Windows khóa ZIP tạm khi dọn file. Coordinator đã resume (pause_new=false), 2.800/7.000 mẫu đã xác minh. Shard 2 recovery đang running; shard 7 chờ owner lquangnhtt trống.


## 2026-11-10 — Thử 2 GPU jobs trên cùng tài khoản
- Theo yêu cầu 'ok thử đi', nhóm kiểm tra shard 7 còn running trên lquangnhtt và shard 14 input đã ready (403 tệp), quota còn 98.110 giây, lớn hơn mức cần 19.200 giây cho hai job và dự phòng. Trước khi gửi có 4 GPU job toàn đợt, giới hạn tổng vẫn 5.
- Đã gửi shard 14 version 1 trên lquangnhtt. Kaggle báo hasError=false, trạng thái running lúc 14:03 khi shard 7 cũng còn running. Coordinator nhận lại shard 14 và theo dõi bình thường. Không đổi plan inference, sample IDs hoặc cấu hình; đây là thử một lần vượt giới hạn điều phối 1 job/tài khoản.


## 2026-11-10 — Ba tài khoản còn slot
- Theo yêu cầu, đã gửi shard 15 lên aajayagh làm job thứ hai khi shard 10 còn chạy; quota trước gửi 73.003 giây, ngưỡng hai job 19.200 giây. Kaggle nhận version 1, coordinator theo dõi.
- Shard 16 đã chuẩn bị local đủ 350 input và 50 clean WAV; đang upload/kiểm tra dataset riêng tư trên tietietie trước khi gửi GPU job.
- Shard 12 từ sirvesta128 được chuyển tuyến dự kiến sang mnhcutephoomaique, giữ 350 ID, seed/cấu hình/template. Vì RAM thấp, bắt đầu chuẩn bị local sau khi shard 15/16 hoàn tất phần CPU. Worker background sẽ đóng gói, upload riêng tư, gửi và thu kết quả khi qua các cổng kiểm tra.
- Lỗi Windows khóa parquet cache làm dừng staging tại shard 15 và 16; đã chạy tiếp từ checkpoint. Sửa cleanup để đóng handle, thử xóa lại và nếu còn khóa thì ghi cleanup pending, không hủy shard.
- Tại 14:44, 5.250/7.000 mẫu đã được xác minh; các shard mới chưa tính vào số này.


- 15:11: Đã gửi đủ ba tài khoản còn lại: shard 15 aajayagh running, shard 16 tietietie running, shard 12 reroute mnhcutephoomaique running (Kaggle version 1, hasError=false). Shard 12 dataset riêng tư đủ 403 tệp, 350 mẫu; quota trước gửi 98.214 giây. Shard 7 recovery đã xác minh. Tổng hiện 5.950/7.000 mẫu xác minh; ba shard 12/15/16 còn chờ kết quả.


## 2026-11-10 — KL/Mel-SSIM cho 7.000 mẫu
- Inference đã xác minh đủ 7.000/7.000 mẫu.
- Chạy KL/Mel-SSIM trên Kaggle CPU cho 16 shard production (5.586 mẫu); pilot 7 mẫu đối chiếu với máy local, sai khác lớn nhất 7.41e-14.
- Gộp với 14 mẫu benchmark và 1.400 mẫu cũ; xác minh 7.000 ID, 1.000 nguồn x 7 biến thể, 42.000 dòng metric. File: artifacts/14_run7000/evaluation/kl_ssim/kaggle/per_sample_7000.csv, summary_7000.csv, completion.json.
- Trung bình All: input KLx1000 5.1299 / SSIM 0.7770; VAE 5.2752 / 0.5457; SonicMaster 0.8928 / 0.6091.
- PQ/FAD-CLAP chưa nằm trong bảng KL/SSIM này; cần xử lý riêng.

## 2026-11-11 — Hoàn tất PQ, FAD-CLAP và Tables 1–2 cho 7.000 mẫu
- PQ/CLAP: 16/16 shard và benchmark14 được xác minh; gộp 22.000 CLAP embeddings, 22.000 bản ghi PQ; FAD được tính toàn bộ theo nhóm. All SonicMaster: FAD 0,072564 (paper 0,073), PQ 7,705057 (paper 7,705).
- Tables 1–2: 19 loại lỗi, 11.000 cặp mẫu–thuộc tính, 33.000 dòng nhánh input/VAE/SonicMaster; 16/16 shard Kaggle CPU và 14 benchmark được xác minh. Kết quả `artifacts/14_run7000/evaluation/attributes/`.
- Ngoại lệ: clip SonicMaster 0,152606 so với paper 1,506; cùng chỉ số trên 1.400 mẫu cũ là 0,001394. Cần audit giao thức/kích thước cohort, chưa diễn giải là model vượt paper.
- Các thống kê cặp: KL tốt hơn input 5.881/7.000, PQ cao hơn 5.346/7.000, SSIM cao hơn 1.063/7.000; 1.211/11.000 cặp thuộc tính xấu hơn input. Đây là tín hiệu metric, chưa phải kết luận nghe.
- Báo cáo hoàn chỉnh: `BAO_CAO_REPRODUCE_7000.md`; dữ liệu chi tiết và hash trong `artifacts/14_run7000/evaluation/`.


## 2026-11-20 — Extension `10_extension` (Codec-Bypass Hybrid): Hoàn tất Phase 0–3 & Khóa tham số

- **Phase 0:** Kiểm kê 7.000 mẫu (`artifacts/10_extension/inputs_manifest.csv`). 40 nguồn VRG nằm ở tập `val` (không có output `run7000`), nên đã lấy 40 nguồn từ tập `test` có đủ `(x, s, clean)` tại máy (`seed=20261119`, 280 mẫu Dev), tách rời hoàn toàn khỏi 960 nguồn Confirm (6.720 mẫu). Đã sinh 280 VAE roundtrip cho Dev (`artifacts/10_extension/dev/vae_wav/`).
- **Phase 1 (Kaggle T4, `maitvng`):** Benchmark 4 cấu hình tách giọng trên 10 nguồn Dev có vocal (`artifacts/10_extension/phase1_separator/summary.csv`). **Winner:** `bs_roformer_1297` (`model_bs_roformer_ep_317_sdr_12.9755.ckpt`, median leakage trên `s` = `-32.62 dB`, `14.69 s/clip`, `1589 MB` VRAM), vượt `htdemucs_ft` (`-30.54 dB`), `mel_band_roformer_kim` (`-31.14 dB`), và `ensemble_bs_mel` (`-31.62 dB`). Đã tách và tải 560 stem FLAC của Dev.
- **Phase 2 & Unit tests:** Hoàn tất `scripts/extension/` (`routing.py`, `branch_r.py`, `branch_p.py`, `mask.py`, `mix.py`, `run_pipeline.py`, `eval_dev.py`, `eval_tables123.py`, `eval_pq_fad.py`), 6/6 unit tests pass.
- **Phase 3 (Dev Gate — ĐẠT TOÀN BỘ):**
  - Định tuyến khóa: `P` (`comp`, `punch`, `stereo`, `volume`), `R` (`EQ` trừ `xband`, `Reverb`, `noise`), `S` (`clip`, `xband`).
  - Grid search (`artifacts/10_extension/dev/grid.csv`): chọn `A5` với `clamp=[0.2, 0.8]`, `smooth_ms=50.0`.
  - Kết quả `A5` vs `A0` (280 mẫu Dev): **SSIM** `0.6279` vs `0.6064` (`Δ = +0.0215`, đạt `≥ +0.01`); **KL×1000** `1.1490` vs `1.1064` (`+3.8%`, đạt `≤ +10%`); **PQ (Audiobox)** `7.6388` vs `7.6452` (`Δ = -0.0064`, CI95% `[-0.0168, +0.0050]`, đạt `≥ -0.05`); **FAD-CLAP** `0.2174` vs `0.2129` (`+2.1%`, đạt `≤ +10%`); **Bảng 1–2 Attribute Guards:** `0` vi phạm trên cả 15 thuộc tính (`punch -7.7%`, `comp -2.6%`, `boom -2.0%`).
- **Khóa:** Đã ghi `artifacts/10_extension/lock.json` (kèm SHA-256 mọi script) và khóa 20 nguồn vocal Confirm vào `splits.json` cho Phase 5. Chuẩn bị chạy Phase 4 (Confirm 6.720 mẫu) trên Kaggle.


## 2026-11-23 — Hybrid-v4 Pilot (`V4-0` → `V4-4`), Tài Liệu & Hình Minh Họa

### 1. Bối cảnh & Thiết lập Split (`V4` Step 0–1)
- **Mục tiêu:** Xây dựng kiến trúc neural nhẹ giữ nguyên sóng gốc `x` theo đường tắt STFT tuyến tính (không qua VAE), dùng đầu ra SonicMaster `s` (`A0`) làm bản mẫu dẫn đường (`Codec-Bypass`).
- **Chia tách dữ liệu tách nguồn tuyệt đối (`artifacts/10_extension/v4/pilot_splits.json`, `seed=20261122`, `overlap=0`):**
  - `Pilot Train`: `400` nguồn (`2,800` cặp) từ `confirm_sources`.
  - `Pilot Val`: `80` nguồn (`560` cặp) từ `confirm_sources` (chọn `best.pt`).
  - `Pilot Test (Dev 280)`: toàn bộ `40` nguồn × 7 variants = `280` mẫu từ `dev_sources` (đối chiếu 1-1 với `A0` và `Hybrid-v3`).
  - `Locked Confirm`: `280` nguồn (`1,960` cặp) khóa kín cho bước xác nhận cuối cùng.
- **Staging:** Đóng gói `480` nguồn (`3,360` clip 5.0s float16 + lag + RMS) thành dataset public Kaggle `aajayagh/sonicmaster-v4-pilot-5s`.
- **Unit tests & Benchmark (`GTX 1650`):** `scripts/extension/tests/test_v4_step0.py` kiểm tra STFT/iSTFT, M/S↔L/R, forced identity, silence, OLA chunking, và các biến thể `v4_0..v4_4` (tất cả `11/11` tests pass).

### 2. Tiến hóa kiến trúc & Kết quả thực nghiệm trên `Dev 280` (20 epochs Kaggle T4, `batch_size=12`)

| Phiên bản | Thay đổi cốt lõi so với bản trước | Params | SSIM ↑ | KL×1000 ↓ | FAD-CLAP ↓ | Audiobox PQ ↑ | Guard Fails / 12 |
|---|---|---:|---:|---:|---:|---:|---:|
| **`A0` (SonicMaster)** | Baseline generative (qua VAE) | — | — | — | `0.2083` | `7.645` | `0/12` |
| **`V4-0` (`NoGuide`)** | Chỉ `x` → `Encoder` → `4×DualAxisBlock` → `DirectMap` (không có `s`) | `0.99M` | `0.7495` | `5.1708` | — | — | — |
| **`V4-1` (`DirectMap`)** | Thêm teacher `s` vào Encoder; vẫn tự vẽ toàn bộ STFT | `1.27M` | `0.8023` | `2.6238` | `0.2507` | `7.0820` | `10/12` |
| **`V4-2` (`MF7 + Gate`)** | Thay `DirectMap` bằng đường tắt `Z_x` + lọc phức `2×2 M/S` 7 khung (`-5..+1`, ~70 ms) + Gate `g` (`bias=-3.5`, `λ_g=0.002`, `λ_ms=0.2`) | `1.68M` | **`0.8325`** | `1.2963` | `0.1848` | `7.3530` | `6/12` |
| **`V4-3` (`Dilated MF9 + MelKL`)** | Giãn tầm lọc lên **9 tap `{-18,-12,-8,-5,-3,-2,-1,0,+1}` (~220 ms)**; **Free Gate** (`bias=-1.5`, `λ_g=0.0`); thêm **`MelKLLoss` (`λ_kl=2.0`)**, tăng `λ_ms=0.5`; **Output RMS match `rms(s)`** | `1.69M` | `0.8236` | `1.1135` | **`0.1775`** | **`7.3809`** | **`2/12`** |
| **`V4-4` (`+Additive Residual R`)** | Nền `V4-3` + nhánh **Additive Complex Residual $R_\theta(Z_x, Z_s)$** (zero-init, scale theo `band_rms`); tăng `λ_ms=1.0`, `λ_kl=2.5` | `1.95M` | `0.8284` | **`1.0468`** | `0.1865` | `7.3339` | **`2/12`** |

- **Phân tích `V4-3` (`nthkhanh/sonicmaster-v4-pilot-v4-3-run`, Ep 20):**
  - Sửa triệt để `4/6` guard lỗi của `V4-2`: `boom` (`+0.167 → -0.011`), `mic` (`+0.082 → +0.012`), `vocal` (`+0.181 → -0.030`), `volume` (`+0.087 → +0.005`).
  - `FAD-CLAP = 0.1775` (tốt hơn `A0 = 0.2083` khoảng `-14.8%`), `Reverb_106 ΔPQ` cải thiện từ `-0.7072 → -0.6439`, `Clip_48 ΔPQ` từ `-0.4420 → -0.3991`.
  - Còn 2 guard FAIL: `bright (+0.194)`, `stereo (+0.177)`.
- **Phân tích `V4-4` (`maitvng/sonicmaster-v4-pilot-v4-4-run`, Ep 20):**
  - Nhánh `R` + `λ_ms=1.0` + `λ_kl=2.5` cắt gần một nửa lỗi `stereo` (`+0.177 → +0.094`) và giảm 30% lỗi `bright` (`+0.194 → +0.135`); `KL` xuống thấp nhất (`1.0468`), `SSIM` lên `0.8284`; `10/12` guard đủ mẫu PASS (`punch -0.151`, `comp -0.075`, `clip -0.029`, `boom -0.028`, `xband -0.026`, `vocal -0.007`).
  - Đánh đổi: `g_reverb` thu về `0.0958` (vs `0.1392` ở `V4-3`), làm `PQ` (`7.3339`) và `FAD` (`0.1865`) nhỉnh kém nhẹ `V4-3` (`7.3809` / `0.1775`).
  - Chẩn đoán 2 lỗi còn lại: `l_ms` mới bật từ Stage 2 (Ep 13–20), và dải `bright` (`6–22 kHz`) chỉ chiếm ~1% tổng năng lượng nên `Wav L1` + `MelKL` tuyến tính chưa tạo đủ gradient.

### 3. Tối ưu hạ tầng & Tài liệu bổ sung
- Tách `scripts/extension/v4_infer_dev_gpu.py` (chạy trên GPU `extension-env`, ~15 phút/280 bài) khỏi `v4_eval_pilot_dev.py --from-infer-json` (`eval-env` CPU cho `CLAP` + `PQ`).
- Bổ sung mục **References & Method Provenance** (đã xác minh tác giả & arXiv ID) vào `HYBRID_V2_PA1_REPORT.md`, `HYBRID_V3_REPORT.md`, và `PLAN_HYBRID_V4.md`.
- Tạo tài liệu hướng dẫn trực quan `HYBRID_V4_GUIDE.md` kèm các hình: `artifacts/10_extension/figures/hybrid_v4_architecture.jpg`, `v4_metrics_progression.png`, `v4_results_table.png`.

### 4. Quyết Định / Gợi Ý Của Người Dùng & Nguồn Gốc Bài Báo (Paper Provenance)
- **Các bước & yêu cầu từ người dùng trong phiên:**
  1. Cho chạy `V4-3` và dự toán thời gian (~70–78 phút T4 + ~15 phút chấm Dev), sau đó căn cứ kết quả `stereo` thực tế của `V4-3` để quyết định mở `V4-4` (`Additive Complex Residual`).
  2. Yêu cầu mọi báo cáo/plan kỹ thuật (`HYBRID_V2_PA1_REPORT.md`, `HYBRID_V3_REPORT.md`, `PLAN_HYBRID_V4.md`) **phải ghi rõ References — thành phần nào lấy từ paper nào** (đã xác minh tác giả, hội nghị/năm, arXiv ID).
  3. Xác minh nguồn gốc dữ liệu `train (400) / val (80) / dev (280) / confirm locked (280)` tách nguồn tuyệt đối (`overlap = 0`).
  4. Tạo bản giải thích trực quan toàn bộ kiến trúc `Hybrid-v4` kèm sơ đồ minh họa (`HYBRID_V4_GUIDE.md`, `hybrid_v4_architecture.jpg`), biểu đồ tiến trình (`v4_metrics_progression.png`), và ảnh bảng tổng hợp (`v4_results_table.png`).
  5. Ghi chép đầy đủ lý do thay đổi từng version (`V4-0 → V4-4`) và nguồn gốc bài báo vào nhật ký dự án.
- **Bảng ánh xạ kỹ thuật `Hybrid-v4` → Bài báo gốc:**
  - **Scale-Aware Band Encoder, Complex Mapping (`V4-0/1`), `TF-MoE` (`V4-5`) & VGGish warm-up:** ASLP Group (NPU-ASLP, theo gợi ý/tài liệu người dùng cung cấp), *"The NPU-ASLP System for Music Restoration in the CCF-AATC 2026 Challenge"*, Oct 2026.
  - **Nền tảng `A0` (`s`), Dataset & Bộ metric Bảng 1–3:** Melechovsky et al., *"SonicMaster"*, ICML 2026 (`arXiv:2508.03448`).
  - **Complex Multi-Frame Filtering (`MF7/MF9`) & Psychoacoustic bands:** Schröter et al., *"DeepFilterNet"*, ICASSP 2022 (`arXiv:2110.05588`).
  - **Delayed / Dilated Multi-Frame Taps (`-18..-1, +1` ~220 ms khử reverb):** Nakatani et al., *"WPE (Variance-Normalized Delayed Linear Prediction)"*, IEEE TASLP 2010.
  - **24-Band Split & Dual-Axis Time–Frequency Backbone:** Luo & Yu, *"BSRNN"*, TASLP 2023 (`arXiv:2209.15174`); Wang et al., *"TF-GridNet"*, TASLP 2023 (`arXiv:2211.12433`).
  - **Multi-Resolution STFT Loss:** Yamamoto et al., *"Parallel WaveGAN"*, ICASSP 2020 (`arXiv:1910.11480`).
  - **Metrics đánh giá:** *Audiobox Aesthetics* (`PQ`, Tjandra et al., `arXiv:2502.05139`), *LAION-CLAP* (Wu et al., ICASSP 2023, `arXiv:2211.06687`), *FAD/fadtk* (Gui et al., ICASSP 2024, `arXiv:2311.01616`), *SSIM* (Wang et al., IEEE TIP 2004).
