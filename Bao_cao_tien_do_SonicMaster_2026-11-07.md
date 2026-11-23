BÁO CÁO TIẾN ĐỘ NGHIÊN CỨU

Tái lập SonicMaster<br/>và khảo sát các trường hợp thất bại

Tổng hợp benchmark 1.400 mẫu trên Kaggle và thí nghiệm SC203 v5<br/>Ngày báo cáo: 07/11/2026 · Múi giờ Việt Nam

Tóm tắt

Nghiên cứu triển khai lại quá trình suy luận và đánh giá của SonicMaster bằng checkpoint công bố, đồng thời khảo sát giới hạn của mô hình qua các metric và các trường hợp có dấu hiệu suy giảm. Báo cáo tổng hợp hai nhánh: (A) benchmark 1.400 mẫu test với evaluator upstream; (B) SC203 v5 khảo sát độ lặp lại, bảng metric đa chiều và tham số inference trên cohort EQ/Dynamics riêng.

Kết quả chính: nhánh A hoàn tất bốn notebook inference, thu đủ 1.400 mẫu từ 200 nguồn và tổng hợp điểm. Sai số trung bình ở cả 19 thuộc tính mục tiêu giảm so với input; tuy nhiên, 256/2.200 cặp mẫu–thuộc tính tăng sai số, thuộc 234 mẫu. Đây là ứng viên failure theo metric, chưa được người nghe xác nhận.

Đã điều tra chênh lệch clipping trên đúng 211 mẫu: thay quy ước power làm điểm SonicMaster tăng từ 0,001394 lên 0,228177 (đều ×1.000), nhưng vẫn khác mốc paper 1,506. Vì vậy chưa xác lập tính so sánh trực tiếp của clipping với bảng công bố.

Theo báo cáo SC203 v5 tại 05h38 ngày 07/11/2026, B0 có waveform trùng hash ở sáu lượt cho mỗi nguồn; B1 cho thấy copy-input tốt hơn baseline ở năm metric đo độ giống target, trong khi baseline có PQ cao hơn 0,09230 điểm. B2 đã kiểm chứng 2.850/3.264 clip; phần còn lại và đánh giá held-out chưa được công nhận hoàn tất tại mốc chốt này.

Mức hoàn thành

| Hạng mục | Trạng thái |

| Inference benchmark 1.400 mẫu | Hoàn tất; chưa phải tái huấn luyện từ đầu |

| Tổng hợp metric / diagnostic clipping | Hoàn tất trong phạm vi metric đã chạy |

| Failure cases | Đã lập ứng viên và bộ nghe; chưa xác nhận chủ quan |

| FAD/PQ | Có ở SC203 B1; chưa có trên benchmark 1.400 mẫu |

| VRG / fine-tune mới | Chưa thực hiện; đang hoãn |

Phạm vi SC203 được kế thừa từ PDF do người thực hiện cung cấp [1], không phải kết quả kiểm lại receipt SC203 trong lượt biên soạn này. Nhánh A dựa trên artifact đã thu và kiểm tra trực tiếp [3–6].


---


1. Mục tiêu, dữ liệu và quy trình

1.1. Mục tiêu

Đánh giá mức độ tái lập hành vi của SonicMaster trên dữ liệu test, xác định khác biệt giữa khả năng sửa thuộc tính và độ giống bản tham chiếu, đồng thời xây dựng tập ví dụ phục vụ phân tích failure. Không đặt mục tiêu chứng minh mọi chỉ số đều vượt paper hoặc coi một metric riêng lẻ là đại diện cho chất lượng nghe.

1.2. Benchmark 1.400 mẫu

Tập đã chạy gồm 200 nguồn nhạc, mỗi nguồn có đủ 7 variants, tổng 1.400 mẫu: 800 lỗi đơn, 400 lỗi đôi và 200 lỗi ba. Đây là tập con của benchmark 7.000 mẫu, nên không đồng nhất điểm tuyệt đối với toàn bộ tập paper. Các variants cùng nguồn có tương quan; phân tích bất định lấy mẫu lại theo source_id.

| Bản audio | Vai trò trong đánh giá |

| Clean | Bản tham chiếu trước khi thêm suy giảm nhân tạo; không cấp cho mô hình khi inference. |

| Input | Bản đã bị tác động theo một hoặc nhiều degradation trong manifest. |

| VAE | Input → VAE encode → decode; kiểm tra tác động riêng của tái tạo latent. |

| SonicMaster | Input → VAE encode → biến đổi latent theo prompt → VAE decode. |

1.3. Cấu hình và kiểm chứng

Nhánh A giữ nguyên checkpoint SonicMaster, VAE và T5. Inference dùng Euler 10 bước, guidance 1, seed 0, float32, VAE latent mode, TF32 bật; xuất WAV PCM16, 44,1 kHz, stereo. Chạy trên Kaggle theo bốn shard, mỗi shard 350 mẫu. Không fine-tune hoặc huấn luyện lại backbone.

Đã kiểm đủ sample ID và source ID; đối chiếu prompt, nhãn, run_lock và hash evaluator giữa các shard. Danh sách output có đủ 5.600 WAV; hash trong progress khớp inventory của scorer. Đã kiểm byte/hash độc lập cho bộ diagnostic 211 mẫu × 4 nhánh và bộ nghe 8 mẫu × 4 nhánh; hai tập có thể giao nhau, không cộng thành số file duy nhất.

Chấm theo evaluator upstream, giữ chiều dài gốc và không căn chỉnh gain/lag theo clean. KL dùng chiều output so với clean; clipping và KL được báo theo thang ×1.000. Các giá trị tính được đều hữu hạn. Khoảng tin cậy của delta được tạo bằng 2.000 bootstrap theo nguồn.

Giới hạn kiểm chứng: chưa tải và hash độc lập toàn bộ 5.600 WAV; chưa chứng minh tương đương đầy đủ với môi trường và toàn bộ tập test của tác giả. Đây là reproduction inference/evaluation, không phải tái lập quá trình training.


---


2. Kết quả benchmark 1.400 mẫu

Bảng dưới là trung bình theo mẫu có thuộc tính tương ứng. Với metric thuộc tính và KL, thấp hơn là tốt hơn; SSIM cao hơn là tốt hơn. Các giá trị có đơn vị/thang khác nhau, không được cộng trực tiếp. N là số mẫu của từng metric, không phải số nguồn độc lập. [2–4]

| Metric | N | Input | VAE | SonicMaster | Paper |

| clarity | 74 | 0.027292 | 0.028132 | 0.011638 | 0.0114 |

| boom | 56 | 0.331241 | 0.332632 | 0.082983 | 0.0834 |

| airy | 44 | 0.010316 | 0.010659 | 0.003239 | 0.0019 |

| bright | 77 | 0.013895 | 0.014584 | 0.005179 | 0.0059 |

| dark | 77 | 0.089502 | 0.073664 | 0.008598 | 0.0058 |

| mud | 86 | 0.485108 | 0.506126 | 0.048983 | 0.0388 |

| warm | 77 | 0.447475 | 0.458447 | 0.068350 | 0.0617 |

| vocal | 76 | 0.245864 | 0.245832 | 0.057523 | 0.0576 |

| mic | 109 | 0.290474 | 0.294258 | 0.022640 | 0.0088 |

| xband | 167 | 0.182550 | 0.186909 | 0.041803 | 0.0358 |

| comp | 247 | 0.046009 | 0.045928 | 0.019328 | 0.0193 |

| punch | 85 | 0.142051 | 0.122526 | 0.088993 | 0.0871 |

| small | 80 | 0.507961 | 0.523836 | 0.374663 | 0.3663 |

| big | 75 | 0.517864 | 0.537329 | 0.406038 | 0.3726 |

| mix | 143 | 0.588043 | 0.617338 | 0.414374 | 0.3935 |

| real | 223 | 0.535776 | 0.562160 | 0.354398 | 0.3109 |

| Clip ×1.000* | 211 | 0.010771 | 0.010298 | 0.001394 | 1.5060 |

| volume | 92 | 0.170263 | 0.170020 | 0.057074 | 0.0468 |

| stereo | 201 | 0.389798 | 0.389312 | 0.098102 | 0.1058 |

| KL ×1.000 | 1400 | 5.192990 | 5.317126 | 0.886757 | 0.8880 |

| SSIM ↑ | 1400 | 0.779314 | 0.547511 | 0.611359 | 0.6090 |

* Clipping chưa xác lập tính so sánh với paper; xem mục 3. Mốc paper là snippet evaluation, khác quy mô dữ liệu. Gần nhau về số không chứng minh tương đương thống kê. Mốc thuộc tính đã đối chiếu Table 1/2; KL/SSIM All đã đối chiếu Table 3 trang 8 của PDF gốc.

KL của SonicMaster đạt 0,886757 và SSIM đạt 0,611359. So với input, KL giảm nhưng SSIM cũng giảm; so với VAE, SSIM tăng. Kết quả phù hợp với việc cần đánh giá đồng thời khả năng sửa lỗi, độ bảo toàn và chất lượng cảm nhận, không kết luận chỉ bằng SSIM.


---


3. Failure candidates và kiểm tra clipping

3.1. Các trường hợp cần khảo sát

Có 256/2.200 cặp mẫu–thuộc tính tăng sai số so với input (11,64%), thuộc 234/1.400 mẫu (16,71%). Một mẫu có thể chứa nhiều thuộc tính; hai tỷ lệ có mẫu số khác nhau. Ngưỡng hiện tại là delta > 0, chưa áp tolerance thực tiễn, nên không đồng nhất với tỷ lệ thất bại nghe được.

| Thuộc tính | Cặp xấu hơn / N | Tỷ lệ |

| comp | 65 / 247 | 26.3% |

| clip | 45 / 211 | 21.3% |

| punch | 20 / 85 | 23.5% |

| big | 18 / 75 | 24.0% |

| volume | 13 / 92 | 14.1% |

SSIM giảm so với input ở 1.175/1.400 mẫu, nhưng đây không tự động là audible failure. Đã chuẩn bị bộ nghe có nhãn gồm 8 mẫu discovery, 32 WAV: 6 ứng viên và 2 đối chứng chọn theo SSIM. Hướng dẫn nêu degradation, prompt và cách so clean/input/output; chưa thu phiếu nghe, chưa có kết luận về mất giọng, âm robot hoặc chất lượng cảm nhận.

3.2. Diagnostic clipping trên đúng 211 mẫu

| Nhánh | Upstream ×1.000 | Power=1 ×1.000 | Paper ×1.000 |

| input | 0.010771 | 3.463703 | 5.122 |

| vae | 0.010298 | 3.162535 | 3.871 |

| sonicmaster | 0.001394 | 0.228177 | 1.506 |

Code upstream đưa |STFT|² vào spectral_flatness nhưng giữ power=2 mặc định. Diagnostic giữ mọi bước khác và đặt power=1 cho phổ công suất. FFT=2048, hop=512, mono bằng trung bình hai kênh; không thay benchmark gốc. Điểm upstream tính lại khớp điểm Kaggle đã lưu với rtol=10<super>-6</super> và atol=10<super>-9</super>; hash audio đều đạt.

PDF gốc, Table 2 trang 7, xác nhận mốc 1,506 và hệ số ×1.000. Phụ lục A.5 trang 18 nêu power spectrogram nhưng không chỉ rõ đối số power. Diagnostic tăng điểm SonicMaster khoảng 164 lần, song vẫn thấp hơn mốc paper khoảng 6,6 lần. Khác quy ước giải thích một phần chênh lệch, chưa giải thích hết; không thể tuyên bố phương pháp vượt paper.

Số mẫu clipping xấu hơn input thay đổi từ 45/211 ở upstream sang 39/211 ở diagnostic. Điều này cho thấy việc xếp failure cũng phụ thuộc quy ước metric. Cần tiếp tục đối chiếu giao thức dữ liệu/tiền xử lý trước khi diễn giải chất lượng. [2,5]


---


4. SC203 v5: độ lặp lại và đa metric

Phần này tóm tắt PDF SC203 v5 tại mốc 05h38, 07/11/2026 [1]. Đây là cohort riêng và giao thức riêng; không gộp điểm hoặc số mẫu với benchmark 1.400 mẫu. Các kết luận “đã kiểm chứng” sau đây là trạng thái được báo cáo trong tài liệu nguồn.

4.1. B0 — độ lặp lại

B0 dùng 32 nguồn validation (16 EQ, 16 Dynamics), mỗi nguồn sáu lượt inference, tổng 192 lượt. Cả 32 nguồn có waveform trùng SHA256 giữa sáu lượt. SD của mean mel-SSIM bằng 0 ở ba lượt seed 0 và ba seed 203/1203/2203, trong độ chính xác đã đo. Đã tính lại độc lập 480 giá trị metric.

Copy-input đạt mel-SSIM 0,910448524, baseline 0,633366443. Blend được định nghĩa input + α × (baseline − input); α=0 là input, α=1 là baseline. Trên common-prefix 1.320.960 samples, α=0,25 đạt 0,839886577; α=0,5 đạt 0,747009193; α=0,75 đạt 0,664484096. Không suy ra tối ưu toàn diện từ SSIM.

Noise floor bằng 0 chỉ là quan sát trên cấu hình/thiết bị này với ba lượt mỗi nhóm. Tài liệu ghi nhận VAE latent mode và khởi tạo latent ngẫu nhiên bị comment, phù hợp với quan sát đổi generation seed không đổi output. Không suy rộng sang mọi cấu hình.

4.2. B1 — 64 nguồn validation EQ/Dynamics

| Metric | Tốt hơn | Baseline | Copy-input | Target |

| mel-SSIM | Tăng | 0,631433 | 0,924205 | 1 |

| MR-STFT | Giảm | 0,886066 | 0,675901 | 0 |

| LSD (dB) | Giảm | 14,5544 | 8,71547 | 0 |

| SI-SDR (dB) | Tăng | 2,3346 | 7,4354 | 100* |

| FAD-CLAP | Giảm | 0,362259 | 0,097028 | ≈ 0 |

| Audiobox PQ | Tăng | 7,78076 | 7,68846 | 7,88311 |

* SI-SDR tự so có giá trị +∞, dùng cap 100 dB để trình bày. FAD là thống kê toàn tập, không phải trung bình FAD từng clip; target gần 0 do sai số số học. PQ(target) là tham chiếu nội bộ.

Copy-input tốt hơn baseline ở năm metric độ giống target; baseline tốt hơn copy-input 0,09230 điểm PQ. Chưa có bằng chứng nghe chủ quan tương ứng. Theo PDF, đã kiểm lại 768 giá trị paired, kiểm hash embedding CLAP/PQ và tính lại aggregate; chưa chạy độc lập local inference CLAP/PQ.


---


5. SC203 v5: quét tham số và held-out

5.1. B2 — thiết kế và khối lượng đã kiểm

Thiết kế random sampling cố định PCG64 seed 7203 gồm 48 cấu hình duy nhất và ba baseline repeat trên 64 nguồn, tổng 3.264 clip. Các factor: steps {5,10,20,50}; guidance {1,2,3,5}; solver {Euler,rk4}; prompt {gốc,rút gọn,rỗng}; segment {10,20,30 giây}.

| Shard | Clip mới đã kiểm | Tích lũy | T4-giờ |

| sweep-a1 | 774 | 774 | 3,767 |

| sweep-a2 | 466 | 1.240 | 3,751 |

| sweep-a3 | 412 | 1.652 | 3,781 |

| sweep-a4 | 531 | 2.183 | 3,759 |

| sweep-a5 | 667 | 2.850 | 3,758 |

Tại mốc PDF, đã kiểm chứng 2.850/3.264 clip (87,3%). Carry của lượt cuối ghi nhận đã xử lý đủ 3.264 clip, nhưng 414 clip còn chờ thu hồi và kiểm chứng. Không coi trạng thái kernel COMPLETE là bằng chứng sweep đã được nghiệm thu đầy đủ.

Shard 5 có 666 clip đạt replay trên CPU Linux; một clip sai khác mel-SSIM được tính lại trên T4 và khớp bốn metric. PDF công nhận đạt tolerance 10<super>-7</super>, nhưng nguyên nhân khác biệt số học giữa môi trường chưa được phân lập. Ngân sách đã hạch toán là 21,647/36 T4-giờ; chi phí session cuối còn chờ, nên 14,353 giờ không phải ngân sách còn lại cuối cùng.

5.2. B3 — chưa có kết quả held-out

Đã chuẩn bị 128 nguồn official test mới thuộc EQ/Dynamics và kiểm disjoint trong phạm vi lịch sử SC203 được báo cáo. Sau CP3 dự kiến chọn tối đa hai cấu hình, chấm baseline và cấu hình đã khóa với ba seed/nguồn. Chưa có kết quả hiệu quả held-out tại mốc chốt PDF; chưa công bố cấu hình tối ưu hay parameter importance.

Ngưỡng đăng ký gồm SSIM +0,02; MR-STFT −0,05; LSD −0,5 dB; SI-SDR +1 dB; FAD giảm max(0,01; 10% baseline); PQ +0,10. Kết luận yêu cầu vượt ngưỡng thực tiễn và noise, CI đúng hướng, Holm p&lt;0,05 cùng ràng buộc trade-off.

5.3. Những khác biệt không được bỏ qua

SC203 dùng common-prefix, transformer FP16, T5/VAE FP32 và TF32 tắt. Với segment 10/20 giây, zero-pad tới 30 giây rồi hard-concat là adaptation đã ghi nhận. Nhánh 1.400 mẫu dùng float32, TF32 bật và chiều dài upstream. Vì vậy không quy mọi khác biệt giữa hai nhánh cho mô hình; phải kiểm exposure liên dự án trước mọi tập xác nhận nghiên cứu mới.


---


6. Kết luận và công việc tiếp theo

6.1. Những gì đã làm được

Đã xây dựng và vận hành được chuỗi inference–evaluation trên Kaggle, thu đủ benchmark 1.400 mẫu và kiểm tra tính nhất quán của dữ liệu, cấu hình và điểm. Đã có kết quả đa thuộc tính, danh sách ứng viên failure, bộ nghe và diagnostic clipping tái chạy được. Nhánh SC203 bổ sung bằng chứng về độ lặp lại và sự đánh đổi giữa độ giống target với điểm chất lượng học được.

Kết luận được hỗ trợ là: SonicMaster giảm sai số thuộc tính trung bình trên tập đã chạy, nhưng không cải thiện đồng đều mọi mẫu hoặc mọi metric. Việc cải thiện thuộc tính và việc bảo toàn âm thanh cần đánh giá riêng. Chênh clipping chưa được giải thích đầy đủ; chưa đủ cơ sở tuyên bố reproduce hoàn chỉnh các bảng paper hoặc cải thiện chất lượng cảm nhận.

6.2. Các bước ưu tiên

| Ưu tiên | Công việc | Điều kiện hoàn tất |

| 1 | Làm rõ phần chênh clipping còn lại | Đối chiếu nhãn/tập, tiền xử lý, cấu hình tác giả; ghi rõ nếu chưa xác định được. |

| 2 | Bổ sung FAD-CLAP và PQ cho 1.400 mẫu | Cùng giao thức và checkpoint scorer; FAD tính trên embedding toàn tập. Không mượn điểm B1. |

| 3 | Nghe và phân tích failure | Phiếu nghe thật, vị trí thời gian, phân biệt lỗi mô hình/pipeline; có ca đối chứng. |

| 4 | Đóng B2 và CP3 của SC203 | Kiểm đủ 414 clip còn lại theo mốc nguồn, cập nhật ngân sách, khóa cấu hình trước B3. |

| 5 | Hoàn thiện báo cáo reproduction | Kết quả, bất định, khác giao thức, dữ liệu còn thiếu và bằng chứng truy vết đầy đủ. |

| Sau đó | Xem xét cải tiến VRG | Chỉ khi failure analysis ủng hộ giả thuyết; chưa triển khai trong báo cáo này. |

6.3. Hướng phát triển đang dự kiến

Đã lập kế hoạch VAE Residual Gating: thử bù phần chênh input − VAE roundtrip, trước hết bằng hệ số cố định rồi mới cân nhắc mặt nạ học. Residual có thể chứa cả chi tiết hữu ích lẫn degradation; đây là giả thuyết chưa kiểm chứng. Không train trên test reproduction và không chọn cấu hình bằng held-out đã mở. Theo quyết định hiện tại, VRG được hoãn; chưa có kết quả nâng cấp để báo cáo.

Bộ nghe hiện tại là nghe khám phá có hướng dẫn, không thay thế một thử nghiệm nghe mù. Không tự gán các hiện tượng như âm robot hoặc mất nhạc cụ nếu chưa có người nghe xác nhận.


---


7. Nguồn số liệu và khả năng tái kiểm

Báo cáo tổng hợp từ tài liệu và artifact có tại thời điểm biên soạn. Không suy diễn tiến độ SC203 sau mốc chốt PDF. Các đường dẫn nhánh A bên dưới tính từ thư mục làm việc của dự án; không chứa thông tin xác thực.

[1] Báo cáo SC203 v5

SC203_v5_bao_cao.pdf, người thực hiện cung cấp; 6 trang; chốt 05h38, 07/11/2026. B0: trang 2; B1: trang 3; B2: trang 4; held-out/ngưỡng: trang 5; danh mục receipt: trang 6.

[2] Paper SonicMaster

melechovsky26a.pdf. Table 1/2: trang 7; Table 3: trang 8; metric clipping: A.5 trang 18. Repository: https://github.com/AMAAI-Lab/SonicMaster .

[3] Benchmark và cấu hình

outputs/sonicmaster_audit_v2/run1400/manifest1400.csv; state.json; collected/shard1…shard4/batch350/run_lock.json và scores/provenance.json.

[4] Điểm và kiểm chứng nhánh A

outputs/sonicmaster_audit_v2/run1400/evaluation/: per_sample.csv, paired_changes.csv, summary.csv, validation.json, evidence_lock.json, failure_candidates.csv.

[5] Diagnostic clipping

Trong evaluation/clipping211/: summary.json, per_sample.csv, paper_evidence.json, comparison_statistics.json và receipt/hash từng mẫu. Script: run1400/diagnose_clipping211.py.

[6] Nghe và nhật ký

evaluation/LISTEN.html; listening_audio_receipts.json; SONICMASTER_DIARY.md. Bộ nghe gồm 8 mẫu, chưa có phiếu đánh giá chủ quan.

[7] Kế hoạch nghiên cứu tiếp

outputs/SonicMaster-Residual-Upgrade-Plan.md. Trạng thái planned, chưa thực thi VRG.

Mốc kỹ thuật nhánh benchmark

Code commit: ab32b2d456e461b6bbc188d666caf2eabefa2f31<br/>SonicMaster revision: 93c7fa73c4dd446c9e656af34524b5f6a416afda<br/>VAE revision: f21265c1e2710b3bd2386596943f0007f55f802e<br/>T5 revision: 0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a

Mỗi kết luận phải được đọc cùng phạm vi dữ liệu và giới hạn kiểm chứng. Báo cáo này ghi nhận tiến độ thực nghiệm, không tuyên bố hoàn tất mọi thí nghiệm của paper, mọi phase SC203 hoặc một phương pháp nâng cấp mới.