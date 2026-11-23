# Pilot toàn bài và nối đoạn — đủ 12 bài

Cập nhật 08/11/2026. **Đã hoàn tất inference, phân tích vùng nối và đo PQ toàn bài; đã kiểm chứng dữ liệu. Chưa có đánh giá nghe đầy đủ cho bộ 12 bài.**

## Kết luận dễ hiểu

1. Chưa có bằng chứng vùng nối làm SonicMaster sai âm lượng hơn vùng khác theo tiêu chí đã khóa. Cả 12 bài đều có sai số tại vùng nối thấp hơn vùng đối chứng.
2. Điều đó không có nghĩa output đã đúng: SonicMaster vẫn lớn hơn clean trung bình khoảng 4,90 LU trên cả bài ở lịch chia gốc.
3. PQ toàn bài của SonicMaster thấp hơn input trên 11/12 bài. Đây là observation của pilot giảm âm lượng, chưa xác định do mô hình, mức âm lượng, clamp khi xuất PCM hay phép nối.
4. Theo điều kiện đã đặt, không mở rộng thí nghiệm joins lúc này. Có thể chuyển sang bước VRG khi kiểm timing cache đạt; chưa có hệ số VRG được chọn.

## Dữ liệu và những việc đã làm

- 12 bài, 24 lịch chia (gốc và dịch 5 giây), 300 chunk; không chạy lại hai bài smoke. Độ dài nguồn 108,39–333,43 giây, tức thực tế có bài ngoài khoảng 2–5 phút đề xuất.
- Clean lấy từ demo chính thức của tác giả, chuẩn hóa peak; input giảm gain còn 0,25, prompt “Normalize the audio volume.” Đây là synthetic extension chỉ có amplitude, không phải benchmark full-song của paper, không đại diện EQ/reverb, và không khẳng định nguồn chưa từng được mô hình thấy.
- Giữ chunk 30 giây, overlap 10 giây, stride 20 giây; conditioning đoạn trước, Euler 10 bước, guidance 1, seed 0, float32/TF32 theo config đã lưu.
- Giữ output ghép gốc. Các số dưới đây dùng biến thể đặt chunk đúng mốc input và chuẩn hóa tổng trọng số crossfade; không fit lag/gain theo clean hoặc co giãn tín hiệu.
- Đo loudness/LRA/true peak trên 72 bản toàn bài. Đo Audiobox PQ trên 612 cửa sổ; kiểm lại toàn bộ 612 SHA256 PCM từ chunk tải về, điểm tổng hợp, coverage, độ dài và audio FLOAT dùng phân tích: khớp.
- Xuất 12 biểu đồ và bộ nghe mới 24 nhóm × 4 bản × 20 giây (96 WAV), gồm 1 overlap và 1 control mỗi bài. Chưa tự điền điểm nghe.

## 1. Timeline: vấn đề kỹ thuật đã xác minh

Mỗi chunk input có 1.323.000 frame (30 giây), nhưng decoded có 1.320.960 frame, thiếu 2.040 frame = 46,259 ms. Ghép theo độ dài decoded khiến mốc đầu chunk trôi sớm dần. Ở lịch gốc của 12 bài, độ trôi tại chunk cuối là 0,231–0,740 giây; output còn dài dư 10,253–27,981 giây do cách xử lý đoạn cuối. Cắt đuôi đơn thuần không sửa độ trôi.

Biến thể anchored đặt chunk tại input_start, chia tổng trọng số crossfade, bỏ đúng prefix 5 giây ở lịch dịch và chỉ giữ support input sau khi kiểm coverage. Audio gốc và benchmark 1.400 mẫu không thay đổi.

Kiểm local lag VAE/input: 834/835 cửa sổ đạt correlation ≥0,5 và input RMS >1e-4, khoảng lag −2 đến 393 sample. Có một cửa sổ đáng chú ý ở 870503/lịch0/chunk6/offset2s, lag 393 sample (~8,91 ms), correlation 0,600; một cửa sổ tương quan rất thấp ở 1536323/lịch0/chunk10/offset12s không đạt tin cậy. Vì vậy không mang kết luận ±2 sample của smoke sang cả 12 bài, không công nhận sample-exact alignment và không chấm waveform SDR/SSIM toàn bài như số benchmark.

## 2. Vùng nối so với vùng đối chứng

D = sai số loudness tuyệt đối tại overlap − sai số tại control đã ghép tương đồng. D > 0 nghĩa là vùng nối sai hơn; D < 0 nghĩa là vùng nối gần clean hơn. Cửa sổ 3 giây, hop 0,1 giây; bỏ 35 giây hai đầu; loại vùng clean ≤−60 LUFS. Control cùng bài/lịch, chênh loudness clean ≤2 LU và onset count ≤2, không tái sử dụng control.

Khoảng tin cậy thăm dò từ 2.000 bootstrap theo **nguồn bài**, không coi hàng nghìn cửa sổ chồng lấn hoặc hai lịch chia là mẫu độc lập.

| Nhánh | Lịch | D trung bình (LU) | CI95% theo nguồn | Số bài D > 0 | Chênh log-phổ (dB) |
|---|---|---:|---|---:|---:|
| Input | 0s | -0.000 | [-0.000; +0.000] | 5/12 | -0.029 |
| Input | 5s | +0.000 | [-0.000; +0.000] | 5/12 | +0.019 |
| VAE | 0s | +0.460 | [+0.361; +0.563] | 12/12 | +1.529 |
| VAE | 5s | +0.470 | [+0.374; +0.571] | 12/12 | +1.570 |
| SonicMaster | 0s | -0.448 | [-0.560; -0.351] | 0/12 | -0.991 |
| SonicMaster | 5s | -0.486 | [-0.597; -0.387] | 0/12 | -0.955 |

D của input thực tế chỉ dao động cỡ 1e-6 LU, được xem như 0; không diễn giải dấu của sai số số học. Control ghép được 52,9–98,5% cửa sổ mục tiêu ở lịch gốc và 67,6–91,4% ở lịch dịch; phần không ghép được vẫn được ghi trong matching_coverage.json.

**Quyết định:** SonicMaster không đạt điều kiện mở rộng (≥8/12 D dương, mean ≥0,5 LU và CI thấp >0, kèm bằng chứng bổ sung). Kết quả hiện tại là 0/12 bài D dương ở cả hai lịch. Dừng mở rộng joins theo endpoint này, không cố tìm thêm bài để có lỗi.

VAE có D dương, nhưng VAE chỉ là codec roundtrip, không được yêu cầu tự sửa input giảm âm lượng. Không dùng kết quả đó để kết luận SonicMaster có lỗi nghe được.

## 3. Toàn bài: âm lượng và PQ

LUFS thể hiện mức loudness; LRA là độ biến thiên loudness, không có chiều tốt/xấu tuyệt đối; true peak là ước lượng đỉnh tín hiệu, đơn vị dBTP. PQ là điểm dự đoán chất lượng của Audiobox, cao hơn là tốt theo bộ chấm, không thay đánh giá người nghe.

| Nhánh | Lịch | Loudness (LUFS) | LRA (LU) | True peak trung bình (dBTP) | PQ |
|---|---|---:|---:|---:|---:|
| Clean | — | -10.100 | 5.050 | +0.467 | 7.7739 |
| Input | — | -22.150 | 5.058 | -11.575 | 7.7906 |
| VAE | 0s | -21.942 | 4.950 | -7.783 | 7.7255 |
| SonicMaster | 0s | -5.192 | 4.367 | +5.425 | 7.5681 |
| VAE | 5s | -21.967 | 4.983 | -7.800 | 7.7330 |
| SonicMaster | 5s | -5.183 | 4.392 | +5.200 | 7.5720 |

Bảng là trung bình đều theo 12 bài. Loudness dùng FFmpeg EBU R128 (log làm tròn 0,1); chênh +4,895 LU dùng số pyloudnorm chưa làm tròn trong matched_summary.json. Hai cách đều cho khoảng +4,9 LU.

So với clean, input thấp khoảng 12,04 LU; SonicMaster tăng lên và vượt clean khoảng 4,90 LU. Sai số mức âm lượng tổng thể giảm, nhưng chưa khôi phục đúng mức tham chiếu. Prompt không nêu một LUFS đích cụ thể, nên cần phân biệt khoảng cách với clean và việc tuân thủ một đích âm lượng được chỉ rõ.

Ở lịch gốc, output float SonicMaster có trung bình 0,8214% sample-channel vượt [-1,1], từng bài khoảng 0,104–1,461%. True peak trung bình +5,425 dBTP, lớn nhất +7,2 dBTP. Đây là tín hiệu float trước clamp; không gọi tỷ lệ vượt biên là metric clipping của paper. Clean đã peak-normalize vẫn có true peak dương nhẹ do đỉnh giữa các sample; true peak và sample peak khác nhau.

PQ được tính trên bản clamp rồi xuất PCM16, theo cửa sổ 30 giây liên tiếp phủ hết bài; cửa sổ cuối zero-pad, trọng số là thời lượng âm thanh thật. Sau đó trung bình đều theo bài. Đây là protocol mở rộng, không so trực tiếp với PQ full-song của paper hay PQ trên 1.400 snippet.

| Chênh PQ | Lịch | Mean | CI95% nguồn | Số bài chênh dương |
|---|---|---:|---|---:|
| SonicMaster − Input | 0s | -0.2225 | [-0.3044; -0.1467] | 1/12 |
| SonicMaster − Clean | 0s | -0.2059 | [-0.2688; -0.1479] | 0/12 |
| SonicMaster − Input | 5s | -0.2186 | [-0.3005; -0.1389] | 1/12 |
| SonicMaster − Clean | 5s | -0.2020 | [-0.2676; -0.1440] | 0/12 |

PQ input và clean gần nhau dù loudness khác nhiều: không dùng PQ đơn lẻ để kết luận input không bị giảm âm lượng. PQ của SonicMaster giảm có thể liên quan nhiều yếu tố; chưa có ablation để quy nguyên nhân cho clamp, VAE, backbone hay joins. Hai lịch là cùng nguồn, không phải hai xác nhận độc lập.

## 4. Diagnostic năng lượng crossfade

Tại 20% giữa overlap, năng lượng hỗn hợp SonicMaster thấp hơn trung bình năng lượng có trọng số của hai chunk khoảng 0,444 dB (lịch0), 0,434 dB (lịch5); VAE khoảng 0,407/0,396 dB. Đây là phép đo bổ sung sau quan sát smoke, không phải threshold nghe hay metric paper.

Năng lượng giảm khi trộn hai waveform chưa hoàn toàn giống nhau. Vì output SonicMaster đang lớn hơn clean, một vùng giảm mức có thể lại làm sai số với clean nhỏ hơn. Do đó D âm không đồng nghĩa không có biến thiên theo lịch chia; diagnostic này cũng chưa chứng minh biến thiên đó nghe được hoặc do lỗi riêng của SonicMaster.

## 5. Quan sát theo từng bài, lịch gốc

| Source | D SonicMaster (LU) | Loudness SM − clean (LU) | Vượt biên float (%) | PQ input | PQ SonicMaster |
|---|---:|---:|---:|---:|---:|
| 1115233 | -0.251 | +5.637 | 0.604 | 8.2461 | 8.1228 |
| 1464612 | -0.328 | +4.476 | 1.461 | 8.1983 | 7.9876 |
| 1495501 | -0.425 | +6.279 | 0.196 | 7.9708 | 7.8019 |
| 1536323 | -0.289 | +5.312 | 0.964 | 8.2133 | 7.6547 |
| 1585651 | -0.597 | +3.910 | 1.266 | 7.3103 | 7.0765 |
| 1615369 | -0.302 | +6.007 | 0.451 | 8.1006 | 7.9084 |
| 1745816 | -0.346 | +5.208 | 0.104 | 8.4328 | 8.2739 |
| 1795816 | -0.851 | +4.169 | 0.887 | 7.2246 | 6.9444 |
| 1807462 | -0.288 | +4.690 | 0.984 | 7.7854 | 7.5099 |
| 1829529 | -0.386 | +4.672 | 1.150 | 8.1427 | 7.8165 |
| 264050 | -0.567 | +3.053 | 1.365 | 6.7555 | 6.8360 |
| 870503 | -0.749 | +5.331 | 0.424 | 7.1068 | 6.8842 |

## 6. Nghe và giới hạn

Người dùng đã nói về bộ smoke: “mình chỉ thấy khác nhau âm lượng giữa các bản thôi”. Giữ nguyên lời nhận xét; chưa có mã trial/mức chắc chắn hay xác nhận nghe đủ. Không gán cho 24 nhóm của bộ mới. Chưa có lỗi nối được xác nhận bằng nghe.

Bộ mới: artifacts/09_join_pilot/listening_full12/LISTEN.html. Một gain chung 0,439979 cho cả 96 file tránh clipping khi nghe, giữ tương quan mức âm lượng; không normalize từng bản. Có file mapping riêng. Nghe là thăm dò một người, không phải MOS chính thức. Các ô nhận xét trên HTML không tự lưu.

Cohort chỉ 12 nguồn demo, synthetic amplitude; các phép đo thăm dò không chứng minh tổng quát trên full-song thật hoặc 19 nhiệm vụ của paper. Matching khóa sau khi xem unmatched smoke, vì vậy không gọi là preregistration hoàn toàn độc lập. Không tính FAD trên 12 bài, không lấy trung bình FAD shard. Không có claim novelty: paper đã đánh giá full-song.

## 7. Bước tiếp theo và bằng chứng

Phần tính toán của B đã xong; phản hồi nghe đủ 12 bài vẫn pending. Không mở rộng joins theo tiêu chí đã khóa. VRG cache đã đủ 28 mẫu nhưng 2 mẫu chưa qua timing gate kể cả kiểm dày hơn; giữ nguyên dữ liệu và chưa chạy hệ số, chưa infer252 hoặc mở confirmation20. Chi tiết VRG_PILOT_REPORT.md.

- analysis_full12/pilot_decision.json, summary.json, matched_summary.json, matching_coverage.json: quyết định và endpoint.
- analysis_full12/whole_song_summary.json, whole_song_loudness.json, ebu_receipts/: PQ tổng hợp và loudness toàn bài.
- analysis_full12/pq_verification.json, pq_verification_progress.json: kiểm đủ 612 hash PCM và 72 stream.
- download_pq/join_full12_pq/: output Kaggle PQ; status ghi 532,53 giây trong script, GPU Tesla T4.
- analysis_full12/timeline_audit.json, overlap_energy_summary.json, plots/: timeline và diagnostics.
- listening_smoke/human_observations.json: phản hồi cũ; listening_full12/: bộ mới chưa có điểm nghe.

Các đường dẫn tương đối trong mục này nằm dưới artifacts/09_join_pilot. Bản báo cáo smoke cũ được giữ riêng; báo cáo mới không thay số reproduce 1.400 mẫu.
