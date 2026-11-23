# Tổng hợp observation từ kết quả reproduce SonicMaster

## Phạm vi

Thí nghiệm sử dụng checkpoint công bố để chạy inference/evaluation trên 1.400 mẫu từ 200 nguồn, mỗi nguồn có 7 variants: 800 mẫu một tác động, 400 mẫu hai tác động và 200 mẫu ba tác động. Đã kiểm hash 5.600 WAV thuộc bốn nhánh clean, input, VAE và SonicMaster. Chưa tái huấn luyện mô hình. Paper đánh giá 7.000 mẫu, nên đây là reproduction trên tập con. SC203 và VRG không được gộp vào kết quả này.

Tài liệu tổng hợp quan sát từ số liệu và lời nhận xét của người dùng; không phân loại failure cases. Các giải thích chưa được kiểm chứng được ghi dưới dạng giới hạn hoặc giả thuyết.

## 1. Table 1: cân bằng phổ EQ

Cả 10 thuộc tính EQ đều giảm sai số trung bình so với input. Điều này cho thấy các đặc trưng phổ được đo tiến gần clean hơn, không đồng nghĩa mọi khía cạnh cảm nhận đều cải thiện. Các cột dưới là sai số, thấp hơn tốt hơn.

| Thuộc tính | Input | VAE | SonicMaster | Paper SonicMaster |
|---|---:|---:|---:|---:|
| clarity | 0.027292 | 0.028132 | 0.011638 | 0.011400 |
| boom | 0.331241 | 0.332632 | 0.082983 | 0.083400 |
| airy | 0.010316 | 0.010659 | 0.003239 | 0.001900 |
| bright | 0.013895 | 0.014584 | 0.005179 | 0.005900 |
| dark | 0.089502 | 0.073664 | 0.008598 | 0.005800 |
| mud | 0.485108 | 0.506126 | 0.048983 | 0.038800 |
| warm | 0.447475 | 0.458447 | 0.068350 | 0.061700 |
| vocal | 0.245864 | 0.245832 | 0.057523 | 0.057600 |
| mic | 0.290474 | 0.294258 | 0.022640 | 0.008800 |
| xband | 0.182550 | 0.186909 | 0.041803 | 0.035800 |

Clarity, boom, bright và vocal có giá trị tương đối gần mốc paper về mặt số học. Mic lệch rõ; airy, dark và mud cũng cao hơn mốc paper. Mức độ gần paper không đồng đều giữa các thuộc tính. Khác tập dữ liệu và số mẫu từng thuộc tính là yếu tố cần xét, chưa phải nguyên nhân đã chứng minh. Không dùng giá trị thấp hơn trên tập con để tuyên bố vượt paper.

## 2. Table 2: reverb, dynamics, amplitude và stereo

Theo evaluator upstream, cả 9 thuộc tính đều giảm sai số trung bình so với input. Comp và punch gần mốc paper; reverb và volume còn cao hơn mốc paper. Clipping có giới hạn so sánh riêng ở mục 3.

| Thuộc tính | Input | VAE | SonicMaster | Paper SonicMaster |
|---|---:|---:|---:|---:|
| small | 0.507961 | 0.523836 | 0.374663 | 0.366300 |
| big | 0.517864 | 0.537329 | 0.406038 | 0.372600 |
| mix | 0.588043 | 0.617338 | 0.414374 | 0.393500 |
| real | 0.535776 | 0.562160 | 0.354398 | 0.310900 |
| comp | 0.046009 | 0.045928 | 0.019328 | 0.019300 |
| punch | 0.142051 | 0.122526 | 0.088993 | 0.087100 |
| clip ×1.000 (*) | 0.010771 | 0.010298 | 0.001394 | 1.506000 |
| volume | 0.170263 | 0.170020 | 0.057074 | 0.046800 |
| stereo | 0.389798 | 0.389312 | 0.098102 | 0.105800 |

## 3. Clipping: chênh lệch chưa giải thích đầy đủ

Trên 211 mẫu có nhãn clipping, SonicMaster đạt upstream ×1.000 = 0,001394; diagnostic đổi sang power=1 đạt 0,228177; paper báo 1,506. Thay quy ước làm số tăng khoảng 164 lần nhưng vẫn thấp hơn mốc paper khoảng 6,6 lần.

Đã kiểm hash audio và tính lại điểm upstream. Kết quả cho thấy spectral flatness nhạy với quy ước lũy thừa; chưa chứng minh paper dùng power=1 và chưa giải thích hết chênh lệch. Diagnostic không thay điểm benchmark. Không diễn giải clipping thấp hơn là chất lượng tốt hơn paper.

## 4. Table 3: đánh giá tổng thể

FAD/KL thấp hơn tốt hơn; SSIM/PQ cao hơn tốt hơn. KL nhân 1.000. FAD tính trên toàn bộ embedding của từng nhóm, không lấy trung bình FAD các shard.

| Metric | Input | VAE | SonicMaster | Paper SonicMaster |
|---|---:|---:|---:|---:|
| FAD | 0.150533 | 0.246570 | 0.109114 | 0.073000 |
| KL ×1.000 | 5.192990 | 5.317126 | 0.886757 | 0.888000 |
| SSIM | 0.779314 | 0.547511 | 0.611359 | 0.609000 |
| PQ | 7.021603 | 6.873516 | 7.693823 | 7.705000 |

- FAD giảm so với input: phân bố đặc trưng CLAP của output gần clean hơn theo phép đo này.
- KL giảm mạnh: biểu diễn phổ trung bình gần clean hơn theo evaluator.
- PQ tăng: Audiobox dự đoán chất lượng output cao hơn input; đây chưa phải kết luận của nhiều người nghe.
- SSIM thấp hơn input nhưng cao hơn VAE: cấu trúc log-mel được cải thiện so với VAE roundtrip, chưa đạt mức tương đồng của input.

KL, SSIM và PQ khá gần số paper về mặt số học; chưa có kiểm định tương đương thống kê. FAD All còn cao hơn paper; input và VAE cũng có FAD cao hơn mốc paper. Cỡ tập, thành phần nguồn và giao thức embedding là yếu tố cần kiểm soát; chưa xác định đóng góp của từng yếu tố.

## 5. Khác biệt giữa nhóm một tác động và nhiều tác động

| Nhóm | FAD input | FAD SonicMaster |
|---|---:|---:|
| single | 0.100944 | 0.104824 |
| double_triple | 0.242771 | 0.123558 |
| all | 0.150533 | 0.109114 |

Nhóm single có FAD output hơi cao hơn input; nhóm double+triple có FAD giảm rõ. Paper cũng ghi nhận FAD single tăng từ input 0,061 lên SonicMaster 0,069. Đây là xu hướng được quan sát lại. Chưa suy ra quy luật “càng nhiều tác động càng xử lý hiệu quả” vì thành phần và mức độ tác động giữa các nhóm khác nhau.

## 6. Ảnh hưởng của VAE roundtrip

So với input, VAE có FAD/KL cao hơn, SSIM/PQ thấp hơn. SonicMaster cải thiện cả bốn chỉ số so với VAE. Như vậy encode/decode đã làm thay đổi audio và bước xử lý tiếp theo cải thiện các chỉ số này so với VAE roundtrip. Chưa đủ bằng chứng quy mọi thay đổi nghe được cho riêng VAE.

## 7. Mức cải thiện không đồng đều giữa các mẫu

Dù cả 19 thuộc tính giảm sai số trung bình, có 256/2.200 cặp mẫu–thuộc tính tăng sai số, thuộc 234/1.400 mẫu (ngưỡng delta > 0). Ngoài ra 1.175/1.400 mẫu có SSIM output thấp hơn input. Đây là thống kê metric, không được chuyển thành tỷ lệ chất lượng nghe kém. Các variants cùng nguồn có tương quan, không phải 1.400 nguồn độc lập.

## 8. Nhận xét nghe của người dùng

Đã có nhận xét 7/8 mẫu được chọn có chủ đích. Đây là nghe khám phá có hướng dẫn từ một người nghe, chưa nghe mù/cân bằng âm lượng, không đại diện ngẫu nhiên cho toàn tập.

| Mẫu | Nhận xét người dùng |
|---|---|
| 1283374_deg2 | Đã tăng âm lượng lên gần clean; giọng hát gãy, vỡ, không mượt như clean. VAE rất nhỏ, hầu như không nghe thấy. |
| 792131_deg2 | SonicMaster nhỏ hơn input và VAE nên khó nghe rõ; phần nhạc ngân như kèn mượt hơn input khá nhiều. |
| 1577531_deg1 | SonicMaster rất gần clean, nhưng tiếng gõ ở clean mạnh và rõ hơn. |
| 1909745_deg6 | SonicMaster cảm giác vẫn khá vang so với clean. |
| 1640646_deg6 | Âm gõ ở clean thanh và mượt hơn; SonicMaster còn rè và sạn. |
| 1811880_deg5 | Khá sát clean. |
| 1115233_deg5 | Khá giống clean, không nghe thấy sự khác biệt. |

Hai quan sát về quan hệ giữa metric và nghe:

- 1283374_deg2: volume và SSIM cùng cải thiện, nhưng người nghe vẫn nhận ra giọng gãy/vỡ. Metric mục tiêu chưa bao quát hết cảm nhận.
- 1811880_deg5 và 1115233_deg5: SSIM giảm nhưng được nhận xét gần clean. SSIM giảm chưa đủ để kết luận chất lượng nghe giảm.

792131_deg2 và VAE của 1283374_deg2 khó đánh giá do âm lượng thấp. Chưa xác định các khác biệt được nghe thấy là thay đổi mới hay đặc điểm đã có trong input. Metric và nghe trực tiếp cung cấp thông tin bổ sung cho nhau.

## 9. Mức độ tái lập và giới hạn

Đã tái lập các hàng Input, Reconstructed Input/VAE và SonicMaster trong phần snippet của Tables 1–3 trên tập con. Chưa chạy các baseline khác (Mel2Mel+Diffwave, Text2FX, HPSS, WPE), full-song, toàn bộ 7.000 mẫu, training hoặc listening study theo thiết kế paper.

Nhiều xu hướng chính của paper được quan sát lại: cải thiện trung bình các thuộc tính, KL và PQ; SSIM thấp hơn input nhưng cao hơn VAE; FAD nhóm single không cải thiện so với input. Các độ lệch về FAD, một số thuộc tính và clipping vẫn cần trình bày rõ. Chưa coi ba bảng khớp hoàn toàn.

## Nguồn số liệu

- Paper: melechovsky26a.pdf, Tables 1–2 trang 7 và Table 3 trang 8; số đã đối chiếu từ bản trích xuất PDF lưu trong dự án làm việc.
- artifacts/07_run1400/evaluation/summary.csv, per_sample.csv, paper_comparison.csv.
- artifacts/07_run1400/evaluation/fad_pq/: FAD/PQ version 2 và xác minh audio.
- artifacts/07_run1400/evaluation/clipping211/: diagnostic clipping.
- artifacts/07_run1400/evaluation/human_listening_observations.json: lời nhận xét người dùng.
- BAO_CAO_REPRODUCE_TABLES_1_2_3.md: bảng đối chiếu chi tiết.

## Các thí nghiệm mở rộng sau reproduction

Các báo cáo sau dùng cohort và giao thức mở rộng riêng; không thay số liệu reproduction 1.400 mẫu ở trên.

- [Audit clipping](CLIP_AUDIT_REPORT_V2.md): kiểm công thức và phần chênh chưa giải thích.
- [Pilot toàn bài và vùng nối](JOIN_PILOT_REPORT.md): timeline và quyết định không mở rộng theo endpoint đã khóa.
- [Đối chứng âm lượng/xuất WAV](LEVEL_EXPORT_CONTROLS_REPORT.md): kiểm tác động gain và hard clipping lên PQ.
- [Pilot VRG và trộn input](VRG_PILOT_REPORT.md): điểm validation, guard và quyết định sau PQ.


## Kiểm chứng bổ sung sau pilot

<!-- vrg-followup-report-link -->

Đã hoàn tất bootstrap FAD, kiểm SSIM trước/sau xuất WAV và PQ cùng loudness. Xem `D:/SonicMaster_Reproducing/VERIFICATION_FOLLOWUP_REPORT.md`. Đây là phân tích thăm dò sau quan sát; không thay điểm hoặc quyết định pilot gốc. Nghe thật còn pending.
