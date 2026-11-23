# SonicMaster — Table 1 và Table 2 trên 7.000 mẫu

Kết quả đoạn nhạc (snippet), ba nhánh đã chạy. **Mọi metric là sai số so với bản sạch, càng thấp càng tốt.** N là số mẫu có thuộc tính đó. Δ dương nghĩa là sai số SonicMaster của đợt 7.000 cao hơn con số paper. Metric clip đã nhân 1.000. Cột paper và cột 1.400 là kết quả SonicMaster. Các thuộc tính có thang đo khác nhau, vì vậy so chênh lệch trong từng hàng.

## Table 1 — EQ

| Nhóm | Metric | N | Input | VAE-only | SonicMaster 7.000 | SonicMaster 1.400 | SonicMaster paper | Δ 7.000 − paper |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| EQ | clarity | 346 | 0.027863 | 0.028574 | 0.012278 | 0.011638 | 0.011400 | +0.000878 |
| EQ | boom | 267 | 0.356125 | 0.360645 | 0.086813 | 0.082983 | 0.083400 | +0.003413 |
| EQ | airy | 222 | 0.004907 | 0.005072 | 0.001797 | 0.003239 | 0.001900 | -0.000103 |
| EQ | bright | 361 | 0.013588 | 0.014334 | 0.006059 | 0.005179 | 0.005900 | +0.000159 |
| EQ | dark | 372 | 0.085549 | 0.069040 | 0.006745 | 0.008598 | 0.005800 | +0.000945 |
| EQ | mud | 382 | 0.470565 | 0.491743 | 0.047494 | 0.048983 | 0.038800 | +0.008694 |
| EQ | warm | 414 | 0.443683 | 0.452365 | 0.066286 | 0.068350 | 0.061700 | +0.004586 |
| EQ | vocal | 316 | 0.256407 | 0.255671 | 0.062387 | 0.057523 | 0.057600 | +0.004787 |
| EQ | mic | 556 | 0.260421 | 0.259292 | 0.021345 | 0.022640 | 0.008800 | +0.012545 |
| EQ | xband | 850 | 0.188514 | 0.195428 | 0.039667 | 0.041803 | 0.035800 | +0.003867 |

## Table 2 — Reverb, Dynamics, Amplitude, Stereo

| Nhóm | Metric | N | Input | VAE-only | SonicMaster 7.000 | SonicMaster 1.400 | SonicMaster paper | Δ 7.000 − paper |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Reverb | small | 428 | 0.528631 | 0.545837 | 0.395804 | 0.374663 | 0.366300 | +0.029504 |
| Reverb | big | 389 | 0.515616 | 0.535171 | 0.399754 | 0.406038 | 0.372600 | +0.027154 |
| Reverb | mix | 763 | 0.584043 | 0.613112 | 0.416390 | 0.414374 | 0.393500 | +0.022890 |
| Reverb | real | 1096 | 0.533369 | 0.558906 | 0.353129 | 0.354398 | 0.310900 | +0.042229 |
| Dynamics | comp | 1194 | 0.048837 | 0.048845 | 0.019529 | 0.019328 | 0.019300 | +0.000229 |
| Dynamics | punch | 453 | 0.141887 | 0.123585 | 0.087926 | 0.088993 | 0.087100 | +0.000826 |
| Amplitude | clip | 1143 | 0.402286 | 0.336524 | 0.152606 | 0.001394 | 1.506000 | -1.353394 |
| Amplitude | volume | 404 | 0.183709 | 0.183456 | 0.052928 | 0.057074 | 0.046800 | +0.006128 |
| Stereo | stereo | 1044 | 0.417473 | 0.416940 | 0.109505 | 0.098102 | 0.105800 | +0.003705 |

## Metric lệch và cần giải thích

| Ưu tiên | Metric | Chênh lệch và cách đọc |
|---|---|---|
| 1 | **clip** | 0,152606 so với 1,506000, thấp hơn khoảng **9,9 lần**, nhưng cao hơn khoảng **109 lần** so với 1.400 mẫu. Cả Input và VAE cũng thấp hơn paper rõ rệt. Trên 211 mẫu cũ, đổi spectral flatness từ power=2 sang power=1 làm điểm SonicMaster tăng từ 0,001394 lên 0,228177. Điều này cho thấy phép đo nhạy với quy ước, chưa xác định paper dùng power nào. Cần kiểm cùng ID, độ dài, scale ×1.000, và phân phối điểm theo shard. |
| 2 | **mic (EQ)** | 0,021345 so với 0,008800, cao khoảng **2,43 lần**. Số 1.400 cũng cao (0,022640). Cần so trên cùng mẫu và kiểm từng bước evaluator EQ. Chưa xác định nguyên nhân. |
| 3 | **Reverb: real, small, big, mix** | Cả bốn sai số đều cao hơn paper khoảng **5,8–13,6%**. Lệch lớn nhất trong nhóm là **real** (0,353129 so với 0,310900, N=1.096). Cần so matched IDs và kiểm tham chiếu sạch, độ dài, preprocessing và phiên bản evaluator. |
| 4 | **mud, dark, volume** | Cao hơn paper lần lượt khoảng **22,4%, 16,3%, 13,1%**. Nên ghi trong phần chênh lệch phụ. Với dark, chênh tuyệt đối nhỏ (+0,000945). |

**Gần paper:** comp (0,019529 vs 0,019300), punch (0,087926 vs 0,087100), bright (0,006059 vs 0,005900) và stereo (0,109505 vs 0,105800). Cả 19 hàng đều giảm sai số trung bình so với Input trong đợt 7.000, kể cả các hàng còn lệch paper. Đây là so sánh số báo cáo, chưa có kiểm định chênh lệch trên cặp output của paper.

Nguồn: [Báo cáo 7.000 mẫu](D:/SonicMaster_Reproducing/BAO_CAO_REPRODUCE_7000.md) và [đối chiếu Tables 1–3 trên 1.400 mẫu](D:/SonicMaster_Reproducing/BAO_CAO_REPRODUCE_TABLES_1_2_3.md).
