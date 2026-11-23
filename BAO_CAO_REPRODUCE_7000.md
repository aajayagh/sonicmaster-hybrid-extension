# Báo cáo tái lập SonicMaster trên 7.000 mẫu test

Nguyễn Thái Bảo — 25125051  
Ngô Thảo Anh — 25125003  
Lê Quang Nhật — 25125047

Cập nhật: 2026-11-11 (giờ Việt Nam).

## 1. Phạm vi và độ tin cậy của kết quả

- Nhóm chạy inference trên đúng 7.000 biến thể của 1.000 bản nhạc nguồn trong tập test chính thức (7 biến thể/nguồn). Không huấn luyện lại model.
- Cấu hình: SonicMaster commit `ab32b2d`, seed 0, Euler 10 bước, guidance 1, float32, VAE latent mode; WAV stereo 44,1 kHz PCM16. Output đã được kiểm tra hash.
- 1.400 mẫu của đợt trước được giữ nguyên; 14 mẫu benchmark và 5.586 mẫu production còn lại được bổ sung. Không chạy lại inference cho 1.400 mẫu cũ.
- KL và Mel-SSIM: 5.586 mẫu mới chấm trên Kaggle CPU, pilot 7 mẫu khớp bản tính tại máy đến sai khác tối đa `7,41e-14`. 14 mẫu benchmark chấm tại máy.
- PQ và FAD-CLAP: 5.586 mẫu mới chấm trên Kaggle GPU. Đã xác minh 22.000 embedding CLAP và 22.000 bản ghi PQ duy nhất cho toàn bộ 7.000 mẫu. FAD tính một lần trên toàn bộ embedding theo từng nhóm, không lấy trung bình FAD của các shard.
- Đã kiểm phép gộp FAD bằng cách tính lại từ embedding 1.400 mẫu đã lưu: sai khác lớn nhất so với FAD cũ là `1,37e-8` do độ chính xác tính số.
- 19 metric theo loại lỗi (Tables 1–2): dùng đúng bundle hàm evaluator của đợt 1.400 mẫu (`SHA256 ee3f888c…`); 5.586 mẫu mới chấm trên Kaggle CPU, 14 mẫu benchmark tại máy.
- Các biến thể từ cùng một bản nhạc có liên quan với nhau. Khi đọc thống kê từng mẫu, không coi 7.000 mẫu là 7.000 bản nhạc độc lập.

## 2. Table 3 — Chất lượng tổng thể

FAD-CLAP và KL càng thấp càng tốt; Mel-SSIM và PQ càng cao càng tốt. KL đã nhân 1.000. P là số paper, R1.400 là đợt trước, R7.000 là kết quả hiện tại.

| Nhóm | Nhánh | FAD P | FAD R1.400 | FAD R7.000 | KL P | KL R1.400 | KL R7.000 | SSIM P | SSIM R1.400 | SSIM R7.000 | PQ P | PQ R1.400 | PQ R7.000 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| single | input | 0.061000 | 0.100944 | 0.062630 | 3.859000 | 4.006905 | 3.858048 | 0.838000 | 0.842695 | 0.837644 | 7.321000 | 7.339644 | 7.321641 |
| single | vae | 0.139000 | 0.186470 | 0.139654 | 3.990000 | 4.124325 | 3.990726 | 0.574000 | 0.577235 | 0.574712 | 7.172000 | 7.179219 | 7.172391 |
| single | sonicmaster | 0.069000 | 0.104824 | 0.068535 | 0.696000 | 0.686268 | 0.697537 | 0.624000 | 0.627649 | 0.624172 | 7.743000 | 7.731483 | 7.743145 |
| double_triple | input | 0.184000 | 0.242771 | 0.186591 | 6.827000 | 6.774437 | 6.825623 | 0.696000 | 0.694804 | 0.696197 | 6.632000 | 6.597549 | 6.632568 |
| double_triple | vae | 0.290000 | 0.354734 | 0.291128 | 6.984000 | 6.907527 | 6.987914 | 0.507000 | 0.507880 | 0.506993 | 6.501000 | 6.465913 | 6.501728 |
| double_triple | sonicmaster | 0.082000 | 0.123558 | 0.081046 | 1.145000 | 1.154076 | 1.153111 | 0.589000 | 0.589640 | 0.588927 | 7.654000 | 7.643610 | 7.654274 |
| all | input | 0.106000 | 0.150533 | 0.108184 | 5.131000 | 5.192990 | 5.129866 | 0.777000 | 0.779314 | 0.777024 | 7.026000 | 7.021603 | 7.026324 |
| all | vae | 0.196000 | 0.246570 | 0.195719 | 5.273000 | 5.317126 | 5.275235 | 0.546000 | 0.547511 | 0.545690 | 6.885000 | 6.873516 | 6.884964 |
| all | sonicmaster | 0.073000 | 0.109114 | 0.072564 | 0.888000 | 0.886757 | 0.892783 | 0.609000 | 0.611359 | 0.609067 | 7.705000 | 7.693823 | 7.705057 |

Nhánh VAE là input đi qua encode/decode của VAE; paper gọi là “Reconstructed Input”. Các số R1.400 và R7.000 là hai kích thước tập khác nhau, không phải hai lần train model.

### Các điểm thành phần Audiobox trên toàn tập

Đây là các đầu ra bổ sung của bộ dự đoán Audiobox, không phải các cột riêng trong Table 3 của paper. CE (mức dễ thưởng thức), CU (mức hữu dụng của nội dung), PC (độ phức tạp sản xuất) và PQ (chất lượng sản xuất) đều cao hơn khi bộ dự đoán đánh giá tốt hơn.

| Nhánh | CE | CU | PC | PQ |
|---|---:|---:|---:|---:|
| clean | 7.288564 | 7.645451 | 6.166086 | 7.885602 |
| input | 6.390073 | 6.809181 | 5.862936 | 7.026324 |
| vae | 6.365118 | 6.699615 | 5.840799 | 6.884964 |
| sonicmaster | 7.239948 | 7.498381 | 6.197888 | 7.705057 |

## 3. Tables 1–2 — Sai số theo từng loại lỗi

Các giá trị là sai số so với bản sạch; **thấp hơn tốt hơn**. Cột Paper chỉ ghi SonicMaster. Cột R1.400 và R7.000 cũng là SonicMaster. Với `clip`, evaluator nhân 1.000.

| Nhóm | Lỗi | N R7.000 | Input R7.000 | VAE R7.000 | SonicMaster R7.000 | SonicMaster R1.400 | SonicMaster Paper |
|---|---|---:|---:|---:|---:|---:|---:|
| EQ | clarity | 346 | 0.027863 | 0.028574 | 0.012278 | 0.011638 | 0.011400 |
| EQ | boom | 267 | 0.356125 | 0.360645 | 0.086813 | 0.082983 | 0.083400 |
| EQ | airy | 222 | 0.004907 | 0.005072 | 0.001797 | 0.003239 | 0.001900 |
| EQ | bright | 361 | 0.013588 | 0.014334 | 0.006059 | 0.005179 | 0.005900 |
| EQ | dark | 372 | 0.085549 | 0.069040 | 0.006745 | 0.008598 | 0.005800 |
| EQ | mud | 382 | 0.470565 | 0.491743 | 0.047494 | 0.048983 | 0.038800 |
| EQ | warm | 414 | 0.443683 | 0.452365 | 0.066286 | 0.068350 | 0.061700 |
| EQ | vocal | 316 | 0.256407 | 0.255671 | 0.062387 | 0.057523 | 0.057600 |
| EQ | mic | 556 | 0.260421 | 0.259292 | 0.021345 | 0.022640 | 0.008800 |
| EQ | xband | 850 | 0.188514 | 0.195428 | 0.039667 | 0.041803 | 0.035800 |
| Reverb | small | 428 | 0.528631 | 0.545837 | 0.395804 | 0.374663 | 0.366300 |
| Reverb | big | 389 | 0.515616 | 0.535171 | 0.399754 | 0.406038 | 0.372600 |
| Reverb | mix | 763 | 0.584043 | 0.613112 | 0.416390 | 0.414374 | 0.393500 |
| Reverb | real | 1096 | 0.533369 | 0.558906 | 0.353129 | 0.354398 | 0.310900 |
| Dynamics | comp | 1194 | 0.048837 | 0.048845 | 0.019529 | 0.019328 | 0.019300 |
| Dynamics | punch | 453 | 0.141887 | 0.123585 | 0.087926 | 0.088993 | 0.087100 |
| Amplitude | clip | 1143 | 0.402286 | 0.336524 | 0.152606 | 0.001394 | 1.506000 |
| Amplitude | volume | 404 | 0.183709 | 0.183456 | 0.052928 | 0.057074 | 0.046800 |
| Stereo | stereo | 1044 | 0.417473 | 0.416940 | 0.109505 | 0.098102 | 0.105800 |

## 4. Quan sát và những con số cần giải thích

1. **Xu hướng chính được tái lập trên toàn tập:** SonicMaster đạt FAD 0.072564 (paper 0,073), KL×1.000 0.892783 (paper 0,888), SSIM 0.609067 (paper 0,609) và PQ 7.705057 (paper 7,705). Số R7.000 gần paper hơn R1.400 rõ nhất ở FAD, phù hợp với việc FAD ước lượng phân phối từ kích thước và thành phần tập mẫu.
2. **FAD của nhóm lỗi đơn tăng nhẹ:** input 0.062630 → SonicMaster 0.068535. Paper cũng có 0,061 → 0,069. Đây là xu hướng của phép đo trên nhóm này, không đủ để kết luận từng audio bị xử lý tệ hơn.
3. **Mel-SSIM giảm dù các metric khác tăng:** input 0.777024 → SonicMaster 0.609067; VAE còn thấp hơn (0.545690). Có 4,841/7.000 mẫu có KL tốt hơn input nhưng SSIM thấp hơn. Hai metric đo các khía cạnh khác nhau của phổ, không chuyển số SSIM giảm thành tỷ lệ nghe kém.
4. **Mức cải thiện không đều:** KL tốt hơn input ở 5,881/7.000 mẫu; PQ cao hơn ở 5,346/7.000 mẫu; SSIM cao hơn ở 1,063/7.000 mẫu. Đây là so sánh từng cặp với ngưỡng chênh lệch > 0, không phải nhãn failure case nghe được.
5. **VAE tạo ra hao hụt đo được:** PQ input 7.026324, VAE 6.884964, SonicMaster 7.705057; FAD tương ứng 0.108184, 0.195719, 0.072564. Điều này cho thấy encode/decode làm thay đổi audio và bước model cải thiện các metric so với nhánh VAE. Chưa tách riêng nguyên nhân của mọi khác biệt khi nghe.
6. **Clipping cần kiểm tra riêng:** `clip` SonicMaster R7.000 = 0.152606, R1.400 = 0.001394, paper = 1.506000. Giá trị R7.000 tăng rất mạnh so với R1.400 vì 1.400 mẫu cũ vẫn giữ nguyên trong phép gộp; cần xem vì sao cohort bổ sung có điểm lớn hơn. Cả Input, VAE và SonicMaster trên 7.000 mẫu đều thấp hơn paper xấp xỉ 10–13 lần, nên chưa thể quy chênh lệch cho riêng model. Diagnostic trên 211 mẫu cũ đổi spectral-flatness `power=2` sang `power=1` đã tăng số nhưng chưa giải thích hết. Không diễn giải giá trị cực thấp là vượt paper về chất lượng clipping.
7. **Metric theo thuộc tính cũng có ngoại lệ từng mẫu:** 1,211/11,000 cặp mẫu–thuộc tính có sai số SonicMaster cao hơn input, thuộc 1,121/7.000 mẫu. Trung bình từng thuộc tính và từng bản nhạc phải được trình bày tách biệt.
8. **Các thuộc tính lệch số paper nhiều nhất ngoài clipping:** `real` R7.000 0.353129 so với paper 0.310900 (N=1096), `small` R7.000 0.395804 so với paper 0.366300 (N=428), `big` R7.000 0.399754 so với paper 0.372600 (N=389), `mix` R7.000 0.416390 so với paper 0.393500 (N=763), `mic` R7.000 0.021345 so với paper 0.008800 (N=556). Cần xét độ lệch cùng số lượng mẫu, không gán nguyên nhân khi chưa kiểm tra giao thức và audio.

### Mẫu nên nghe/nhìn phổ để kiểm tra

Đây là danh sách ưu tiên kiểm tra dựa trên metric, **chưa phải failure case đã xác nhận bằng nghe**.
- PQ giảm mạnh: `262598_deg5` (-1.198), `1807454_deg7` (-1.175), `2213447_deg7` (-1.048), `2152566_deg7` (-0.986), `703978_deg4` (-0.971).
- SSIM giảm mạnh: `1787052_deg1` (-0.551), `1558368_deg1` (-0.539), `1689821_deg4` (-0.523), `739122_deg1` (-0.497), `1419792_deg3` (-0.492).
- KL tăng mạnh: `1383080_deg1` (+7.323), `2004967_deg7` (+4.451), `1997783_deg4` (+3.982), `2172466_deg1` (+3.031), `1732536_deg2` (+2.632).

## 5. Giới hạn và việc còn lại

- Báo cáo tái lập hàng Input, Reconstructed Input/VAE và SonicMaster cho phần snippet của Tables 1–3. Chưa chạy lại các baseline Mel2Mel+Diffwave, Text2FX, HPSS, WPE; chưa tái lập training hay listening study của paper.
- Metric trung bình và các danh sách ứng viên không thay cho đánh giá nghe mù hoặc xác nhận lỗi trên từng audio. Nên nghe các ứng viên sau khi cân bằng độ lớn và xem waveform/mel spectrogram để chỉ ra thời điểm lỗi.
- FAD-CLAP là khoảng cách giữa hai tập embedding; không thể suy ra FAD của từng audio. PQ là điểm dự đoán của mô hình Audiobox, không phải phiếu của người nghe.
- Tất cả điểm ở đây là trên các đoạn khoảng 30 giây; chưa chứng minh chất lượng xử lý cả bài hát hoặc chất lượng ở ranh giới ghép đoạn.

## 6. File kiểm chứng

- `artifacts/14_run7000/autopilot/heartbeat.json`: trạng thái inference đã xác minh.
- `artifacts/14_run7000/evaluation/kl_ssim/kaggle/`: điểm KL/SSIM từng mẫu, bảng nhóm và biên nhận hoàn tất.
- `artifacts/14_run7000/evaluation/fad_pq/`: CLAP embeddings, PQ records từng shard, FAD/PQ toàn tập và biên nhận hash.
- `artifacts/14_run7000/evaluation/attributes/`: 19 metric theo thuộc tính từng mẫu, bảng nhóm và biên nhận hash.
- `artifacts/14_run7000/evaluation/diagnostics_7000.json`: đếm các thay đổi từng cặp và ID ưu tiên nghe.
- `artifacts/14_run7000/evaluation/candidate_audio_levels_7000.json`: kiểm âm lượng/đỉnh sóng cho một số ứng viên có metric bất thường; chưa phải đánh giá nghe.
- `BAO_CAO_REPRODUCE_TABLES_1_2_3.md`: đối chiếu paper của đợt 1.400 mẫu và diễn giải giao thức.

**Kết luận:** Nhóm đã tái lập gần sát các hàng SonicMaster của Table 3 trên đủ 7.000 mẫu; Tables 1–2 đã được đo trên cùng tập. Các điểm cần làm rõ nhất là sai lệch metric clipping và các mẫu mà PQ/KL cải thiện nhưng SSIM hoặc chi tiết âm thanh vẫn có thể kém. Xác nhận bằng nghe và hình phổ là bước tiếp theo.
