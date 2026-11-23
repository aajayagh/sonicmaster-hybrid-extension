# Báo cáo reproduction SonicMaster: đối chiếu Tables 1–3 và phân tích failure cases

Nguyễn Thái Bảo — 25125051  
Ngô Thảo Anh — 25125003  
Lê Quang Nhật — 25125047

## 1. Mục tiêu và phạm vi

Tái lập inference/evaluation từ checkpoint công bố, đối chiếu kết quả với ba bảng chính của paper và phân tích các trường hợp chưa phục hồi tốt. Báo cáo dùng benchmark 1.400 mẫu từ 200 nguồn (7 variants/nguồn): 800 lỗi đơn, 400 lỗi đôi, 200 lỗi ba. Paper đánh giá 7.000 mẫu; đây là reproduction trên tập con, không phải tái lập training hay toàn bộ thí nghiệm paper. SC203 không được gộp vào kết quả này. VRG chưa triển khai.

Chạy bốn shard inference, mỗi shard 350 mẫu; checkpoint SonicMaster/VAE/T5 công bố; Euler 10 bước, guidance 1, seed 0, float32, VAE latent mode, TF32 bật. WAV stereo 44,1 kHz PCM16, khoảng 30 giây. Đã kiểm hash 5.600 WAV trên Kaggle. Chấm theo evaluator upstream, không chỉnh gain/lag theo clean. Clean chỉ là tham chiếu; VAE là tái tạo input qua encode/decode.

## 2. Cách đọc bảng

Các bảng dưới đặt số paper và số tái lập cạnh nhau, chuyển vị để dễ đọc. P = paper (snippet evaluation), R = reproduction 1.400 mẫu. Không thay số paper bằng số của nhóm. Reconstructed Input của paper tương ứng nhánh VAE. Delta = R SonicMaster − P SonicMaster; đây là chênh số học giữa hai tập, không phải kiểm định vượt paper. Các baseline bên ngoài và full-song được liệt kê là chưa tái lập ở mục 6.

## 3. Table 1 — EQ

Sai số thấp hơn là tốt hơn. Clip đã nhân 1.000 ở cả hai cột. N là số mẫu có thuộc tính tương ứng.

| Metric | N | Input P | Input R | VAE P | VAE R | SonicMaster P | SonicMaster R | Delta SM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clarity | 74 | 0.023800 | 0.027292 | 0.024300 | 0.028132 | 0.011400 | 0.011638 | +0.000238 |
| boom | 56 | 0.360100 | 0.331241 | 0.371700 | 0.332632 | 0.083400 | 0.082983 | -0.000417 |
| airy | 44 | 0.004900 | 0.010316 | 0.005100 | 0.010659 | 0.001900 | 0.003239 | +0.001339 |
| bright | 77 | 0.014300 | 0.013895 | 0.015100 | 0.014584 | 0.005900 | 0.005179 | -0.000721 |
| dark | 77 | 0.089300 | 0.089502 | 0.072800 | 0.073664 | 0.005800 | 0.008598 | +0.002798 |
| mud | 86 | 0.456000 | 0.485108 | 0.474900 | 0.506126 | 0.038800 | 0.048983 | +0.010183 |
| warm | 77 | 0.434500 | 0.447475 | 0.445600 | 0.458447 | 0.061700 | 0.068350 | +0.006650 |
| vocal | 76 | 0.252500 | 0.245864 | 0.252500 | 0.245832 | 0.057600 | 0.057523 | -0.000077 |
| mic | 109 | 0.239300 | 0.290474 | 0.237900 | 0.294258 | 0.008800 | 0.022640 | +0.013840 |
| xband | 167 | 0.178200 | 0.182550 | 0.185400 | 0.186909 | 0.035800 | 0.041803 | +0.006003 |

**Giải thích Table 1:** Sai số trung bình cả 10 thuộc tính EQ giảm so với input. Boom, bright và vocal cho số gần mốc paper; mic lệch rõ (0,022640 so với 0,008800), airy và dark cũng cao hơn mốc paper. Chưa có bằng chứng xác định nguyên nhân riêng từng thuộc tính. Khác tập và số lượng mẫu thuộc tính là các yếu tố cần kiểm soát, không phải nguyên nhân đã chứng minh. Input của hai tập cũng có điểm khác nhau.
## 4. Table 2 — Reverb, Dynamics, Amplitude, Stereo

Sai số thấp hơn là tốt hơn. Clip đã nhân 1.000 ở cả hai cột. N là số mẫu có thuộc tính tương ứng.

| Metric | N | Input P | Input R | VAE P | VAE R | SonicMaster P | SonicMaster R | Delta SM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| small | 80 | 0.445700 | 0.507961 | 0.468600 | 0.523836 | 0.366300 | 0.374663 | +0.008363 |
| big | 75 | 0.424300 | 0.517864 | 0.450700 | 0.537329 | 0.372600 | 0.406038 | +0.033438 |
| mix | 143 | 0.504500 | 0.588043 | 0.543300 | 0.617338 | 0.393500 | 0.414374 | +0.020874 |
| real | 223 | 0.463900 | 0.535776 | 0.490800 | 0.562160 | 0.310900 | 0.354398 | +0.043498 |
| comp | 247 | 0.049600 | 0.046009 | 0.049400 | 0.045928 | 0.019300 | 0.019328 | +0.000028 |
| punch | 85 | 0.120000 | 0.142051 | 0.059000 | 0.122526 | 0.087100 | 0.088993 | +0.001893 |
| clip | 211 | 5.122000 | 0.010771 | 3.871000 | 0.010298 | 1.506000 | 0.001394 | không diễn giải |
| volume | 92 | 0.181300 | 0.170263 | 0.181000 | 0.170020 | 0.046800 | 0.057074 | +0.010274 |
| stereo | 201 | 0.418300 | 0.389798 | 0.418100 | 0.389312 | 0.105800 | 0.098102 | -0.007698 |

**Giải thích Table 2:** Các thuộc tính đều giảm sai số trung bình so với input trên tập đã chạy. Comp và punch gần mốc paper; reverb và volume còn cao hơn. Clipping là ngoại lệ về khả năng so sánh: upstream SonicMaster ×1000 = 0,001394 so với paper 1,506. Diagnostic trên đúng 211 mẫu đổi power=1 cho kết quả 0,228177; tăng khoảng 164 lần nhưng vẫn thấp hơn paper khoảng 6,6 lần. Đây là bằng chứng metric nhạy với quy ước, chưa chứng minh paper dùng power=1 và chưa giải thích hết chênh lệch. Giữ nguyên upstream, không tuyên bố tốt hơn paper về clipping.

## 5. Table 3 — FAD, KL, SSIM, PQ

FAD và KL thấp tốt; SSIM và PQ cao tốt. KL nhân 1.000. FAD tính toàn tập từng nhóm, không trung bình shard. Clean được chấm một lần mỗi nguồn rồi dùng cho các variants; 4.400 bản ghi PQ duy nhất, đầy đủ và hữu hạn.

### single

| Nhánh | FAD P | FAD R | KL P | KL R | SSIM P | SSIM R | PQ P | PQ R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| input | 0.061000 | 0.100944 | 3.859000 | 4.006905 | 0.838000 | 0.842695 | 7.321000 | 7.339644 |
| vae | 0.139000 | 0.186470 | 3.990000 | 4.124325 | 0.574000 | 0.577235 | 7.172000 | 7.179219 |
| sonicmaster | 0.069000 | 0.104824 | 0.696000 | 0.686268 | 0.624000 | 0.627649 | 7.743000 | 7.731483 |

### double_triple

| Nhánh | FAD P | FAD R | KL P | KL R | SSIM P | SSIM R | PQ P | PQ R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| input | 0.184000 | 0.242771 | 6.827000 | 6.774437 | 0.696000 | 0.694804 | 6.632000 | 6.597549 |
| vae | 0.290000 | 0.354734 | 6.984000 | 6.907527 | 0.507000 | 0.507880 | 6.501000 | 6.465913 |
| sonicmaster | 0.082000 | 0.123558 | 1.145000 | 1.154076 | 0.589000 | 0.589640 | 7.654000 | 7.643610 |

### all

| Nhánh | FAD P | FAD R | KL P | KL R | SSIM P | SSIM R | PQ P | PQ R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| input | 0.106000 | 0.150533 | 5.131000 | 5.192990 | 0.777000 | 0.779314 | 7.026000 | 7.021603 |
| vae | 0.196000 | 0.246570 | 5.273000 | 5.317126 | 0.546000 | 0.547511 | 6.885000 | 6.873516 |
| sonicmaster | 0.073000 | 0.109114 | 0.888000 | 0.886757 | 0.609000 | 0.611359 | 7.705000 | 7.693823 |

Clean reference PQ: paper 7,886; reproduction 7,890967.

**Giải thích Table 3:** KL All 0,886757 gần paper 0,888; SSIM All 0,611359 gần 0,609; PQ All 7,693823 gần 7,705. Đây là độ gần số học, chưa có kiểm định tương đương. FAD All 0,109114 cao hơn paper 0,073. Input và VAE cũng có FAD cao hơn paper, nên không quy toàn bộ chênh FAD cho khả năng restoration. FAD phụ thuộc tập mẫu và ước lượng phân phối; khác cỡ tập, thành phần nguồn và giao thức embedding là các yếu tố cần kiểm soát để giải thích, chưa xác định được đóng góp từng yếu tố.

Xu hướng được tái lập: trên toàn tập SonicMaster cải thiện FAD, KL và PQ so với input, trong khi SSIM thấp hơn input và cao hơn VAE. Riêng nhóm single, paper có FAD SonicMaster 0,069 > input 0,061; reproduction cũng có 0,104824 > 0,100944. Vì vậy không nên gọi hiện tượng này là lỗi bất thường chỉ xuất hiện ở reproduction.

## 6. Những hàng của ba bảng chưa tái lập

- Table 1: chưa chạy lại Mel2Mel+Diffwave, Text2FXcos, Text2FXdir; chưa chạy full-song.
- Table 2: chưa chạy lại HPSS 6/12 dB, WPE, Mel2Mel+Diffwave; chưa chạy full-song.
- Table 3: chưa chạy lại Mel2Mel+Diffwave và full-song.
- Chưa dùng toàn bộ 7.000 mẫu, chưa tái lập huấn luyện, ablation hoặc listening study của paper.

Do đó báo cáo tái lập các hàng Input/VAE/SonicMaster trong phần snippet của Tables 1–3 trên tập con. Không tuyên bố hoàn tất toàn bộ ba bảng theo mọi baseline và mọi chế độ.

## 7. Failure cases: metric và nhận xét nghe

Có 256/2.200 cặp mẫu–thuộc tính tăng sai số, thuộc 234/1.400 mẫu. Đây là ứng viên theo metric với ngưỡng delta > 0, không phải tỷ lệ failure nghe được. 1.175/1.400 mẫu có SSIM output thấp hơn input; SSIM không phải chất lượng nghe tổng thể.

Người dùng đã nghe 7/8 mẫu được chọn có chủ đích, có hướng dẫn, chưa nghe mù/cân bằng loudness. Không suy rộng tỷ lệ từ bộ nghe ra cả tập. Nhận xét là của người dùng; chưa đủ để xác định artifact mới do mô hình hay lỗi input còn sót.

## 1283374_deg2

Nhãn: volume. Prompt: Make the audio louder and more powerful.

Người nghe: Đã tăng âm lượng lên gần clean; giọng hát gãy, vỡ, không mượt như clean. VAE rất nhỏ, hầu như không nghe thấy.

| Metric | Input | VAE | SonicMaster |
|---|---:|---:|---:|
| volume | 0.0638106 | 0.063789 | 0.00512323 |
| kl_x1000 | 1.37981 | 2.18978 | 0.935164 |
| ssim | 0.461893 | 0.318964 | 0.708361 |

Volume và SSIM cải thiện nhưng giọng vẫn gãy/vỡ: metric tốt hơn chưa bảo đảm hết lỗi nghe được. Chưa xác định lỗi mới hay còn sót.

## 792131_deg2

Nhãn: clip. Prompt: Reduce the gritty or crushed character.

Người nghe: SonicMaster nhỏ hơn input và VAE nên khó nghe rõ; phần nhạc ngân như kèn mượt hơn input khá nhiều.

| Metric | Input | VAE | SonicMaster |
|---|---:|---:|---:|
| clip | 0.00922119 | 0.0111057 | 0.0104786 |
| kl_x1000 | 0.00611215 | 0.283803 | 0.524817 |
| ssim | 0.987509 | 0.682868 | 0.68588 |

SSIM và clipping upstream xấu hơn nhưng nhạc ngân được nhận xét mượt hơn. Âm lượng thấp và vấn đề quy ước clipping hạn chế kết luận.

## 1577531_deg1

Nhãn: real. Prompt: Bring the sound closer and more focused.

Người nghe: SonicMaster rất gần clean, nhưng tiếng gõ ở clean mạnh và rõ hơn.

| Metric | Input | VAE | SonicMaster |
|---|---:|---:|---:|
| real | 0.397932 | 0.383524 | 0.282515 |
| kl_x1000 | 7.05376 | 6.65603 | 0.784149 |
| ssim | 0.289579 | 0.272647 | 0.56629 |

Reverb real và SSIM cải thiện, còn tiếng gõ kém rõ/mạnh. Có tiến bộ nhưng chưa bảo toàn hoàn toàn chi tiết.

## 1909745_deg6

Nhãn: small;airy. Prompt: Give the mix a breathier, more open feel. Clean this off any echoes!

Người nghe: SonicMaster cảm giác vẫn khá vang so với clean.

| Metric | Input | VAE | SonicMaster |
|---|---:|---:|---:|
| small | 0.406728 | 0.436086 | 0.379011 |
| airy | 0.00217644 | 0.00255924 | 0.00377096 |
| kl_x1000 | 2.30762 | 2.46829 | 1.08416 |
| ssim | 0.764286 | 0.504581 | 0.523091 |

Reverb small cải thiện nhẹ nhưng người nghe vẫn thấy vang; airy xấu hơn, SSIM giảm. Chưa phục hồi hoàn toàn, không đủ bằng chứng vang tăng so với input.

## 1640646_deg6

Nhãn: big;clip. Prompt: Please, reduce the strong echo in this song. Reduce the gritty or crushed character.

Người nghe: Âm gõ ở clean thanh và mượt hơn; SonicMaster còn rè và sạn.

| Metric | Input | VAE | SonicMaster |
|---|---:|---:|---:|
| big | 0.280595 | 0.31214 | 0.385396 |
| clip | 0.00108386 | 0.00093397 | 5.22955e-06 |
| kl_x1000 | 1.41544 | 1.62686 | 0.941564 |
| ssim | 0.398508 | 0.31816 | 0.502922 |

Reverb big xấu hơn, clipping upstream và SSIM cải thiện. Âm gõ còn rè/sạn theo người nghe. Không dùng metric reverb để giải thích nguyên nhân rè.

## 1811880_deg5

Nhãn: mix;bright. Prompt: Remove the church effect, please, lol. Enhance the presence of the upper frequencies.

Người nghe: Khá sát clean.

| Metric | Input | VAE | SonicMaster |
|---|---:|---:|---:|
| mix | 0.701143 | 0.767204 | 0.301136 |
| bright | 0.00111699 | 0.00158201 | 0.00364724 |
| kl_x1000 | 3.67215 | 5.24727 | 1.96956 |
| ssim | 0.591805 | 0.436589 | 0.520992 |

Reverb mix cải thiện, bright xấu hơn và SSIM giảm; người nghe thấy gần clean. Metric xấu đi chưa đủ để gọi là failure nghe được.

## 1115233_deg5

Nhãn: mix;boom. Prompt: Make it boom! Please, reduce the strong echo in this song.

Người nghe: Khá giống clean, không nghe thấy sự khác biệt.

| Metric | Input | VAE | SonicMaster |
|---|---:|---:|---:|
| mix | 0.37756 | 0.364033 | 0.289315 |
| boom | 0.0650502 | 0.0658419 | 0.0680332 |
| kl_x1000 | 2.20373 | 1.70443 | 1.35458 |
| ssim | 0.799802 | 0.631367 | 0.632258 |

Reverb mix cải thiện, boom hơi xấu hơn và SSIM giảm; người nghe không nhận thấy khác biệt. Manifest có hidden clipping mức 2; không suy ra waveform giống nhau.


## 8. Kết luận và công việc còn mở

Đã vận hành inference/evaluation, kiểm hash audio và đối chiếu đầy đủ các metric của ba bảng cho ba nhánh đã chạy. Reproduction giữ được nhiều xu hướng chính của paper; một số giá trị gần mốc công bố, nhưng FAD và một số thuộc tính còn lệch. Clipping chưa thể so sánh trực tiếp. Không dùng các số thấp hơn trên tập con để tuyên bố vượt paper.

Đánh giá nghe bổ sung bằng chứng về giọng gãy/vỡ, reverb còn dư và âm gõ chưa mượt/rõ, đồng thời có ca nghe gần clean. Ví dụ volume được sửa tốt nhưng giọng vẫn chưa mượt cho thấy metric mục tiêu không bao quát hết chất lượng. Các ca SSIM giảm nhưng nghe gần clean cho thấy không thể phân loại failure chỉ bằng SSIM.

Báo cáo hiện đủ để trình bày reproduction trong phạm vi đã định cùng giới hạn. Muốn tuyên bố tái lập đầy đủ Tables 1–3 cần thêm các baseline, full-song và giao thức/tập tương ứng. Cần tiếp tục làm rõ clipping và đánh giá nghe có kiểm soát nếu muốn kết luận mạnh hơn. VRG vẫn hoãn.

## 9. Nguồn và bằng chứng

- Paper người dùng cung cấp: melechovsky26a.pdf; Table 1–2 trang 7, Table 3 trang 8. Số được đối chiếu với bản trích xuất đã lưu của PDF.
- manifest1400.csv: ID nguồn, nhãn, prompt và chia shard.
- evaluation/summary.csv, per_sample.csv: điểm metric và thống kê.
- evaluation/fad_pq/: kết quả Kaggle version 2, embeddings, PQ records, phiên bản thư viện và xác minh audio.
- evaluation/clipping211/: diagnostic clipping 211 mẫu.
- evaluation/human_listening_observations.json và failure_listening_evidence.json: lời nhận xét và liên kết điểm từng mẫu.
- evaluation/LISTEN.html: audio phục vụ đối chiếu. SONICMASTER_DIARY.md: nhật ký.
