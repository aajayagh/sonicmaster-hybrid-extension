# SonicMaster — Tổng hợp công việc nhóm SC203

Tổng hợp từ năm tài liệu được cung cấp. Trạng thái các việc đang chạy lấy theo báo cáo tuần chốt **14:30 ngày 16/11/2026, giờ Việt Nam**. Báo cáo reproduce chốt ngày 11/11, báo cáo SkipChunk ngày 14/11 và báo cáo nghe ngày 15/11. Các kết quả dưới đây là số từ tài liệu nguồn.

## 1. Mỗi tài liệu dùng để đọc gì?

| Tài liệu | Nội dung | Đọc khi muốn biết |
|---|---|---|
| [Reproduce 7.000 mẫu](D:/SonicMaster_Reproducing/BAO_CAO_REPRODUCE_7000.md) | Quy trình tái lập, đối chiếu Tables 1–3, điểm bất thường và nơi lưu bằng chứng | Nhóm đã chạy lại phần nào của paper, khớp đến đâu? Đọc §1–2 trước, rồi §4–5. |
| [Listening 100](D:/SonicMaster_Reproducing/output/pdf/SC203_SonicMaster_Listening_100.pdf) | Cách nghe, nhóm lỗi, ví dụ cụ thể và metric chấm lại trên đúng đoạn có ghi chú | Những lỗi nào nghe thấy thực tế? |
| [SkipChunk Report](C:/Users/ADMIN/Downloads/SonicMaster_SkipChunk_Report.pdf) | Thí nghiệm bật/tắt cờ bỏ chunk dư, bảng theo mốc thời gian, hình waveform và cách kiểm lại | Vì sao 10 giây cuối có thể bị ảnh hưởng? Đọc tr. 2 và 4–6 trước. |
| [Tổng hợp hướng B1](C:/Users/ADMIN/Downloads/TONG_HOP_HUONG_B1.pdf) | Ý tưởng chuyển phần SonicMaster đã sửa lên input gốc, kết quả và tiêu chí dừng | Vì sao SSIM tăng mà phương pháp vẫn thất bại? |
| [Báo cáo tuần 10–16/11](C:/Users/ADMIN/Downloads/Bao_cao_tuan_10-16_11_SC203_gui_TA.pdf) | Tiến độ, thử nghiệm tìm nguyên nhân, ba pilot bị dừng và kế hoạch còn lại | Đọc trước để nắm tiến độ, đọc lại sau khi hiểu bốn tài liệu kia. |

Nếu mới đọc lần đầu, đi theo thứ tự **reproduce → listening → Phase R → SkipChunk → B1 và các pilot**. Phase R nằm trong báo cáo tuần.

### Bốn phiên bản audio cần phân biệt

| Ký hiệu | Hiểu đơn giản |
|---|---|
| **C / Clean / GT** | Bản sạch dùng làm mốc. |
| **X / Input** | Bản có lỗi đưa vào mô hình. |
| **V / VAE-only** | Đưa X qua bước nén rồi giải nén, chưa thực hiện yêu cầu sửa nhạc. |
| **S / SonicMaster** | Nén X, cho bộ sinh chỉnh biểu diễn nén theo prompt, rồi giải nén ra audio. |

Thêm **Cv = VAE(C)** trong Phase R. Đưa ngay bản sạch qua VAE giúp đo riêng mức thay đổi do quá trình nén và giải nén. Một **nguồn** là một bản nhạc gốc. Nhiều **clip/biến thể** có thể đến từ cùng nguồn, nên số clip và số bản nhạc độc lập khác nhau.

## 2. Tái lập paper trên 7.000 mẫu test

**Đã làm.** Chạy đủ 7.000 biến thể từ 1.000 nguồn test chính thức, bảy biến thể mỗi nguồn. Tái sử dụng 1.400 output cũ, bổ sung 14 mẫu benchmark và 5.586 mẫu production. Dùng checkpoint công bố, commit `ab32b2d`, Euler 10 bước, guidance 1, seed 0, FP32 và VAE latent mode. Output WAV stereo 44,1 kHz PCM16 được kiểm hash. KL/SSIM và 19 metric theo lỗi được chấm trên CPU cho phần production, PQ/CLAP trên GPU. FAD được tính từ toàn bộ embedding trong từng nhóm. Báo cáo tuần ghi nhận đã sinh lại 20 mẫu và khớp tuyệt đối với output đã dùng.

| Table 3, toàn tập | Input | VAE-only | SonicMaster | SonicMaster trong paper |
|---|---:|---:|---:|---:|
| FAD-CLAP ↓ | 0,108184 | 0,195719 | **0,072564** | 0,073 |
| KL × 1.000 ↓ | 5,129866 | 5,275235 | **0,892783** | 0,888 |
| Mel-SSIM ↑ | 0,777024 | 0,545690 | **0,609067** | 0,609 |
| Audiobox PQ ↑ | 7,026324 | 6,884964 | **7,705057** | 7,705 |

**Diễn giải.** Table 3 gần số paper. SonicMaster cải thiện KL và PQ so với input, đồng thời mel-SSIM thấp hơn. Có **4.841/7.000** mẫu vừa tốt lên về KL vừa giảm SSIM. Từng mẫu, KL tốt hơn ở 5.881, PQ cao hơn ở 5.346, SSIM cao hơn ở 1.063. FAD của nhóm lỗi đơn tăng nhẹ **0,062630 → 0,068535**, cùng chiều với paper. FAD của nhóm lỗi kép/ba giảm **0,186591 → 0,081046**. Báo cáo cũng lưu CE, CU và PC từ Audiobox để tham khảo bổ sung.

**Tables 1–2.** Đã đo 19 thuộc tính thuộc EQ, reverb, dynamics, biên độ và stereo. Sai số trung bình của SonicMaster thấp hơn input trên từng hàng, nhưng **1.211/11.000 cặp mẫu–thuộc tính** xấu đi, trải trên 1.121 mẫu. Reverb và `mic` còn lệch paper. Điểm clipping của SonicMaster là **0,152606** so với **1,506000** trong paper và **0,001394** của đợt 1.400 mẫu. Input và VAE cũng lệch paper ở clipping. Thử đổi tham số spectral flatness trên 211 mẫu làm điểm tăng nhưng chưa giải thích hết. Danh sách PQ giảm, SSIM giảm và KL tăng mạnh trong báo cáo là danh sách ưu tiên nghe.

**Phạm vi.** Đây là tái lập inference và metric cho các đoạn khoảng 30 giây. Training, các baseline Mel2Mel+DiffWave, Text2FX, HPSS, WPE, listening study của paper và đánh giá cả bài chưa được tái lập ở đây. Các file kiểm chứng và bảng đủ 19 hàng nằm ở §3, §6 của nguồn. [Reproduce 7.000 mẫu](D:/SonicMaster_Reproducing/BAO_CAO_REPRODUCE_7000.md) · [Báo cáo tuần, tr. 1](C:/Users/ADMIN/Downloads/Bao_cao_tuan_10-16_11_SC203_gui_TA.pdf)

## 3. Nghe lại 100 mẫu để biết model có thể làm sai gì

**Phương pháp.** Nghe 100 mẫu đầu trong bộ 190, gồm 49 mẫu nhãn vocal và 51 instrumental. Mỗi mẫu có năm đoạn 6 giây, tổng 500 đoạn. Ghi chú trống được tính là chưa nghe thấy lỗi. Mỗi mẫu được đếm một lần cho mỗi loại lỗi, và có thể thuộc nhiều loại.

| Lỗi ghi nhận | Số mẫu / 100 | Số đoạn / 500 |
|---|---:|---:|
| Âm bị chìm, mất tiếng hoặc kém rõ | 39 | 79 |
| Giọng robot hoặc giống Auto-Tune | 38 | 130 |
| Rè, vỡ hoặc sạn | 17 | 25 |
| Thay đổi âm lượng | 15 | 36 |
| Vang hoặc đổi đuôi âm | 4 | 10 |
| Khác | 6 | 7 |

Có **68/100 mẫu** và **234/500 đoạn** có ít nhất một lỗi được ghi. Nhóm 39 mẫu bao gồm cả vocal hoặc âm khác bị chìm. Ví dụ, hai tiếng gõ không còn nghe rõ ở `2137218_deg2`, đoạn 5, còn tiếng trống kém rõ ở `2032431_deg1`, đoạn 5, được ghi là đã có từ input. Trong kiểm tra tiếp trên 10 đoạn vocal, người nghe nhận ra giọng robot xuất hiện lần đầu ở VAE-only ở các trường hợp đã kiểm.

| Nhóm đoạn | Metric | Input | VAE-only | SonicMaster |
|---|---|---:|---:|---:|
| Robot, 130 đoạn | FAD / KL×1.000 | 0,115 / 6,128 | 0,243 / 6,223 | 0,118 / 1,012 |
| | Mel-SSIM / PQ | 0,872 / 7,100 | 0,566 / 6,730 | 0,611 / 7,446 |
| Âm chìm, 79 đoạn | FAD / KL×1.000 | 0,125 / 5,337 | 0,246 / 5,446 | 0,139 / 1,117 |
| | Mel-SSIM / PQ | 0,850 / 7,323 | 0,562 / 7,011 | 0,610 / 7,586 |

Hai nhóm có 23 đoạn trùng, tổng 186 đoạn riêng từ 60 mẫu. Các nhánh được chấm đúng cùng mốc thời gian. Cả bốn metric phục hồi phần nào từ VAE-only sang SonicMaster. So với input, KL/PQ cải thiện, còn SSIM/FAD của hai nhóm vẫn kém hơn. Điều này giải thích vì sao điểm toàn bản phối tốt lên mà một nhạc cụ cụ thể vẫn khó nghe. Các tỷ lệ chỉ mô tả 100 mẫu đã xem. Báo cáo chưa tách tỷ lệ chỉ riêng nhạc cụ và tỷ lệ lỗi do model tạo thêm. [Listening 100](D:/SonicMaster_Reproducing/output/pdf/SC203_SonicMaster_Listening_100.pdf)

## 4. Phase R — Tìm phần nào của pipeline làm mất chi tiết

**Phương pháp.** So sánh input, bản sạch, hai bản chỉ qua codec và SonicMaster. Pilot 40 nguồn train báo **khoảng 97,5%** mức tụt Mel-SSIM theo phép phân rã của Phase R quy về codec. Mở rộng lên 100 nguồn train, bootstrap 1.000 lần theo nguồn, cho **97,87% [97,20%; 98,46%]** ở codec và **2,13%** ở bộ sinh.

| Nhánh, 100 nguồn train | Mel-SSIM | PQ | KL×1.000 | ViSQOL |
|---|---:|---:|---:|---:|
| Bản sạch C | 1,000 | 7,895 | 0 | 4,732 |
| Bản sạch qua codec Cv | 0,657 | 7,832 | 0,33 | 4,031 |
| Input X | 0,773 | 7,067 | 5,32 | 3,822 |
| Input qua codec V | 0,545 | 6,939 | 5,46 | 3,406 |
| SonicMaster S | 0,614 | 7,728 | 0,70 | 3,854 |

**Diễn giải.** Bản sạch qua codec cũng mất điểm rõ rệt, dù chưa có yêu cầu chỉnh nhạc. SonicMaster tăng PQ và giảm KL so với input, trong khi ViSQOL chỉ nhích từ 3,82 lên 3,85. Con số 97,87% mô tả mức tụt **Mel-SSIM theo thiết kế Phase R**. Kiểm tra nghe vocal ở mục 3 là bằng chứng riêng về điểm xuất hiện lỗi ở các đoạn đã nghe. Trên pilot 40 nguồn, B1 cũng ghi các codec so sánh có Mel-SSIM tái dựng cao hơn Oobleck trong thử nghiệm đó (EnCodec 0,781, DAC 0,764, SAME-L 0,718), chưa phải kết quả thay codec vào toàn pipeline. [Báo cáo tuần, tr. 2](C:/Users/ADMIN/Downloads/Bao_cao_tuan_10-16_11_SC203_gui_TA.pdf) · [Tổng hợp hướng B1, §6](C:/Users/ADMIN/Downloads/TONG_HOP_HUONG_B1.pdf)

## 5. SkipChunk — Phát hiện vấn đề ở đoạn cuối

Với clip 30 giây và overlap 10 giây, code lấy chunk **0–30 giây**, rồi chunk **20–50 giây**, trong đó chunk thứ hai chỉ có 10 giây audio thật và 20 giây đệm 0. Kết quả chunk hai được trộn vào 10 giây cuối. SkipChunk dừng khi chunk hiện tại đã phủ hết tín hiệu.

### Đối chứng bật/tắt trong SonicBoost

Giữ các tối ưu khác bật, chỉ đổi `SKIP_REDUNDANT_CHUNKS`. Hai lượt cùng manifest, ID, prompt, hash input/GT, worker, Euler 10, guidance 1, seed 0 và hai T4 theo cấu hình lưu. Cohort gồm **64 biến thể từ 10 nguồn train**. Notebook đánh giá gốc downmix về mono 22.050 Hz và chấm theo từng vùng.

| Toàn clip | Skip OFF | Skip ON | Số clip tốt hơn khi ON |
|---|---:|---:|---:|
| SI-SDR ↑ | −2,0362 dB | 0,8281 dB | 64/64 |
| RMSE ↓ | 0,11039 | 0,09754 | 52/64 |
| Tương quan waveform ↑ | 0,62178 | 0,71294 | 64/64 |
| LSD ↓ | 10,9624 dB | 10,6379 dB | 58/64 |

SI-SDR trung bình tăng **2,86 dB** cả clip, gần như không đổi ở 0–20 giây và tăng **7,40 dB** ở 20–30 giây. Bootstrap theo 10 nguồn cho mức tăng toàn clip trung bình theo nguồn **3,195 dB [2,027; 4,538]**. Skip ON ngắn hơn **46,26 ms**. Chấm lại ở cùng độ dài trên bốn cặp FLAC của **một nguồn** vẫn cùng chiều (SI-SDR −8,177 → −6,963 dB). Gói chỉ có bốn cặp audio chung để kiểm tra trực tiếp. Báo cáo chưa có số tốc độ ON/OFF từ log, PQ, FAD hay nghe mù cho **chính phép thử 64 clip**.

Phụ lục A0–A5 là lịch sử các cấu hình giữ model trong bộ nhớ, FP16 VAE, AMP, SkipChunk và chia hai GPU. Số 10,31 / 5,08 / 2,65 giây mỗi mẫu là mốc tham khảo từ các cấu hình cũ. Gói kèm notebook, CSV ghép cặp, script chấm và bốn ví dụ FLAC. [SkipChunk Report, tr. 2–9](C:/Users/ADMIN/Downloads/SonicMaster_SkipChunk_Report.pdf)

### Kiểm tra bổ sung với code chính thức

Báo cáo tuần ghi một phép thử riêng trên **12 nguồn train, 24 biến thể**. So code chính thức với bản chỉ chạy một chunk, 20 giây đầu gần như không đổi. Trong 10 giây cuối, code chính thức thấp hơn **0,293 PQ [−0,469; −0,140]** và **0,166 ViSQOL [−0,227; −0,113]**. Bản sửa vị trí ghép tiến gần bản một chunk, nhưng chưa chứng minh tương đương. Báo cáo tuần xác nhận bộ reproduce của nhóm không bị ảnh hưởng bởi lỗi này. Hiệu quả trên nhạc dài liên tục còn mở. [Báo cáo tuần, tr. 1–2](C:/Users/ADMIN/Downloads/Bao_cao_tuan_10-16_11_SC203_gui_TA.pdf)

### Ghi chú thực nghiệm về 5 cờ SpeedUp

Phần này tổng hợp ghi chú của người chạy SpeedUp về cấu hình đã dùng và thời gian quan sát. Có thể dùng benchmark kèm báo cáo SkipChunk để thử từng cờ hoặc cả bộ cờ trong cùng môi trường.

| Cờ | Tiết kiệm ở đâu | Điều đã quan sát |
|---|---|---|
| Giữ model trong bộ nhớ (persistent worker) | Mỗi worker tải model một lần rồi xử lý nhiều mẫu. | Trong các lượt được mô tả, cách cũ tải lại khoảng 4–7 giây/mẫu. Bảng A0–A1 cho metric gần như giống nhau. |
| Chia mẫu cho hai GPU | Hai GPU xử lý các mẫu khác nhau song song. | Khi chỉ một GPU nhận việc, GPU thứ hai bị bỏ trống. Lợi ích phụ thuộc mức cân bằng tải và chi phí đọc, ghi. |
| VAE FP16 | Giảm độ chính xác số ở bước nén và giải nén. | Bảng lịch sử ghi thay đổi nhỏ ở metric. Ghi chú người chạy đề cập đã thử hơn 30 cấu hình trên 64 mẫu. FP16 có dải giá trị và độ chính xác thấp hơn FP32. |
| AMP, giữ T5 FP32 | Dùng độ chính xác hỗn hợp cho các phép tính phù hợp khi inference, giữ bộ mã hóa văn bản T5 ở FP32. | Bảng A0–A5 ghi AMP FP16. Ghi chú người chạy và bảng cũ đều cho thấy điểm thay đổi ít trong các lượt thử. |
| Skip redundant chunks | Dừng khi chunk hiện tại đã phủ đến cuối audio. | Với clip 30 giây, tránh xử lý lại 10 giây cuối trong một chunk có thêm 20 giây đệm 0. Đối chứng ghi SI-SDR toàn clip từ −2,04 lên 0,83 dB, riêng 20–30 giây tăng 7,40 dB. |

Người chạy ghi nhận **3.264 mẫu trong khoảng 48–49 phút** khi bật bộ cờ, tương đương **0,88–0,90 giây/mẫu theo thời gian toàn lượt** nếu thời lượng này bao trọn 3.264 mẫu chạy song song. Đợt thử vài chục đến vài trăm mẫu được báo khoảng **5–20 phút**. Mức tăng tốc **28–34×** và ước tính **khoảng 2× mỗi cờ** là ghi nhận của người chạy. Tài liệu hiện có chưa cung cấp log đối chứng cùng GPU, cùng tập mẫu và cùng cách tính thời gian để kiểm tra các ước tính đó. Không nhân các hệ số từng cờ để dự đoán tốc độ chung.

Việc bỏ chunk dư giảm từ hai lượt xử lý chunk xuống một lượt trên clip 30 giây. Chunk thứ hai có ít audio thật hơn và được trộn dần vào cuối output, phù hợp với vùng suy giảm quan sát được. Với bài dài, cần giữ các chunk có thêm audio chưa xử lý và kiểm tra chất lượng ở chỗ ghép. Người chạy nghe thử vài mẫu và chưa phân biệt rõ với baseline khi bật bộ cờ. Đây là nhận xét nghe sơ bộ. Mốc thời gian trên áp dụng cho inference theo ghi chú. Chi phí và tác động của phần evaluation chưa được khảo sát thêm.

## 6. B1 — Chuyển phần đã sửa về input gốc

**Ý tưởng.** V và S đều qua VAE. Thử lấy tỉ số giữa phổ của S và V để ước lượng phần sửa, rồi áp lên X nhằm giữ chi tiết gốc. B1-0 chuyển biên độ phổ và giữ pha X. B1-C chuyển tỉ số phức. C* làm mượt tỉ số rồi trộn, được chọn trên validation và mang tính khám phá vì trượt ngưỡng compression tại đó. B1-0/B1-C khóa công thức và tiêu chí trước khi chấm test. Sau ba lần lỗi hạ tầng, nhóm viết S2-lite và kiểm lại số baseline trước khi chấm.

| 1.400 clip test / 200 nguồn | Input | SonicMaster | B1-0 | B1-C | C* |
|---|---:|---:|---:|---:|---:|
| Mel-SSIM ↑ | 0,779 | 0,611 | 0,750 | 0,724 | 0,801 |
| PQ ↑ | 7,02 | 7,69 | 7,14 | 7,11 | 7,43 |
| KL×1.000 ↓ | 5,19 | 0,89 | 1,69 | 1,51 | 1,00 |
| MR-STFT ↓ | 1,136 | 0,903 | 0,949 | 0,887 | 0,752 |
| LSD ↓ | 14,74 | 13,98 | 14,33 | 13,31 | 11,60 |
| FAD-CLAP ↓, mô tả theo giao thức B1 | 0,198 | 0,344 | 0,386 | 0,426 | 0,329 |
| Sai số volume ↓ | 0,170 | 0,057 | 0,129 | 0,130 | 0,107 |
| Sai số stereo ↓ | 0,390 | 0,098 | 0,223 | 0,098 | 0,251 |

**Cả ba FAIL.** SSIM tăng có nhất quán theo bootstrap nguồn, nhưng PQ tụt hơn mức cho phép 0,10 và KL vượt 1,05 lần baseline. Các điều kiện còn lại xét việc vượt input và blend 0,5 về fidelity, báo cáo đủ nhóm, giữ chất lượng sửa lỗi mục tiêu và vượt residual λ=1. Một số điều kiện của C* thiếu mẫu số validation cho các loại lỗi vắng mặt, báo NA. Dãy FAD B1 chỉ dùng trong giao thức B1, vì các số không khớp bảng reproduce của đợt 1.400.

**Vì sao có thể thất bại?** Input vốn có SSIM cao. Kéo audio về input tăng độ giống phổ và đồng thời có thể giữ lỗi đầu vào. Ví dụ volume cần tăng khoảng 70 lần nhưng tỉ số bị kẹp ở 30, khiến output vẫn nhỏ hơn bản sạch (RMS 0,056 so với 0,156). Biến đổi tỉ số phổ cũng khó mang theo phần sửa lỗi phức tạp như clipping, nén hoặc reverb. B1 dẫn tới Phase R và hướng thử thích ứng decoder. [Tổng hợp hướng B1, §2–7](C:/Users/ADMIN/Downloads/TONG_HOP_HUONG_B1.pdf)

## 7. Ba pilot tiếp theo và công việc còn mở

Các pilot sau B1 dùng **24 nguồn train dev**, bootstrap 10.000 lần và hiệu chỉnh Bonferroni, với cổng quyết định định trước.

| Pilot | Làm gì | Kết quả và quyết định |
|---|---|---|
| M02 | Bơm lại phần dư bị codec làm mất, có cổng chặn | ViSQOL +0,0009, FAD xấu thêm 0,044 vượt biên 0,02. **KILL**. |
| M03 | Bơm phần dư theo hướng trực giao với phần sửa lỗi | PQ +0,0007, ViSQOL +0,011, CI chứa 0. **KILL**. |
| M05 | Hiệu chỉnh đáp ứng tần số codec | ViSQOL giảm 0,013. **KILL**. |

**Siêu tham số.** Đã lưu output và thời gian cho **62 cấu hình × 64 nguồn train = 3.968 lần đo**. Có 22 cấu hình đổi từng tham số (OFAT) và 40 cấu hình tổ hợp. Đã chấm chất lượng 6/40 tổ hợp, PQ quan sát 7,32–7,80, FAD 0,32–0,50. Chưa chốt thứ hạng tham số, tương tác fANOVA hoặc mức dao động do seed. Báo cáo tuần ghi khoảng 16.600 dòng nhật ký run và 31,2/32 giờ quota đã giữ chỗ cho nghiên cứu siêu tham số.

**Phân loại lỗi tự động cho mentor.** Đã dựng panel **192 mẫu**, cân bằng 6 nhóm lỗi × vocal/instrumental × 16, thêm 16 mẫu lặp và đủ năm phiên bản audio. Đã xong 6/8 phần chấm metric. Kế hoạch là lập bảng lỗi input/codec/bộ sinh, so SonicMaster với VAE-only, và tìm thêm lỗi phổ biến ngoài giọng robot. Bộ cepstrum đạt AUC 1,0 trên lỗi nhân tạo nhưng báo nhầm 29,5% với audio vô hại đã qua codec. Qwen2-Audio cho 0/80 câu trả lời hợp lệ và bị loại. Tại mốc chốt chưa có bộ phát hiện qua kiểm định. Đây là quy trình tiếp theo, song song với ghi chú nghe thủ công đã có ở mục 3.

**Decoder và tiến độ.** Pilot GPU sửa decoder gặp crash, hướng này tạm đóng băng. Tài liệu B1 đề xuất so FGD (decoder nhìn thêm input và có cổng theo lỗi) với DFT (fine-tune decoder không nhìn input), chỉ phát triển trên train trước khi khóa cấu hình và chấm test mới. Tài liệu ghi dự kiến dùng 100 nguồn test còn nguyên cho đánh giá cuối (700 clip đã sinh, chưa chấm ở mốc B1). Báo cáo tuần đặt mốc 17/11 chốt detector, 19/11 chốt phân loại lỗi và số siêu tham số, 20/11 báo cáo mentor, 21–23/11 chọn pilot decoder có huấn luyện. Lỗi DNS/chuyển dữ liệu Kaggle và ngân sách GPU là rủi ro tiến độ. Việc xem xét venue ICML 2027 được ghi trong báo cáo tuần, phụ thuộc kết quả phương pháp. [Báo cáo tuần, tr. 2–4](C:/Users/ADMIN/Downloads/Bao_cao_tuan_10-16_11_SC203_gui_TA.pdf) · [Tổng hợp hướng B1, §6–7](C:/Users/ADMIN/Downloads/TONG_HOP_HUONG_B1.pdf)

## 8. Bảng tra nhanh khi đọc kết quả

| Chỉ số | Đọc thế nào |
|---|---|
| Mel-SSIM ↑ | Độ giống cấu trúc của ảnh phổ mel so với bản sạch. |
| KL ↓ | Độ khác giữa các phân bố phổ theo evaluator. Cột ×1.000 chỉ đổi thang hiển thị. |
| FAD-CLAP ↓ | Khoảng cách giữa hai tập đặc trưng audio. Đọc theo nhóm và cùng giao thức. |
| PQ ↑ / ViSQOL ↑ | PQ là điểm dự đoán chất lượng sản xuất. ViSQOL so audio với bản tham chiếu theo thước đo khách quan. |
| SI-SDR ↑ / tương quan ↑ | Mức bám waveform tham chiếu. SI-SDR bỏ qua khác biệt hệ số gain toàn tín hiệu. |
| RMSE ↓ / LSD ↓ / MR-STFT ↓ | Lần lượt là sai khác biên độ waveform, phổ log và phổ ở nhiều độ phân giải, theo công thức từng thí nghiệm. |
| CI 95% theo nguồn | Khoảng bất định ước lượng bằng lấy mẫu lại theo bài gốc, phù hợp khi một bài có nhiều biến thể. |
| FAIL / KILL / đang chạy | Trượt tiêu chí đánh giá / dừng hướng thử theo tiêu chí / chưa có kết quả cuối. |

## 9. Kết luận từ năm tài liệu

1. **Baseline đã có trên đủ test chính thức.** Bốn điểm chính của SonicMaster gần paper. Clipping và một số thuộc tính còn chênh lệch cần giải thích.
2. **Đã có hai loại bằng chứng về hạn chế.** Nghe thủ công ghi nhận giọng robot và âm bị chìm. Phase R cho thấy codec chiếm phần lớn mức tụt Mel-SSIM theo phép phân rã đã dùng. SkipChunk phát hiện một vấn đề riêng ở phần trộn cuối clip.
3. **Các hướng cải tiến cần giữ cả độ trung thực lẫn khả năng sửa lỗi.** B1 tăng SSIM nhưng trượt PQ và KL. M02/M03/M05 cũng trượt các cổng đã khóa. Báo cáo tuần chưa công nhận phương pháp cải tiến nào thắng.
4. **Việc cần chốt tiếp** là kiểm định detector, hoàn tất chấm siêu tham số, và thử decoder có đối chứng khi có pilot đủ điều kiện. Với SkipChunk, cần đánh giá thêm trên nhạc dài, cùng độ dài chấm và có log nếu muốn tuyên bố tốc độ.


