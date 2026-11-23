# Điều tra clipping và đánh giá full-song

## Clipping trong code

SonicMaster/evaluation/evaluate_control_multiple_degs_mass.py có spectral_flatness và nhánh clip: S=abs(STFT)**2 (FFT2048, hop512), librosa.feature.spectral_flatness(S=S) với power mặc định 2, lấy trung bình rồi absolute difference với clean. Phụ lục A.5 trang18 mô tả power spectrogram và absolute error nhưng không chốt đối số power. Với S đã là power, librosa hướng dẫn power=1. Upstream tương đương lũy thừa 4 trước floor số học. Đây là khác biệt quy ước có bằng chứng, chưa chứng minh bản dùng tạo bảng paper khác code công bố.

211 mẫu: upstream input/VAE/output ×1000 = 0.010770756/0.010298032/0.001393785; power1 = 3.463702933/3.162535069/0.228176709; paper = 5.122/3.871/1.506. Hash và tái tính upstream khớp; đổi power giải thích một phần nhưng không hết. Không phải chỉ thiếu ×1000.

Các yếu tố chưa được loại trừ: tập/nhãn clipping và hidden clipping, vị trí crop, mono so với tính từng kênh, floor amin và silence, chuẩn hóa, mean(abs(frame difference)) so với abs(mean difference), phiên bản code/evaluator của tác giả. Không chọn công thức vì khớp số. Cần so sánh có kiểm soát trên cùng audio; nếu không có môi trường gốc thì phải giữ kết luận chưa xác định nguyên nhân hoàn toàn.

## Metric bổ sung trong nghiên cứu declipping

Survey của Zaviska và cộng sự sử dụng SDR và metric cảm nhận; website báo cáo delta SDR trên phần clipped, PEAQ, PEMO-Q và Rnonlin. Có thể thêm SDR waveform và delta SDR, SI-SDR để tách ảnh hưởng gain, cùng lỗi vùng clipped/unclipped nếu có mask thật từ quá trình tạo dữ liệu. SDR nhạy alignment; SI-SDR bỏ qua scale nên không thay volume. Không suy mask clipping từ ngưỡng |x| gần 1 sau normalize. PEAQ/PEMO-Q cần đúng implementation và điều kiện sử dụng, không giả lập bằng metric khác. Metric bổ sung không thay cột Clip paper.

## Full-song không phải khoảng trống chưa được paper làm

Tables1–3 đã có full-song; code inference_fullsong.py dùng đoạn30s, overlap10s, stride20s, conditioning từ output đoạn trước và crossfade. Appendix A.10 thừa nhận volume không nhất quán theo thời gian. Hướng nghiên cứu có thể khảo sát: metric trung bình che khuất suy giảm cục bộ và chất lượng tại vùng nối. Đây mới là giả thuyết nghiên cứu, chưa khẳng định novelty toàn lĩnh vực.

Phát hiện từ code: pad đoạn cuối đủ30s; lưu output sau crossfade không thấy trim về chiều dài input. Demo cũng có thời lượng output dài hơn clean. Chưa chứng minh toàn bộ chênh thời lượng do padding; không dùng hiện tượng này để giải thích chênh clipping snippet hiện tại.

## Đo đã bắt đầu và giao thức tiếp theo

Đã hoàn tất đo 13 bộ demo full-song tác giả công bố: duration, RMS và spectral-flatness upstream/power1 trên toàn file. Kết quả nằm ở artifacts/fullsong_demo_audit/global_diagnostics.json. Đây là audit demo, không phải inference do nhóm tự chạy; paired_alignment_verified=False. Điểm toàn file với khác thời lượng chỉ là diagnostic.

Lượt full-song có kiểm soát tiếp theo cần: clean/input đầy đủ cùng timeline; khóa cohort trước khi xem điểm; inference upstream và nhánh trim phần padding được ghi riêng; kiểm duration/lag; tính metric toàn bài và cửa sổ thời gian; so vùng crossfade với vùng ngoài crossfade; đo sai lệch loudness so với clean thay vì phạt động lực tự nhiên của nhạc. Báo cáo từng bài và bootstrap theo bài, không coi cửa sổ là nguồn độc lập. FAD cần định nghĩa rõ embedding nhiều cửa sổ và phân phối toàn tập; không gọi lấy10s tự động là đo cả bài. Giữ đoạn không căn gain cho đánh giá volume, ghi riêng mọi phép alignment diagnostic. Chưa khởi chạy GPU full-song mới.

## Nguồn

- https://librosa.org/doc/0.11.0/generated/librosa.feature.spectral_flatness.html
- https://arxiv.org/abs/2007.07663
- https://rajmic.github.io/declipping2020/
- Paper người dùng cung cấp: Tables1–3; Appendix A.5, A.10.
- SonicMaster/inference_fullsong.py; evaluation/evaluate_control_multiple_degs_mass.py.
