# Audit clipping v2 — kết quả đo

Đã chấm lại211 mẫu, kiểm hash đủ4 nhánh, upstream khớp baseline. 12 mẫu smoke xác nhận magnitude/power2 tương đương power-spectrum/power1. Kiểm silence không NaN.

| Nhánh | Upstream ×1000 | Power1 ×1000 | Power1 frame MAE ×1000 | Power1 bỏ silence ×1000 |
|---|---:|---:|---:|---:|
| input | 0.010770756 | 3.463702933 | 3.498316498 | 3.464656949 |
| vae | 0.010298032 | 3.162535069 | 3.210487712 | 3.164555282 |
| sonicmaster | 0.001393785 | 0.228176709 | 0.305248839 | 0.228229118 |

Paper input/VAE/SM =5.122/3.871/1.506. Đổi power làm thay đổi lớn nhưng frame-MAE và bỏ silence theo clean RMS>-60dBFS vẫn không giải thích hết chênh. SM power1 frame-MAE0.305249 vẫn thấp hơn1.506 khoảng4.9 lần. Bỏ silence ở upstream đổi SM đáng kể nhưng power1 gần như không đổi, cho thấy floor/power nhạy với năng lượng thấp. Không chọn biến thể để khớp paper.

Lịch sử GitHub API của evaluator chỉ tìm thấy commit ce16adf60179 (first code), chưa tìm được môi trường/receipt tạo Table2. Chưa xác định được nguyên nhân toàn bộ chênh.

SDR/SI-SDR đã tính trên common-prefix stereo với công thức khóa trong protocol.json. Chưa kiểm lag nên chỉ là diagnostic, chưa kết luận chất lượng waveform. Dữ liệu group nằm summary.json; raw per-sample trong receipts/. Nguồn script scripts/research_pilots/clip_audit_v2.py.

Dữ liệu chi tiết: artifacts/08_clip_audit_v2/.

## Bổ sung kiểm chứng trên 12 mẫu smoke — 08/11/2026

Đã lưu flatness từng frame, environment, 108 cửa sổ kiểm lag và đối chiếu support chung cho cả bốn nhánh trong artifacts/08_clip_audit_v2/supplement/. Không thay bảng 211 mẫu ở trên.

Mean power1 ×1000 trên chính 12 mẫu này, từ full support sang common support của cả bốn nhánh: input 5,735470 → 5,741672; VAE 5,391311 → 5,389713; SonicMaster 0,237699 → 0,237426. Thay đổi nhỏ trên smoke; không suy ra đây là số toàn bộ211 hoặc công thức của tác giả.

Tạo36 cặp synthetic từ12 clean, threshold cố định0,25/0,5/0,75 sau peak normalization, lưu mask thật. Lỗi vùng reliable bằng0 đúng theo phép tạo clipping; vùng clipped đo riêng. Không chạy inference mới cho36 cặp này và không gán mask synthetic cho dữ liệu benchmark gốc. Fixture silence/sine/clipped-sine cho kết quả finite và xác nhận công thức magnitude-power2 tương đương power-spectrum-power1.

Roundtrip PCM16 của các WAV đã lưu cho sai số0 trên12 mẫu. Điều này xác nhận lần đọc/ghi lại không đổi các WAV ấy; không thể loại trừ clipping/quantization đã xảy ra trước lúc lưu WAV gốc. SDR/SI-SDR trên cùng support đã tính lại trong supplement/common_support.csv; vẫn chỉ diagnostic vì kiểm tương quan không tự chứng minh mọi output phục hồi cùng pha hoặc cùng alignment. Không dịch theo clean để tăng điểm.
