# Đối chứng âm lượng và xuất WAV — 12 bài

Ngày08/11/2026. Đã hoàn tất96 stream/816 cửa sổ PQ; toàn bộ816 hash PCM khớp đối chiếu độc lập tại máy. Bộ nghe24 trial đã sẵn sàng, chưa có phản hồi.

## Mục đích và cách kiểm soát

Bản SonicMaster cũ lớn hơn clean khoảng4,9LU và có đỉnh vượt biên. Hai thí nghiệm bổ sung kiểm tra việc cắt đỉnh khi xuất và so với chỉnh volume đơn giản. Dùng lại12 bài synthetic amplitude-only, lịch chia0, không chạy lại SonicMaster hoặc sửa kết quả cũ.

**Khối A:** cùng output float s, so 0,25×s với 0,25×clip(s,−1,1). Hai nhánh dùng đúng cùng gain và cùng PCM16 cuối cùng; khác nhau ở thao tác cắt đỉnh được chèn trước gain. Không so bản to/clipped với bản nhỏ/unclipped rồi quy mọi khác biệt cho clipping.

**Khối B:** dùng một gain cố định trên cả bài để đưa từng bản clean/input/VAE/SonicMaster tới−18LUFS, được chọn trước khi chấm. Gain chỉ dựa trên loudness của chính bản đó, không fit với clean. Đây là một điểm vận hành thí nghiệm, không phải chuẩn hay mức tối ưu. Input→−18 là đối chứng chỉ chỉnh volume; SM→−18 là SonicMaster kèm bước gain sau xử lý.

Tất cả48 bản ở khối B đạt mục tiêu mà không chạm giới hạn peak0,98; sai số loudness sau PCM16 tối đa 0.003788LU. Gain làm bằng scalar cả bài, không limiter hay gain theo đoạn. Vì prompt inference gốc không nêu LUFS đích, không dùng phép đo này để kết luận mô hình vi phạm một yêu cầu số cụ thể.

## Kết quả trung bình theo bài

| Nhánh | PQ ↑ | Loudness sau PCM (LUFS) | LRA (LU) | True peak lớn nhất (dBTP) |
|---|---:|---:|---:|---:|
| Clean ×0,25 | 7.790588 | -22.1800 | 5.058 | -11.1 |
| Input ×0,25 | 7.698443 | -34.2209 | 5.050 | -23.2 |
| SonicMaster ×0,25, giữ đỉnh | 7.610553 | -17.2852 | 4.358 | -4.9 |
| SonicMaster cắt đỉnh rồi ×0,25 | 7.588974 | -17.3142 | 4.333 | -10.6 |
| Clean → −18 LUFS | 7.786914 | -17.9993 | 5.050 | -3.3 |
| Input → −18 LUFS (gain-only) | 7.786868 | -18.0000 | 5.050 | -3.3 |
| VAE → −18 LUFS | 7.733335 | -18.0002 | 4.950 | -0.9 |
| SonicMaster → −18 LUFS | 7.596922 | -17.9999 | 4.358 | -5.4 |

## So sánh trực tiếp

Mỗi số chênh được tính trong cùng bài rồi trung bình đều theo12 nguồn. CI95% thăm dò:2.000 bootstrap theo source, seed20261108; không coi816 cửa sổ là816 nguồn độc lập.

| Phép so sánh (trái − phải) | Chênh PQ | CI95% nguồn | Số bài chênh dương |
|---|---:|---|---:|
| SonicMaster ×0,25, giữ đỉnh − SonicMaster cắt đỉnh rồi ×0,25 | +0.021579 | [+0.011961; +0.031775] | 12/12 |
| SonicMaster → −18 LUFS − Input → −18 LUFS (gain-only) | -0.189945 | [-0.269836; -0.116531] | 0/12 |
| SonicMaster → −18 LUFS − VAE → −18 LUFS | -0.136412 | [-0.216682; -0.067619] | 2/12 |
| Input → −18 LUFS (gain-only) − Clean → −18 LUFS | -0.000046 | [-0.000115; +0.000033] | 5/12 |

## Diễn giải

1. Tránh hard clipping ở cùng gain cho chênh PQ trung bình +0.021579. Đây là tác động của thao tác cắt đỉnh trong đối chứng đã khóa. Không suy nó giải thích toàn bộ chênh PQ cũ, không lấy tỷ lệ phần trăm đóng góp giữa các phép so có mức âm lượng khác nhau, và không khẳng định mức chênh này nghe được.

2. Khi đưa về cùng mức đích, SonicMaster vẫn có PQ thấp hơn gain-only: chênh -0.189945. Do đó chênh mức loudness toàn bài và cắt đỉnh lúc xuất không giải thích hết khoảng cách điểm ở pilot này. Chưa quy phần còn lại cho VAE, backbone hay joins; VAE roundtrip cũng có thay đổi và khác waveform clean.

3. Input chỉ bị giảm gain, nên cách chỉ tăng/giảm gain được kỳ vọng giữ gần nguyên nội dung. Kết quả này ủng hộ việc dùng DSP đơn giản làm baseline bắt buộc cho bài toán volume, không chứng minh DSP giải quyết EQ/reverb hoặc thay thế toàn bộ SonicMaster.

4. Điểm PQ cũng thay đổi giữa các mức gain trên cùng nội dung. Không dùng PQ làm phép đo đúng mức âm lượng hay dùng một mình để suy ra chất lượng cảm nhận. Giữ kết quả loudness, peak, PQ và nghe riêng.

5. Đây là kiểm tra sau quan sát trên nguồn demo tác giả đã tiếp xúc; không có tập xác nhận độc lập hoặc kiểm định nghe chính thức. Mục tiêu−18 chỉ được thử một mức, không quét để chọn mức cho điểm đẹp.

## Bộ nghe và dữ liệu

- listening/LISTEN.html:24 nhóm/48 WAV, mỗi nhóm hai bản cùng20 giây, ẩn danh. Có12 nhóm cắt đỉnh và12 nhóm gain-only so SM; vị trí cắt bằng hash trước khi xem PQ.
- loudness toàn bài giống nhau không bảo đảm từng đoạn20 giây cùng loudness. Không gán phản hồi nghe smoke trước đây cho bộ mới.
- PROTOCOL.md và gain_lock.json: quy tắc và48 source hash/gain đã khóa.
- analysis/verification.json:816 hash, coverage và điểm tổng hợp đã kiểm.
- analysis/summary.json, local_measurements.json, local_window_hashes.json: số đầy đủ; ebu_receipts/: logFFmpeg.
- download/level_export_controls/:7 artifact Kaggle; notebook143360498 v1, thời gian script633,88s. Không chạy inference SonicMaster mới.

Đường dẫn tương đối nằm trong artifacts/11_level_export_controls. Không đổi baseline1.400 mẫu hoặc báo cáo joins12 gốc.
