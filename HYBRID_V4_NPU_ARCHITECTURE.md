# Kiến trúc NPU-ASLP và Hybrid-v4

**NPU-ASLP học cách tự dựng phổ sạch từ một bản thu lỗi. Hybrid-v4.4 học cách sửa phổ của bản thu lỗi với sự hướng dẫn của bản SonicMaster.** Phần Hybrid-v4 lấy rõ nhất từ NPU-ASLP là cách **chia phổ thành từng dải, chuẩn hóa từng dải nhưng vẫn giữ thông tin mạnh yếu**, rồi cho mạng nhìn theo cả **tần số** lẫn **thời gian**. Hai hệ dùng những cơ chế khác nhau để tạo phổ đầu ra. [Paper NPU-ASLP, mục 2.1](https://arxiv.org/html/2610.15464v1)

Mình lấy **v4.4** làm phiên bản chính để giải thích. Cuối bài có bảng đối chiếu v4.0 đến v4.5 để bạn thấy ý tưởng nào xuất hiện ở đâu.

## 1. NPU-ASLP xử lý một bản nhạc như thế nào?

Đầu vào của NPU-ASLP là **một bản nhạc mono 16 kHz**, kiểu nhạc phát qua loa rồi được điện thoại thu lại. Mạng học từ các cặp bản thu lỗi và bản sạch đã căn thời gian. Khi chạy trên nhạc mới, nó chỉ cần bản thu lỗi. [Paper NPU-ASLP, mục 2 và 3.1](https://arxiv.org/html/2610.15464v1)

| Module | Nó làm gì và vì sao cần? |
|---|---|
| **STFT và chia 80 dải** | Cắt âm thanh thành các khung ngắn, rồi biểu diễn mỗi khung bằng các hệ số tần số. Mỗi hệ số là số phức, chứa cả **biên độ và pha**. Sau đó gom các tần số lân cận thành 80 dải để mạng xử lý theo nhóm. |
| **Scale-aware encoder** | Với mỗi dải tại mỗi khung, mạng nhận **phổ đã chia cho RMS của dải** và **`log(RMS)` được đưa vào riêng**. Ví dụ, cùng một cấu trúc treble xuất hiện ở hai mức âm lượng khác nhau thì sau chuẩn hóa, hình dạng có thể giống nhau. `log(RMS)` cho mạng biết dải nào thực sự yếu hơn. |
| **Backbone TF-MoE** | Sáu khối xử lý theo hai trục. Trục **tần số** liên hệ các dải cùng lúc, chẳng hạn nốt bass và các họa âm của nó. Trục **thời gian** theo dõi diễn biến của một dải, chẳng hạn tiếng trống tắt nhanh hay kéo đuôi do phòng vang. Trong mỗi lớp MoE, bộ định tuyến chọn **một trong 12 expert** cho từng token. |
| **Complex mapping decoder và iSTFT** | Decoder dự đoán trực tiếp phần thực và phần ảo của phổ sạch. Nhờ vậy, nó có thể đưa ra một hệ số khác 0 ngay cả khi hệ số tương ứng trong input gần 0. iSTFT chuyển phổ đã dự đoán về sóng âm. |

[Kiến trúc NPU-ASLP và các hình 1–4](https://arxiv.org/html/2610.15464v1)

**Chi tiết kỹ thuật**

- **STFT.** Cửa sổ Hann 512 mẫu và hop 128 mẫu ở 16 kHz tương ứng 32 ms và 8 ms. Phổ một phía có 257 bin, được chia thành 80 dải kiểu Mel. Mỗi dải ở mỗi khung là một token.
- **Encoder.** Với dải rộng $W_b$ bin, ghép phần thực và ảo thành $\mathbf z\in\mathbb R^{2W_b}$. Tính $r=\sqrt{\|\mathbf z\|_2^2/(2W_b)+\varepsilon}$, rồi chiếu vector $[\mathbf z/r,\log(r+\varepsilon)]$ xuống 32 chiều bằng phép chiếu riêng của dải. Sau encoder, lưới đặc trưng có dạng `[T, 80, 32]`.
- **Backbone.** Mỗi trục dùng module kiểu Conformer với attention bốn head, convolution kernel 31 và hai lớp MoE. Router tính điểm cho 12 expert, chọn expert có xác suất lớn nhất cho token hiện tại và nhân đầu ra của expert với xác suất được chọn. Loss cân bằng tải khuyến khích phân bổ token qua các expert.
- **Decoder.** Mỗi bin đầu ra cần hai số thực để biểu diễn một hệ số phức. Mô hình học dự đoán hệ số này từ đặc trưng đầu vào. Khi suy luận, NPU-ASLP chạy chunk 10 giây với bước 5 giây và ghép các đoạn chồng nhau. [NPU-ASLP, mục 2.1 và 3.1](https://arxiv.org/html/2610.15464v1)

**Encoder** biến audio thành đặc trưng để mạng phân tích. **Backbone** gom thông tin từ các dải và khung lân cận. **Decoder** quyết định phổ đầu ra. Ba vai trò này cũng có trong Hybrid-v4.

## 2. Hybrid-v4.4 xử lý hai bản audio như thế nào?

```text
x = bản thu lỗi ─────┬─→ SonicMaster(x, prompt) ──→ s = bản sửa gợi ý
                    │                               │
                    └──────────────┬────────────────┘
                                   ↓
                         Căn thời gian và mức tín hiệu
                                   ↓
                              Mid/Side + STFT
                          ┌────────┴────────┐
                          ↓                 ↓
                   Encoder → Backbone    Phổ gốc của x và s
                          ↓                 ↓
                        Hệ số lọc, gate, phần dư R
                                   ↓
                           Phổ đầu ra → iSTFT
                                   ↓
                              Bản nhạc stereo
```

Khi train, v4.4 nhận `x` và `s`, rồi loss so đầu ra với **bản clean thật**. Khi suy luận, nó chỉ cần `x` và `s`. Prompt đi vào SonicMaster. V4.4 nhận thông tin về yêu cầu chỉnh sửa gián tiếp qua bản audio `s`.

### Module 1 — Căn thời gian và mức tín hiệu

V4.4 có lúc trộn hai hệ số phức từ `x` và `s`. Nếu cùng một tiếng trống đến sớm hơn ở một bản, pha của chúng cũng lệch. Khi cộng, một số thành phần có thể triệt tiêu nhau. Vì vậy pipeline **ước lượng độ trễ giữa hai waveform rồi dịch `s`**, dùng zero padding ở mép lộ ra.

Sau đó, pipeline đưa hai bản về mức RMS dễ so sánh và vẫn chuyển cho encoder **mức RMS gốc của từng bản cùng chênh lệch gain**. Nhờ vậy, khi SonicMaster làm bài to hơn, encoder vẫn biết mức thay đổi đó dù hai tín hiệu đã được chuẩn hóa.

**Chi tiết kỹ thuật**

- Căn trễ dùng tương quan chéo FFT trên tín hiệu mono lấy trung bình hai kênh, ưu tiên cửa sổ giữa dài tối đa 10 giây. Pipeline tìm đỉnh trong khoảng ±512 mẫu, khoảng ±11,6 ms ở 44,1 kHz. Một độ dịch chung được áp lên cả hai kênh của `s`.
- Gọi $r_x,r_s$ là RMS toàn bài, tính chung hai kênh. Gain căn mức của `x` là $a=\operatorname{clip}(r_s/r_x,0.1,16)$. Sau đó $x_{\rm in}=a\,x/\max(r_s,\varepsilon)$ và $s_{\rm in}=s/\max(r_s,\varepsilon)$. Các đặc trưng toàn bài gồm $\log r_x$, $\log r_s$ và $20\log_{10}(r_s/r_x)/24$ được chặn trong $[-2,2]$. Code dùng epsilon để ổn định phép tính trên tín hiệu nhỏ.

### Module 2 — Mid/Side, STFT và hai đường tín hiệu

V4.4 đổi stereo từ trái/phải sang **Mid/Side**. Mid tỉ lệ với `L + R`, Side tỉ lệ với `L − R`. Nếu hai kênh giống hệt nhau thì Side bằng 0. Biểu diễn này giúp mạng nhìn trực tiếp vào thành phần chung và thành phần khác nhau giữa hai kênh.

STFT biến mỗi kênh thành **1.025 bin tần số**, với cửa sổ 2.048 mẫu và bước 512 mẫu ở 44,1 kHz, tương ứng khoảng **46 ms mỗi cửa sổ và 11,6 ms giữa hai khung**. Từ đây có hai đường:

- **Đường phân tích** dùng phổ đã nén biên độ theo lũy thừa `0,3` để các thành phần nhỏ dễ được mạng quan sát hơn. Pha vẫn được giữ trong các hệ số phức.
- **Đường tạo đầu ra** giữ phổ phức ở thang tuyến tính `Zx` và `Zs`. Decoder dùng trực tiếp hai phổ này để lọc và phối trộn.

Đây là chỗ thực hiện **codec bypass**. Chi tiết từ `x` có đường đi `STFT → chỉnh phổ → iSTFT`. Phần lấy từ `s` vẫn mang những gì SonicMaster đã tạo, kể cả artifact nếu `s` có.

**Chi tiết kỹ thuật**

- Đổi kênh bằng $M=(L+R_{\rm right})/\sqrt 2$ và $S=(L-R_{\rm right})/\sqrt 2$. Ký hiệu $R_{\rm right}$ là kênh phải. Phần dư decoder ở module 5 dùng ký hiệu $R$.
- STFT dùng cửa sổ Hann và `center=True`. Tensor phổ của mỗi bản có dạng `[B, 2, 1025, T]`, trong đó `B` là số mẫu trong batch, `2` là Mid và Side, `T` là số khung.
- Nén biên độ thực hiện bằng $Z_c=Z(|Z|+\varepsilon)^{0.3}/(|Z|+\varepsilon)$. Hệ số nhân là số thực dương, nên giữ góc pha của $Z$. Decoder MF9 nhận $Z_x,Z_s$ trước phép nén này.

### Module 3 — Dual Scale-Aware Band Encoder

V4.4 gom 1.025 bin thành **24 dải có độ rộng khác nhau**. Ở mỗi khung, encoder làm ba việc cho từng dải:

1. Chuẩn hóa phổ của `x` và `s` **riêng từng bản** theo RMS của dải, rồi lấy phần thực và phần ảo của cả Mid lẫn Side.
2. Đưa lại thông tin về **mức của từng bản**, **mức thay đổi giữa hai bản**, **chênh lệch tỷ lệ Side/Mid** và **RMS toàn bài**.
3. Chiếu tất cả thành **64 giá trị đặc trưng**. Mỗi dải có phép chiếu riêng vì số bin trong các dải khác nhau.

Ví dụ, ở đoạn có tiếng cymbal, `x` có thể vẫn giữ hình dạng phổ của cú đánh nhưng năng lượng dải cao quá nhỏ. `s` có thể mạnh hơn ở dải đó. Encoder thấy **cả hình dạng phổ lẫn chênh lệch mức**, rồi chuyển thông tin đó cho backbone. Việc có tăng dải cao hay không được quyết định ở decoder. Đây là phần mở rộng trực tiếp nhất từ **Scale-Aware** của NPU-ASLP sang hai đầu vào.

**Chi tiết kỹ thuật**

- Với dải rộng $W_b$ bin, mỗi bản đóng góp $4W_b$ số thực từ `Mid.real`, `Mid.imag`, `Side.real` và `Side.imag`. Hai bản cho tổng cộng $8W_b$ số.
- Gọi $r_{x,b},r_{s,b}$ là RMS dải tính trên phổ đã nén, chung Mid/Side ở khung hiện tại. Bốn đặc trưng cấp dải là $\log r_{x,b}$, $\log r_{s,b}$, $\log(r_{s,b}/r_{x,b})$ và chênh lệch $\log(\mathrm{RMS}_{\rm Side}/\mathrm{RMS}_{\rm Mid})$ giữa `s` và `x`. Ba đặc trưng toàn bài từ module 1 nâng tổng số đặc trưng bổ sung lên bảy.
- Phép chiếu riêng của mỗi dải dùng `LayerNorm → Linear(8W_b + 7, 64) → SiLU`. Ghép 24 dải cho đầu ra `[B, T, 24, 64]`. [Scale-Aware của NPU-ASLP](https://arxiv.org/html/2610.15464v1)

### Module 4 — Backbone gồm bốn khối xử lý theo hai trục

Mỗi khối làm ba việc theo thứ tự:

| Thành phần | Nó nhìn vào đâu? | Vai trò |
|---|---|---|
| **Cross-band attention** | Cả 24 dải **trong cùng một khung** | Cho dải cao nhìn xuống dải trung và bass để hiểu quan hệ giữa họa âm, transient và cân bằng phổ tại thời điểm đó. |
| **BiGRU theo thời gian** | Các khung trước và sau **trong cùng một dải**, trong phạm vi chunk | Theo dõi một sự kiện qua nhiều khung. Ví dụ, phần năng lượng hiện tại là một nốt mới hay đuôi của tiếng vừa phát. |
| **Dense FFN** | Đặc trưng tại từng vị trí thời gian–dải tần | Biến đổi đặc trưng sau khi hai trục đã trao đổi thông tin. Mọi vị trí dùng chung FFN của khối đó. |

Ý tưởng **nhìn theo hai trục** giống NPU-ASLP. Cách triển khai trong v4.4 là **attention theo dải + BiGRU theo thời gian + Dense FFN**. Trong NPU-ASLP, hai trục dùng các module kiểu Conformer với **Top-1 MoE**.

**Chi tiết kỹ thuật**

- Attention bốn head chuyển tensor thành `[B·T, 24, 64]`. Mỗi chuỗi attention gồm 24 dải của một khung.
- BiGRU đổi thứ tự tensor thành `[B·24, T, 64]` để quét theo thời gian. Mỗi hướng có hidden size 64. Đầu ra hai hướng ghép thành 128 chiều rồi được chiếu về 64.
- Dense FFN có kích thước `64 → 256 → 64` và GELU ở giữa. Mỗi bước dùng LayerNorm trước phép biến đổi và cộng kết quả trở lại đầu vào của bước đó. Sau bốn khối, kích thước vẫn là `[B, T, 24, 64]`. [Backbone NPU-ASLP](https://arxiv.org/html/2610.15464v1)

### Module 5 — Decoder MF9, gate và phần dư `R`

Đây là phần quyết định âm thanh đầu ra của v4.4. Từ đặc trưng đã có ngữ cảnh, decoder sinh ra ba thứ:

$$
Z_{\text{out}} = (1-g)\,F(Z_x) + g\,Z_s + R
$$

| Thành phần | Ý nghĩa trong audio |
|---|---|
| **`F(Zx)` — lọc phổ input** | Bắt đầu từ `Zx`, rồi cộng các hiệu chỉnh lấy từ **khung hiện tại và các khung lân cận**. MF9 có chín vị trí thời gian, trải khoảng **220 ms** từ vị trí xa nhất tới vị trí gần nhất ở phía còn lại. Ở mỗi vị trí, ma trận phức `2×2` cho phép **Mid tác động lên Side và Side tác động lên Mid**, đồng thời chỉnh cả biên độ và pha. |
| **`g` — trộn với SonicMaster** | `g` nằm trong khoảng 0–1 và thay đổi theo **khung, bin tần số và kênh Mid/Side**. `g` gần 0 ưu tiên input đã lọc. `g` gần 1 ưu tiên `s`. Trong công thức, `R` vẫn được cộng sau bước trộn. |
| **`R` — phần dư cộng trực tiếp** | Mạng dự đoán thêm một hệ số phức cho từng kênh. Độ lớn được đặt theo RMS của **cả `x` và `s` trong dải đó**. Vì đây là phép cộng, nó có thể bổ sung vào bin đang gần 0. Nó cũng có thể hiệu chỉnh một hệ số đang có thông qua biên độ và pha của phần được cộng. |

Ví dụ, nếu **Side của `x` gần bằng 0**, decoder có ba đường để tạo Side đầu ra. Nó có thể dùng **Mid → Side** trong bộ lọc, lấy **Side từ SonicMaster** qua `g`, hoặc cộng thêm **`R` ở kênh Side**. Đây là cách nhìn cụ thể nhất về việc ba thành phần phối hợp. `g` của v4.4 điều khiển **tỷ lệ trộn hai phổ**. Router trong NPU-ASLP điều khiển **mạng con nào sẽ xử lý đặc trưng**.

**Chi tiết kỹ thuật**

- **Các đầu dự đoán.** Một MLP dùng chung có kích thước `64 → 128 → 64`. Đầu riêng từng dải sinh $14W_b$ số trên mỗi khung, gồm tám số cho ma trận phức tại khung hiện tại, hai logit gate và bốn số cho phần thực/ảo của phần dư Mid/Side, tính trên mỗi bin.
- **Lọc đa khung.** Tập offset là $\{-18,-12,-8,-5,-3,-2,-1,0,+1\}$. Ở bin $f$, khung $t$, bộ lọc có dạng $F(Z_x)_{f,t}=Z_{x,f,t}+\sum_k\Delta W_{k,f,t}\,Z_{x,f,t+k}$, với $Z_{x,f,t}$ là vector phức hai kênh. Ma trận tại $k=0$ được dự đoán theo bin. Với các offset còn lại, mỗi dải được chia thành bốn dải con và những bin cùng dải con chia sẻ hệ số. Khung nằm ngoài biên được đệm 0. Các thành phần thực và ảo của hệ số hiệu chỉnh dùng `2·tanh`.
- **Gate.** $g=\sigma(g_{\mathrm{raw}}+b_g)$, với $b_g=-1.5$ ở checkpoint v4.4 đã dùng. Có một giá trị cho Mid và một cho Side tại mỗi bin và khung. Phép trộn thực hiện trên số phức, nên kết quả phụ thuộc cả biên độ lẫn quan hệ pha giữa hai phổ.
- **Phần dư.** Mỗi thành phần thực và ảo của phần dư chạy qua `2·tanh` rồi được nhân với $\sqrt{\operatorname{mean}(|Z_x|^2+|Z_s|^2)+10^{-6}}$. Trung bình lấy qua hai kênh và các bin trong dải tại khung đang xét. Khi kiểm tra `R = 0 / 0,5 / 1`, công thức thực tế dùng $\alpha R$ với $\alpha\in\{0,0.5,1\}$.

Nếu bạn thấy class tên `MF7Decoder` trong code, class đó được dùng chung. **V4.4 khởi tạo class này với chín tap MF9 và bật `R`**. Trong thí nghiệm vừa chạy, `R = 0 / 0,5 / 1` là ba hệ số nhân lên **phần dư mà mạng đã dự đoán**.

### Module 6 — Ghép lại waveform và học từ bản clean

Sau decoder, **iSTFT** đổi phổ về waveform Mid/Side, rồi pipeline đổi lại thành trái/phải. Khi bài dài, v4.4 xử lý theo **chunk 8 giây chồng 50%** và ghép bằng cửa sổ Hann để làm mượt chỗ nối. Cuối cùng, mức RMS toàn bài được khớp theo SonicMaster. Vì vậy, độ to tổng thể của v4.4 còn chịu ảnh hưởng trực tiếp từ `s`.

Trong lúc train, **waveform L1** đo sai khác mẫu sóng, **MR-STFT** so phổ ở nhiều độ phân giải, và **complex loss** so cả phần thực lẫn phần ảo. **Mel-KL và log-mel** tạo thêm áp lực để khớp phân bố phổ. **Loss Side/Mid**, bật từ epoch 13 trong pilot v4.4, kéo tỷ lệ độ rộng stereo về gần bản clean. Các loss này giải thích mạng được khuyến khích học gì, còn phép chấm PQ là một bước đánh giá sau train.

**Chi tiết kỹ thuật**

- **Ghép waveform.** iSTFT dùng cùng kích thước FFT, hop và cửa sổ với bước STFT. Sau khi đổi lại sang stereo, mỗi chunk được nhân cửa sổ Hann, cộng ở vị trí tương ứng và chia cho tổng trọng số tại từng mẫu. Pipeline khôi phục thang RMS rồi cân RMS đầu ra với `s` khi cả hai có đủ năng lượng.
- **Loss phổ và waveform.** MR-STFT dùng cỡ FFT 512, 1024 và 2048 trên cả L/R lẫn Mid/Side. Nó kết hợp spectral convergence với L1 trên log-biên độ. Complex loss dùng STFT Mid/Side nén lũy thừa, còn waveform L1 so từng mẫu L/R.
- **Cấu hình pilot.** Loss bắt đầu bằng $L_{\rm MRSTFT}+0.5L_{\rm complex}+0.5L_{\rm wav}+2.5L_{\rm mel}$. $L_{\rm mel}$ dùng KL giữa hai phân bố Mel đã chuẩn hóa theo năng lượng, cộng $0.1$ lần log-mel L1. Từ epoch 13, thêm L1 giữa hai tỷ lệ RMS Side/Mid với trọng số $1.0$. Code có các hạng regularization gắn với gate và bộ lọc ở stage 2. Cấu hình v4.4 lịch sử đặt trọng số hạng gate bằng 0.
- **Train loss và bảng kết quả.** Hàm train tính $D_{\rm KL}(p_{\rm clean}\|p_{\rm out})$ bên trong $L_{\rm mel}$. Bảng chấm Dev dùng giao thức riêng của dự án, với thứ tự `KL(output, clean)`.

## 3. Những gì từ NPU-ASLP đã đi vào Hybrid-v4

| Ý tưởng từ NPU-ASLP | Mức độ áp dụng trong Hybrid-v4 |
|---|---|
| **Band splitting và Scale-Aware** | **Đã áp dụng từ v4.0.** V4 dùng 24 dải stereo và v4.1 trở đi mở rộng sang hai bản `x`, `s`, thêm đặc trưng chênh lệch mức và stereo. |
| **Mô hình hóa theo thời gian và tần số** | **Đã áp dụng ở backbone Dense.** V4 dùng cross-band attention và BiGRU. |
| **Dự đoán phổ phức trực tiếp** | **Thử ở v4.0 và v4.1.** NPU dự đoán toàn bộ phổ sạch. Hai bản v4 này dự đoán **phần thay đổi cộng vào phổ nén của input**, rồi giải nén. V4.0 chỉ nhận `x`, v4.1 nhận thêm `s`. Từ v4.2 chuyển sang lọc phổ tuyến tính của `x` và trộn với `s`. |
| **Top-1 MoE** | **Nằm trong thiết kế v4.5.** Các checkpoint v4.0–v4.4 dùng Dense FFN. |
| **VGGish perceptual loss và warm-up** | **Nằm trong phần thử nghiệm tùy chọn của plan.** Loss của checkpoint v4.4 hiện tại dùng các thành phần mô tả ở trên. |

[NPU-ASLP, phần phương pháp](https://arxiv.org/html/2610.15464v1)

| Phiên bản | Đầu vào | Cách tạo phổ đầu ra |
|---|---|---|
| **v4.0** | Bản lỗi | DirectMap cộng hiệu chỉnh vào phổ nén của input. |
| **v4.1** | Bản lỗi và SonicMaster | DirectMap với encoder hai đầu vào. |
| **v4.2** | Bản lỗi và SonicMaster | Lọc phức Mid/Side với bảy tap MF7, rồi trộn với SonicMaster qua gate. |
| **v4.3** | Bản lỗi và SonicMaster | Lọc phức với chín tap MF9 giãn cách, trộn qua gate và cân RMS theo SonicMaster. |
| **v4.4** | Bản lỗi và SonicMaster | MF9, gate và cân RMS như v4.3, kèm phần dư phức cộng sau bước trộn. |
| **v4.5** | Bản lỗi và SonicMaster | Thiết kế dự kiến dùng decoder được chọn sau đối chứng, với bốn expert Top-1 thay cho Dense FFN. |

Gói lại bằng một ví dụ. **Nếu một bài bị tối nhưng vocal trong input vẫn tự nhiên**, NPU-ASLP nhìn vào bản thu và tự dự đoán phổ sạch theo những gì đã học. **V4.4** nhìn vào cả bản thu và bản SonicMaster, dùng encoder và backbone để xác định vùng cần sửa, rồi có thể **tăng treble trên tín hiệu gốc**, **mượn một phần phổ của SonicMaster**, hoặc **cộng hiệu chỉnh `R`**. Phần **đọc và hiểu phổ** là nơi v4 học nhiều nhất từ NPU-ASLP. Phần **kết hợp tín hiệu để tạo bản cuối** là thiết kế riêng của Hybrid-v4.

## 4. Kết quả thực nghiệm

### So sánh trên 280 clip Dev

Tập Dev gồm **280 clip từ 40 bài nhạc nguồn**. Các hệ được chấm trên cùng clip và cùng số frame cho mỗi ID. Bảng dùng thứ tự `KL(output, clean)`. FAD-CLAP dùng đoạn giữa tối đa 10 giây ở cùng vị trí cho mỗi clip. Audiobox PQ được tính trên toàn bộ khoảng frame chung.

| Hệ | Mel-SSIM ↑ | KL × 1000 ↓ | FAD-CLAP ↓ | Audiobox PQ ↑ | PQ reverb ↑ (106 clip) | Guard trượt |
|---|---:|---:|---:|---:|---:|---|
| SonicMaster | 0,6064 | 1,1064 | 0,1674 | **7,6452** | **7,5946** | Mốc so sánh |
| Hybrid-v3 v3c có nhãn | 0,7398 | 1,0720 | **0,1001** | 7,6330 | **7,5946** | 0/12 |
| Hybrid-v3 Blind E1-tight | 0,7277 | 1,0599 | 0,1159 | 7,5568 | 7,3972 | 0/12 |
| Hybrid-v4.3 | 0,8211 | 1,1181 | 0,1131 | 7,3809 | 6,9507 | 2/12 |
| Hybrid-v4.4 | **0,8260** | **1,0475** | 0,1304 | 7,3339 | 6,8798 | 2/12 |

So với SonicMaster, **v4.4 cải thiện Mel-SSIM, KL và FAD-CLAP**, còn Audiobox PQ giảm **0,3113 điểm** trên toàn tập và **0,7148 điểm** ở nhóm có reverb. Cả v4.3 và v4.4 trượt hai guard **Bright** và **Stereo**. Guard so sai số thuộc tính với mốc SonicMaster theo ngưỡng của benchmark. Điểm PQ là đánh giá của Audiobox. Mức độ giảm tiếng vocal robot cần được kiểm chứng riêng bằng thử nghe.

### Thử nghiệm hệ số phần dư trên cùng checkpoint v4.4

Cả ba mức dùng checkpoint epoch 20 và cùng 280 clip Dev. Hệ số `α` nhân với phần dư đã dự đoán, tức decoder cộng `αR` vào phổ sau bước trộn. Bảng này dùng các bản v4.4 tái xuất cho thí nghiệm hệ số.

| Hệ | Mel-SSIM ↑ | KL × 1000 ↓ | FAD-CLAP ↓ | Audiobox PQ ↑ | PQ reverb ↑ (106 clip) |
|---|---:|---:|---:|---:|---:|
| SonicMaster | 0,6064 | 1,1064 | 0,1674 | **7,6452** | **7,5946** |
| v4.4, α = 0 | 0,8240 | 1,5046 | **0,1181** | 7,3379 | 6,8858 |
| v4.4, α = 0,5 | **0,8269** | 1,1556 | 0,1218 | 7,3452 | 6,8968 |
| v4.4, α = 1 | 0,8260 | **1,0475** | 0,1304 | 7,3340 | 6,8800 |

Tắt phần dư làm KL tăng khoảng **44%** so với cấu hình gốc `α = 1`. Mức `α = 0,5` đạt PQ cao nhất trong ba cấu hình v4.4, song vẫn thấp hơn SonicMaster khoảng **0,70 điểm** trên nhóm reverb. Kết quả ủng hộ **giữ phần dư ở checkpoint hiện tại** vì lợi ích về KL. Phép thử đo tác động của hệ số phần dư lúc suy luận. Kiến trúc bỏ hẳn nhánh phần dư sẽ cần một lượt huấn luyện và đánh giá riêng.
