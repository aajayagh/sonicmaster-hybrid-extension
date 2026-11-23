# Kế hoạch bàn giao · Tái lập SonicMaster trên Kaggle từ đầu

## Lệnh giao việc cho agent

> Tạo một dự án riêng tư mới trên Kaggle để tái lập bài *SonicMaster: Towards Controllable All-in-One Music Restoration and Mastering*, bản ICML 2026. Bắt đầu ở bước 00 và thực hiện theo các điều kiện nghiệm thu bên dưới. Chỉ sử dụng mã nguồn, dữ liệu và trọng số chính thức hoặc các phụ thuộc đã xác minh. Không mở, sửa, chạy, gắn làm input hay lấy kết quả từ bất kỳ notebook/dataset thực nghiệm nào của nhóm trước đây. Không thêm refiner, tinh chỉnh mô hình hoặc chọn prompt theo điểm test. Mỗi lần báo cáo phải phân biệt số tác giả công bố, số tự đo và phần chưa thực hiện. Tiếp tục những nhánh độc lập khi một nhánh thiếu tài nguyên. Không điền số paper vào ô kết quả chạy lại.

## 1. Đích đến và thứ tự ưu tiên

| Mốc | Sản phẩm nghiệm thu | Được kết luận gì? |
|---|---|---|
| M0 | Chạy checkpoint trên các mẫu WAV chính thức, xác nhận tham số nạp đúng, có đầu ra nghe được | Đã tái lập thao tác suy luận trên mẫu |
| M1 · ưu tiên cao nhất | Đầu ra checkpoint Large trên toàn bộ test snippet chính thức; hàng input, VAE và SonicMaster trong Bảng 1–3; số mẫu và bảng chênh lệch với paper | Đã đo lại các hàng tương ứng trên đoạn ngắn |
| M2 | Chạy các phương pháp đối chứng, bài dài, RemFX, piano và ablation có đủ đầu vào; mỗi ô có bằng chứng | Đã tái lập những so sánh được đánh dấu hoàn tất |
| M3 | Huấn luyện backbone mới, rồi chấm trên test đã cố định | Tái lập huấn luyện đến mức cấu hình và tài nguyên thực tế cho phép |
| M4 | Hai nghiên cứu nghe độc lập nếu tuyển được người | Bằng chứng cảm nhận trên người nghe mới |

M1 độc lập với M3. Có thể hoàn thành đánh giá checkpoint trước khi quyết định dành tài nguyên cho huấn luyện. Thứ tự này là lựa chọn triển khai trên Kaggle. Mọi mục tiêu khoa học vẫn nằm trong bảng theo dõi kể cả khi chưa làm được.

## 2. Nguồn và phiên bản ban đầu

- [Paper của tác giả](https://arxiv.org/abs/2508.0338). Dùng PDF ICML 2026 người dùng cung cấp làm chuẩn cho số bảng, cách nhân hệ số và các ngoại lệ. Nếu agent từ xa không có PDF, kiểm tra phiên bản tải về có Bảng 1–4, 8–10, Hình 2, 3, 11 và đối chiếu các số mốc bên dưới.
- [Repo tác giả, commit đã kiểm tra](https://github.com/AMAAI-Lab/SonicMaster/tree/ab32b2d456e461b6bbc188d666caf2eabefa2f31). Có mã suy luận, huấn luyện, tạo dữ liệu, đánh giá và các bộ WAV mẫu clean/degraded/inferred.
- [Trọng số SonicMaster](https://huggingface.co/amaai-lab/SonicMaster/tree/93c7fa73c4dd446c9e656af34524b5f6a416afda). Suy luận dùng model.safetensors. Ghi SHA-256 thực tế khi tải.
- [Dữ liệu chính thức](https://huggingface.co/datasets/amaai-lab/SonicMasterDataset/tree/04cc0dd51ff748b41373308d9ee152796c840233). Có metadata, JSONL và các shard Parquet. Kiểm tra schema thực tế trước khi viết downloader.
- [VAE](https://huggingface.co/stabilityai/stable-audio-open-1.0) dùng thư mục vae; [bộ mã hóa văn bản](https://huggingface.co/google/flan-t5-large). Pin revision riêng cho cả hai, tokenizer và các mô hình chấm điểm. Dùng Kaggle Secrets cho token cần thiết. Kiểm tra quyền truy cập trước khi bật GPU; không in token hay sao chép trọng số bị hạn chế sang tài sản công khai.

## 3. Cấu trúc job và phụ thuộc

Tạo namespace mới, ví dụ sonicmaster-paper-repro-v1. Mỗi job là một notebook/script có version, đọc input từ output **đã commit** của job trước hoặc dataset riêng mới có version cụ thể. Dùng /kaggle/input để đọc, /kaggle/working cho sản phẩm cần giữ; thử mở lại output từ một phiên mới trước khi chạy dài.

| Job | GPU | Input | Output chính |
|---|---|---|---|
| 00_inventory | Không | Nguồn chính thức | source-lock.json, bảng quyền truy cập, requirements thực tế, giấy phép và danh mục thí nghiệm |
| 01_manifest | Không | Metadata chính thức | manifests, thống kê split/variant/attribute, danh sách shard và báo cáo trùng ID |
| 02_smoke | Có | 00, WAV mẫu, weights | audio đầu ra, so sánh với WAV tác giả, log nạp weights, runtime/VRAM |
| 03_stage_audio | Không | 01 | shard audio đã kiểm hash, bảng mapping clean/degraded, phiên bản dữ liệu |
| 04_infer | Có | 02, 03 | WAV suy luận theo shard, VAE roundtrip, bảng kê và hash |
| 05_score | CPU và GPU nếu CLAP/PQ cần | 01, 03, 04 | điểm từng mẫu, embedding, thống kê cấp tập, Bảng 1–3 |
| 06_comparisons | Tùy nhánh | 00–05 và tài sản bổ sung | hàng baseline, full-song, RemFX, piano, ablation |
| 07_train | Có, sau quyết định tài nguyên | train/val đúng split, thiết lập khóa | latent, checkpoint mới, log và kết quả test riêng |
| 08_listening | Không bắt buộc GPU | Mẫu đã khóa, output của các hệ thống | giao diện thử nghe, dữ liệu ẩn danh, thống kê |
| 09_report | Không | Output đã commit | report.md, bảng CSV, audio mẫu, danh sách sai lệch và phần bị chặn |

## 4. Bước 00–01 · Chốt dữ liệu trước khi chạy model

1. Ghi revision, hash, tên tài sản, cách tải, điều kiện sử dụng vào source-lock.json. Lập bảng từng hàng/cột của paper với trạng thái planned, running, measured, partial hoặc blocked và lý do. Không công bố dữ liệu/weights ngoài quyền phân phối.
2. Đọc metadata trước audio. Hugging Face gọi toàn bộ container là train, còn dataset card mô tả meta.split là nhãn phân chia thí nghiệm. Đọc giá trị thực tế, kiểu meta (chuỗi JSON hay object), cách ánh xạ metadata sang row/shard. Nếu thiếu split, truy tìm manifest chính thức trước khi chấm.
3. Mỗi dòng manifest cần sample_id, source_id, split, variant, số lỗi được gắn nhãn, nhóm/thuộc tính, prompt và alt_prompt, hidden_clipping, shard/row, định danh hai file audio. Lưu sample rate, số kênh, chiều dài, checksum sau giải mã hoặc tải. Giữ nguyên prompt chính thức; ghi quy tắc chọn khi có hai prompt, không thử cả hai rồi chọn điểm đẹp hơn.
4. Kiểm tra source_id không giao nhau giữa các split; ID duy nhất; bản clean khớp giữa các variant của một nguồn. Paper mô tả 1.000 nguồn test × 7 = 7.000 cặp, dự kiến 4.000 single, 3.000 double+triple. Đếm thật và xuất chênh lệch. Bảy variant gồm 4 single, 2 double và 1 triple. Khi phân nhóm, tách hidden_clipping khỏi danh sách lỗi có prompt theo quy tắc script gốc; lưu cả hai trường để kiểm toán.
5. Nếu không khớp số liệu, khoanh nguyên nhân: phiên bản, thiếu shard, variant trùng, split khác hay chính sách lọc. Có thể chạy subset để gỡ lỗi, chọn cố định theo source và bao phủ nhóm lỗi, gắn nhãn pilot. Không tạo split ngẫu nhiên rồi gọi là test của tác giả.
6. Tính dung lượng từ metadata/HEAD và đo một shard trước khi tải hàng loạt. Chọn đọc có chọn lọc qua Parquet hoặc tải từng shard nếu range requests không ổn định; lưu theo checksum, dùng file .part và rename sau khi kiểm tra. Retry có giới hạn với 429/5xx, backoff + jitter, không bỏ im lặng mẫu lỗi. Ưu tiên dữ liệu test cho M1; chỉ tải train khi thực hiện M3.

## 5. Bước 02 · Kiểm chứng suy luận trên ví dụ có đáp án tác giả

- Kiểm tra Python, Torch, torchaudio, CUDA, diffusers, transformers, thư viện audio thực sự dùng trong phiên Kaggle. README đề xuất Python 3.13; thử môi trường chính thức trước, khóa lại môi trường chạy được và ghi mọi thay đổi. Chuyển đường dẫn /mastering/... sang tham số Kaggle. Dùng secret hoặc cache local thay cho cấu hình token trong mã nguồn. Lưu patch/diff; tách sửa đường dẫn khỏi thay đổi thuật toán.
- Nạp model.safetensors theo configs/tangoflux_config.yaml. infer_single.py dùng strict=False: ghi missing_keys/unexpected_keys; giải thích từng nhóm. Nếu các khối backbone không được nạp, dừng trước full run. Kiểm tra load VAE, text encoder, tokenizer, eval mode, hữu hạn của tensor; không chạy bằng weights ngẫu nhiên nếu tải lỗi.
- Khởi đầu với infer_single.py, ví dụ dòng lệnh dưới đây. Thay biến bằng đường dẫn/tệp thực và prompt đã tra theo ID:

~~~bash
python infer_single.py --ckpt "$CKPT" --input "$INPUT_WAV" --prompt "$PROMPT" --output "$OUTPUT_WAV" --config configs/tangoflux_config.yaml --fs 44100 --chunk_duration 30 --overlap_duration 10 --num_inference_steps 10 --guidance_scale 1.0 --solver Euler --seed 0
~~~

- Với nhiều ID từ samples/, ghép đúng orig, input và infer bằng ID + variant. Nghe và kiểm tra WAV thực (repo có dùng Git LFS; phát hiện file pointer). So output tự tạo với bản infer chính thức, và so cả hai với orig. Ghi prompt, RMS hai kênh, thời lượng, peak/clipping, mel-SSIM và sai số dạng sóng. Chỉ đòi bằng byte khi toàn bộ môi trường/seed/đường inference trùng khớp. Kiểm tra hai lượt cùng seed, lệch hoặc hoán đổi kênh, zero/NaN, lệch thời gian. Không căn chỉnh tùy ý theo clean để tăng điểm.
- Đọc đường thực thi model trước khi vector hóa. Paper đặt degraded ở t=1, clean ở t=0; dùng solver upstream và kiểm chứng trên mẫu. Đo ít nhất vài clip sau warm-up để ước tính giây/clip, VRAM, bytes/clip. Chạy 1 clip/GPU trước, tăng batch sau khi kiểm chứng tương đương. Nếu dùng precision khác để vừa bộ nhớ, đối chiếu cùng clip với thiết lập chuẩn và ghi rõ.

**Điều kiện chuyển bước:** audio hợp lệ, toàn bộ khóa trọng số quan trọng được nạp, các chênh lệch với ví dụ tác giả có biên bản, runtime thực tế cho phép chọn shard an toàn. Nếu chỉ chạy được pilot, vẫn nộp log cùng nguyên nhân.

## 6. Bước 03–04 · Chạy benchmark theo shard

1. Tính ngân sách từ pilot: giờ GPU xấp xỉ số clip × giây GPU/clip ÷ 3600, nhân riêng cho từng phương pháp, cộng CLAP/PQ, tải dữ liệu và lưu output. Đọc quota và dung lượng còn lại trong tài khoản tại thời điểm chạy; không mặc định hai GPU, một số giờ cố định, hoặc thời lượng phiên cố định.
2. Chia theo source_id, giữ cả 7 variant cùng shard. Chạy job pilot đầu cuối trên một shard; lưu input, GT, reconstructed input (degraded → VAE encode/decode) và SonicMaster với đúng prompt. Mỗi WAV có mapping duy nhất tới method/seed/config/sample_id; ghi định dạng, precision và quy tắc chuẩn hóa. Giữ stereo cho toàn bộ xử lý cần stereo.
3. Với phần đoạn ngắn dùng đường batch snippet gốc nếu khả thi; so vài mẫu với infer_single và mã eval trước khi mở rộng. Tắt mọi clean cue lấy từ chính GT test. Nhánh full-song chỉ sử dụng tham chiếu mà giao thức inference cho phép. Ghi seed, cách tạo latent và mọi padding/cropping.
4. Mỗi shard tạo index, số ID mong đợi, số hoàn tất, checksum, thời gian và danh sách lỗi. Khi restart, chỉ skip file đúng cấu hình/hash và vượt kiểm tra audio; không dùng sự tồn tại của file làm bằng chứng. Nếu shard thiếu mẫu, giữ kết quả ở trạng thái partial và sửa trước khi lập bảng cuối. Lưu dưới dạng output có version trước khi phiên Kaggle hết hạn. Bắt đầu shard nhỏ, tăng theo dung lượng đo được.

## 7. Bước 05 · Tái lập Bảng 1–3

Chạy mã trong evaluation/ từ commit đã pin. Đọc cả công thức phụ lục A.5 và code. Khi khác nhau, giữ một kết quả theo code và ghi cách làm theo paper thành biến thể riêng nếu cần. Ghi số dòng được tính cho mỗi ô, tên đơn vị, split và filters; đối chiếu sample IDs, cách tách single/multiple, cách gộp và trục stereo trước khi kết luận. Không tự lấy trung bình tất cả cột lỗi khác đơn vị.

| Nhóm / bảng | Cách đo cần triển khai |
|---|---|
| EQ, Bảng 1 | Tám thuộc tính clarity/boom/airy/bright/dark/muddy/warm/vocals: sai số tỷ lệ năng lượng dải mục tiêu so với GT, biên dải theo A.5. Mic và X-band: cosine distance giữa phân bố năng lượng 9 dải. Sai số thấp tốt. |
| Dynamics, Bảng 2 | Compression: sai số độ lệch chuẩn RMS theo frame 2048/hop 1024. Punch: sai số trung bình onset strength. |
| Reverb, Bảng 2 | Small, Big, Mixed, Real: khoảng cách Euclid của modulation spectrum chuẩn hóa. Đối chiếu STFT 1024/hop 512 và 20 modulation bins ở A.5. |
| Amplitude và Stereo, Bảng 2 | Clip: sai số spectral flatness; nhân 1000 khi trình bày. Volume: sai số RMS. Stereo: sai số tỷ lệ RMS(side)/RMS(mid), giữ riêng L/R và epsilon đúng code. |
| KL và SSIM, Bảng 3 | Dùng extract_kl_ssim_mass.py, kiểm tra thứ tự GT/output tại caller. Mã đọc mono 44,1 kHz, mel 128, FFT 2048, hop 512, chuyển dB theo cực đại. KL dùng trung bình log-mel theo thời gian rồi chuẩn hóa như code; lưu raw và chỉ nhân 1000 khi trình bày. SSIM crop về kích thước chung, min–max từng ma trận, data_range=1. Bất kỳ thay đổi công thức đều phải có nhãn riêng. |
| FAD-CLAP, Bảng 3 | Theo extract_clap_gt.py và extract_fad_mass.py; pin CLAP checkpoint, fusion, resample, cách gộp cửa sổ, số GT duy nhất và chính sách lặp GT. Lưu embedding để ghép nhiều shard rồi tính FAD một lần trên toàn bộ tập so sánh. Không trung bình FAD của từng clip/shard. |
| PQ, Bảng 3 | Theo extract_aesthetics_scores.py và Audiobox Aesthetics chính thức; ghi revision, preprocessing, aggregation, raw điểm từng bài. Chấm cả clean để kiểm tra mốc tham chiếu. |

Trước full scoring, test GT với chính nó (sai số gần 0, SSIM gần 1 trên tín hiệu không suy biến), tín hiệu mono/stereo có kiểm soát, giảm volume biết trước và cặp bị đảo ID. Ghi trường hợp im lặng hoặc NaN; không tự loại mẫu mà không báo số lượng. Làm rõ mỗi hàng dùng clip đơn lỗi hay tập nào theo caller và manifest thực tế.

**Các mốc đối chiếu, không phải số mặc định.** Bảng 3, snippet/All, thứ tự FAD↓, KL×1000↓, SSIM↑, PQ↑: input = 0.106 / 5.131 / 0.777 / 7.026; reconstructed = 0.196 / 5.273 / 0.546 / 6.885; SonicMaster = 0.073 / 0.888 / 0.609 / 7.705. PQ clean = 7.886. Bảng 1 snippet/SonicMaster: Mic 0.0088, X-band 0.0358. Bảng 2 snippet/SonicMaster: Clip×1000 1.506, Stereo 0.1058. Từng bảng cuối phải có cột paper, measured, delta, N, protocol, evidence. Tách cột Single / Double+Triple / All. Không hứa bằng số đã làm tròn.

**Khi lệch:** kiểm tra GT/input trước vì hai hàng này không phụ thuộc checkpoint SonicMaster. Nếu input lệch: kiểm tra split, decoder, sample rate, mono, normalization, cách gộp GT, cách nhân 1000. Nếu VAE lệch: kiểm tra version, scale latent, sample/mode. Nếu chỉ output lệch: kiểm tra keys, cấu hình Large, prompt, solver, steps, seed và precision. Định nghĩa dung sai trước khi chấm full test, dựa trên repeatability của metric và mức làm tròn; tránh chỉnh cấu hình theo mức gần con số paper. Báo “đã đo lại” riêng với “khớp số” và “khớp xu hướng”.

## 8. Bước 06 · Các thí nghiệm bổ sung, theo thứ tự khả thi

| Nhánh | Protocol và điều kiện nghiệm thu |
|---|---|
| Baseline EQ | Text2FXcos và Text2FXdir trên cùng ID, prompt và lỗi EQ; 600 vòng, LR 0.01; phương án dir cần target/contrast prompt và EQ 6 dải theo A.3. Lưu từng cặp prompt, nguồn code và weights. Nếu phải tạo contrast prompt mới, đánh dấu khác biệt. |
| Baseline reverb | WPE order 30; HPSS attenuation 6 và 12 dB theo script; Mel2Mel + DiffWave đúng checkpoint/phương pháp. Chỉ tính hàng thuộc phạm vi baseline. Nếu không có weights công khai hoặc split tương thích, ghi blocked cụ thể cho hàng đó. |
| Full-song, phần dưới Bảng 1–3 | Xác minh danh tính bài, audio nguyên bản, danh sách test, degradation và prompt full-song. Chạy inference_fullsong.py sau khi kiểm path; các đoạn 30 s, overlap 10 s, cue từ đầu ra trước và crossfade đúng implementation. Kiểm tra clipping, lệch âm lượng tại điểm nối và độ dài. Báo full-song riêng. Nếu chỉ có 30 s hoặc không có danh sách tác giả, ghi không thể đối chiếu chính xác; không nối nhiều bài ngắn giả thành full-song. |
| RemFX, Hình 2 | Tìm release/test split giao thức RemFX, cặp clean/effected, hiệu ứng Dynamics và Reverb, cách tính SI-SDR theo nguồn. Kiểm tra alignment, stereo và aggregation. Báo dB cùng N. Mốc paper là 47.11 / 45.76 dB. Chia riêng baseline tự chạy và số trích dẫn cho DPTNet/UMX/DCUNet/TCN/HDemucs. |
| Piano, Bảng 4 | Từ nguồn historical solo piano/BABE-2 trích dẫn trong paper, xác minh đúng recordings và đầu ra Original/LTAS-EQ/BEHM-GAN/BABE/BABE-2. Chạy SonicMaster zero-shot, ghi prompt; tính CE/CU/PC/PQ và độ phân tán đúng bảng. Nếu thiếu prompt/ID ghi comparable chỉ tới mức nào. |
| Ablation có thể dùng cùng checkpoint | So Euler 1/10/100 và Runge–Kutta 10 bước nếu repo hỗ trợ đúng solver; có prompt, empty prompt và shuffle prompt theo hoán vị cố định trong test. Ghi seed, số điểm đánh giá mạng và thời gian. Tạo Bảng 8–10 theo cùng metric, so với default trên cùng ID. |
| Ablation cần model khác | Small 2/6, Medium 4/12, Medium 6/6, Large 6/18; training có/không audio condition, cue 5/10/15 s. Phân biệt bỏ cue lúc inference với bỏ cue lúc train. Kiểm kê checkpoint cho từng biến thể; muốn đo biến thể chưa phát hành thì phải huấn luyện nó. |

## 9. Bước 07 · Huấn luyện từ đầu nếu cần tái lập quá trình học

1. Dùng dataset đã phát hành cho nhánh chính. Nếu muốn kiểm chứng cả quá trình dựng dataset: bắt đầu từ Jamendo + lọc Audiobox theo genre, crop 30 s, áp 19 hiệu ứng theo xác suất và khoảng tham số ở A.4, tạo prompt và alt_prompt; lưu provenance, clip offsets, RNG và hash. Khi thiếu snapshot nguồn/seed chính thức, nhánh này chỉ là dựng tập tương đương.
2. Kiểm tra source-disjoint train/val/test; nếu paper không cung cấp val list, công bố train-derived split cố định, giữ test riêng. Preencode bằng preencode_latents_acce2.py, lưu latent đúng sampling/scaling, shape, dtype, ID và metadata qua các shard. Kiểm tra decode một số latent so với đường chạy trực tiếp.
3. So paper, code và YAML trước khi chạy: paper báo 5 L40S, 40 epoch, tổng batch 80; tangoflux_config.yaml ở commit đã pin ghi 201 epoch, per-device 16, LR 5e-4, warmup 1000. Ghi cấu hình thực sự đi vào optimizer, scheduler, stopping rule và liệu vòng lặp tính cả epoch 0. Nhánh paper đặt 40 epoch, global batch 80; nếu đổi GPU, dùng gradient accumulation và kiểm tra số lần optimizer.step. Ghi mọi khác biệt phần cứng, precision và runtime.
4. Backbone Large 6 MM-DiT + 18 DiT; VAE và text encoder frozen. Audit hướng flow degraded→clean, timestep sampling, loss vận tốc, text dropout 10%, generic prompt 10% khác và clean cue 10 s ở 25% mẫu theo paper. Chỉ dùng cue sạch trong ngữ cảnh protocol cho phép. Ghi điểm nào paper/code chưa giải quyết được trước run.
5. Pilot train trên một tập nhỏ để xác nhận gradient, finite loss, VRAM và round-trip save/resume. Checkpoint đủ model, optimizer, scheduler, scaler nếu có, RNG và dataloader progress. So bước tiếp theo của run liên tục và run resume. Tính GPU-giờ và storage cho **40 epoch** từ pilot; nếu quota thực không khả thi, công bố dự toán và xin quyết định trước khi tiêu tài nguyên cho full train. Pilot ngắn có nhãn riêng.
6. Chọn checkpoint bằng quy tắc validation cố định trước test; chấm weights tự train bằng jobs 04–05 và đặt cạnh weights tác giả với cùng test. Không khởi tạo backbone bằng weights SonicMaster cho nhánh train-from-scratch; VAE/text encoder pretrained là thành phần chính thức.

## 10. Bước 08 · Đánh giá nghe

- Nghiên cứu 1: 43 cặp (38 single, 3 double, 2 triple), 12 người trong paper (7 chuyên gia âm nhạc, 5 nhà nghiên cứu MIR), thang 1–7 cho text relevance, chất lượng trước/sau, consistency và preference. Theo A.7, giao diện đầu tiên cho biết vị trí input/output. Nếu thêm nhánh mù hoặc random hóa vị trí, ghi rõ khác giao thức gốc. Cân bằng điều kiện phát/volume theo protocol; không loudness-match tùy tiện khi đang chấm volume.
- Nghiên cứu 2: 20 người, 10 mẫu X-band EQ và 10 reverb; so với baseline phù hợp, giữ danh tính phương pháp kín nếu thiết kế cho phép. Ghi seed chọn mẫu, thứ tự, hướng dẫn, tai nghe, đồng thuận và dữ liệu ẩn danh. Để người phụ trách duyệt việc tuyển người; agent có thể tự chuẩn bị giao diện và protocol, không tự gửi lời mời.
- Báo MOS, 95% CI, phép paired t-test theo paper; ghi đơn vị phân tích là người, clip hoặc lượt chấm. Báo số phiếu và số người riêng. Ghi bất nhất A.8: SonicMaster 191/200 lượt reverb, văn bản và Hình 11 khác nhau ở phiếu của đối thủ. Với panel mới, so xu hướng, mẫu số và độ bất định. Khi chưa có người tham gia, chỉ bàn giao protocol, không thay phiếu bằng LLM hoặc PQ.

## 11. Bàn giao và tiêu chí dừng

- Nộp README có thứ tự chạy, version Kaggle của mỗi input/output, thời gian, chi phí GPU thực, câu lệnh và cấu hình chính xác. Nộp source-lock, environment-lock, manifest, inventory từng thí nghiệm, log lỗi/retry, bảng per-sample và bảng tổng hợp. Pin upstream; nếu cần patch, lưu diff và hash cả trước lẫn sau.
- Tạo report ngắn với bảng **paper / chạy lại / chênh lệch / N / phạm vi / bằng chứng**. Dẫn tới CSV hoặc artifact tạo ra từng hàng. Nộp vài bộ clean–degraded–restored chọn trước theo ID, dùng WAV và prompt đúng, thêm ví dụ thất bại nếu có. Không dùng demo output của tác giả như đầu ra tự chạy.
- Khi thiếu quota, quyền VAE, checkpoint đối chứng hoặc full-song manifest: ghi hạng mục, tài sản thiếu, đã kiểm nguồn nào, ảnh hưởng tới hàng nào và bước tiếp theo. Có thể hoàn tất các hàng còn lại; không gắn nhãn “tái lập toàn bộ” khi mới đạt M1.
- Việc đầu tiên của agent: tạo repo/notebook mới cho 00–01, nộp source-lock + thống kê test + bảng tài sản/khả năng truy cập. Sau đó chạy 02 trên GPU ngắn. Chỉ mở rộng sau khi có audio kiểm chứng, dự toán tài nguyên và một shard scoring end-to-end.
