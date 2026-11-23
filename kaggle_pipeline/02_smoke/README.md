# Hướng Dẫn Thực Thi Bước 02 (`02_smoke_test`) Trên Kaggle GPU

Tệp notebook [02_smoke_test.ipynb](file:///d:/SonicMaster_Reproducing/kaggle_pipeline/02_smoke/02_smoke_test.ipynb) được thiết kế tự động hoàn toàn (self-contained) để chạy trên Kaggle GPU (T4 hoặc P100).

---

## Cách 1: Tải lên giao diện Web Kaggle (Thực hiện thủ công nhanh nhất)

1. Mở trình duyệt và truy cập [kaggle.com/code](https://www.kaggle.com/code) $\to$ Nhấn **New Notebook**.
2. Trên thanh menu của Notebook:
   - Chọn **File** $\to$ **Import Notebook** $\to$ Chọn tệp `02_smoke_test.ipynb` tại đường dẫn máy của bạn:  
     `D:\SonicMaster_Reproducing\kaggle_pipeline\02_smoke\02_smoke_test.ipynb`.
3. Cấu hình môi trường ở thanh menu bên phải (**Notebook Settings**):
   - **Accelerator:** Chọn **GPU T4 x1** (hoặc GPU P100).
   - **Internet:** Bật **Internet on** (để tải checkpoint và code).
4. Nhấn **Run All** (hoặc **Save Version** $\to$ **Quick Save / Run**).
5. Khi chạy xong:
   - Tải tệp `smoke_test_report.json` và `smoke_test_summary.md` từ thư mục `/kaggle/working/` về để chuyển sang Bước 03.

---

## Cách 2: Sử dụng Kaggle CLI (Nếu đã cấu hình Kaggle Token)

1. Cập nhật `YOUR_KAGGLE_USERNAME` trong tệp `kernel-metadata.json`.
2. Chạy lệnh:
   ```bash
   kaggle kernels push -p kaggle_pipeline/02_smoke
   ```
3. Theo dõi trạng thái:
   ```bash
   kaggle kernels status YOUR_KAGGLE_USERNAME/sonicmaster-02-smoke
   ```
