# EDA Module Documentation (`labs/lab2/eda/eda.py`)

## 1. Purpose
- View sample images per class.
- Inspect image size and brightness / channel distributions.
- Count the number of images per class (check class balance).
- Detect duplicate, corrupted, or abnormal images.

## 2. Input - Output
- **Input:**
  - `data_root` (`str`): Path to the directory where raw CIFAR-10 data is stored or downloaded.
  - `out_dir` (`str`): Directory path where generated figure artifacts will be saved.
- **Output:**
  - Generates 2 figures saved in `out_dir`:
    1. Class distribution bar chart (`cifar10_class_distribution.png`): Number of images in each class.
    2. Sample grid image (`cifar10_sample_grid.png`): Sample images across each class (10 classes x 3 examples).
  - Returns a statistics dictionary:
    - `n_train`: Number of training images (50,000).
    - `n_test`: Number of test images (10,000).
    - `class_counts`: Sample count per class (5,000 each in train set).
    - `image_size`: Dimensions of raw images `(32, 32, 3)`.
    - `channel_mean`: Per-channel mean normalized to `[0, 1]` (`[0.4914, 0.4822, 0.4465]`).
    - `channel_std`: Per-channel standard deviation normalized to `[0, 1]` (`[0.2470, 0.2435, 0.2616]`).

## 3. Success_valuable
- Answers core questions about the dataset (dimensions, balance, value ranges).
- Identifies potential issues before training.
- Provides evidence-based rationale for downstream preprocessing decisions.
- Formulates clear hypotheses on dataset characteristics and model challenges.
- Results are fully reproducible with documented source and impact.

## 4. How to do
Execute `run_eda(data_root, out_dir)`:
1. **Load raw dataset:** Call `torchvision.datasets.CIFAR10` for train and test sets with `transform=None` and `download=True`.
2. **Compute metrics and distributions:**
   - Extract sample counts (`n_train`, `n_test`) and raw image dimensions `(32, 32, 3)`.
   - Count label frequencies in `train_raw.targets` to verify dataset balance.
   - Scale pixel values to `[0, 1]` (`float32`) and compute channel-wise mean and standard deviation across RGB channels.
3. **Generate visualization artifacts:**
   - Create a 10x3 sample grid using `matplotlib` (backend `"Agg"`) and save to `cifar10_sample_grid.png`.
   - Create a bar chart of class counts and save to `cifar10_class_distribution.png`.
4. **Return structured summary:** Return the dictionary containing `n_train`, `n_test`, `class_counts`, `image_size`, `channel_mean`, and `channel_std`.

---

**Tiếng Việt**

# Tài liệu Module EDA (`labs/lab2/eda/eda.py`)

## 1. Purpose (Mục đích)
- Xem mẫu ảnh theo từng class.
- Kiểm tra phân phối kích thước ảnh, độ sáng / giá trị các kênh màu.
- Đếm số lượng ảnh của mỗi class (kiểm tra cân bằng dữ liệu).
- Tìm kiếm các ảnh bị trùng lặp hoặc bị hỏng/lỗi.

## 2. Input - Output (Đầu vào - Đầu ra)
- **Input:**
  - `data_root` (`str`): Đường dẫn thư mục chứa dữ liệu thô CIFAR-10.
  - `out_dir` (`str`): Thư mục lưu các file ảnh biểu đồ sinh ra.
- **Output:**
  - Xuất ra 2 file ảnh trong `out_dir`:
    1. Ảnh số lượng các ảnh trong từng class (`cifar10_class_distribution.png`).
    2. Ảnh các mẫu ảnh trực quan trong từng class (`cifar10_sample_grid.png`).
  - Trả về dictionary chứa các thông số:
    - `n_train`: Số lượng ảnh tập train (50.000).
    - `n_test`: Số lượng ảnh tập test (10.000).
    - `class_counts`: Số lượng ảnh của từng class (5.000 ảnh/class).
    - `image_size`: Kích thước ảnh gốc `(32, 32, 3)`.
    - `channel_mean`: Giá trị trung bình trên từng kênh màu khi đưa về `[0, 1]` (`[0.4914, 0.4822, 0.4465]`).
    - `channel_std`: Độ lệch chuẩn trên từng kênh màu khi đưa về `[0, 1]` (`[0.2470, 0.2435, 0.2616]`).

## 3. Success_valuable (Tiêu chuẩn đạt yêu cầu)
- Trả lời được các câu hỏi cốt lõi về dữ liệu.
- Phát hiện vấn đề bất thường trước khi train.
- Đưa ra quyết định có căn cứ (cơ sở chọn chuẩn hóa, kích thước resize cho mô hình).
- Có hypothesis (giả thuyết) rõ ràng về dữ liệu và bài toán.
- Kết quả tái lập được (reproducible) và có tài liệu, nguồn gốc dữ liệu rõ ràng.

## 4. How to do (Cách thực hiện)
Thực thi hàm `run_eda(data_root, out_dir)`:
1. **Nạp dữ liệu thô:** Dùng `torchvision.datasets.CIFAR10` nạp tập train và test nguyên bản không áp dụng transform (`download=True`).
2. **Tính toán thống kê & phân phối:**
   - Lấy số lượng mẫu (`n_train`, `n_test`) và kích thước ảnh gốc `(32, 32, 3)`.
   - Đếm số lượng ảnh theo từng nhãn trong `train_raw.targets` để xác nhận dữ liệu cân bằng.
   - Chuyển mảng pixel sang `float32` chia 255 về đoạn `[0, 1]` và tính `mean`, `std` trên 3 kênh RGB.
3. **Vẽ và xuất biểu đồ:**
   - Tạo lưới 10 hàng x 3 cột thể hiện ảnh mẫu từng lớp và lưu vào file `cifar10_sample_grid.png`.
   - Vẽ biểu đồ cột phân phối 10 lớp và lưu vào file `cifar10_class_distribution.png`.
4. **Trả về kết quả:** Đóng gói toàn bộ chỉ số vào dictionary trả về.
