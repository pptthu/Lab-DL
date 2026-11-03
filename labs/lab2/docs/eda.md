# EDA Module Documentation (`labs/lab2/eda/eda.py`)

## 1. Overview
The `eda` module is responsible for Exploratory Data Analysis (EDA) on the CIFAR-10 dataset for Lab 2. It inspects the dataset characteristics, computes statistical properties across image color channels, and visualizes sample grids and class distributions before training.

- **File location:** `labs/lab2/eda/eda.py`
- **Assigned owner:** Member A
- **Dependencies:** `torchvision`, `torchvision.datasets`, `matplotlib`, `numpy`, `pathlib`

---

## 2. Constants & Data Structures

### `CIFAR10_CLASSES`
A tuple containing the 10 class names in CIFAR-10 in standard label order (index 0 to 9):
```python
CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer", 
    "dog", "frog", "horse", "ship", "truck"
)
```

---

## 3. Functions & API Contract

### `run_eda(data_root: str, out_dir: str) -> dict`
Performs read-only inspection of the CIFAR-10 dataset and generates visual artifacts.

#### Arguments:
- `data_root` (`str`): Path to the directory where CIFAR-10 raw data is stored or downloaded.
- `out_dir` (`str`): Path to the directory where visual artifact PNGs will be saved.

#### Returns:
A dictionary containing dataset statistics:
- `n_train` (`int`): Number of training images (50,000).
- `n_test` (`int`): Number of test images (10,000).
- `class_counts` (`dict[str, int]`): Sample count per class in the training split (5,000 each).
- `image_size` (`tuple`): Raw image shape `(32, 32, 3)` (height, width, RGB channels).
- `channel_mean` (`list[float]`): Per-channel mean values scaled to `[0, 1]`, rounded to 4 decimals (approx. `[0.4914, 0.4822, 0.4465]`).
- `channel_std` (`list[float]`): Per-channel standard deviation scaled to `[0, 1]`, rounded to 4 decimals (approx. `[0.2470, 0.2435, 0.2616]`).

#### Generated Artifacts:
1. `cifar10_sample_grid.png`: A 10x3 grid displaying 3 sample images for each of the 10 classes.
2. `cifar10_class_distribution.png`: A bar chart showing the frequency of each class in the training dataset.

---

## 4. Key Implementation Details & Defense Q&A
- **Q: Why calculate channel mean and std on train data only?**
  - **A:** To prevent data leakage. The test set represents unseen data in real deployment, so any normalization constants must be calculated solely from the training distribution.
- **Q: Why use `matplotlib.use("Agg")`?**
  - **A:** `"Agg"` is a non-interactive backend. It ensures headless environments (CI/CD servers, terminal runs, remote servers without a display) can render and save figures without throwing display/GUI errors.
- **Q: What is the shape and dtype of raw CIFAR-10 data?**
  - **A:** `train_raw.data` is a NumPy array of shape `(50000, 32, 32, 3)` with `uint8` values ranging from 0 to 255.

---

**Tiếng Việt**

# Tài liệu Module EDA (`labs/lab2/eda/eda.py`)

## 1. Tổng quan
Module `eda` thực hiện nhiệm vụ khám phá và thống kê dữ liệu (Exploratory Data Analysis) trên tập dữ liệu CIFAR-10 trong Lab 2. Module này kiểm tra quy mô dữ liệu, tính toán đặc trưng thống kê các kênh màu (mean, std), và xuất ra các biểu đồ mẫu ảnh và phân bố nhãn.

- **Vị trí file:** `labs/lab2/eda/eda.py`
- **Người phụ trách:** Member A
- **Thư viện phụ thuộc:** `torchvision`, `torchvision.datasets`, `matplotlib`, `numpy`, `pathlib`

---

## 2. Hằng số & Cấu trúc dữ liệu

### `CIFAR10_CLASSES`
Bộ tuple chứa tên 10 lớp của CIFAR-10 theo đúng thứ tự nhãn từ 0 đến 9:
```python
CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer", 
    "dog", "frog", "horse", "ship", "truck"
)
```

---

## 3. Các hàm & Giao diện lập trình (API)

### `run_eda(data_root: str, out_dir: str) -> dict`
Thực hiện đọc và phân tích tập dữ liệu CIFAR-10 mà không làm biến đổi dữ liệu, đồng thời lưu các biểu đồ phân tích.

#### Tham số đầu vào:
- `data_root` (`str`): Thư mục lưu trữ hoặc tải về dữ liệu CIFAR-10 thô.
- `out_dir` (`str`): Thư mục lưu các biểu đồ hình ảnh định dạng PNG.

#### Đầu ra:
Một từ điển (dictionary) chứa thông tin thống kê:
- `n_train` (`int`): Số lượng ảnh tập train (50.000).
- `n_test` (`int`): Số lượng ảnh tập test (10.000).
- `class_counts` (`dict[str, int]`): Số lượng mẫu trên mỗi lớp (5.000 mẫu/lớp).
- `image_size` (`tuple`): Kích thước ảnh gốc `(32, 32, 3)`.
- `channel_mean` (`list[float]`): Giá trị trung bình trên từng kênh màu khi chuẩn hóa về `[0, 1]`.
- `channel_std` (`list[float]`): Độ lệch chuẩn trên từng kênh màu khi chuẩn hóa về `[0, 1]`.

#### Biểu đồ sinh ra:
1. `cifar10_sample_grid.png`: Lưới 10 hàng x 3 cột hiển thị 3 ảnh ví dụ cho mỗi lớp.
2. `cifar10_class_distribution.png`: Biểu đồ cột biểu diễn phân phối số lượng mẫu của từng lớp trong tập train.

---

## 4. Chi tiết kỹ thuật & Câu hỏi vấn đáp thường gặp
- **Hỏi: Tại sao chỉ tính mean và std trên tập train mà không gộp tập test?**
  - **Trả lời:** Để tránh rò rỉ thông tin (data leakage). Tập test đóng vai trò dữ liệu chưa từng thấy trong thực tế, mọi thông số tiền xử lý và chuẩn hóa phải tính toán hoàn toàn độc lập từ tập train.
- **Hỏi: Ý nghĩa của lệnh `matplotlib.use("Agg")` là gì?**
  - **Trả lời:** `"Agg"` là backend không tương tác (non-interactive). Nó cho phép vẽ và lưu file ảnh trong môi trường chạy ngầm, terminal hoặc máy chủ không có giao diện hiển thị (GUI) mà không gặp lỗi kết nối màn hình.
- **Hỏi: Dữ liệu thô CIFAR-10 có shape và kiểu dữ liệu gì?**
  - **Trả lời:** `train_raw.data` là một mảng NumPy kích thước `(50000, 32, 32, 3)`, kiểu `uint8` với giá trị pixel từ 0 đến 255.
