# Features Module Documentation (`labs/lab2/features/features.py`)

## 1. Purpose
- Build torchvision image transformation pipelines for pre-trained CNN backbones.
- Apply ImageNet normalization (`IMAGENET_MEAN`, `IMAGENET_STD`) and spatial resizing.
- Provide optional Data Augmentation (RandomResizedCrop, RandomHorizontalFlip) solely for training.
- Wrap subsets dynamically via `TransformedSubset` to prevent data leakage and keep raw datasets unmodified.
- Construct PyTorch `DataLoader` instances for train, validation, and test sets.

## 2. Input - Output
- **Input:**
  - `train_set`, `test_set`: Raw CIFAR-10 datasets loaded via `preprocessing.load_raw_dataset`.
    - **Source directory:** `labs/lab2/raw_data/` (resolved via `shared.paths.raw_data_dir("lab2")`).
  - `train_idx`, `val_idx` (`list[int]`): Partitioned sample index lists loaded from split JSON artifact:
    - **Standard path:** `labs/lab2/artifacts/splits/train_val_split.json` (via `preprocessing.load_split`).
  - `image_size` (`int`): Spatial resolution, default `224` (or `32` / `64` for faster CPU testing).
  - `augment` (`bool`): Enables training data augmentation (default: `False`).
  - `batch_size` (`int`): Mini-batch size (default: `64`).
- **Output:**
  - Transformation pipelines:
    - `train_transform`: Resize + (Augmentation if enabled) + ToTensor + ImageNet Normalization.
    - `eval_transform`: Resize + ToTensor + ImageNet Normalization.
  - In-memory PyTorch DataLoaders:
    - `train_loader`: 704 batches (with `shuffle=True`, `num_workers=0`).
    - `val_loader`: 79 batches (with `shuffle=False`, `num_workers=0`).
    - `test_loader`: 157 batches (with `shuffle=False`, `num_workers=0`).
  - Batch data specifications (Milestone Handoff to Member B):
    - `images`: Tensor of shape `[B, 3, H, W]` with dtype `torch.float32`.
    - `labels`: Tensor of shape `[B]` containing integer class targets in range `[0, 9]`.

## 3. Success
- Transforms output tensors of shape `(3, image_size, image_size)` with dtype `torch.float32`.
- Normalization matches ImageNet statistics (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
- Training transformations with augmentation do not contaminate validation or test evaluations.
- DataLoaders operate safely across operating systems with `num_workers=0` (preventing multiprocessing deadlocks on Windows).
- Handoff criteria satisfied: `next(iter(train_loader))` yields `[B, 3, H, W]` images and valid labels `0-9`.

## 4. How to do
Execute the features pipeline sequentially:
1. **Load raw datasets and split indices:**
   - Retrieve `train_set` and `test_set` from `labs/lab2/raw_data/`.
   - Read `train_idx` (45,000) and `val_idx` (5,000) from `labs/lab2/artifacts/splits/train_val_split.json` using `load_split()`.
2. **Build transformation pipelines:**
   - Call `build_transforms(image_size=224, augment=False)` to obtain `(train_transform, eval_transform)`.
3. **Assemble DataLoaders:**
   - Call `make_loaders(train_set, test_set, train_idx, val_idx, train_tf, eval_tf, batch_size=64)`.
   - Internally wraps data using `TransformedSubset` for train and val splits, and `TransformedDataset` for test split.
4. **Deliver to downstream training:**
   - Provide `train_loader`, `val_loader`, and `test_loader` directly to the `train` module.

---

**Tiếng Việt**

# Tài liệu Module Đặc Trưng & Pipeline Dữ Liệu (`labs/lab2/features/features.py`)

## 1. Purpose (Mục đích)
- Xây dựng pipeline biến đổi ảnh (torchvision transforms) phù hợp cho các mạng CNN pre-trained.
- Chuẩn hóa ảnh theo thông số ImageNet (`IMAGENET_MEAN`, `IMAGENET_STD`) và thay đổi kích thước ảnh (resizing).
- Cung cấp tùy chọn làm giàu dữ liệu (Data Augmentation: RandomResizedCrop, RandomHorizontalFlip) chỉ riêng cho tập train.
- Tạo các lớp bọc dữ liệu động (`TransformedSubset`) giúp áp dụng transform riêng biệt mà không làm thay đổi dataset thô ban đầu.
- Tạo các đối tượng PyTorch `DataLoader` (train, val, test) phục vụ huấn luyện và đánh giá.

## 2. Input - Output (Đầu vào - Đầu ra)
- **Input (Đầu vào):**
  - `train_set`, `test_set`: Dữ liệu CIFAR-10 thô nạp qua `preprocessing.load_raw_dataset`.
    - **Thư mục nguồn dữ liệu:** `labs/lab2/raw_data/` (truy xuất qua `shared.paths.raw_data_dir("lab2")`).
  - `train_idx`, `val_idx` (`list[int]`): Danh sách chỉ số mẫu nạp từ file JSON phân chia:
    - **Đường dẫn chuẩn:** `labs/lab2/artifacts/splits/train_val_split.json` (thông qua `preprocessing.load_split`).
  - `image_size` (`int`): Kích thước ảnh mục tiêu, mặc định `224` (hoặc `32` / `64` nếu chạy test nhanh trên CPU).
  - `augment` (`bool`): Có bật Data Augmentation cho tập train hay không (mặc định: `False`).
  - `batch_size` (`int`): Kích thước mỗi batch dữ liệu (mặc định: `64`).
- **Output (Đầu ra):**
  - Các pipeline biến đổi ảnh:
    - `train_transform`: Resize + (Augmentation nếu bật) + ToTensor + ImageNet Normalization.
    - `eval_transform`: Resize + ToTensor + ImageNet Normalization.
  - Các đối tượng PyTorch DataLoader trong bộ nhớ:
    - `train_loader`: 704 batch (bật `shuffle=True`, `num_workers=0`).
    - `val_loader`: 79 batch (tắt `shuffle=False`, `num_workers=0`).
    - `test_loader`: 157 batch (tắt `shuffle=False`, `num_workers=0`).
  - Dữ liệu mỗi batch trả về (Mốc bàn giao cho Member B):
    - `images`: Tensor kích thước `[B, 3, H, W]`, kiểu dữ liệu `torch.float32`.
    - `labels`: Tensor kích thước `[B]`, chứa các nhãn nguyên trong khoảng `[0, 9]`.

## 3. Success_valuable (Tiêu chuẩn đạt yêu cầu)
- Đầu ra của transform tạo thành công Tensor kích thước `(3, image_size, image_size)` kiểu `torch.float32`.
- Chuẩn hóa chính xác theo phân phối ImageNet (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
- Các phép Data Augmentation không làm rò rỉ hoặc can thiệp vào quá trình đánh giá của tập validation và test.
- DataLoader hoạt động ổn định trên môi trường Windows nhờ thiết lập an toàn `num_workers=0`.
- Thỏa mãn điều kiện bàn giao nhóm: Lấy batch đầu tiên `next(iter(train_loader))` trả đúng shape `[B, 3, H, W]` và nhãn từ 0 đến 9.

## 4. How to do (Cách thực hiện)
Quy trình thực thi module gồm các bước tuần tự:
1. **Nạp dữ liệu thô và chỉ số split:**
   - Lấy `train_set` và `test_set` từ thư mục `labs/lab2/raw_data/`.
   - Đọc danh sách `train_idx` (45.000) và `val_idx` (5.000) từ file `labs/lab2/artifacts/splits/train_val_split.json` bằng hàm `load_split()`.
2. **Khởi tạo transform:**
   - Gọi `build_transforms(image_size=224, augment=False)` để nhận về bộ `(train_transform, eval_transform)`.
3. **Tạo DataLoaders:**
   - Gọi `make_loaders(train_set, test_set, train_idx, val_idx, train_tf, eval_tf, batch_size=64)`.
   - Module tự động bọc dữ liệu qua `TransformedSubset` cho tập train và val, và `TransformedDataset` cho tập test.
4. **Bàn giao sang module huấn luyện:**
   - Cung cấp trực tiếp các đối tượng `train_loader`, `val_loader`, `test_loader` cho module `train`.
