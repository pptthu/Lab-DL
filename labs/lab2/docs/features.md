# Features Module Documentation (`labs/lab2/features/features.py`)

## 1. Overview
The `features` module provides custom dataset wrappers and transformation pipelines for transfer learning on CIFAR-10. Because pre-trained CNN backbones (ResNet, DenseNet, VGG) were trained on ImageNet, images must be resized, converted to tensors, and normalized using standard ImageNet mean and standard deviation. This module also constructs PyTorch `DataLoader` instances with decoupled train/eval transformations.

- **File location:** `labs/lab2/features/features.py`
- **Assigned owner:** Member A
- **Dependencies:** `torch`, `torch.utils.data.DataLoader`, `torch.utils.data.Dataset`, `torchvision.transforms`

---

## 2. Constants & Data Structures

### ImageNet Normalization Constants
```python
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
```
Pre-trained models expect 3-channel RGB inputs normalized per channel with these statistics.

---

## 3. Classes & Functions

### `class TransformedSubset(Dataset)`
A lightweight PyTorch dataset wrapper around an underlying dataset and a list of indices, dynamically applying a specified transform at item retrieval time (`__getitem__`).

- **Constructor:** `__init__(base_dataset, indices: list[int], transform=None)`
- **`__len__()`**: Returns `len(self.indices)`.
- **`__getitem__(idx: int)`**: Retrieves sample at `self.indices[idx]` from `base_dataset`, executes `transform(image)` if defined, and returns `(image, label)`.

### `class TransformedDataset(Dataset)`
A PyTorch dataset wrapper around an entire dataset applying a specified transform dynamically.

- **Constructor:** `__init__(base_dataset, transform=None)`
- **`__getitem__(idx: int)`**: Retrieves sample at index `idx`, executes `transform(image)`, and returns `(image, label)`.

---

### `build_transforms(image_size: int = 224, augment: bool = False) -> tuple`
Builds composition pipelines of torchvision transforms for training and evaluation.

#### Arguments:
- `image_size` (`int`): Target spatial dimension `(image_size, image_size)`. Default is `224` (standard for ImageNet backbones). Smaller sizes (e.g. 64 or 128) can be specified for fast CPU testing.
- `augment` (`bool`): If `True`, enables light data augmentation on the training pipeline.

#### Pipelines:
- **`eval_transform`:**
  - `transforms.Resize((image_size, image_size))`
  - `transforms.ToTensor()`
  - `transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)`
- **`train_transform`:**
  - If `augment=True`: `RandomResizedCrop(image_size, scale=(0.8, 1.0))`, `RandomHorizontalFlip()`, `ToTensor()`, `Normalize`.
  - If `augment=False`: `Resize((image_size, image_size))`, `ToTensor()`, `Normalize`.

#### Returns:
- `(train_transform, eval_transform)`

---

### `make_loaders(train_set, test_set, train_idx: list[int], val_idx: list[int], train_tf, eval_tf, batch_size: int = 64) -> tuple[DataLoader, DataLoader, DataLoader]`
Builds data loaders for training, validation, and testing.

#### Arguments:
- `train_set`: Raw CIFAR-10 training dataset.
- `test_set`: Raw CIFAR-10 test dataset.
- `train_idx`: Indices for the training split.
- `val_idx`: Indices for the validation split.
- `train_tf`: Transformation applied to training images.
- `eval_tf`: Deterministic transformation applied to validation and test images.
- `batch_size` (`int`): Batch size (default: 64).

#### Returns:
- `(train_loader, val_loader, test_loader)`

---

## 4. Key Implementation Details & Defense Q&A
- **Q: Why use `TransformedSubset` instead of PyTorch's built-in `torch.utils.data.Subset`?**
  - **A:** In PyTorch, `torch.utils.data.Subset` delegates `__getitem__` directly to the parent dataset. If the parent dataset already has a transform attached, all subsets share that exact transform. `TransformedSubset` allows the base dataset to remain raw while applying `train_tf` (with augmentation) to the training split and `eval_tf` (without augmentation) to the validation split.
- **Q: Why normalize using ImageNet mean/std rather than CIFAR-10 mean/std?**
  - **A:** The CNN backbones were pre-trained on ImageNet. Their early convolutional filters are tuned to activate properly on input features matching the ImageNet color and contrast distribution. Feeding differently normalized images will disrupt feature extraction.
- **Q: Why is `shuffle=False` for `val_loader` and `test_loader`?**
  - **A:** Validation and test phases are deterministic evaluation passes. Shuffling does not affect loss or accuracy metrics, increases memory overhead, and makes debugging harder.

---

**Tiếng Việt**

# Tài liệu Module Đặc Trưng & Pipeline Dữ Liệu (`labs/lab2/features/features.py`)

## 1. Tổng quan
Module `features` phụ trách xây dựng quy trình tiền xử lý và trích xuất đặc trưng (data transformation pipeline) cho kỹ thuật transfer learning trên CIFAR-10. Do các mạng tiền huấn luyện (ResNet, DenseNet, VGG) được huấn luyện trên ImageNet, ảnh CIFAR-10 cần được đưa về kích thước phù hợp, chuyển sang dạng tensor và chuẩn hóa theo bộ thông số mean và std của ImageNet. Module này cũng đóng gói các `DataLoader` phục vụ quá trình huấn luyện.

- **Vị trí file:** `labs/lab2/features/features.py`
- **Người phụ trách:** Member A
- **Thư viện phụ thuộc:** `torch`, `torch.utils.data.DataLoader`, `torch.utils.data.Dataset`, `torchvision.transforms`

---

## 2. Hằng số & Cấu trúc dữ liệu

### Bộ chuẩn hóa ImageNet
```python
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
```
Mọi ảnh đầu vào 3 kênh RGB cần được chuẩn hóa với các giá trị này để phù hợp với phân phối dữ liệu huấn luyện ban đầu của mô hình tiền huấn luyện.

---

## 3. Các lớp & Hàm chính

### `class TransformedSubset(Dataset)`
Lớp kế thừa `Dataset` của PyTorch dùng để bọc một tập con các chỉ số (subset), đồng thời áp dụng transform trực tiếp khi truy xuất mẫu thông qua hàm `__getitem__`.

- **Mục đích:** Khắc phục nhược điểm của `torch.utils.data.Subset` khi muốn dùng 2 bộ transform khác nhau cho train và validation từ cùng một dataset cha.

### `class TransformedDataset(Dataset)`
Lớp bọc toàn bộ dataset và áp dụng một bộ transform xác định.

---

### `build_transforms(image_size: int = 224, augment: bool = False) -> tuple`
Xây dựng chuỗi các phép biến đổi ảnh (transforms composition) cho tập train và tập đánh giá.

#### Tham số:
- `image_size` (`int`): Kích thước ảnh mục tiêu `(image_size, image_size)`, mặc định `224` theo chuẩn ImageNet. Có thể giảm xuống (ví dụ 64 hoặc 128) để test nhanh trên CPU.
- `augment` (`bool`): Nếu `True`, thêm các kỹ thuật làm giàu dữ liệu (Data Augmentation) gồm cắt ngẫu nhiên có tỉ lệ (`RandomResizedCrop`) và lật ảnh ngang ngẫu nhiên (`RandomHorizontalFlip`).

#### Đầu ra:
- `(train_transform, eval_transform)`: Bộ 2 pipeline transform.

---

### `make_loaders(train_set, test_set, train_idx, val_idx, train_tf, eval_tf, batch_size=64) -> tuple`
Tạo các đối tượng `DataLoader` cho train, validation và test.

- `train_loader`: Sử dụng `train_tf`, bật `shuffle=True` để xáo trộn dữ liệu qua mỗi epoch.
- `val_loader` & `test_loader`: Sử dụng `eval_tf`, đặt `shuffle=False` để giữ thứ tự cố định.

---

## 4. Chi tiết kỹ thuật & Câu hỏi vấn đáp thường gặp
- **Hỏi: Tại sao phải chuẩn hóa theo ImageNet mean/std thay vì mean/std của CIFAR-10?**
  - **Trả lời:** Vì các trọng số (weights) của backbone được học dựa trên phân phối của ImageNet. Để các filter ban đầu kích hoạt chính xác các đặc trưng biên và góc cạnh, dữ liệu đưa vào cần có cùng phân phối chuẩn hóa với dữ liệu tiền huấn luyện.
- **Hỏi: Tại sao tập validation không được áp dụng Data Augmentation?**
  - **Trả lời:** Validation dùng để đo lường khách quan khả năng tổng quát hóa của mô hình trên dữ liệu thực tế. Việc thêm biến đổi ngẫu nhiên vào tập validation sẽ làm mất tính ổn định và tính so sánh của độ đo qua các epoch.
