# Preprocessing Module Documentation (`labs/lab2/preprocessing/preprocessing.py`)

## 1. Overview
The `preprocessing` module handles raw data loading without transforms, deterministic train/validation index splitting, and persisting split indices to disk as JSON. This ensures that experimental splits remain reproducible across runs and collaborators.

- **File location:** `labs/lab2/preprocessing/preprocessing.py`
- **Assigned owner:** Member A
- **Dependencies:** `torchvision.datasets`, `numpy`, `json`, `pathlib`

---

## 2. Functions & API Contract

### `load_raw_dataset(data_root: str, download: bool = True) -> tuple`
Loads the CIFAR-10 dataset without transforms attached.

#### Arguments:
- `data_root` (`str`): Path where dataset is stored.
- `download` (`bool`): Whether to download CIFAR-10 if not present locally (default: `True`).

#### Returns:
- `(train_set, test_set)`: A tuple of two `torchvision.datasets.CIFAR10` instances with `transform=None`.

---

### `make_split(n_samples: int, val_ratio: float = 0.1, seed: int = 42) -> tuple[list[int], list[int]]`
Generates reproducible, randomized train and validation split index lists.

#### Arguments:
- `n_samples` (`int`): Total count of samples to split (e.g., 50,000 for CIFAR-10 training set).
- `val_ratio` (`float`): Proportion assigned to validation set, where `0.0 < val_ratio < 1.0` (default: `0.1`).
- `seed` (`int`): Random seed for `np.random.RandomState` to guarantee deterministic partitioning (default: `42`).

#### Returns:
- `(train_idx, val_idx)`: Tuple of integer lists corresponding to dataset indices for training and validation subsets.

#### Raises:
- `ValueError`: If `val_ratio` is not strictly between 0.0 and 1.0.

---

### `save_split(train_idx: list[int], val_idx: list[int], path: str) -> None`
Saves the split indices into a structured JSON file.

#### Arguments:
- `train_idx` (`list[int]`): List of training sample indices.
- `val_idx` (`list[int]`): List of validation sample indices.
- `path` (`str`): Destination file path for the JSON artifact (e.g., `artifacts/splits/train_val_split.json`).

#### Output JSON format:
```json
{
  "train_idx": [ ... ],
  "val_idx": [ ... ],
  "n_train": 45000,
  "n_val": 5000
}
```

---

### `load_split(path: str) -> tuple[list[int], list[int]]`
Reads previously saved train/val split indices from disk.

#### Arguments:
- `path` (`str`): File path to the JSON split file.

#### Returns:
- `(train_idx, val_idx)`: Reconstructed tuple of index lists.

#### Raises:
- `FileNotFoundError`: If the file path does not exist.

---

## 3. Key Implementation Details & Defense Q&A
- **Q: Why separate raw loading and transform application?**
  - **A:** Applying transforms during dataset loading couples data representation to a specific pipeline. By keeping raw dataset loading transform-free, the training set can be cleanly split into train and validation subsets, allowing training data to use data augmentation while validation uses deterministic evaluation transforms.
- **Q: Why save the split indices to a JSON file?**
  - **A:** In collaborative projects, teammate Member A and teammate Member B must train and evaluate on the exact same subsets. Hardcoding or generating splits on the fly without persistence can cause subtle discrepancies if different environments or random seeds are used.
- **Q: Why use `np.random.RandomState(seed)` instead of `np.random.seed(seed)`?**
  - **A:** Using an isolated `RandomState` object scopes the random number generation locally and avoids mutating the global NumPy random state, which could interfere with other libraries or scripts.

---

**Tiếng Việt**

# Tài liệu Module Tiền Xử Lý (`labs/lab2/preprocessing/preprocessing.py`)

## 1. Tổng quan
Module `preprocessing` phụ trách việc nạp dữ liệu thô (chưa kèm transform), chia tách tập train/validation theo tỷ lệ xác định với tính tái lập cao (reproducible), và lưu/nạp danh sách chỉ số (split indices) dưới định dạng JSON.

- **Vị trí file:** `labs/lab2/preprocessing/preprocessing.py`
- **Người phụ trách:** Member A
- **Thư viện phụ thuộc:** `torchvision.datasets`, `numpy`, `json`, `pathlib`

---

## 2. Các hàm & Giao diện lập trình (API)

### `load_raw_dataset(data_root: str, download: bool = True) -> tuple`
Nạp tập dữ liệu CIFAR-10 thô không kèm bất kỳ phép biến đổi (transform) nào.

#### Tham số đầu vào:
- `data_root` (`str`): Đường dẫn lưu trữ thư mục dữ liệu.
- `download` (`bool`): Có tải dữ liệu về nếu chưa tồn tại hay không (mặc định: `True`).

#### Đầu ra:
- `(train_set, test_set)`: Bộ tuple chứa 2 đối tượng `torchvision.datasets.CIFAR10` với tham số `transform=None`.

---

### `make_split(n_samples: int, val_ratio: float = 0.1, seed: int = 42) -> tuple[list[int], list[int]]`
Tạo danh sách chỉ số mẫu cho tập train và validation một cách ngẫu nhiên nhưng có thể tái lập được nhờ cố định seed.

#### Tham số đầu vào:
- `n_samples` (`int`): Tổng số mẫu cần chia (50.000 mẫu đối với tập train CIFAR-10).
- `val_ratio` (`float`): Tỷ lệ dành cho tập validation (mặc định: `0.1`, tức 10% = 5.000 mẫu).
- `seed` (`int`): Hạt giống ngẫu nhiên cho `np.random.RandomState` (mặc định: `42`).

#### Đầu ra:
- `(train_idx, val_idx)`: Bộ gồm 2 danh sách số nguyên chứa chỉ số mẫu của tập train và tập val.

---

### `save_split(train_idx: list[int], val_idx: list[int], path: str) -> None`
Ghi danh sách chỉ số split ra file JSON trên đĩa cứng, tự tạo thư mục cha nếu chưa có.

#### Tham số đầu vào:
- `train_idx` (`list[int]`): Danh sách chỉ số tập train.
- `val_idx` (`list[int]`): Danh sách chỉ số tập validation.
- `path` (`str`): Đường dẫn file JSON cần lưu (ví dụ: `artifacts/splits/train_val_split.json`).

---

### `load_split(path: str) -> tuple[list[int], list[int]]`
Đọc lại danh sách chỉ số split từ file JSON đã lưu từ trước.

#### Tham số đầu vào:
- `path` (`str`): Đường dẫn file JSON cần nạp.

#### Đầu ra:
- `(train_idx, val_idx)`: Danh sách chỉ số tập train và tập val.

---

## 3. Chi tiết kỹ thuật & Câu hỏi vấn đáp thường gặp
- **Hỏi: Tại sao không gán transform ngay khi tạo dataset mà phải tách riêng?**
  - **Trả lời:** Nếu gán transform vào dataset ban đầu, cả tập train và tập validation sẽ dùng chung một pipeline transform. Việc tách rời cho phép ta dùng Dataset Wrapper để áp dụng Data Augmentation riêng cho train, trong khi validation chỉ dùng phép resize và chuẩn hóa cố định.
- **Hỏi: Việc lưu file split ra JSON có ý nghĩa gì khi làm việc nhóm?**
  - **Trả lời:** Giúp đảm bảo sự nhất quán tuyệt đối giữa Member A và Member B. Khi Member B thực hiện huấn luyện và so sánh mô hình, họ sẽ dùng đúng tập dữ liệu validation mà Member A đã định nghĩa mà không bị xáo trộn.
- **Hỏi: Tại sao dùng `np.random.RandomState(seed)` thay vì `np.random.seed(seed)`?**
  - **Trả lời:** `RandomState` tạo ra một bộ sinh số ngẫu nhiên cô lập cục bộ, không làm thay đổi trạng thái ngẫu nhiên toàn cục của chương trình, tránh gây ảnh hưởng ngoài ý muốn đến các module khác.
