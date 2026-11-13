# Preprocessing Module Documentation (`labs/lab2/preprocessing/preprocessing.py`)

## 1. Purpose
- Load raw CIFAR-10 dataset without transforms attached.
- Deterministically partition dataset into training and validation splits.
- Guarantee zero data leakage between train and validation sets (`set_train.isdisjoint(set_val)`).
- Persist split indices to disk as a standardized JSON artifact for reproducible handoff between team members (Member A -> Member B).

## 2. Input - Output
- **Input:**
  - `data_root` (`str`): Path to directory holding raw CIFAR-10 data.
    - Default path: `labs/lab2/raw_data/` (resolved via `shared.paths.raw_data_dir("lab2")`).
  - `download` (`bool`): Flag to automatically download dataset if not present locally (default: `True`).
  - `n_samples` (`int`): Total sample count to partition (50,000 for CIFAR-10 train set).
  - `val_ratio` (`float`): Validation ratio (default: `0.1` -> 10%).
  - `seed` (`int`): Random seed for deterministic shuffling (default: `42`).
  - `path` (`str`): Destination file path for split JSON artifact.
    - Standard path: `labs/lab2/artifacts/splits/train_val_split.json` (resolved via `shared.paths.artifact_dir("lab2", "splits") / "train_val_split.json"`).
- **Output:**
  - In-memory data objects:
    - `(train_set, test_set)`: Raw `torchvision.datasets.CIFAR10` instances with `transform=None`.
    - `(train_idx, val_idx)`: Partitioned integer index lists (45,000 train, 5,000 val).
  - Persistent JSON file saved at:
    - Destination path: `labs/lab2/artifacts/splits/train_val_split.json`
    - Contains: `train_idx` (list of 45,000 ints), `val_idx` (list of 5,000 ints), `n_train` (45000), `n_val` (5000).

## 3. Success_valuable
- Raw datasets are loaded untransformed (`transform is None`), preserving raw image pixels.
- Zero data overlap between train and validation indices (`train_idx ∩ val_idx = ∅`).
- Complete coverage of samples (`len(train_idx) + len(val_idx) == n_samples`).
- Splitting is 100% reproducible across machines and runs when using identical seed (`seed=42`).
- Saved JSON artifact file exists, is valid JSON, and can be cleanly reloaded by downstream modules.

## 4. How to do
Execute the preprocessing pipeline sequentially:
1. **Load raw datasets:** Call `load_raw_dataset(data_root="labs/lab2/raw_data", download=True)` to obtain `train_set` (50,000 samples) and `test_set` (10,000 samples) with raw pixels untouched.
2. **Partition indices:** Call `make_split(n_samples=50000, val_ratio=0.1, seed=42)`. Uses `np.random.RandomState(seed)` to shuffle and divide into 45,000 training and 5,000 validation indices.
3. **Persist split artifact:** Call `save_split(train_idx, val_idx, path="labs/lab2/artifacts/splits/train_val_split.json")` to write the JSON file (automatically creating parent directories if needed).
4. **Handoff & reload:** Downstream training and feature extraction modules call `load_split(path="labs/lab2/artifacts/splits/train_val_split.json")` to retrieve identical indices.

---

**Tiếng Việt**

# Tài liệu Module Tiền Xử Lý (`labs/lab2/preprocessing/preprocessing.py`)

## 1. Purpose (Mục đích)
- Nạp tập dữ liệu CIFAR-10 thô nguyên bản (chưa áp dụng bất kỳ transform nào).
- Phân chia tập dữ liệu thành 2 tập train và validation một cách tất định, có thể tái lập được.
- Đảm bảo không có sự rò rỉ dữ liệu giữa tập train và validation (`set_train.isdisjoint(set_val)`).
- Lưu trữ danh sách chỉ số phân chia (split indices) ra file JSON chuẩn hóa làm căn cứ bàn giao giữa các thành viên (Member A -> Member B).

## 2. Input - Output (Đầu vào - Đầu ra)
- **Input (Đầu vào):**
  - `data_root` (`str`): Đường dẫn thư mục chứa dữ liệu thô CIFAR-10.
    - Đường dẫn chuẩn: `labs/lab2/raw_data/` (truy xuất qua `shared.paths.raw_data_dir("lab2")`).
  - `download` (`bool`): Cho phép tự động tải về nếu dữ liệu chưa có sẵn (mặc định: `True`).
  - `n_samples` (`int`): Tổng số lượng mẫu cần phân chia (50.000 mẫu của tập train CIFAR-10).
  - `val_ratio` (`float`): Tỷ lệ chia tập validation (mặc định: `0.1` tức 10% = 5.000 mẫu).
  - `seed` (`int`): Hạt giống ngẫu nhiên để tái lập phân chia (mặc định: `42`).
  - `path` (`str`): Đường dẫn file lưu split JSON.
    - Đường dẫn chuẩn: `labs/lab2/artifacts/splits/train_val_split.json` (truy xuất qua `shared.paths.artifact_dir("lab2", "splits") / "train_val_split.json"`).
- **Output (Đầu ra):**
  - Đối tượng trong bộ nhớ:
    - `(train_set, test_set)`: Hai bộ `CIFAR10` nguyên bản với `transform=None`.
    - `(train_idx, val_idx)`: Bộ gồm 2 danh sách số nguyên chỉ số mẫu (45.000 mẫu train, 5.000 mẫu val).
  - File JSON lưu trữ trên đĩa cứng tại đường dẫn:
    - Đường dẫn file: `labs/lab2/artifacts/splits/train_val_split.json`
    - Cấu trúc file: `train_idx` (45.000 phần tử), `val_idx` (5.000 phần tử), `n_train` (45000), `n_val` (5000).

## 3. Success_valuable (Tiêu chuẩn đạt yêu cầu)
- Dữ liệu thô được nạp đúng nguyên bản (`transform is None`), không làm méo mó hoặc thay đổi phân phối ảnh.
- Tuyệt đối không rò rỉ dữ liệu giữa tập train và val (`train_idx ∩ val_idx = ∅`).
- Tổng số lượng mẫu được bảo toàn nguyên vẹn (`len(train_idx) + len(val_idx) == n_samples`).
- Kết quả phân chia có tính tái lập 100% khi chạy lại cùng seed (`seed=42`).
- File JSON được tạo hợp lệ, rõ ràng, sẵn sàng để Member B nạp vào huấn luyện.

## 4. How to do (Cách thực hiện)
Quy trình thực thi module gồm các bước:
1. **Nạp dữ liệu thô:** Gọi `load_raw_dataset(data_root="labs/lab2/raw_data", download=True)` để lấy `train_set` (50.000 ảnh) và `test_set` (10.000 ảnh) nguyên bản.
2. **Chia tách chỉ số:** Gọi `make_split(n_samples=50000, val_ratio=0.1, seed=42)`. Hàm sử dụng `np.random.RandomState(seed)` để xáo trộn chỉ số và cắt thành 45.000 chỉ số train và 5.000 chỉ số val.
3. **Lưu file JSON:** Gọi `save_split(train_idx, val_idx, path="labs/lab2/artifacts/splits/train_val_split.json")` để ghi kết quả ra đĩa (tự động tạo thư mục cha `artifacts/splits/` nếu chưa có).
4. **Bàn giao và nạp lại:** Các module sau (`features`, `train`) gọi `load_split(path="labs/lab2/artifacts/splits/train_val_split.json")` để nạp đúng chỉ số đã thống nhất.
