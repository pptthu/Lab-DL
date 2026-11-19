# preprocessing — Tải dữ liệu và chia train/val (v1)

- **Created**: 2026-11-04T14:49:03+07:00
- **Last Updated**: 2026-11-19T13:35:00+07:00 (Đại)
- **Người phụ trách**: Member A (Đại)
- **Status**: Done

---

## Name
Preprocessing — tải CIFAR-10 thô, chia train/val cố định có thể tái lập, lưu và nạp split JSON

## Background
Đề Practice 2: Cần chuẩn bị dữ liệu cho CNNs. Nhóm sử dụng CIFAR-10 từ `torchvision.datasets`.
Tập validation (10% = 5.000 mẫu) được tách từ 50.000 mẫu train gốc; 10.000 mẫu test giữ nguyên độc lập cho việc đánh giá cuối. Phân chia được cố định qua `train_val_split.json` làm hợp đồng chuyển giao (handoff) giữa Member A và Member B.

## Goals / Purpose
- `load_raw_dataset` trả tuple `(train_set, test_set)` nguyên bản không qua transforms (`transform=None`)
- `make_split` chia phân tách 45.000 train / 5.000 val với `val_ratio=0.1`, seed 42 cố định
- `save_split` lưu JSON cấu trúc `train_idx`, `val_idx`, `n_train`, `n_val` tại `artifacts/splits/train_val_split.json`
- `load_split` nạp lại chính xác danh sách chỉ số từ file JSON
- Không làm:
  - Lấy mẫu validation từ tập test — tránh rò rỉ dữ liệu (test chỉ dùng ở `evaluate`)
  - Resize ảnh, ToTensor hay chuẩn hoá — việc của `features`
  - Thay đổi seed hoặc tỉ lệ split ngẫu nhiên — làm mất tính tái lập của các mốc thí nghiệm
  - Commit mảng dữ liệu thô CIFAR-10 lên git (được loại trừ bởi `.gitignore`)

## Input / Output
- **Input:** `data_root` (thư mục `labs/lab2/raw_data/`), tham số `download` (bool); `n_samples` (50.000), `val_ratio` (0.1), `seed` (42)
- **Output:** `(train_set, test_set)` thô; `train_val_split.json` chứa 45.000 `train_idx` và 5.000 `val_idx`
  - Code nạp qua `torchvision.datasets.CIFAR10` (`preprocessing/preprocessing.py:27-28`), lưu JSON vào `artifacts/splits/train_val_split.json` (`main.py:193-200`).

## How to do it (general plan)
1. `load_raw_dataset`: Nạp CIFAR-10 gốc train và test với `transform=None`
2. `make_split`: Dùng `np.random.RandomState(seed)` hoán vị 50.000 chỉ số, cắt `val_size = int(round(n_samples * val_ratio))`
3. `save_split`: Ghi dict chỉ số ra file JSON với `indent=2`
4. `load_split`: Nạp lại danh sách chỉ số từ file JSON và kiểm tra tính toàn vẹn
5. Smoke run: Kiểm tra `train_idx ∩ val_idx = ∅`, tổng phần tử đúng 50.000, 2 lần gọi cùng seed cho kết quả y hệt
6. Normal run trong pipeline `main.py`: nếu file JSON đã tồn tại thì nạp lại, nếu chưa thì tạo mới

## Pipeline
```
data_root → load_raw_dataset → train_set (50k), test_set (10k)
  → make_split(seed=42) → train_idx (45k), val_idx (5k)
  → save_split → train_val_split.json
  → load_split (handoff cho features)
```

## Detailed plan
- **Tái lập tất định (Deterministic split):** Dùng `np.random.RandomState(seed)` cục bộ trong hàm `make_split`, tránh can thiệp seed toàn cục của NumPy (`preprocessing/preprocessing.py:45`)
- **Chống rò rỉ dữ liệu:** `set(train_idx).isdisjoint(set(val_idx))` và `len(train_idx) + len(val_idx) == n_samples`
- **Hàm hiện có:**
  - `load_raw_dataset(data_root: str, download: bool = True) -> tuple` (`preprocessing/preprocessing.py:17`)
  - `make_split(n_samples: int, val_ratio: float = 0.1, seed: int = 42) -> tuple` (`:32`)
  - `save_split(train_idx: list[int], val_idx: list[int], path: str) -> None` (`:54`)
  - `load_split(path: str) -> tuple[list[int], list[int]]` (`:76`)
- **Chỗ code chưa khớp plan:** không có (mọi hàm đúng hợp đồng)
- **Gotchas:**
  - Kiểm tra `val_ratio` trong khoảng `(0.0, 1.0)`, sai ném `ValueError` (`preprocessing/preprocessing.py:42-43`)
  - Đường dẫn file split JSON không tồn tại khi gọi `load_split` sẽ ném `FileNotFoundError` (`:87-88`)

## Success criteria
- **Test:** `python -m pytest labs/lab2/tests/test_preprocessing.py -q` đạt (1 test)
- **Smoke run:** Tạo split từ 50.000 mẫu, kiểm tra file `train_val_split.json` được tạo, đọc lại khớp 100%
- **Normal run:** 45.000 train / 5.000 val / 10.000 test; file `train_val_split.json` được lưu cố định tại `artifacts/splits/`

## Links
- Desc lab: [../project_description.md](../project_description.md)
- Module liên quan: [eda.md](eda.md), [features.md](features.md)
- Tiến độ và bằng chứng chạy: [../progress_status.md](../progress_status.md) (mục Handoff A -> B dòng 17, Audit dòng 28)
- Code: `labs/lab2/preprocessing/preprocessing.py` | Test: `labs/lab2/tests/test_preprocessing.py`
