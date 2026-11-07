# Lab 1 / preprocessing: Data source, loading and split (phase 2 of 4, data part)

- **Motivation/Background**: The old Kaggle CSV did not follow the official train/test split; every result depends on the data source and on one shared split.
- **Purpose**: Specify the `preprocessing` module and `download_official.py`: data source, CSV loading, the stratified train/validation split and the single split policy.
- **Overview Pipeline**: IDX files (torchvision, MD5) -> 2 CSV -> `load_csv` -> `make_split` -> `save_split` -> `train_val_split.json` (committed, read-only) -> `features`.
- **Detailed Plan**: Name; Purpose; Goals (Mục tiêu); Input / Output; How to do it (general plan); Pipeline; Public API; Success criteria & retrial; Smoke / Normal run; Risks; Detailed plan / gotchas; Integration; Links; Vietnamese version.
- **References**: [STATUS.md](STATUS.md), [PROMPTS.md](PROMPTS.md), [AUDIT.md](AUDIT.md), torchvision (`datasets.FashionMNIST`), pandas, numpy, `shared/paths.py`, `labs/lab1/preprocessing/preprocessing.py`, `labs/lab1/preprocessing/download_official.py`, `labs/lab1/tests/test_preprocessing.py`.
- **Created**: 2026-10-13T00:00:00+07:00
- **Last Updated**: 2026-11-07T00:00:00+07:00

---

## Table of Contents

- [Name](#name)
- [Purpose](#purpose)
- [Goals (Mục tiêu)](#goals-mục-tiêu)
- [Input / Output](#input--output)
- [How to do it (general plan)](#how-to-do-it-general-plan)
- [Pipeline](#pipeline)
- [Public API](#public-api)
- [Success criteria & retrial](#success-criteria--retrial)
- [Smoke / Normal run](#smoke--normal-run)
- [Risks](#risks)
- [Detailed plan / gotchas](#detailed-plan--gotchas)
- [Integration](#integration)
- [Links](#links)
- [Tiếng Việt](#tiếng-việt)

---

## Name

- **Title:** Data Source, Preprocessing & Split
- **Execution order:** 2 of 4 (eda -> preprocessing -> features -> model); `download_official.py` runs once before everything
- **Created:** 2026-10-13, author: Toàn (Member A)
- **Updated:** 2026-11-07 by Toàn (Member A)
- **Description:** Download the official Fashion-MNIST, read the CSV into checked arrays, split train/validation once (stratified) and save the split for the whole team.
- **Status:** Done (Run on the official data, 2026-11-02 and 2026-11-04)
- **Owner:** Member A (Toàn), assigned by the team leader

## Purpose

The old Kaggle CSV did not follow the official train/test split (most Kaggle test images are official training images; 43 duplicate rows), so on 2026-11-02 the team switched to the official Zalando files. Validation must come from the training data so the test set is used only once at the end. If every run re-split the data, experiments could not be compared, so the split is made **once**, saved, and treated as read-only (single split policy).

## Goals (Mục tiêu)

- Done means:
  - `raw_data/fashion-mnist_train.csv` (60,000 rows) and `fashion-mnist_test.csv` (10,000 rows), columns `label`, `pixel1`..`pixel784`, identical to the IDX files pixel by pixel.
  - `load_csv` returns images `uint8 [N,28,28]` and labels `int64 [N]`; wrong columns or values outside 0-255 raise an error.
  - `make_split`: 54,000 / 6,000 (600 per class in validation), disjoint, covering all 60,000.
  - `artifacts/splits/train_val_split.json` committed; regenerating it gives the identical file.
- Not in this phase: normalization ([features.md](../features/features.md)); splitting the test file (the 10,000 official test images stay the test set).

## Input / Output

- **Input:** 4 IDX files from https://github.com/zalandoresearch/fashion-mnist, fetched by `torchvision.datasets.FashionMNIST(download=True)` (MD5 checked) into `labs/lab1/raw_data/FashionMNIST/raw/` (not committed).
- **Input of `load_csv`:** `labs/lab1/raw_data/fashion-mnist_train.csv` and `fashion-mnist_test.csv` (written by `download_official.py`).
- **Output:**
  - `labs/lab1/raw_data/fashion-mnist_train.csv`, `fashion-mnist_test.csv` (not committed; old Kaggle files renamed `*_kaggle.csv`, unused).
  - arrays in memory.
  - `labs/lab1/artifacts/splits/train_val_split.json` = `{"train_idx": [...], "val_idx": [...]}` (**committed**).

## How to do it (general plan)

1. `download_official.py`: torchvision downloads the IDX files and checks their MD5; reshape `ds.data` to `[-1, 784]`, insert `label`, write CSV; read back with `load_csv` and compare every pixel and label (`np.array_equal`), check N/10 per class.
2. `load_csv`: check the `label` column and 784 pixel columns; check 0-255 **before** casting to `uint8` (uint8 wraps 256 -> 0); reshape to `[-1,28,28]`.
3. `make_split(labels, val_ratio=0.1, seed=42)`: per class, a seeded permutation; `round(n * 0.1)` indices go to validation; train = the rest (`setdiff1d`).
4. `save_split`: numpy ints -> Python ints, write JSON. `main.py` and the notebook call `load_split` if the file exists and only create it otherwise.

## Pipeline

```
python -m labs.lab1.preprocessing.download_official
  -> torchvision FashionMNIST (IDX + MD5) -> 2 CSV -> self-check with load_csv
python -m labs.lab1.preprocessing.preprocessing
  -> load_csv(train) -> make_split(labels, 0.1, 42) -> save_split -> train_val_split.json
main.py: load_csv(train), load_csv(test) -> load_split(json)
```

## Public API

### `load_csv(csv_path: str) -> tuple`

**Parameters:** `csv_path` (str): path to a Fashion-MNIST CSV.
**Returns:** `(images, labels)`: `uint8` array `[N,28,28]`, `int64` array `[N]`.
**Raises:** `ValueError`: no `label` column; pixel columns != 784; values outside 0-255.

### `make_split(labels, val_ratio: float = 0.1, seed: int = 42) -> tuple`

**Parameters:** `labels` (int64 array `[N]`); `val_ratio` (float in (0, 1)); `seed` (int).
**Returns:** `(train_idx, val_idx)`: sorted, disjoint `int64` index arrays covering all N samples; stratified per class.
**Raises:** `ValueError`: `val_ratio` not in (0, 1).

### `save_split(train_idx, val_idx, path: str) -> None`

**Parameters:** index arrays; `path` (.json). Parent folder is created if missing.
**Returns:** None; writes `{"train_idx": [...], "val_idx": [...]}`.

### `load_split(path: str) -> tuple`

**Parameters:** `path` (.json).
**Returns:** `(train_idx, val_idx)` as `int64` arrays.
**Raises:** `FileNotFoundError` if the file does not exist.

### Script `download_official.py`

`python -m labs.lab1.preprocessing.download_official`: no arguments; writes the 2 CSV files and asserts they match the IDX files.

**Example:**
```python
from labs.lab1.preprocessing.preprocessing import load_csv, load_split
from shared.paths import raw_data_dir, artifact_dir

images, labels = load_csv(str(raw_data_dir("lab1") / "fashion-mnist_train.csv"))
train_idx, val_idx = load_split(str(artifact_dir("lab1", "splits") / "train_val_split.json"))
print(images.shape, len(train_idx), len(val_idx))   # (60000, 28, 28) 54000 6000
```

## Success criteria & retrial

| Criterion | Correct value | If not met |
|---|---|---|
| CSV vs IDX | identical pixels and labels; 6,000 / 1,000 per class | the script's asserts stop it; delete the CSV and run again; still wrong -> tell the team leader |
| CSV MD5 | train `8ff2cced1c624159b21dac4f735b4215`, test `78e18be2098f1256a07ce6f91a4d436e` | check the pandas version / how the CSV is written |
| dtype / shape | `uint8 [60000,28,28]`, `int64 [60000]` | fix `load_csv`, rerun the tests |
| Split sizes | 54,000 / 6,000, 600 per class in validation | check `val_ratio` and `seed` |
| Disjoint, complete | 0 shared indices, union = 60,000 | the unit test fails; fix `make_split` |
| Reproducible | regenerated split = committed file | **never overwrite the JSON**; find the cause (different CSV? different seed?) |

## Smoke / Normal run

| Type | Command | Expected |
|---|---|---|
| Unit test | `python -m pytest labs/lab1/tests/test_preprocessing.py -q` | 6 tests pass (fake CSV, no download) |
| Smoke run | regenerate `make_split(labels, 0.1, 42)` from the CSV and compare with `train_val_split.json` | identical, 54,000 / 6,000 |
| Normal run | `python -m labs.lab1.preprocessing.download_official` (only when the CSV is missing); `python -m labs.lab1.main` stage 2 | MD5 as above; "Split loaded: train=54000, val=6000, test=10000" |

## Risks

- Overwriting `train_val_split.json` -> every reported number changes silently; the file is read-only, regenerate only with the team leader.
- Casting to `uint8` before the range check -> 256 wraps to 0 silently; the check runs first.
- Validation taken from the test file -> test no longer independent; validation comes from the 60,000 training images only.
- Different CSV on another machine -> different split; MD5 of both CSV files recorded.

## Detailed plan / gotchas

- Never write data paths by hand; use `raw_data_dir("lab1")` from `shared/paths.py`.
- On Windows with a Vietnamese path, set `PYTHONIOENCODING=utf-8` before running the scripts, otherwise the last `print` can fail with `UnicodeEncodeError` (the files are written before it).
- Stratification keeps validation balanced; a random split would move each class by about ±22 images around 600.
- 6,000 validation images give accuracy a standard error of about ±0.4 percentage points; smaller differences are not conclusive (why `experiment` runs 5 seeds).
- **Rules for AI agents:** may edit `preprocessing.py`, `download_official.py`, `tests/test_preprocessing.py`, this file. Must not edit other modules, raw data, other owners' notebook cells; `train_val_split.json` only on purpose, with the team leader. Signatures `load_csv`, `make_split`, `save_split`, `load_split` are a contract (`project_description.md`). Prohibited: validation from the test file; changing `val_ratio` or `seed` without the team leader; committing CSV/IDX files; downloads inside unit tests.

## Integration

This module is used by:
- `features` (arrays and indices)
- `labs/lab1/main.py` stage 2, `labs/lab1/extended_study.py`
- `notebook/lab1_presentation.ipynb` Stage 2 (cell `[3]`)

## Links

- Progress: `labs/lab1/progress_status.md`
- Previous / next phase: [eda.md](../eda/eda.md) / [features.md](../features/features.md)

---

## Tiếng Việt

### Tên

- **Tên:** Nguồn dữ liệu, tiền xử lý và chia dữ liệu
- **Thứ tự thực hiện:** 2 trên 4 (eda → preprocessing → features → model); `download_official.py` chạy một lần trước tất cả
- **Tạo:** 13/10/2026, tác giả Toàn (Member A)
- **Cập nhật:** 07/11/2026 bởi Toàn (Member A)
- **Mô tả:** tải Fashion-MNIST chính thức, đọc CSV thành mảng đã kiểm, chia train/validation một lần (phân tầng) và lưu split cho cả nhóm.
- **Trạng thái:** xong (đã chạy trên dữ liệu chính thức, 02/11/2026 và 04/11/2026)
- **Người phụ trách:** Member A (Toàn), do nhóm trưởng giao. Không tự giao việc cho mình; xem GitHub Issue và `progress_status.md`, hỏi nhóm trưởng.

### Purpose (lý do)

CSV Kaggle cũ không theo cách chia train/test chính thức (phần lớn ảnh test Kaggle là ảnh train chính thức; 43 dòng trùng), nên từ 02/11/2026 nhóm dùng file gốc của Zalando. Validation phải lấy từ dữ liệu train để tập test chỉ dùng một lần ở cuối. Nếu mỗi lần chạy lại chia một kiểu, các thí nghiệm không so được với nhau, nên split chỉ tạo **một lần**, lưu file, coi là read-only (single split policy).

### Mục tiêu

- Xong khi:
  - Có `raw_data/fashion-mnist_train.csv` (60.000 dòng) và `fashion-mnist_test.csv` (10.000 dòng), cột `label`, `pixel1`..`pixel784`, khớp từng pixel với file IDX.
  - `load_csv` trả ảnh `uint8 [N,28,28]` và nhãn `int64 [N]`; sai cột hoặc giá trị ngoài 0–255 thì báo lỗi.
  - `make_split`: 54.000 / 6.000 (600 ảnh mỗi lớp ở validation), không giao nhau, gộp lại đủ 60.000.
  - `artifacts/splits/train_val_split.json` đã commit; tạo lại ra đúng y file đó.
- Không làm: chuẩn hóa ([features.md](../features/features.md)); chia file test (10.000 ảnh test chính thức giữ nguyên làm tập test).

### Input / Output

- **Input:** 4 file IDX từ https://github.com/zalandoresearch/fashion-mnist, tải bằng `torchvision.datasets.FashionMNIST(download=True)` (kiểm MD5) vào `labs/lab1/raw_data/FashionMNIST/raw/` (không commit).
- **Input của `load_csv`:** `labs/lab1/raw_data/fashion-mnist_train.csv` và `fashion-mnist_test.csv` (do `download_official.py` ghi ra).
- **Output:**
  - `labs/lab1/raw_data/fashion-mnist_train.csv`, `fashion-mnist_test.csv` (không commit; file Kaggle cũ đổi tên `*_kaggle.csv`, không dùng).
  - Mảng trong bộ nhớ.
  - `labs/lab1/artifacts/splits/train_val_split.json` = `{"train_idx": [...], "val_idx": [...]}` (**có commit**).

### Cách làm

1. `download_official.py`: torchvision tải file IDX và kiểm MD5; reshape `ds.data` về `[-1, 784]`, chèn cột `label`, ghi CSV; đọc lại bằng `load_csv`, so từng pixel và nhãn (`np.array_equal`), kiểm mỗi lớp đúng N/10.
2. `load_csv`: kiểm cột `label` và 784 cột pixel; kiểm 0–255 **trước** khi ép `uint8` (uint8 quay vòng 256 → 0); reshape về `[-1,28,28]`.
3. `make_split(labels, val_ratio=0.1, seed=42)`: mỗi lớp xáo bằng hoán vị có seed; `round(n * 0.1)` chỉ số vào validation; train là phần còn lại (`setdiff1d`).
4. `save_split`: đổi số nguyên numpy sang int Python, ghi JSON. `main.py` và notebook gọi `load_split` nếu file đã có, chỉ tạo mới khi chưa có.

### Pipeline

Như sơ đồ ở phần tiếng Anh: `download_official` → 2 CSV → `load_csv` → `make_split` → `save_split` → `train_val_split.json`; `main.py` chỉ `load_split`.

### Public API (hàm công khai)

| Hàm | Tham số | Trả về | Báo lỗi |
|---|---|---|---|
| `load_csv(csv_path: str)` | đường dẫn CSV | `(images uint8 [N,28,28], labels int64 [N])` | `ValueError`: thiếu cột `label`, số cột pixel khác 784, giá trị ngoài 0–255 |
| `make_split(labels, val_ratio=0.1, seed=42)` | mảng nhãn `int64 [N]`, tỉ lệ val trong (0, 1), seed | `(train_idx, val_idx)`: mảng chỉ số `int64` đã sắp, không giao nhau, phân tầng theo lớp | `ValueError`: `val_ratio` ngoài (0, 1) |
| `save_split(train_idx, val_idx, path: str)` | 2 mảng chỉ số, đường dẫn `.json` (tự tạo thư mục cha) | None; ghi file JSON | — |
| `load_split(path: str)` | đường dẫn `.json` | `(train_idx, val_idx)` kiểu `int64` | `FileNotFoundError` nếu file không tồn tại |
| script `download_official.py` | không có tham số | ghi 2 CSV và tự assert khớp IDX | `AssertionError` nếu không khớp |

Ví dụ: xem đoạn code ở phần tiếng Anh (in ra `(60000, 28, 28) 54000 6000`).

### Tiêu chí thành công & làm lại

| Tiêu chí | Giá trị đúng | Không đạt thì |
|---|---|---|
| CSV so với IDX | khớp từng pixel, từng nhãn; 6.000 / 1.000 mỗi lớp | assert trong script dừng lại; xóa CSV, chạy lại; vẫn sai → báo nhóm trưởng |
| MD5 của CSV | train `8ff2cced1c624159b21dac4f735b4215`, test `78e18be2098f1256a07ce6f91a4d436e` | kiểm phiên bản pandas / cách ghi CSV |
| dtype / shape | `uint8 [60000,28,28]`, `int64 [60000]` | sửa `load_csv`, chạy lại test |
| Kích thước split | 54.000 / 6.000, 600 mỗi lớp ở validation | kiểm `val_ratio`, `seed` |
| Không giao nhau, đủ | 0 chỉ số chung, gộp đủ 60.000 | unit test báo lỗi; sửa `make_split` |
| Tái lập | split tạo lại = file đã commit | **không ghi đè JSON**; tìm nguyên nhân (CSV khác? seed khác?) |

### Smoke / Normal run

| Loại | Lệnh | Kết quả đúng |
|---|---|---|
| Unit test | `python -m pytest labs/lab1/tests/test_preprocessing.py -q` | 6 test đạt (CSV giả, không tải) |
| Smoke run | tạo lại `make_split(labels, 0.1, 42)` từ CSV rồi so với `train_val_split.json` | giống hệt, 54.000 / 6.000 |
| Normal run | `python -m labs.lab1.preprocessing.download_official` (chỉ khi chưa có CSV); `python -m labs.lab1.main` stage 2 | MD5 như trên; "Split loaded: train=54000, val=6000, test=10000" |

### Rủi ro

- Ghi đè `train_val_split.json` -> mọi số báo cáo âm thầm thay đổi; file chỉ đọc, chỉ tạo lại khi nhóm trưởng đồng ý.
- Ép `uint8` trước khi kiểm khoảng giá trị -> 256 âm thầm thành 0; kiểm trước rồi mới ép.
- Lấy validation từ file test -> test không còn độc lập; validation chỉ lấy từ 60.000 ảnh train.
- Máy khác có CSV khác -> split khác; đã ghi MD5 của 2 file CSV.

### Lưu ý / lỗi hay gặp

- Không viết tay đường dẫn dữ liệu; dùng `raw_data_dir("lab1")` trong `shared/paths.py`.
- Trên Windows có đường dẫn tiếng Việt, đặt `PYTHONIOENCODING=utf-8` trước khi chạy script, nếu không lệnh `print` cuối có thể lỗi `UnicodeEncodeError` (file đã được ghi trước đó).
- Phân tầng giữ validation cân bằng; chia ngẫu nhiên thì mỗi lớp lệch khoảng ±22 ảnh quanh 600.
- 6.000 ảnh validation cho sai số chuẩn của accuracy khoảng ±0,4 điểm phần trăm; chênh lệch nhỏ hơn mức này chưa kết luận được (lý do `experiment` chạy 5 seed).
- **Quy tắc cho AI:** được sửa `preprocessing.py`, `download_official.py`, `tests/test_preprocessing.py`, file này. Cấm sửa module khác, data thô, ô notebook của người khác; chỉ sửa `train_val_split.json` khi có chủ đích và đã hỏi nhóm trưởng. Chữ ký 4 hàm là hợp đồng (`project_description.md`). Cấm lấy validation từ file test; cấm đổi `val_ratio` hay `seed` khi chưa hỏi nhóm trưởng; cấm commit file CSV/IDX; cấm tải dữ liệu trong unit test.

### Nơi sử dụng (Integration)

- `features` (mảng ảnh, nhãn và chỉ số)
- `labs/lab1/main.py` stage 2, `labs/lab1/extended_study.py`
- `notebook/lab1_presentation.ipynb` Stage 2 (ô `[3]`)

### Liên kết

- Tiến độ: `labs/lab1/progress_status.md`
- Giai đoạn trước / sau: [eda.md](../eda/eda.md) / [features.md](../features/features.md)
