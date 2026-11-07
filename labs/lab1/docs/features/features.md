# Lab 1 / features: Normalization, Dataset, DataLoader (phase 3 of 4, data part)

- **Motivation/Background**: An MLP needs normalized float tensors in batches; statistics computed on validation or test images would leak into training.
- **Purpose**: Specify the `features` module: train-only normalization, the Dataset wrapper and the three DataLoaders.
- **Overview Pipeline**: arrays + split indices -> `compute_mean_std` (train only) -> `build_transform` -> `FashionDataset` x 3 -> `make_loaders` -> `train`, `evaluate`, `visualize`, `experiment`.
- **Detailed Plan**: Name; Purpose; Goals (Mục tiêu); Input / Output; How to do it (general plan); Pipeline; Public API; Success criteria & retrial; Smoke / Normal run; Risks; Detailed plan / gotchas; Integration; Links; Vietnamese version.
- **References**: [STATUS.md](STATUS.md), [PROMPTS.md](PROMPTS.md), [AUDIT.md](AUDIT.md), PyTorch (`torch.utils.data.Dataset`, `DataLoader`), numpy, `labs/lab1/features/features.py`, `labs/lab1/tests/test_features.py`.
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

- **Title:** Normalization & DataLoaders
- **Execution order:** 3 of 4 (eda -> preprocessing -> features -> model)
- **Created:** 2026-10-13, author: Toàn (Member A)
- **Updated:** 2026-11-07 by Toàn (Member A)
- **Description:** Compute mean/std on the training split only, turn images into normalized `[1,28,28]` tensors, and wrap the three splits in DataLoaders.
- **Status:** Done (Run on the official data, 2026-11-04)
- **Owner:** Member A (Toàn), assigned by the team leader

## Purpose

An MLP trains more stably when its input has mean 0 and std 1. Computing mean/std on validation or test images leaks their information into training. The images are grayscale, so one mean and one std are enough.

## Goals (Mục tiêu)

- Done means:
  - `compute_mean_std(images[train_idx])` returns `(mean, std)` on the 0-1 scale.
  - `build_transform`: uint8 `[28,28]` -> float32 `[1,28,28]`, `(x/255 - mean) / std`; `std <= 0` raises an error.
  - `FashionDataset(images, labels, indices, transform)`; `indices=None` uses every row (test).
  - `make_loaders(..., batch_size=64)`: shuffle the training loader only.
- Not in this phase: augmentation (the brief does not ask for it); saving mean/std to a file (recomputed each run, cheap).

## Input / Output

- **Input:** `images`, `labels`, `train_idx`, `val_idx` from [preprocessing.md](../preprocessing/preprocessing.md); test images and labels.
- **Saved to disk:** nothing; this module writes no file (mean/std are recomputed every run).
- **Output (in memory, handed to `train`, `evaluate`, `visualize`, `experiment`):** `train_loader` (844 batches), `val_loader` (94), `test_loader` (157); each batch is images `[64,1,28,28]` float32 and labels `[64]` int64.

## How to do it (general plan)

1. `mean = images.mean(float64) / 255`, `std = images.std(float64) / 255` on the training images only.
2. One `build_transform(mean, std)` shared by all three splits.
3. Three `FashionDataset`s share the image array (no copy) and select rows with `indices`.
4. `DataLoader(shuffle=True)` for train; `shuffle=False` for validation and test (fixed order keeps predictions aligned with images for the confusion matrix and the error analysis).

## Pipeline

```
compute_mean_std(images[train_idx]) -> build_transform(mean, std)
  -> FashionDataset x 3 (train_idx / val_idx / whole test)
  -> make_loaders(batch_size=64) -> (train_loader, val_loader, test_loader)
```

## Public API

### `compute_mean_std(train_images) -> tuple`

**Parameters:** `train_images`: `uint8` array `[N,28,28]`, training indices only.
**Returns:** `(mean, std)`: floats on the 0-1 pixel scale.

### `build_transform(mean: float, std: float)`

**Parameters:** `mean`, `std` from `compute_mean_std`.
**Returns:** callable: `uint8 [28,28]` -> `float32` tensor `[1,28,28]`, `(x/255 - mean) / std`.
**Raises:** `ValueError`: `std <= 0`.

### `class FashionDataset(images, labels, indices=None, transform=None)`

**Parameters:** shared image and label arrays; `indices` selects the rows of this split (`None` = every row, used for test); `transform` from `build_transform`.
**Returns:** `__getitem__(i)` -> `(tensor [1,28,28], int label)`; `__len__()` -> number of selected rows.

### `make_loaders(train_ds, val_ds, test_ds, batch_size: int = 64) -> tuple`

**Parameters:** three `FashionDataset`s; `batch_size`.
**Returns:** `(train_loader, val_loader, test_loader)`; only `train_loader` shuffles.

**Example:**
```python
from labs.lab1.features.features import compute_mean_std, build_transform, FashionDataset, make_loaders

mean, std = compute_mean_std(images[train_idx])
tf = build_transform(mean, std)
train_ds = FashionDataset(images, labels, indices=train_idx, transform=tf)
val_ds = FashionDataset(images, labels, indices=val_idx, transform=tf)
test_ds = FashionDataset(test_images, test_labels, indices=None, transform=tf)
train_loader, val_loader, test_loader = make_loaders(train_ds, val_ds, test_ds, batch_size=64)
x, y = next(iter(train_loader))   # x [64,1,28,28] float32, y [64] int64
```

## Success criteria & retrial

| Criterion | Correct value | If not met |
|---|---|---|
| mean / std | 0.2859 / 0.3530 | check that validation/test images were not passed to `compute_mean_std` |
| After normalization (train) | mean ~ 0, std ~ 1 | fix `build_transform` |
| Batch | `[64,1,28,28]` float32, labels int64 0-9 | check `unsqueeze(0)` and dtype |
| Shuffle | train only | fix `make_loaders` |

## Smoke / Normal run

| Type | Command | Expected |
|---|---|---|
| Unit test | `python -m pytest labs/lab1/tests/test_features.py -q` | 6 tests pass |
| Smoke run | `compute_mean_std(images[train_idx])` on the official data, read one train batch | mean 0.2859, std 0.3530; batch `[64,1,28,28]` float32 |
| Normal run | `python -m labs.lab1.main` stage 3 | "Train-only normalization statistics: mean=0.2859, std=0.3530" |

## Risks

- mean/std computed on validation or test -> leakage; `main.py` passes `images[train_idx]` only.
- Shuffling validation/test -> predictions no longer aligned with images in the confusion matrix and error analysis.
- Wrong shape or dtype (`[28,28]` uint8 instead of `[1,28,28]` float32) -> model error or silent wrong scale; covered by unit tests.

## Detailed plan / gotchas

- Always pass the training indices to `compute_mean_std`; the same statistics are then used for all three splits.
- The last training batch has 48 images (54,000 = 843 x 64 + 48), so `train` averages the epoch loss by images, not by batches.
- `drop_last=False` (default): no validation or test image is dropped.
- **Rules for AI agents:** may edit `features.py`, `tests/test_features.py`, this file. Must not edit other modules, raw data, the committed split file, other owners' notebook cells. The 4 signatures are a contract (`project_description.md`). Prohibited: mean/std on validation or test (leakage); shuffling validation or test; augmentation the brief does not ask for.

## Integration

This module is used by:
- `train`, `evaluate`, `visualize`, `experiment` (the 3 DataLoaders)
- `labs/lab1/main.py` stage 3, `labs/lab1/extended_study.py`
- `notebook/lab1_presentation.ipynb` Stage 3 (cell `[4]`)

## Links

- Progress: `labs/lab1/progress_status.md`
- Previous / next phase: [preprocessing.md](../preprocessing/preprocessing.md) / [model.md](../model/model.md)

---

## Tiếng Việt

### Tên

- **Tên:** Chuẩn hóa và DataLoader
- **Thứ tự thực hiện:** 3 trên 4 (eda → preprocessing → features → model)
- **Tạo:** 13/10/2026, tác giả Toàn (Member A)
- **Cập nhật:** 07/11/2026 bởi Toàn (Member A)
- **Mô tả:** tính mean/std chỉ trên tập train, biến ảnh thành tensor `[1,28,28]` đã chuẩn hóa, đóng gói 3 tập thành DataLoader.
- **Trạng thái:** xong (đã chạy trên dữ liệu chính thức, 04/11/2026)
- **Người phụ trách:** Member A (Toàn), do nhóm trưởng giao. Không tự giao việc cho mình; xem GitHub Issue và `progress_status.md`, hỏi nhóm trưởng.

### Purpose (lý do)

MLP học ổn định hơn khi đầu vào có mean 0, std 1. Tính mean/std trên ảnh validation hoặc test là để thông tin của tập đánh giá lọt vào lúc train (rò rỉ dữ liệu). Ảnh xám nên chỉ cần một mean và một std.

### Mục tiêu

- Xong khi:
  - `compute_mean_std(images[train_idx])` trả `(mean, std)` trên thang 0–1.
  - `build_transform`: uint8 `[28,28]` → float32 `[1,28,28]`, `(x/255 - mean) / std`; `std <= 0` thì báo lỗi.
  - `FashionDataset(images, labels, indices, transform)`; `indices=None` dùng mọi dòng (tập test).
  - `make_loaders(..., batch_size=64)`: chỉ xáo tập train.
- Không làm: augmentation (đề không yêu cầu); lưu mean/std ra file (mỗi lần chạy tính lại, rất nhanh).

### Input / Output

- **Input:** `images`, `labels`, `train_idx`, `val_idx` từ [preprocessing.md](../preprocessing/preprocessing.md); ảnh và nhãn test.
- **Lưu ra đĩa:** không có; module này không ghi file nào (mean/std tính lại mỗi lần chạy).
- **Output (trong bộ nhớ, giao cho `train`, `evaluate`, `visualize`, `experiment`):** `train_loader` (844 batch), `val_loader` (94), `test_loader` (157); mỗi batch gồm ảnh `[64,1,28,28]` float32 và nhãn `[64]` int64.

### Cách làm

1. `mean = images.mean(float64) / 255`, `std = images.std(float64) / 255`, chỉ trên ảnh train.
2. Một `build_transform(mean, std)` dùng chung cho cả 3 tập.
3. Ba `FashionDataset` dùng chung mảng ảnh (không sao chép), chọn dòng bằng `indices`.
4. `DataLoader(shuffle=True)` cho train; `shuffle=False` cho validation và test (giữ thứ tự để dự đoán khớp đúng ảnh khi làm confusion matrix và phân tích lỗi).

### Pipeline

Như sơ đồ ở phần tiếng Anh: `compute_mean_std` → `build_transform` → 3 `FashionDataset` → `make_loaders(64)` → 3 DataLoader.

### Public API (hàm công khai)

| Hàm / lớp | Tham số | Trả về | Báo lỗi |
|---|---|---|---|
| `compute_mean_std(train_images)` | mảng `uint8 [N,28,28]`, chỉ ảnh train | `(mean, std)`: số thực trên thang 0–1 | — |
| `build_transform(mean, std)` | mean, std từ hàm trên | hàm: `uint8 [28,28]` → tensor `float32 [1,28,28]`, `(x/255 - mean) / std` | `ValueError`: `std <= 0` |
| `FashionDataset(images, labels, indices=None, transform=None)` | mảng ảnh và nhãn dùng chung; `indices` chọn dòng của tập (`None` = mọi dòng, dùng cho test); `transform` | `__getitem__(i)` → `(tensor [1,28,28], nhãn int)`; `__len__()` → số dòng được chọn | — |
| `make_loaders(train_ds, val_ds, test_ds, batch_size=64)` | 3 `FashionDataset`; `batch_size` | `(train_loader, val_loader, test_loader)`; chỉ `train_loader` xáo | — |

Ví dụ: xem đoạn code ở phần tiếng Anh (batch `x [64,1,28,28]` float32, `y [64]` int64).

### Tiêu chí thành công & làm lại

| Tiêu chí | Giá trị đúng | Không đạt thì |
|---|---|---|
| mean / std | 0,2859 / 0,3530 | kiểm có lỡ truyền ảnh validation/test vào `compute_mean_std` không |
| Sau chuẩn hóa (train) | mean ≈ 0, std ≈ 1 | sửa `build_transform` |
| Batch | `[64,1,28,28]` float32, nhãn int64 0–9 | kiểm `unsqueeze(0)` và dtype |
| Xáo | chỉ train | sửa `make_loaders` |

### Smoke / Normal run

| Loại | Lệnh | Kết quả đúng |
|---|---|---|
| Unit test | `python -m pytest labs/lab1/tests/test_features.py -q` | 6 test đạt |
| Smoke run | `compute_mean_std(images[train_idx])` trên dữ liệu chính thức, đọc 1 batch train | mean 0,2859, std 0,3530; batch `[64,1,28,28]` float32 |
| Normal run | `python -m labs.lab1.main` stage 3 | "Train-only normalization statistics: mean=0.2859, std=0.3530" |

### Rủi ro

- Tính mean/std trên val hoặc test -> rò rỉ; `main.py` chỉ truyền `images[train_idx]`.
- Xáo val/test -> dự đoán không còn khớp ảnh khi làm confusion matrix và phân tích lỗi.
- Sai shape hoặc dtype (`[28,28]` uint8 thay vì `[1,28,28]` float32) -> model lỗi hoặc thang giá trị sai; đã có unit test.

### Lưu ý / lỗi hay gặp

- Luôn truyền chỉ số train vào `compute_mean_std`; cùng bộ thống kê đó dùng cho cả 3 tập.
- Batch train cuối chỉ có 48 ảnh (54.000 = 843 × 64 + 48), nên `train` tính loss epoch theo số ảnh, không theo số batch.
- `drop_last=False` (mặc định): không bỏ ảnh validation hay test nào.
- **Quy tắc cho AI:** được sửa `features.py`, `tests/test_features.py`, file này. Cấm sửa module khác, data thô, file split đã commit, ô notebook của người khác. Chữ ký 4 hàm/lớp là hợp đồng (`project_description.md`). Cấm tính mean/std trên validation hoặc test (rò rỉ dữ liệu); cấm xáo validation hoặc test; cấm thêm augmentation khi đề không yêu cầu.

### Nơi sử dụng (Integration)

- `train`, `evaluate`, `visualize`, `experiment` (3 DataLoader)
- `labs/lab1/main.py` stage 3, `labs/lab1/extended_study.py`
- `notebook/lab1_presentation.ipynb` Stage 3 (ô `[4]`)

### Liên kết

- Tiến độ: `labs/lab1/progress_status.md`
- Giai đoạn trước / sau: [preprocessing.md](../preprocessing/preprocessing.md) / [model.md](../model/model.md)
