# Lab 1 / eda: Exploratory Data Analysis (phase 1 of 4, data part)

- **Motivation/Background**: The data must be checked before any training; wrong columns, missing cells or duplicate images would silently corrupt every later number.
- **Purpose**: Specify the `eda` module: what it measures, its input/output, how to run and verify it, and when it counts as done.
- **Overview Pipeline**: training CSV -> `run_eda` (column check, 6 statistics, 2 figures) -> numbers recorded in `progress_status.md` -> `preprocessing`.
- **Detailed Plan**: Name; Purpose; Goals (Mục tiêu); Input / Output; How to do it (general plan); Pipeline; Public API; Figures, per-split statistics and data issues; Success criteria & retrial; Smoke / Normal run; Risks; Detailed plan / gotchas; Integration; Links; Vietnamese version.
- **References**: [STATUS.md](STATUS.md), [PROMPTS.md](PROMPTS.md), [AUDIT.md](AUDIT.md), pandas, matplotlib (`matplotlib.figure.Figure`), `labs/lab1/eda/eda.py`, `labs/lab1/tests/test_eda.py`, `project_description.md`.
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
- [Figures, per-split statistics and data issues](#figures-per-split-statistics-and-data-issues)
- [Success criteria & retrial](#success-criteria--retrial)
- [Smoke / Normal run](#smoke--normal-run)
- [Risks](#risks)
- [Detailed plan / gotchas](#detailed-plan--gotchas)
- [Integration](#integration)
- [Links](#links)
- [Tiếng Việt](#tiếng-việt)

---

## Name

- **Title:** Exploratory Data Analysis
- **Execution order:** 1 of 4 (eda -> preprocessing -> features -> model)
- **Created:** 2026-10-13, author: Toàn (Member A)
- **Updated:** 2026-11-07 by Toàn (Member A)
- **Description:** Read the training CSV (read-only), return 6 data-quality statistics and save 2 check figures.
- **Status:** Done (Run on the official data, 2026-11-04)
- **Owner:** Member A (Toàn), assigned by the team leader

## Purpose

The data must be checked before training: row count, class balance, pixel range, missing cells, duplicate rows. Balanced classes make accuracy a fair metric (random guess = 10%). Duplicate images across splits would leak data. EDA only measures and reports; any fix is a separate, recorded decision.

## Goals (Mục tiêu)

- Done means:
  - `run_eda` returns `n_rows`, `class_counts`, `pixel_min`, `pixel_max`, `n_missing`, `n_duplicates`.
  - `class_counts.png` and `sample_grid.png` are saved in `out_dir`.
  - A CSV with wrong columns raises `ValueError` immediately.
- Not in this phase: modifying or deleting rows; running EDA on the test file to make decisions.

## Input / Output

- **Input:** `labs/lab1/raw_data/fashion-mnist_train.csv` (official Zalando data, written by `preprocessing/download_official.py`, see [preprocessing.md](../preprocessing/preprocessing.md)); `out_dir` for figures.
- **Output:** dict of 6 statistics (printed by `main.py`; numbers recorded in [STATUS.md](STATUS.md) and `labs/lab1/progress_status.md`); `labs/lab1/artifacts/figures/class_counts.png`, `sample_grid.png` (not committed).

## How to do it (general plan)

1. Read the CSV with pandas; `_check_columns`: a `label` column and exactly 784 pixel columns.
2. Compute: `len(df)`; `value_counts().sort_index()`; `min().min()` / `max().max()`; `isna().sum().sum()`; `duplicated().sum()` (label + all pixels).
3. Plot with `matplotlib.figure.Figure` (no window, no memory leak): bar chart per class; 2x5 grid of the first image of each class with `vmin=0, vmax=255`.

## Pipeline

```
eda.run_eda(train_csv, out_dir)
  -> _check_columns -> 6 statistics
  -> _plot_class_counts -> class_counts.png
  -> _plot_sample_grid  -> sample_grid.png
```

## Public API

### `run_eda(train_csv: str, out_dir: str) -> dict`

Explore the training CSV (read-only) and save 2 figures.

**Parameters:**
- `train_csv` (str): path to the training CSV (`label` + 784 pixel columns).
- `out_dir` (str): folder for the figures; created if missing.

**Returns:**
- `dict`: `n_rows` (int), `class_counts` (dict label -> count), `pixel_min`, `pixel_max`, `n_missing`, `n_duplicates` (int).

**Raises:**
- `ValueError`: no `label` column, or the number of pixel columns is not 784.

**Example:**
```python
from labs.lab1.eda.eda import run_eda
from shared.paths import raw_data_dir, artifact_dir

stats = run_eda(str(raw_data_dir("lab1") / "fashion-mnist_train.csv"), str(artifact_dir("lab1", "figures")))
print(stats["n_rows"], stats["n_duplicates"])
```

Private helpers: `_check_columns(df)`, `_plot_class_counts(class_counts, out_path)`, `_plot_sample_grid(df, out_path)`.

## Figures, per-split statistics and data issues

Figures written by `main.py` stage 1 (not committed; regenerate with `python -m labs.lab1.main`). Bar charts and image grids only, no pie charts.

![Samples per class](../../artifacts/figures/class_counts.png)

![First image of each class](../../artifacts/figures/sample_grid.png)

| Split | Images | Per class | Pixel mean (0-1) | Source |
|---|---|---|---|---|
| train | 54,000 | 5,400 | 0.2859 | `main.py` stage 3 (2026-11-04) |
| validation | 6,000 | 600 | 0.2874 | measured separately (2026-11-06) |
| test | 10,000 | 1,000 | 0.2868 | measured separately (2026-11-06) |

Identical images across splits: train-val 0, train-test 0, val-test 0 -> no leakage from duplicates.

**Data issues:**

1. The old Kaggle CSV did not follow the official split (most Kaggle test images are official training images, 43 duplicate rows) -> replaced by the official data on 2026-11-02.
2. 50.2% of the training pixels are 0 (black background) -> low mean 0.2859 (measured separately on the 54,000 training images, 2026-11-06).
3. T-shirt/top, Shirt, Pullover and Coat look alike (mean brightness 82.9, 84.4, 96.1, 98.1 on the 0-255 scale) -> the hardest classes; `evaluate` confirms Shirt is the worst class (70.17%).
4. No missing cells, no duplicate rows, classes perfectly balanced -> no cleaning or class weighting needed.

## Success criteria & retrial

| Criterion | Correct value | If not met |
|---|---|---|
| `n_rows` | 60,000 | re-check the download (see [preprocessing.md](../preprocessing/preprocessing.md)) |
| `class_counts` | 6,000 per class | record it, tell the team leader |
| pixel range | 0-255 | corrupted data: download again |
| `n_missing` | 0 | record it; do not fill values |
| `n_duplicates` | 0 | record it, **do not delete**; measure the impact separately (done for the old Kaggle CSV: `logbook/duplicate_impact.ipynb`) |

## Smoke / Normal run

| Type | Command | Expected |
|---|---|---|
| Unit test | `python -m pytest labs/lab1/tests/test_eda.py -q` | 3 tests pass (fake CSV, `tmp_path`) |
| Smoke run | `python -c "import tempfile; from labs.lab1.eda.eda import run_eda; print(run_eda('labs/lab1/raw_data/fashion-mnist_train.csv', tempfile.mkdtemp()))"` | `n_rows` 60000, 6000 per class, pixels 0-255, `n_missing` 0, `n_duplicates` 0 |
| Normal run | `python -m labs.lab1.main` stage 1 | "EDA completed: 60000 rows, 10 classes, 0 duplicates." |

## Risks

- Wrong file or format (e.g. the old Kaggle CSV) -> every later number is wrong; mitigated by `_check_columns` and the MD5 check in `download_official.py`.
- Duplicate images across splits -> optimistic test accuracy; measured 0 / 0 / 0 on the official data.
- Using the test file to make decisions -> biased test number; EDA reads the training CSV only.

## Detailed plan / gotchas

- `duplicated()` counts extra rows, not pairs (an image seen 3 times counts 2).
- pandas `min`/`max` skip NaN, so empty cells do not break the statistics.
- **Rules for AI agents:** may edit `labs/lab1/eda/eda.py`, `labs/lab1/tests/test_eda.py`, this file. Must not edit other modules, raw data, the committed split file, figures of real runs, other owners' notebook cells. Signature `run_eda(train_csv: str, out_dir: str) -> dict` is a contract (`project_description.md`). Prohibited: modifying the CSV; EDA on the test file to make decisions; tests writing to the real `artifacts/figures/`. Never assign work to yourself; ask the team leader.

## Integration

This module is used by:
- `labs/lab1/main.py` stage 1
- `notebook/lab1_presentation.ipynb` Stage 1 (cell `[2]`)

## Links

- Progress: `labs/lab1/progress_status.md`
- Next phase: [preprocessing.md](../preprocessing/preprocessing.md)

---

## Tiếng Việt

### Tên

- **Tên:** Khám phá dữ liệu (EDA)
- **Thứ tự thực hiện:** 1 trên 4 (eda → preprocessing → features → model)
- **Tạo:** 13/10/2026, tác giả Toàn (Member A)
- **Cập nhật:** 07/11/2026 bởi Toàn (Member A)
- **Mô tả:** đọc CSV train (chỉ đọc), trả 6 thống kê chất lượng dữ liệu và lưu 2 hình kiểm tra.
- **Trạng thái:** xong (đã chạy trên dữ liệu chính thức, 04/11/2026)
- **Người phụ trách:** Member A (Toàn), do nhóm trưởng giao. Không tự giao việc cho mình; xem GitHub Issue và `progress_status.md`, hỏi nhóm trưởng.

### Purpose (lý do)

Phải kiểm dữ liệu trước khi train: số dòng, cân bằng lớp, khoảng pixel, ô thiếu, dòng trùng. Lớp cân bằng thì accuracy là thước đo công bằng (đoán bừa = 10%). Ảnh trùng giữa các tập là rò rỉ dữ liệu. EDA chỉ đo và báo; sửa (nếu cần) là một quyết định riêng, có ghi lại.

### Mục tiêu

- Xong khi:
  - `run_eda` trả `n_rows`, `class_counts`, `pixel_min`, `pixel_max`, `n_missing`, `n_duplicates`.
  - Lưu `class_counts.png` và `sample_grid.png` vào `out_dir`.
  - CSV sai cột thì báo `ValueError` ngay.
- Không làm: sửa hoặc xóa dòng; EDA trên file test để ra quyết định.

### Input / Output

- **Input:** `labs/lab1/raw_data/fashion-mnist_train.csv` (dữ liệu chính thức của Zalando, do `preprocessing/download_official.py` ghi ra); `out_dir` để lưu hình.
- **Output:** dict 6 thống kê (`main.py` in ra; số được ghi vào [STATUS.md](STATUS.md) và `labs/lab1/progress_status.md`); `labs/lab1/artifacts/figures/class_counts.png`, `sample_grid.png` (không commit).

### Cách làm

1. Đọc CSV bằng pandas; `_check_columns`: phải có cột `label` và đúng 784 cột pixel.
2. Tính: `len(df)`; `value_counts().sort_index()`; `min().min()` / `max().max()`; `isna().sum().sum()`; `duplicated().sum()` (xét cả nhãn và mọi pixel).
3. Vẽ bằng `matplotlib.figure.Figure` (không mở cửa sổ, không rò bộ nhớ): biểu đồ cột số mẫu mỗi lớp; lưới 2x5 ảnh đầu tiên của mỗi lớp với `vmin=0, vmax=255`.

### Pipeline

Như sơ đồ ở phần tiếng Anh: `run_eda` → `_check_columns` → 6 thống kê → `class_counts.png`, `sample_grid.png`.

### Public API (hàm công khai)

**`run_eda(train_csv: str, out_dir: str) -> dict`**: khám phá CSV train (chỉ đọc) và lưu 2 hình.

- **Tham số:** `train_csv` (str): đường dẫn CSV train (`label` + 784 cột pixel); `out_dir` (str): thư mục lưu hình, tự tạo nếu chưa có.
- **Trả về:** `dict`: `n_rows` (int), `class_counts` (dict nhãn -> số mẫu), `pixel_min`, `pixel_max`, `n_missing`, `n_duplicates` (int).
- **Báo lỗi:** `ValueError` khi không có cột `label` hoặc số cột pixel khác 784.
- **Ví dụ:** xem đoạn code ở phần tiếng Anh.
- Hàm nội bộ: `_check_columns(df)`, `_plot_class_counts(class_counts, out_path)`, `_plot_sample_grid(df, out_path)`.

### Hình, thống kê theo từng tập và vấn đề của dữ liệu

Hình do `main.py` stage 1 ghi ra (không commit; tạo lại bằng `python -m labs.lab1.main`). Chỉ dùng biểu đồ cột và lưới ảnh, không dùng biểu đồ tròn.

![Số mẫu mỗi lớp](../../artifacts/figures/class_counts.png)

![Ảnh đầu tiên của mỗi lớp](../../artifacts/figures/sample_grid.png)

| Tập | Số ảnh | Mỗi lớp | Pixel trung bình (0-1) | Nguồn |
|---|---|---|---|---|
| train | 54.000 | 5.400 | 0,2859 | `main.py` stage 3 (04/11/2026) |
| validation | 6.000 | 600 | 0,2874 | đo riêng (06/11/2026) |
| test | 10.000 | 1.000 | 0,2868 | đo riêng (06/11/2026) |

Ảnh giống hệt giữa các tập: train-val 0, train-test 0, val-test 0 -> không rò rỉ do ảnh trùng.

**Vấn đề của dữ liệu:**

1. CSV Kaggle cũ không theo split chính thức (phần lớn ảnh test Kaggle là ảnh train chính thức, 43 dòng trùng) -> thay bằng dữ liệu chính thức ngày 02/11/2026.
2. 50,2% pixel train bằng 0 (nền đen) -> mean thấp 0,2859 (đo riêng trên 54.000 ảnh train, 06/11/2026).
3. T-shirt/top, Shirt, Pullover, Coat trông giống nhau (độ sáng trung bình 82,9, 84,4, 96,1, 98,1 trên thang 0-255) -> các lớp khó nhất; `evaluate` xác nhận Shirt là lớp kém nhất (70,17%).
4. Không có ô thiếu, không có dòng trùng, các lớp cân bằng tuyệt đối -> không cần làm sạch hay đặt trọng số lớp.

### Tiêu chí thành công & làm lại

| Tiêu chí | Giá trị đúng | Không đạt thì |
|---|---|---|
| `n_rows` | 60.000 | kiểm lại bước tải dữ liệu (xem [preprocessing.md](../preprocessing/preprocessing.md)) |
| `class_counts` | 6.000 mỗi lớp | ghi lại, báo nhóm trưởng |
| khoảng pixel | 0–255 | dữ liệu hỏng: tải lại |
| `n_missing` | 0 | ghi lại; không tự điền giá trị |
| `n_duplicates` | 0 | ghi lại, **không xóa**; đo ảnh hưởng riêng (đã làm với CSV Kaggle cũ: `logbook/duplicate_impact.ipynb`) |

### Smoke / Normal run

| Loại | Lệnh | Kết quả đúng |
|---|---|---|
| Unit test | `python -m pytest labs/lab1/tests/test_eda.py -q` | 3 test đạt (CSV giả, `tmp_path`) |
| Smoke run | lệnh `python -c ...` ở phần tiếng Anh | `n_rows` 60000, 6000 mỗi lớp, pixel 0-255, `n_missing` 0, `n_duplicates` 0 |
| Normal run | `python -m labs.lab1.main` stage 1 | "EDA completed: 60000 rows, 10 classes, 0 duplicates." |

### Rủi ro

- Sai file hoặc sai định dạng (ví dụ CSV Kaggle cũ) -> mọi số phía sau đều sai; đã chặn bằng `_check_columns` và kiểm MD5 trong `download_official.py`.
- Ảnh trùng giữa các tập -> accuracy test cao ảo; đo trên dữ liệu chính thức: 0 / 0 / 0.
- Dùng file test để ra quyết định -> số test bị lệch; EDA chỉ đọc CSV train.

### Lưu ý / lỗi hay gặp

- `duplicated()` đếm số dòng thừa, không phải số cặp (một ảnh xuất hiện 3 lần đếm 2).
- `min`/`max` của pandas bỏ qua NaN, nên ô trống không làm hỏng thống kê.
- **Quy tắc cho AI:** được sửa `eda.py`, `tests/test_eda.py`, file này. Cấm sửa module khác, data, file split đã commit, hình của lần chạy thật, ô notebook của người khác. Chữ ký `run_eda(train_csv, out_dir) -> dict` là hợp đồng (`project_description.md`). Cấm sửa CSV, cấm EDA trên file test để ra quyết định, test không được ghi vào `artifacts/figures/` thật.

### Nơi sử dụng (Integration)

- `labs/lab1/main.py` stage 1
- `notebook/lab1_presentation.ipynb` Stage 1 (ô `[2]`)

### Liên kết

- Tiến độ: `labs/lab1/progress_status.md`
- Giai đoạn tiếp theo: [preprocessing.md](../preprocessing/preprocessing.md)
