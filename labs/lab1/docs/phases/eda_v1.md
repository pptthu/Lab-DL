# eda — Khám phá dữ liệu (v1)

- **Created**: 2026-11-04T14:49:03+07:00
- **Last Updated**: 2026-11-14T01:41:23+07:00 (Thư)
- **Người phụ trách**: Member A (Toàn)
- **Status**: Done

---

## Name
EDA — kiểm tra file train, phân bố lớp, ảnh mẫu

## Background
Đề P1: "load Fashion-MNIST". Trước khi train cần biết dữ liệu có đúng định dạng, đủ lớp, cân bằng không.
EDA chỉ chạy trên file train để không đưa thông tin của test vào quyết định.

## Goals / Purpose
- `run_eda` trả dict: `n_rows`, `class_counts`, `pixel_min`, `pixel_max`, `n_missing`, `n_duplicates`
- Lưu 2 hình: `class_counts.png` (số ảnh mỗi lớp) và `sample_grid.png` (ảnh mẫu)
- Không làm:
  - Sửa hay xoá dòng trùng trong CSV — chỉ đếm; dữ liệu giữ nguyên như bản phát hành (`eda.md` cũ)
  - Chạy EDA trên file test để ra quyết định — tránh rò rỉ thông tin
  - Chuẩn hoá, chia train/val — việc của `features`, `preprocessing`

## Input / Output
- **Input:** `fashion-mnist_train.csv` (60.000 dòng, cột `label` + 784 cột pixel) từ [preprocessing_v1.md](preprocessing_v1.md)
- **Output:** dict thống kê; 2 file PNG
  - Code đọc `data/raw/fashion-mnist_train.csv`, ghi hình vào `outputs/figures/` (`main.py:96-97`, `main.py:104`).

## How to do it (general plan)
1. Đọc CSV bằng pandas, kiểm cột (`label` + 784 pixel)
2. Đếm số dòng, số ảnh mỗi lớp, min/max pixel, ô thiếu, dòng trùng
3. Vẽ biểu đồ cột số ảnh mỗi lớp và lưới ảnh mẫu
4. Smoke run trên CSV train thật, ghi hình vào thư mục tạm
5. Normal run trong pipeline `main.py`

## Pipeline
```
fashion-mnist_train.csv → _check_columns → thống kê (dict)
  → _plot_class_counts → class_counts.png
  → _plot_sample_grid  → sample_grid.png
```

## Detailed plan
- **Chỉ file train:** mọi quyết định về dữ liệu dựa trên train; test để dành cho `evaluate`
- **Dòng trùng:** chỉ đếm, không xoá — giữ dữ liệu như bản chính thức để so sánh được với kết quả của người khác
- **Hàm hiện có:** `run_eda(train_csv: str, out_dir: str) -> dict` (`eda/eda.py:25`); hàm phụ `_check_columns` (`:63`), `_plot_class_counts` (`:72`), `_plot_sample_grid` (`:85`)
- **Chỗ code chưa khớp plan:** không có (đường dẫn đã đổi ở O5 — đổi thư mục `artifacts/`, `raw_data/` sang `data/`, `outputs/`, 2026-11-14)
- **Gotchas:**
  - Sai cột thì `ValueError` (`eda/eda.py:66`, `:69`)
  - CSV Kaggle cũ có 43 dòng trùng; file chính thức có 0 (`progress_status.md:42`)

## Success criteria
- **Test:** `python -m pytest labs/lab1/tests/test_eda.py -q` đạt (3 test, CSV giả)
- **Smoke run:** `run_eda` trên CSV train thật ghi đủ 2 PNG vào thư mục tạm
- **Normal run:** `n_rows` = 60000, 6000 ảnh mỗi lớp, pixel 0–255, `n_missing` = 0, `n_duplicates` = 0

## Links
- Desc lab: [../lab1_description_v1.md](../lab1_description_v1.md)
- Module liên quan: [preprocessing_v1.md](preprocessing_v1.md), [features_v1.md](features_v1.md)
- Tiến độ và bằng chứng chạy: [../../progress_status.md](../../progress_status.md) (dữ liệu chính thức: 60000 dòng, 6000 mỗi lớp, 0 trùng, dòng 42)
- Code: `labs/lab1/eda/eda.py` | Test: `labs/lab1/tests/test_eda.py`
