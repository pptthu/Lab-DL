# Lab 1 / eda: guide for AI agents

## For AI agents (read first)

| Field | Value |
|---|---|
| Module | `labs/lab1/eda/eda.py` |
| Owner | Member A (Toàn), assigned by the team leader |
| Status | done in `progress_status.md` (Run on the official data, 2026-11-04) |
| Assigned work | see the GitHub Issues and `progress_status.md` of the lab; never assign work to yourself, ask the team leader |
| Files you may edit | `labs/lab1/eda/eda.py`, `labs/lab1/tests/test_eda.py`, this guide |
| Files you must not edit | other modules, raw data, committed split file, figures of real runs, other owners' notebook cells |
| Agreed signatures | `run_eda(train_csv: str, out_dir: str) -> dict` (same as `project_description.md`) |
| Depends on / used by | the training CSV in `raw_data_dir("lab1")` / `main.py` stage 1, notebook Stage 1 |
| Prohibited | modifying the CSV; running EDA on the test file to make decisions; tests writing to the real `artifacts/figures/` (use `tmp_path`) |
| Checks | Unit: `python -m pytest labs/lab1/tests/test_eda.py -q` (fake CSV). Smoke: `run_eda` on the real training CSV into a temporary folder |
| Done when | unit tests pass; output recorded in `progress_status.md`; this guide updated |

## What the brief requires

- Brief requirement: load and explore the Fashion-MNIST data before training.
- Team choice: official Fashion-MNIST from https://github.com/zalandoresearch/fashion-mnist (the dataset link list came from another group, not from the lecturer), written as CSV by `preprocessing/download_official.py`.

## How the module works (current code)

- Reads the CSV with pandas, checks the columns (`label` + 784 pixel columns).
- Returns `n_rows`, `class_counts`, `pixel_min`, `pixel_max`, `n_missing`, `n_duplicates`.
- Saves a class bar chart (`class_counts.png`) and a sample image grid (`sample_grid.png`) in `out_dir`.

## Inputs, outputs and pitfalls

- Wrong columns raise an error (`_check_columns`).
- Duplicate rows are counted, not removed (decision: keep the data as released).

## How to verify

| Type | Command | Expected |
|---|---|---|
| Unit test | `python -m pytest labs/lab1/tests/test_eda.py -q` | all pass |
| Smoke test | `python -c "import tempfile; from labs.lab1.eda.eda import run_eda; print(run_eda('labs/lab1/raw_data/fashion-mnist_train.csv', tempfile.mkdtemp()))"` | `n_rows` 60000, 6000 per class, pixels 0-255, `n_missing` 0, `n_duplicates` 0 (Run, 2026-11-04) |
| Real integration run | `python -m labs.lab1.main` stage 1 | "EDA completed: 60000 rows, 10 classes, 0 duplicates." (Run, 2026-11-04) |

## Open questions / limitations

- None for this module. The old Kaggle CSVs (43 duplicate rows) are no longer used.

---

**Tiếng Việt**

- **Việc được giao:** xem GitHub Issue và `progress_status.md` của lab; không tự giao việc cho mình, hỏi nhóm trưởng.
- **Dành cho AI:** người phụ trách Member A (Toàn), do nhóm trưởng giao. Được sửa `eda.py`, `tests/test_eda.py`, file này; cấm sửa module khác, data, file split đã commit, hình của lần chạy thật. Chữ ký `run_eda(train_csv, out_dir) -> dict`, khớp `project_description.md`. Cấm sửa CSV, cấm dùng file test để ra quyết định, test phải ghi hình vào `tmp_path`.
- **Đề:** nạp và khám phá dữ liệu trước khi train. Nhóm chọn Fashion-MNIST chính thức của Zalando (danh sách link do nhóm khác tổng hợp, không phải của cô).
- **Cách hoạt động:** kiểm tra cột, trả về số dòng, số mẫu mỗi lớp, khoảng pixel, số giá trị thiếu, số dòng trùng; lưu 2 hình.
- **Kiểm chứng:** 60.000 dòng, 6.000 mỗi lớp, 0 dòng trùng, 0 giá trị thiếu (đã chạy, 04/11/2026).
