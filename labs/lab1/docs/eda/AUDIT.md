# Lab 1 / eda: audit (code vs spec)

- **Motivation/Background**: Step 8 of the workflow: the code is checked against the spec before the version is approved.
- **Purpose**: Record each claim of [eda.md](../eda/eda.md), the evidence in the code or a run, and the result.
- **Overview Pipeline**: spec claim -> code / command -> OK or not done (open question).
- **Detailed Plan**: Audit table; Vietnamese version.
- **References**: [eda.md](../eda/eda.md), `labs/lab1/tests/`.
- **Created**: 2026-11-07T00:00:00+07:00
- **Last Updated**: 2026-11-07T00:00:00+07:00

---

## Table of Contents

- [Audit (code vs description)](#audit-code-vs-description)
- [Tiếng Việt](#tiếng-việt)

---

## Audit (code vs description)

Checked on 2026-11-07 with `.venv`: `pytest` of the 4 data/model test files + `test_smoke.py` -> 30 passed; after adding the line-by-line comments to the `.py` files, `pytest labs/lab1/tests` -> 52 passed and the code AST is unchanged.

| Description says | Code / evidence | Result |
|---|---|---|
| read-only, never modify the CSV | only `read_csv`, no write to the CSV | OK |
| 6 statistics returned | `stats` has the 6 keys in `run_eda` | OK |
| wrong columns -> `ValueError` | `_check_columns`; `test_run_eda_rejects_wrong_columns` | OK |
| 2 figures in `out_dir` | end of `run_eda` (`_plot_class_counts`, `_plot_sample_grid`); `test_run_eda_stats_and_figures` | OK |
| 60,000 rows, 6,000 per class, 0-255, 0 missing, 0 duplicates | re-run on 2026-11-07: identical values | OK |
| tests write to `tmp_path` only | 3 tests use `tmp_path` | OK |

---

## Tiếng Việt

### Audit (đối chiếu code với mô tả)

Kiểm ngày 07/11/2026 bằng `.venv`: `pytest` 4 file test data/model + `test_smoke.py` → 30 test đạt; sau khi thêm comment từng dòng vào file `.py`, `pytest labs/lab1/tests` → 52 test đạt và AST của code không đổi.

| Mô tả ghi | Code / bằng chứng | Kết quả |
|---|---|---|
| chỉ đọc, không sửa CSV | chỉ có `read_csv`, không ghi vào CSV | Đạt |
| trả 6 thống kê | `stats` có đủ 6 khóa trong `run_eda` | Đạt |
| sai cột → `ValueError` | `_check_columns`; test `test_run_eda_rejects_wrong_columns` | Đạt |
| 2 hình trong `out_dir` | cuối `run_eda` (`_plot_class_counts`, `_plot_sample_grid`); test `test_run_eda_stats_and_figures` | Đạt |
| 60.000 dòng, 6.000 mỗi lớp, 0–255, 0 ô thiếu, 0 dòng trùng | chạy lại 07/11/2026: đúng các số này | Đạt |
| test chỉ ghi vào `tmp_path` | 3 test đều dùng `tmp_path` | Đạt |
