# Lab 1 / preprocessing: audit (code vs spec)

- **Motivation/Background**: Step 8 of the workflow: the code is checked against the spec before the version is approved.
- **Purpose**: Record each claim of [preprocessing.md](../preprocessing/preprocessing.md), the evidence in the code or a run, and the result.
- **Overview Pipeline**: spec claim -> code / command -> OK or not done (open question).
- **Detailed Plan**: Audit table; Vietnamese version.
- **References**: [preprocessing.md](../preprocessing/preprocessing.md), `labs/lab1/tests/`.
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
| official data, MD5 train `8ff2cced...4215`, test `78e18be2...436e` | `md5sum` on 2026-11-07: identical | OK |
| check 0-255 before `uint8` | `load_csv` | OK |
| stratified 10%, seed 42, 600 per class | `make_split`; re-run 2026-11-07: 54,000 / 6,000, 600 per class | OK |
| regenerated split = committed JSON | `make_split(labels, 0.1, 42)` vs `load_split(...)` on 2026-11-07: identical | OK |
| `main.py` only loads the split if it exists | `main.py`: `if split_file.is_file(): load_split` | OK |
| no download in unit tests | `test_preprocessing.py` uses fake CSV files | OK |
| JSON stores seed and ratio (submission template) | JSON has only `train_idx`, `val_idx` | Not done, open question (team leader) |

---

## Tiếng Việt

### Audit (đối chiếu code với mô tả)

Kiểm ngày 07/11/2026 bằng `.venv`: `pytest` 4 file test data/model + `test_smoke.py` → 30 test đạt; sau khi thêm comment từng dòng vào file `.py`, `pytest labs/lab1/tests` → 52 test đạt và AST của code không đổi.

| Mô tả ghi | Code / bằng chứng | Kết quả |
|---|---|---|
| dữ liệu chính thức, MD5 train `8ff2cced...4215`, test `78e18be2...436e` | `md5sum` ngày 07/11/2026: khớp | Đạt |
| kiểm 0–255 trước khi ép `uint8` | `load_csv` | Đạt |
| phân tầng 10%, seed 42, 600 mỗi lớp | `make_split`; chạy lại 07/11/2026: 54.000 / 6.000, 600 mỗi lớp | Đạt |
| split tạo lại = JSON đã commit | so `make_split` với `load_split` ngày 07/11/2026: giống hệt | Đạt |
| `main.py` chỉ load split nếu đã có | `main.py`: `if split_file.is_file(): load_split` | Đạt |
| không tải dữ liệu trong unit test | `test_preprocessing.py` dùng CSV giả | Đạt |
| JSON lưu seed và tỉ lệ (theo mẫu bài tham khảo) | JSON chỉ có `train_idx`, `val_idx` | Chưa, câu hỏi mở (nhóm trưởng) |
