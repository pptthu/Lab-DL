# Lab 1 / features: audit (code vs spec)

- **Motivation/Background**: Step 8 of the workflow: the code is checked against the spec before the version is approved.
- **Purpose**: Record each claim of [features.md](../features/features.md), the evidence in the code or a run, and the result.
- **Overview Pipeline**: spec claim -> code / command -> OK or not done (open question).
- **Detailed Plan**: Audit table; Vietnamese version.
- **References**: [features.md](../features/features.md), `labs/lab1/tests/`.
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
| mean/std on train only | `main.py`: `compute_mean_std(images[train_idx])` | OK |
| mean 0.2859, std 0.3530 | re-run 2026-11-07: 0.2859 / 0.353 | OK |
| batch `[64,1,28,28]` float32, labels int64 | re-run 2026-11-07: `[64,1,28,28]` float32, `[64]` int64 | OK |
| 844 / 94 / 157 batches | re-run 2026-11-07: 844, 94, 157 | OK |
| shuffle train only | `make_loaders`; `test_make_loaders_batches_and_shuffle_only_train` | OK |
| `std <= 0` raises | `build_transform`; `test_build_transform_rejects_zero_std` | OK |
| no augmentation | transform = scale + normalize only | OK |

---

## Tiếng Việt

### Audit (đối chiếu code với mô tả)

Kiểm ngày 07/11/2026 bằng `.venv`: `pytest` 4 file test data/model + `test_smoke.py` → 30 test đạt; sau khi thêm comment từng dòng vào file `.py`, `pytest labs/lab1/tests` → 52 test đạt và AST của code không đổi.

| Mô tả ghi | Code / bằng chứng | Kết quả |
|---|---|---|
| mean/std chỉ trên train | `main.py`: `compute_mean_std(images[train_idx])` | Đạt |
| mean 0,2859, std 0,3530 | chạy lại 07/11/2026: 0,2859 / 0,353 | Đạt |
| batch `[64,1,28,28]` float32, nhãn int64 | chạy lại 07/11/2026: đúng | Đạt |
| 844 / 94 / 157 batch | chạy lại 07/11/2026: 844, 94, 157 | Đạt |
| chỉ xáo train | `make_loaders`; test `test_make_loaders_batches_and_shuffle_only_train` | Đạt |
| `std <= 0` báo lỗi | `build_transform`; test `test_build_transform_rejects_zero_std` | Đạt |
| không augmentation | transform chỉ chia 255 và chuẩn hóa | Đạt |
