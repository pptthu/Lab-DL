# features — Chuẩn hoá và DataLoader (v1)

- **Created**: 2026-11-04T14:49:03+07:00
- **Last Updated**: 2026-11-14T00:55:48+07:00 (Thư)
- **Người phụ trách**: Member A (Toàn)
- **Status**: Done

---

## Name
Features — mean/std từ train, transform, Dataset, DataLoader

## Background
Đề P1: "transforms" và đưa ảnh vào mạng theo batch. Module nhận mảng ảnh từ `preprocessing`,
trả 3 DataLoader cho `train`, `evaluate`, `visualize`.

## Goals / Purpose
- `compute_mean_std` trên ảnh train trả mean 0,2859, std 0,3530 (thang 0–1)
- `make_loaders` trả 3 loader: train 54.000 / val 6.000 / test 10.000; batch `[64,1,28,28]` float32, nhãn int64
- Chỉ loader train được shuffle; val và test giữ thứ tự cố định
- Không làm:
  - Tính mean/std trên val hay test — rò rỉ thông tin
  - Data augmentation — đề không yêu cầu
  - Chia train/val — việc của `preprocessing`

## Input / Output
- **Input:** ảnh `[N,28,28]` uint8, nhãn `[N]` int64 và `train_idx`, `val_idx` từ [preprocessing_v1.md](preprocessing_v1.md)
- **Output:** `train_loader`, `val_loader`, `test_loader` (đối tượng trong bộ nhớ, không ghi file)

## How to do it (general plan)
1. `compute_mean_std(images[train_idx])` — chỉ ảnh train
2. `build_transform(mean, std)`: uint8 `[28,28]` → float32 `[1,28,28]`, `(x/255 − mean) / std`
3. Tạo 3 `FashionDataset` (train, val dùng `indices`; test dùng toàn bộ file test)
4. `make_loaders(..., batch_size=64)`
5. Smoke run: lấy 1 batch train thật, kiểm shape và kiểu
6. Normal run trong pipeline `main.py`

## Pipeline
```
images, labels, train_idx, val_idx → compute_mean_std(train) → build_transform
  → FashionDataset × 3 → make_loaders(batch 64) → train / evaluate / visualize
```

## Detailed plan
- **Một mean/std toàn cục từ train:** dùng chung cho cả 3 tập — đúng cách làm, tránh rò rỉ
- **Batch 64:** dùng chung cho mọi cấu hình của desc lab
- **Không shuffle val/test:** dự đoán khớp thứ tự ảnh, cần cho confusion matrix và phân tích lỗi
- **Hàm hiện có:**
  - `compute_mean_std(train_images) -> tuple` (`features/features.py:18`)
  - `build_transform(mean: float, std: float)` (`:38`)
  - `class FashionDataset(images, labels, indices=None, transform=None)` (`:60`, `:71`)
  - `make_loaders(train_ds, val_ds, test_ds, batch_size: int = 64) -> tuple` (`:92`)
- **Chỗ code chưa khớp plan:** không có
- **Gotchas:**
  - Luôn truyền ảnh train (`images[train_idx]`) vào `compute_mean_std` (`features.md` cũ; `main.py:94`)
  - `std == 0` thì `ValueError` (`features/features.py:49`)

## Success criteria
- **Test:** `python -m pytest labs/lab1/tests/test_features.py -q` đạt (6 test)
- **Smoke run:** 1 batch train thật có shape `[64,1,28,28]` float32, nhãn int64 trong 0–9
- **Normal run:** mean 0,2859, std 0,3530; số mẫu 54.000 / 6.000 / 10.000

## Links
- Desc lab: [../lab1_description_v1.md](../lab1_description_v1.md)
- Module liên quan: [preprocessing_v1.md](preprocessing_v1.md), [train_v1.md](train_v1.md), [evaluate_v1.md](evaluate_v1.md)
- Tiến độ và bằng chứng chạy: [../../progress_status.md](../../progress_status.md) (mean/std trên dữ liệu chính thức dòng 43; shape loader dòng 88 — lần chạy cũ trên CSV Kaggle)
- Code: `labs/lab1/features/features.py` | Test: `labs/lab1/tests/test_features.py`
