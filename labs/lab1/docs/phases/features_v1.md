# Lab 1 / features: guide for AI agents

## For AI agents (read first)

| Field | Value |
|---|---|
| Module | `labs/lab1/features/features.py` |
| Owner | Member A (Toàn), assigned by the team leader |
| Status | done in `progress_status.md` (Run on the official data, 2026-11-04) |
| Assigned work | see the GitHub Issues and `progress_status.md` of the lab; never assign work to yourself, ask the team leader |
| Files you may edit | `labs/lab1/features/features.py`, `labs/lab1/tests/test_features.py`, this guide |
| Files you must not edit | other modules, raw data, committed split file, other owners' notebook cells |
| Agreed signatures | `compute_mean_std(train_images) -> tuple`; `build_transform(mean: float, std: float)`; `class FashionDataset(images, labels, indices=None, transform=None)`; `make_loaders(train_ds, val_ds, test_ds, batch_size: int = 64) -> tuple` (same as `project_description.md`) |
| Depends on / used by | `preprocessing` output / `train`, `evaluate`, `visualize`, `experiment`, `main.py`, notebook |
| Prohibited | computing mean/std on validation or test images (leakage); shuffling validation or test loaders; adding augmentation that the brief does not ask for |
| Checks | Unit: `python -m pytest labs/lab1/tests/test_features.py -q`. Smoke: build the loaders from the real data and read one batch |
| Done when | unit tests pass; output recorded in `progress_status.md`; this guide updated |

## What the brief requires

- Brief requirement: turn the images into normalized tensors and feed them to the network in batches.
- Team choices: one global mean/std from the **training split only**; batch size 64; shuffle the training loader only.

## How the module works (current code)

- `compute_mean_std`: mean and std of the training images on the 0-1 scale.
- `build_transform`: uint8 `[28,28]` -> float32 tensor `[1,28,28]`, `(x/255 - mean) / std`; `std == 0` raises an error.
- `FashionDataset`: wraps the arrays, optional `indices` select the split, returns `(tensor, int label)`.
- `make_loaders`: `shuffle=True` for train, `shuffle=False` for validation and test (fixed order keeps predictions aligned with images for the confusion matrix and the error analysis).

## Inputs, outputs and pitfalls

- Always pass the training indices to `compute_mean_std`; the same statistics are then used for all three splits.

## How to verify

| Type | Command | Expected |
|---|---|---|
| Unit test | `python -m pytest labs/lab1/tests/test_features.py -q` | all pass |
| Smoke test | `compute_mean_std(images[train_idx])` on the official data | mean 0.2859, std 0.3530 (Run, 2026-11-04); a train batch is `[64,1,28,28]` float32 |
| Real integration run | `python -m labs.lab1.main` stage 3 | "Train-only normalization statistics: mean=0.2859, std=0.3530" (Run, 2026-11-04) |

## Open questions / limitations

- None for this module.

---

**Tiếng Việt**

- **Việc được giao:** xem GitHub Issue và `progress_status.md` của lab; không tự giao việc cho mình, hỏi nhóm trưởng.
- **Dành cho AI:** người phụ trách Member A (Toàn), do nhóm trưởng giao. Được sửa `features.py`, `tests/test_features.py`, file này. Chữ ký 4 hàm/lớp khớp `project_description.md`. Cấm tính mean/std trên val hoặc test (rò rỉ dữ liệu), cấm xáo val/test, cấm thêm augmentation khi đề không yêu cầu.
- **Đề:** chuyển ảnh thành tensor đã chuẩn hóa và đưa vào mạng theo batch. Nhóm chọn mean/std chỉ tính trên train, batch 64, chỉ xáo train.
- **Cách hoạt động:** `(x/255 - mean) / std`; val và test không xáo để dự đoán khớp đúng ảnh khi làm confusion matrix và phân tích lỗi.
- **Kiểm chứng:** mean 0,2859, std 0,3530; batch train `[64,1,28,28]` float32 (đã chạy, 04/11/2026).
