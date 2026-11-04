# Lab 2 / features: guide for AI agents

## For AI agents (read first)

| Field | Value |
|---|---|
| Module | `labs/lab2/features/features.py` |
| Owner | Member A (Đại), assigned by the team leader |
| Status | done in `progress_status.md` (Reported only) |
| Assigned work | see the GitHub Issues and `progress_status.md` of the lab; never assign work to yourself, ask the team leader |
| Files you may edit | `labs/lab2/features/features.py`, `labs/lab2/tests/test_feature.py`, this guide |
| Files you must not edit | other modules, raw data, committed split file, other owners' notebook cells |
| Agreed signatures | `build_transforms(image_size: int = 224, augment: bool = False) -> tuple`; `make_loaders(train_set, test_set, train_idx, val_idx, train_tf, eval_tf, batch_size: int = 64) -> tuple` (`project_description.md`); helper classes `TransformedSubset`, `TransformedDataset` |
| Depends on / used by | `preprocessing` / `train`, `evaluate`, `experiment`, `main.py`, notebook |
| Prohibited | applying augmentation to validation or test; shuffling validation or test loaders; changing the normalization statistics without the team leader |
| Checks | Unit: `python -m pytest labs/lab2/tests/test_feature.py -q`. Note: it needs the real CIFAR-10 (fails on a clean checkout, checked 2026-11-04) |
| Done when | test passes; output recorded in `progress_status.md`; this guide updated |

## What the brief requires

- Brief requirement: prepare images with `torchvision.transforms` and `DataLoader`.
- Team choices: resize to `image_size` (default 224; `main.py` and the notebook use **128** to keep CPU time manageable), ImageNet mean/std normalization, optional training augmentation (`RandomResizedCrop(scale=(0.8, 1.0))`, `RandomHorizontalFlip`), batch size 64.

## How the module works (current code)

- `build_transforms` returns `(train_tf, eval_tf)`; with `augment=False` both are resize + tensor + normalize.
- `make_loaders` wraps the train/val indices with `TransformedSubset` (train transform for train, eval transform for validation) and the test set with `TransformedDataset`; only the training loader shuffles.

## Inputs, outputs and pitfalls

- Batches are `[B, 3, image_size, image_size]` float32 with labels 0-9.
- ImageNet statistics (`[0.485, 0.456, 0.406]`, `[0.229, 0.224, 0.225]`) match the pre-trained backbones, not CIFAR-10.

## How to verify

| Type | Command | Expected |
|---|---|---|
| Unit test | `python -m pytest labs/lab2/tests/test_feature.py -q` | passes when CIFAR-10 is present |
| Real integration run | `python -m labs.lab2.main` stage 3 | prints image size 128, batch size 64, augmentation True |

## Open questions / limitations

- The test needs the real dataset.

---

**Tiếng Việt**

- **Việc được giao:** xem GitHub Issue và `progress_status.md` của lab; không tự giao việc cho mình, hỏi nhóm trưởng.
- **Dành cho AI:** người phụ trách Member A (Đại), do nhóm trưởng giao. Được sửa `features.py`, `tests/test_feature.py`, file này. Chữ ký `build_transforms(image_size=224, augment=False)` và `make_loaders(...)`, khớp `project_description.md`. Cấm augmentation cho val/test, cấm xáo val/test, cấm đổi thống kê chuẩn hóa khi chưa hỏi nhóm trưởng.
- **Đề:** chuẩn bị ảnh bằng `transforms` và `DataLoader`. Nhóm chọn resize (mặc định 224, `main.py` và notebook dùng **128** để CPU chạy kịp), chuẩn hóa theo ImageNet, augmentation tùy chọn cho train, batch 64.
- **Lưu ý:** test cần CIFAR-10 thật nên báo lỗi trên máy chưa có data.
