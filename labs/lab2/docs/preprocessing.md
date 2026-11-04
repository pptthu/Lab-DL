# Lab 2 / preprocessing: guide for AI agents

## For AI agents (read first)

| Field | Value |
|---|---|
| Module | `labs/lab2/preprocessing/preprocessing.py` |
| Owner | Member A (Đại), assigned by the team leader |
| Status | done in `progress_status.md` (Reported only) |
| Assigned work | see the GitHub Issues and `progress_status.md` of the lab; never assign work to yourself, ask the team leader |
| Files you may edit | `labs/lab2/preprocessing/preprocessing.py`, `labs/lab2/tests/test_preprocessing.py`, this guide |
| Files you must not edit | other modules, raw data, other owners' notebook cells. `artifacts/splits/train_val_split.json` only on purpose, with the team leader (every reported number depends on it) |
| Agreed signatures | `load_raw_dataset(data_root: str, download: bool = True) -> tuple`; `make_split(n_samples: int, val_ratio: float = 0.1, seed: int = 42) -> tuple`; `save_split(train_idx, val_idx, path: str)`; `load_split(path: str) -> tuple` (`project_description.md`) |
| Depends on / used by | `torchvision`, numpy / `features`, `main.py`, notebook |
| Prohibited | taking validation from the test set; changing `val_ratio` or `seed` without the team leader; committing CIFAR-10 files |
| Checks | Unit: `python -m pytest labs/lab2/tests/test_preprocessing.py -q`. Note: it calls `load_raw_dataset(download=False)` on the real data, so it **fails** on a clean checkout without CIFAR-10 (checked 2026-11-04) |
| Done when | test passes; output recorded in `progress_status.md`; this guide updated |

## What the brief requires

- Brief requirement: load the dataset and prepare training, validation and test data.
- Team choices: CIFAR-10 train/test from `torchvision`; validation = 10% of the training images, seed 42; the official test set is the test set.

## How the module works (current code)

- `load_raw_dataset`: `CIFAR10(root=data_root, train=True/False, download=download, transform=None)`; data lives in `raw_data_dir("lab2")` (never `./data/raw`).
- `make_split`: `np.random.RandomState(seed).permutation(n_samples)`, then the first `val_ratio` part is validation. This is a **random** split, not stratified (CIFAR-10 is balanced, so classes stay close to 10% each, but not exactly).
- `save_split` / `load_split`: JSON with the two index lists. `main.py` loads the committed file if it exists.

## Inputs, outputs and pitfalls

- With 50,000 training images: 45,000 train / 5,000 validation.

## How to verify

| Type | Command | Expected |
|---|---|---|
| Unit test | `python -m pytest labs/lab2/tests/test_preprocessing.py -q` | passes when CIFAR-10 is present |
| Real integration run | `python -m labs.lab2.main` stage 2 | "Loaded existing split", train 45,000 / val 5,000 / test 10,000 |

## Open questions / limitations

- The test needs the real dataset (see above).
- Some README text may call the split stratified; the code does a random split.

---

**Tiếng Việt**

- **Việc được giao:** xem GitHub Issue và `progress_status.md` của lab; không tự giao việc cho mình, hỏi nhóm trưởng.
- **Dành cho AI:** người phụ trách Member A (Đại), do nhóm trưởng giao. Được sửa `preprocessing.py`, `tests/test_preprocessing.py`, file này; chỉ sửa `train_val_split.json` khi có chủ đích và đã hỏi nhóm trưởng. Chữ ký 4 hàm khớp `project_description.md`. Cấm lấy val từ test, cấm đổi `val_ratio` hay `seed` khi chưa hỏi, cấm commit data.
- **Đề:** nạp dataset, chuẩn bị train, validation và test. Nhóm chọn 10% ảnh train làm validation, seed 42.
- **Cách hoạt động:** `make_split` xáo ngẫu nhiên theo seed rồi cắt 10% làm val, tức là chia **ngẫu nhiên, không phân tầng**; 45.000 / 5.000.
- **Lưu ý:** test gọi `download=False` trên data thật nên **báo lỗi** trên máy chưa có CIFAR-10.
