# Lab 1 / preprocessing: guide for AI agents

## For AI agents (read first)

| Field | Value |
|---|---|
| Module | `labs/lab1/preprocessing/preprocessing.py`, plus the script `labs/lab1/preprocessing/download_official.py` |
| Owner | Member A (Toàn), assigned by the team leader |
| Status | done in `progress_status.md` (Run on the official data, 2026-11-02 and 2026-11-04) |
| Assigned work | see the GitHub Issues and `progress_status.md` of the lab; never assign work to yourself, ask the team leader |
| Files you may edit | `labs/lab1/preprocessing/preprocessing.py`, `labs/lab1/preprocessing/download_official.py`, `labs/lab1/tests/test_preprocessing.py`, this guide |
| Files you must not edit | other modules, raw data, other owners' notebook cells. `artifacts/splits/train_val_split.json` only on purpose, with the team leader (every reported number depends on it) |
| Agreed signatures | `load_csv(csv_path: str) -> tuple`; `make_split(labels, val_ratio: float = 0.1, seed: int = 42) -> tuple`; `save_split(train_idx, val_idx, path: str) -> None`; `load_split(path: str) -> tuple` (same as `project_description.md`) |
| Depends on / used by | pandas, numpy, torchvision (download script) / `features`, `main.py`, `extended_study.py`, notebook |
| Prohibited | taking validation from the test file; changing `val_ratio` or `seed` without the team leader; committing CSV or IDX files; downloads inside unit tests |
| Checks | Unit: `python -m pytest labs/lab1/tests/test_preprocessing.py -q` (fake CSV). Smoke: regenerate the split and compare it with the committed file |
| Done when | unit tests pass; output recorded in `progress_status.md`; this guide updated |

## What the brief requires

- Brief requirement: prepare the data and split it for training, validation and testing.
- Team choices: official Zalando files (https://github.com/zalandoresearch/fashion-mnist) converted to CSV; stratified 90/10 train/validation split of the 60,000 training images, seed 42; the official 10,000 test images are the test set.

## How the module works (current code)

- `download_official.py`: `torchvision.datasets.FashionMNIST(download=True)` fetches the 4 IDX files and checks their MD5; the script writes `fashion-mnist_train.csv` and `fashion-mnist_test.csv` (`label`, `pixel1`..`pixel784`) into `raw_data_dir("lab1")` and reads them back with `load_csv` to compare every pixel and label.
- `load_csv`: checks the `label` column and 784 pixel columns, checks the range 0-255 **before** casting to `uint8`, returns images `[N,28,28]` uint8 and labels `[N]` int64.
- `make_split`: per class, a seeded permutation; `val_ratio` of each class goes to validation, so validation has 600 images per class.
- `save_split` / `load_split`: JSON with `train_idx` and `val_idx`.

## Inputs, outputs and pitfalls

- Never write data paths by hand; use `raw_data_dir("lab1")` from `shared/paths.py`.
- On Windows with a Vietnamese path, set `PYTHONIOENCODING=utf-8` before running the scripts, otherwise the last `print` can fail with `UnicodeEncodeError` (the files are written before it).
- The old Kaggle CSVs can stay locally as `*_kaggle.csv`; they are not used.

## How to verify

| Type | Command | Expected |
|---|---|---|
| Unit test | `python -m pytest labs/lab1/tests/test_preprocessing.py -q` | all pass |
| Smoke test | `python -m labs.lab1.preprocessing.download_official` | train 60,000 rows (6,000 per class), test 10,000 rows (1,000 per class); MD5 train CSV `8ff2cced1c624159b21dac4f735b4215`, test CSV `78e18be2098f1256a07ce6f91a4d436e` (Run, 2026-11-04) |
| Smoke test | regenerate `make_split(labels, 0.1, 42)` from the new CSV and compare with `train_val_split.json` | identical, 54,000 / 6,000 (Verified, 2026-11-04) |

## Open questions / limitations

- One split only; results across splits are not measured.

---

**Tiếng Việt**

- **Việc được giao:** xem GitHub Issue và `progress_status.md` của lab; không tự giao việc cho mình, hỏi nhóm trưởng.
- **Dành cho AI:** người phụ trách Member A (Toàn), do nhóm trưởng giao. Được sửa `preprocessing.py`, `download_official.py`, `tests/test_preprocessing.py`, file này. Chỉ sửa `train_val_split.json` khi có chủ đích và đã hỏi nhóm trưởng (mọi con số phụ thuộc vào nó). Chữ ký 4 hàm khớp `project_description.md`. Cấm lấy val từ file test, cấm đổi `val_ratio` hay `seed` khi chưa hỏi, cấm commit data, cấm tải trong unit test.
- **Đề:** chuẩn bị và chia dữ liệu. Nhóm chọn file gốc Zalando, chia phân tầng 90/10 với seed 42; test là 10.000 ảnh test chính thức.
- **Cách hoạt động:** `download_official.py` tải 4 file IDX (tự kiểm MD5), ghi ra CSV và đọc lại để so từng pixel; `load_csv` kiểm khoảng 0-255 trước khi ép kiểu; `make_split` chia theo từng lớp (600 ảnh val mỗi lớp).
- **Lưu ý:** dùng `raw_data_dir("lab1")`; trên Windows có đường dẫn tiếng Việt thì đặt `PYTHONIOENCODING=utf-8`.
- **Kiểm chứng:** tạo lại split từ CSV mới ra đúng y file đã commit, 54.000 / 6.000 (đã kiểm, 04/11/2026).
