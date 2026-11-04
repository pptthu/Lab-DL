# Lab 2 / eda: guide for AI agents

## For AI agents (read first)

| Field | Value |
|---|---|
| Module | `labs/lab2/eda/eda.py` |
| Owner | Member A (Đại), assigned by the team leader |
| Status | done in `progress_status.md` (Reported only) |
| Assigned work | see the GitHub Issues and `progress_status.md` of the lab; never assign work to yourself, ask the team leader |
| Files you may edit | `labs/lab2/eda/eda.py`, `labs/lab2/tests/test_eda.py`, this guide |
| Files you must not edit | other modules, `main.py` (team leader), raw data, committed split file, other owners' notebook cells |
| Agreed signatures | `run_eda(data_root: str, out_dir: str) -> dict` (same as `project_description.md`) |
| Depends on / used by | `torchvision.datasets.CIFAR10` / `main.py` stage 1, notebook EDA section |
| Prohibited | modifying the dataset; using test images to make decisions; committing generated figures |
| Checks | Unit: `python -m pytest labs/lab2/tests/test_eda.py -q`. Note: this test currently uses the **real** CIFAR-10 (downloads about 170 MB on a clean checkout and writes to the real `artifacts/figures/`) |
| Done when | test passes; output recorded in `progress_status.md`; this guide updated |

## What the brief requires

- Brief requirement: prepare data with `torchvision` before transfer learning.
- Team choice: CIFAR-10 from `torchvision.datasets` (official page https://www.cs.toronto.edu/~kriz/cifar.html; the dataset link list came from another group, not from the lecturer). The archive MD5 `c58f30108f718f92721af3b95e74349a` matches the Toronto file (checked 2026-11-02).

## How the module works (current code)

- Loads train and test with `CIFAR10(..., download=True)` (always downloads if missing, even when `main.py` gets `--no-download`).
- Returns `n_train`, `n_test`, `class_counts`, `image_size`, `channel_mean`, `channel_std`; saves PNG figures in `out_dir`.

## Inputs, outputs and pitfalls

- Expected on CIFAR-10: 50,000 train, 10,000 test, 5,000 per class, images 32x32x3.
- `channel_mean`/`channel_std` describe CIFAR-10; the transforms use the ImageNet statistics because the backbones were pre-trained on ImageNet.

## How to verify

| Type | Command | Expected |
|---|---|---|
| Unit test | `python -m pytest labs/lab2/tests/test_eda.py -q` | passes when CIFAR-10 is present; slow on a clean checkout (over 4 minutes, measured 2026-11-04) |
| Real integration run | `python -m labs.lab2.main` stage 1 | 50,000 train / 10,000 test images |

## Open questions / limitations

- The test depends on the real dataset and writes to real artifacts; a fake-data test in `tmp_path` would make it fast and order-independent (not assigned yet).

---

**Tiếng Việt**

- **Việc được giao:** xem GitHub Issue và `progress_status.md` của lab; không tự giao việc cho mình, hỏi nhóm trưởng.
- **Dành cho AI:** người phụ trách Member A (Đại), do nhóm trưởng giao. Được sửa `eda.py`, `tests/test_eda.py`, file này. Chữ ký `run_eda(data_root, out_dir) -> dict`, khớp `project_description.md`. Cấm sửa dataset, cấm dùng ảnh test để ra quyết định, cấm commit hình.
- **Đề:** chuẩn bị dữ liệu bằng `torchvision`. Nhóm chọn CIFAR-10 (danh sách link do nhóm khác tổng hợp, không phải của cô); MD5 file nén khớp file của Toronto.
- **Cách hoạt động:** luôn tải data nếu thiếu (kể cả khi `main.py` có `--no-download`); trả về số ảnh, số ảnh mỗi lớp, kích thước, mean/std từng kênh; lưu hình.
- **Lưu ý:** test hiện dùng CIFAR-10 thật: trên máy chưa có data sẽ tải 170 MB, chạy hơn 4 phút và ghi vào `artifacts/figures/` thật.
