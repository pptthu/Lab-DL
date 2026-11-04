# Lab 2 / demo: how to run and present the lab

## For AI agents (read first)

| Field | Value |
|---|---|
| Scope | `labs/lab2/main.py`, `labs/lab2/notebook/lab2_presentation.ipynb`, the README |
| Owner | team leader (Thư); `main.py` was wired in PR #48 |
| Status | `main.py` wired end to end (Implemented, tested with fakes in `test_main_pipeline.py`); recorded results come from the notebook (Reported only) |
| Assigned work | see the GitHub Issues and `progress_status.md` of the lab; never assign work to yourself, ask the team leader |
| Files you may edit | `main.py`, the notebook, the README, this guide (team leader's files) |
| Files you must not edit | the module files (owners in each module guide), raw data, committed split file |
| Prohibited | starting `main.py` with the comparison stage or a long notebook run without the team leader (hours on CPU); adding models beyond the three supported ones; committing checkpoints, data or TensorBoard logs; writing numbers that do not come from a recorded run |
| Checks | `python -m pytest labs/lab2/tests -q` (needs CIFAR-10 locally, see below); real runs only when the team agrees |
| Done when | the README, the notebook and `progress_status.md` show the numbers of the same recorded run |

## Current (what runs today)

| Step | Command (repository root) | Notes |
|---|---|---|
| Install | `pip install -r labs/lab2/requirements.txt` | includes `tensorboard` (optional at run time) |
| Tests | `python -m pytest labs/lab2/tests -q` | on 2026-11-04 in a clean checkout: 10 of 12 files pass quickly; `test_eda.py` downloads CIFAR-10 and takes over 4 minutes; `test_preprocessing.py` and `test_feature.py` fail until CIFAR-10 is downloaded |
| Baseline only | `python -m labs.lab2.main --no-experiments` | EDA, split, ResNet-18 (frozen, image 128, 3 epochs), validation and one test evaluation, EDA figures, history CSV and `artifacts/runs/pipeline_summary.json` |
| Full pipeline | `python -m labs.lab2.main` | adds the comparison stage (ResNet-18 and DenseNet-121, frozen vs last block unfrozen); hours on CPU |
| Options | `python -m labs.lab2.main --help` | `--model`, `--epochs`, `--image-size`, `--batch-size`, `--lr`, `--unfreeze-blocks`, `--no-freeze`, `--no-augment`, `--no-download`, `--comparison-models`, ... |
| TensorBoard | `tensorboard --logdir labs/lab2/artifacts/logs` (baseline) or `labs/lab2/artifacts/runs` (comparison) | only if `tensorboard` is installed |
| Notebook | `labs/lab2/notebook/lab2_presentation.ipynb` | saved outputs from the run committed in PR #48 |

Recorded results (notebook, Reported only): ResNet-18 frozen, 3 epochs, image 128: validation 78.18%, test 79.00%. Comparison (2 epochs each, validation): ResNet-18 frozen 77.60%, ResNet-18 last block unfrozen 91.16%, DenseNet-121 frozen 82.46%, DenseNet-121 last block unfrozen 90.22%.

## Intended (what the brief needs)

- Deliverables (team leader, 2026-11-04): the full source code of the three labs and one written report (PDF, written later).
- Tests that run on a clean checkout without downloading CIFAR-10 (fake data for `eda`, `preprocessing`, `features`).
- One recorded run whose numbers appear in the README, the notebook and `progress_status.md`.

## What to show in the oral exam

- Transfer learning vs fine-tuning: frozen backbone + new head, then unfreezing the last block; why fine-tuning helped in the recorded comparison.
- Why ImageNet normalization and resizing to 128 (CPU time) instead of 224.
- Checkpoint selection by validation loss; test used once.
- TensorBoard scalars (`Loss/train`, `Loss/val`, `Accuracy/val`).
- Limits: one seed per configuration; BatchNorm statistics still update under a frozen backbone.

---

**Tiếng Việt**

- **Việc được giao:** xem GitHub Issue và `progress_status.md` của lab; không tự giao việc cho mình, hỏi nhóm trưởng.
- **Dành cho AI:** phần của nhóm trưởng (Thư): `main.py` (đã nối ở PR #48), notebook, README. Cấm sửa file module của người khác, data, split; cấm tự chạy phần so sánh hay notebook dài khi chưa hỏi (mất vài giờ trên CPU); cấm commit checkpoint, data, log TensorBoard; cấm ghi số không lấy từ một lần chạy đã ghi lại.
- **Hiện tại:** `python -m labs.lab2.main --no-experiments` chỉ chạy baseline ResNet-18; chạy không kèm tùy chọn sẽ thêm phần so sánh (vài giờ trên CPU). Trên bản clone sạch, 3 file test cần CIFAR-10 thật (một file tải data hơn 4 phút, hai file báo lỗi nếu chưa có data).
- **Số đã ghi (notebook, chỉ có trong báo cáo):** ResNet-18 đóng băng, 3 epoch, ảnh 128: val 78,18%, test 79,00%. So sánh 2 epoch: ResNet-18 đóng băng 77,60%, mở khối cuối 91,16%; DenseNet-121 đóng băng 82,46%, mở khối cuối 90,22%.
- **Bài nộp:** toàn bộ source của 3 lab và một bài báo cáo (PDF, viết sau).
- **Khi vấn đáp:** transfer learning và fine-tuning; vì sao chuẩn hóa theo ImageNet và dùng ảnh 128; chọn checkpoint theo val loss; TensorBoard; giới hạn: mỗi cấu hình 1 seed, BatchNorm vẫn cập nhật khi đóng băng.
