# Lab 2 / experiment: guide for AI agents

## For AI agents (read first)

| Field | Value |
|---|---|
| Module | `labs/lab2/experiment/experiment.py` |
| Owner | Member B (Bảo), assigned by the team leader |
| Status | done in `progress_status.md`; TensorBoard made optional in PR #48 (Reported only) |
| Assigned work | see the GitHub Issues and `progress_status.md` of the lab; never assign work to yourself, ask the team leader |
| Files you may edit | `labs/lab2/experiment/experiment.py`, `labs/lab2/tests/test_experiment.py`, `labs/lab2/tests/test_tensorboard_optional.py`, this guide |
| Files you must not edit | Member A's modules, saved results and checkpoints, other owners' notebook cells |
| Agreed signatures | `run_comparison(model_names, hparam_grid, loaders, device, log_root, ...)`; `save_results(results, csv_path)` (`project_description.md`). The code adds optional `build_model_fn`, `unfreeze_fn`, `train_model_fn` (for tests) and `enable_tensorboard=True`, plus `make_summary_writer(log_dir)` |
| Depends on / used by | `model`, `train` / `main.py` stage 7, notebook comparison section |
| Prohibited | comparing configurations on the test set; adding models or a larger grid without the team leader; starting the comparison (hours on CPU) without asking |
| Checks | Unit: `python -m pytest labs/lab2/tests/test_experiment.py labs/lab2/tests/test_tensorboard_optional.py -q` (fake models) |
| Done when | tests pass; results recorded in `progress_status.md` and the README; this guide updated |

## What the brief requires

- Brief requirement: experiment with different models and hyperparameters; monitor with TensorBoard.
- Team choices: models ResNet-18 and DenseNet-121 (VGG-16 supported); grid = frozen backbone vs fine-tuning the last block; comparison on **validation** accuracy only.

## How the module works (current code)

- For each model and each grid entry: build, optionally unfreeze, train, record `best_val_acc`, `best_val_loss`, `best_epoch` of the saved checkpoint, `train_seconds`, checkpoint path, whether TensorBoard logged.
- The loaders are built once outside, so a `batch_size` in the grid has no effect. The seed is set once before the loop, not per run.
- `save_results` flattens the hyperparameters into `hp_*` columns of a CSV.

## How to verify

| Type | Command | Expected |
|---|---|---|
| Unit test | see Checks | all pass (2 passed; 5 passed, 1 skipped without `tensorboard`, 2026-11-04) |
| Real integration run | `python -m labs.lab2.main` (only when the team agrees) | recorded in the notebook (2 epochs, image 128), best validation accuracy: ResNet-18 frozen 77.60%, ResNet-18 last block unfrozen 91.16%, DenseNet-121 frozen 82.46%, DenseNet-121 last block unfrozen 90.22% (Reported only) |

## Open questions / limitations

- One seed per configuration: differences of a few points are not tested for significance.
- `batch_size` in the grid is ignored (see above).

---

**Tiếng Việt**

- **Việc được giao:** xem GitHub Issue và `progress_status.md` của lab; không tự giao việc cho mình, hỏi nhóm trưởng.
- **Dành cho AI:** người phụ trách Member B (Bảo), do nhóm trưởng giao. Được sửa `experiment.py`, hai file test liên quan, file này. Chữ ký theo `project_description.md`; code thêm tham số tùy chọn để test và `enable_tensorboard`. Cấm so sánh trên tập test, cấm thêm model hay lưới lớn hơn khi chưa hỏi nhóm trưởng, cấm tự chạy phần so sánh (mất vài giờ trên CPU).
- **Đề:** thử nhiều model và siêu tham số, theo dõi bằng TensorBoard. Nhóm so ResNet-18 và DenseNet-121, đóng băng so với mở khối cuối, chỉ so trên validation.
- **Lưu ý:** `batch_size` trong lưới không có tác dụng vì loader tạo một lần; seed chỉ đặt một lần trước vòng lặp; mỗi cấu hình 1 seed.
- **Kiểm chứng:** notebook ghi (2 epoch, ảnh 128): ResNet-18 đóng băng 77,60%, mở khối cuối 91,16%; DenseNet-121 đóng băng 82,46%, mở khối cuối 90,22% (chỉ có trong báo cáo).
