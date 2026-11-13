# Lab 1 / visualize: guide for AI agents

## For AI agents (read first)

| Field | Value |
|---|---|
| Module | `labs/lab1/visualize/visualize.py` |
| Owner | Member B (Thư), assigned by the team leader |
| Status | done in `progress_status.md` (Run on the official data, 2026-11-04) |
| Assigned work | see the GitHub Issues and `progress_status.md` of the lab; never assign work to yourself, ask the team leader |
| Files you may edit | `labs/lab1/visualize/visualize.py`, `labs/lab1/tests/test_visualize.py`, this guide |
| Files you must not edit | other modules, saved checkpoints, other owners' notebook cells |
| Agreed signatures | `plot_loss_curves(history: dict, out_path: str)`; `show_predictions(model, loader, class_names, device, out_path: str, n: int = 16)`; `plot_confusion_matrix(cm, class_names, out_path: str)` (same as `project_description.md`) |
| Depends on / used by | `train` history, `evaluate.confusion_matrix` / `main.py` stage 7, notebook Stage 8 |
| Prohibited | committing generated figures (`artifacts/figures/` is ignored); tests writing to the real figures folder (use `tmp_path`) |
| Checks | Unit: `python -m pytest labs/lab1/tests/test_visualize.py -q` (files created and non-empty, class-name checks) |
| Done when | unit tests pass; figures looked at by a person after a real run; this guide updated |

## What the brief requires

- Brief requirement: show the training curves and example predictions.
- Team choice: add a confusion matrix heatmap.

## How the module works (current code)

- `plot_loss_curves`: train and validation loss per epoch.
- `show_predictions`: `n` validation images with predicted and true labels, green if correct, red if wrong.
- `plot_confusion_matrix`: annotated heatmap; `class_names` must have one name per class.

## How to verify

| Type | Command | Expected |
|---|---|---|
| Unit test | `python -m pytest labs/lab1/tests/test_visualize.py -q` | all pass |
| Real integration run | `python -m labs.lab1.main` stage 7 | `loss_curves.png`, `predictions_grid.png`, `confusion_matrix.png` in `labs/lab1/artifacts/figures/` (Run, 2026-11-04) |

## Open questions / limitations

- The figures of the 2026-11-04 run still need to be looked at by a person.

---

**Tiếng Việt**

- **Việc được giao:** xem GitHub Issue và `progress_status.md` của lab; không tự giao việc cho mình, hỏi nhóm trưởng.
- **Dành cho AI:** người phụ trách Member B (Thư), do nhóm trưởng giao. Được sửa `visualize.py`, `tests/test_visualize.py`, file này. Chữ ký 3 hàm khớp `project_description.md`. Cấm commit hình sinh ra, cấm test ghi vào thư mục hình thật.
- **Đề:** vẽ đường cong huấn luyện và ví dụ dự đoán; nhóm thêm confusion matrix.
- **Kiểm chứng:** 3 hình được lưu trong `artifacts/figures/` (đã chạy 04/11/2026); hình chưa được người xem lại.
