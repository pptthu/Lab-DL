# Lab 2 / evaluate: guide for AI agents

## For AI agents (read first)

| Field | Value |
|---|---|
| Module | `labs/lab2/evaluate/evaluate.py` |
| Owner | Member B (Bảo), assigned by the team leader |
| Status | done in `progress_status.md` (Reported only) |
| Assigned work | see the GitHub Issues and `progress_status.md` of the lab; never assign work to yourself, ask the team leader |
| Files you may edit | `labs/lab2/evaluate/evaluate.py`, `labs/lab2/tests/test_evaluate.py`, this guide |
| Files you must not edit | Member A's modules, saved checkpoints, other owners' notebook cells |
| Agreed signatures | `evaluate(model, loader, device) -> (avg_loss, accuracy)`; `confusion_matrix(model, loader, device, num_classes: int = 10) -> np.ndarray`; `top_confusions(cm, class_names, k: int = 5) -> list` (`project_description.md`) |
| Depends on / used by | `train` checkpoint, `features` loaders / `main.py` stage 6, notebook |
| Prohibited | choosing a model or a hyperparameter from test results |
| Checks | Unit: `python -m pytest labs/lab2/tests/test_evaluate.py -q` (fake data) |
| Done when | tests pass; results recorded in `progress_status.md`; this guide updated |

## What the brief requires

- Brief requirement: evaluate the model on a test set.
- Team choices: loss and accuracy; confusion matrix (rows = true, columns = predicted) and the most frequent confusions; validation for every choice, test once for the selected checkpoint.

## How the module works (current code)

- `evaluate`: `eval` mode, `torch.no_grad()`, mean loss and accuracy in 0-1.
- `confusion_matrix`: argmax predictions counted into `[C, C]`.
- `top_confusions`: the `k` largest off-diagonal cells as `(true_class, predicted_class, count)`.

## How to verify

| Type | Command | Expected |
|---|---|---|
| Unit test | `python -m pytest labs/lab2/tests/test_evaluate.py -q` | all pass (4 passed, 2026-11-04) |
| Real integration run | `python -m labs.lab2.main` stage 6 | recorded in the notebook (ResNet-18, frozen, 3 epochs, image 128): test 79.00% (7,900 / 10,000), val 78.18% (Reported only) |

## Open questions / limitations

- None for this module.

---

**Tiếng Việt**

- **Việc được giao:** xem GitHub Issue và `progress_status.md` của lab; không tự giao việc cho mình, hỏi nhóm trưởng.
- **Dành cho AI:** người phụ trách Member B (Bảo), do nhóm trưởng giao. Được sửa `evaluate.py`, `tests/test_evaluate.py`, file này. Chữ ký 3 hàm khớp `project_description.md`. Cấm chọn model hay siêu tham số theo kết quả test.
- **Đề:** đánh giá model trên tập test. Nhóm dùng loss, accuracy, confusion matrix và các cặp nhầm nhiều nhất; test chỉ dùng một lần cho checkpoint đã chọn.
- **Kiểm chứng:** notebook ghi ResNet-18 đóng băng, 3 epoch, ảnh 128: test 79,00%, val 78,18% (chỉ có trong báo cáo).
