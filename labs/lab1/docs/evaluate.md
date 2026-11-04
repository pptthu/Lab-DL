# Lab 1 / evaluate: guide for AI agents

## For AI agents (read first)

| Field | Value |
|---|---|
| Module | `labs/lab1/evaluate/evaluate.py` |
| Owner | Member B (Thư), assigned by the team leader |
| Status | done in `progress_status.md` (Run on the official data, 2026-11-04) |
| Assigned work | see the GitHub Issues and `progress_status.md` of the lab; never assign work to yourself, ask the team leader |
| Files you may edit | `labs/lab1/evaluate/evaluate.py`, `labs/lab1/tests/test_evaluate.py`, this guide |
| Files you must not edit | Member A's modules, raw data, saved checkpoints, other owners' notebook cells |
| Agreed signatures | `evaluate(model, loader, device) -> (avg_loss, accuracy)`; `confusion_matrix(model, loader, device, num_classes: int = 10) -> np.ndarray`; `per_class_accuracy(cm) -> dict`; `save_classification_report(cm, class_names, csv_path: str) -> DataFrame` (same as `project_description.md`) |
| Depends on / used by | `train` checkpoint, `features` loaders / `main.py`, `experiment.analyze_seed_runs`, notebook |
| Prohibited | choosing a model or a hyperparameter from test results; tests writing to the real `artifacts/runs/` |
| Checks | Unit: `python -m pytest labs/lab1/tests/test_evaluate.py -q` (dummy loader, perfect model, known matrix, zero division) |
| Done when | unit tests pass; results recorded in `progress_status.md`; this guide updated |

## What the brief requires

- Brief requirement: evaluate the trained model on held-out data.
- Team choices: accuracy and loss on validation and test; confusion matrix (rows = true class, columns = predicted); per-class precision, recall, specificity, F1 saved as CSV. The test set is used only to report the baseline fixed beforehand.

## How the module works (current code)

- `evaluate`: `eval` mode, `torch.no_grad()`, sample-weighted mean loss and accuracy in 0-1.
- `confusion_matrix`: argmax predictions counted into a `[10,10]` array.
- `per_class_accuracy`: diagonal / row sum, 0 for an empty row.
- `save_classification_report`: precision, recall, specificity, F1 per class into a CSV.

## Inputs, outputs and pitfalls

- Use the validation loader for every comparison and the test loader only for the final report.

## How to verify

| Type | Command | Expected |
|---|---|---|
| Unit test | `python -m pytest labs/lab1/tests/test_evaluate.py -q` | all pass |
| Real integration run | `python -m labs.lab1.main` | saved model (seed 42): validation 89.05% (loss 0.3018), test 88.36% (loss 0.3338); validation recall highest Trouser 98.17%, lowest Shirt 70.17% (Run, 2026-11-04, official data) |

## Open questions / limitations

- The test set was seen several times during development, though no choice used it.

---

**Tiếng Việt**

- **Việc được giao:** xem GitHub Issue và `progress_status.md` của lab; không tự giao việc cho mình, hỏi nhóm trưởng.
- **Dành cho AI:** người phụ trách Member B (Thư), do nhóm trưởng giao. Được sửa `evaluate.py`, `tests/test_evaluate.py`, file này. Chữ ký 4 hàm khớp `project_description.md`. Cấm chọn model hay siêu tham số theo kết quả test, cấm test ghi vào `artifacts/runs/` thật.
- **Đề:** đánh giá model trên dữ liệu chưa thấy. Nhóm dùng accuracy và loss trên val và test, confusion matrix, báo cáo từng lớp; test chỉ để báo cáo baseline đã chốt trước.
- **Kiểm chứng:** model đã lưu (seed 42): val 89,05%, test 88,36%; recall cao nhất Trouser 98,17%, thấp nhất Shirt 70,17% (đã chạy 04/11/2026 trên data chính thức).
