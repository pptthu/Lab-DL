# Lab 1 / model: guide for AI agents

## For AI agents (read first)

| Field | Value |
|---|---|
| Module | `labs/lab1/model/model.py` |
| Owner | Member A (Toàn), assigned by the team leader |
| Status | done in `progress_status.md` (Smoke-tested) |
| Assigned work | see the GitHub Issues and `progress_status.md` of the lab; never assign work to yourself, ask the team leader |
| Files you may edit | `labs/lab1/model/model.py`, `labs/lab1/tests/test_model.py`, this guide |
| Files you must not edit | other modules, saved checkpoints, other owners' notebook cells |
| Agreed signatures | `class MLP(hidden_sizes=(256, 128), num_classes: int = 10, dropout: float = 0.0)`, `forward(x [B,1,28,28]) -> logits [B,10]`; `count_parameters(model) -> int` (same as `project_description.md`) |
| Depends on / used by | torch / `train`, `evaluate`, `experiment.build_model`, `main.py`, notebook |
| Prohibited | adding softmax in `forward` (CrossEntropyLoss expects logits); adding a CNN or other architectures (team decision 2026-11-04: Lab 1 is MLP only); changing the default sizes without the team leader (the saved baseline depends on them) |
| Checks | Unit: `python -m pytest labs/lab1/tests/test_model.py -q` (shapes, logits, parameter count, dropout, overfit one batch) |
| Done when | unit tests pass; this guide updated |

## What the brief requires

- Brief requirement: build a multilayer perceptron for Fashion-MNIST in PyTorch.
- Team choices: baseline 784 -> 256 -> 128 -> 10 with ReLU; optional dropout after each hidden layer; `hidden_sizes=()` gives softmax regression (no hidden layer), used as the linear reference in `experiment`.
- Not in scope (team decision): CNN.

## How the module works (current code)

- Flattens the input to 784 values, then `Linear` + `ReLU` (+ `Dropout` if `dropout > 0`) per hidden size, then a final `Linear` to 10 logits.
- `count_parameters` counts trainable parameters only (`requires_grad=True`).

## Inputs, outputs and pitfalls

- Baseline has 235,146 trainable parameters; `larger_network` (512-256-128) is used only in `experiment`.

## How to verify

| Type | Command | Expected |
|---|---|---|
| Unit test | `python -m pytest labs/lab1/tests/test_model.py -q` | all pass |
| Smoke test | `python -c "from labs.lab1.model.model import MLP, count_parameters; print(count_parameters(MLP()))"` | `235146` |

## Open questions / limitations

- An MLP ignores the 2-D layout of the image; this is a known limit of the model, not a task.

---

**Tiếng Việt**

- **Việc được giao:** xem GitHub Issue và `progress_status.md` của lab; không tự giao việc cho mình, hỏi nhóm trưởng.
- **Dành cho AI:** người phụ trách Member A (Toàn), do nhóm trưởng giao. Được sửa `model.py`, `tests/test_model.py`, file này. Chữ ký `MLP(hidden_sizes=(256,128), num_classes=10, dropout=0.0)` trả logits `[B,10]`, `count_parameters(model)`, khớp `project_description.md`. Cấm thêm softmax trong `forward`, cấm thêm CNN hay kiến trúc khác (nhóm đã chốt ngày 04/11: Lab 1 chỉ làm MLP), cấm đổi kích thước mặc định khi chưa hỏi nhóm trưởng.
- **Đề:** xây MLP cho Fashion-MNIST bằng PyTorch. Nhóm chọn 784 -> 256 -> 128 -> 10, ReLU, dropout tùy chọn; `hidden_sizes=()` là softmax regression để so sánh.
- **Kiểm chứng:** baseline có 235.146 tham số huấn luyện được.
