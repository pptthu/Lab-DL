# Lab 2 / train: guide for AI agents

## For AI agents (read first)

| Field | Value |
|---|---|
| Module | `labs/lab2/train/train.py` |
| Owner | Member B (Bảo), assigned by the team leader |
| Status | done in `progress_status.md`; checkpoint reporting fixed in PR #48 (Reported only) |
| Assigned work | see the GitHub Issues and `progress_status.md` of the lab; never assign work to yourself, ask the team leader |
| Files you may edit | `labs/lab2/train/train.py`, `labs/lab2/tests/test_train.py`, `labs/lab2/tests/test_checkpoint_consistency.py`, this guide |
| Files you must not edit | Member A's modules, saved checkpoints, other owners' notebook cells |
| Agreed signatures | `train_model(model, train_loader, val_loader, epochs: int, lr: float, device, ckpt_path: str, writer=None, optimizer_name: str = 'adam', weight_decay: float = 0.0) -> dict`; `save_checkpoint(model, path: str)`; `load_checkpoint(model, path: str, device) -> nn.Module` (`project_description.md`) |
| Depends on / used by | `features` loaders, `model` / `evaluate`, `experiment`, `main.py`, notebook |
| Prohibited | using the test loader during training or for checkpoint selection; committing checkpoints (`*.pt`); starting a full training run without the team leader (slow on CPU) |
| Checks | Unit: `python -m pytest labs/lab2/tests/test_train.py labs/lab2/tests/test_checkpoint_consistency.py -q` (fake data, fast) |
| Done when | tests pass; real run recorded in `progress_status.md`; this guide updated |

## What the brief requires

- Brief requirement: train with a loss function, an optimizer and a training loop; monitor training with TensorBoard.
- Team choices: CrossEntropyLoss; Adam (or SGD) on the parameters with `requires_grad=True`; keep the checkpoint with the **lowest validation loss**.

## How the module works (current code)

- Each epoch: train loop, then validation in `eval` mode; history `train_loss`, `val_loss`, `val_acc` per epoch, plus `best_epoch`, `best_val_loss`, `best_val_acc` from the same epoch as the saved checkpoint.
- If `writer` is given, logs `Loss/train`, `Loss/val`, `Accuracy/val` per epoch. TensorBoard is optional: `experiment.make_summary_writer` returns `None` when `tensorboard` is not installed.
- A checkpoint is always written in the first epoch, even if the loss is NaN.

## How to verify

| Type | Command | Expected |
|---|---|---|
| Unit test | see Checks | all pass (2026-11-04: 4 and 8 passed) |
| Real integration run | `python -m labs.lab2.main --no-experiments` (only when the team agrees) | ResNet-18 baseline; recorded in the notebook: 3 epochs, image 128, val 78.18% (Reported only) |

## Open questions / limitations

- Frozen BatchNorm statistics still update in training mode (see `model.md`).

---

**Tiếng Việt**

- **Việc được giao:** xem GitHub Issue và `progress_status.md` của lab; không tự giao việc cho mình, hỏi nhóm trưởng.
- **Dành cho AI:** người phụ trách Member B (Bảo), do nhóm trưởng giao. Được sửa `train.py`, hai file test train, file này. Chữ ký 3 hàm khớp `project_description.md`. Cấm dùng tập test khi train hay chọn checkpoint, cấm commit file `.pt`, cấm tự chạy train đầy đủ khi chưa hỏi nhóm trưởng.
- **Đề:** huấn luyện với loss, optimizer, vòng lặp; theo dõi bằng TensorBoard. Nhóm giữ checkpoint có **validation loss thấp nhất**; history trả về chỉ số của đúng epoch đó.
- **Cách hoạt động:** TensorBoard là tùy chọn, máy không cài vẫn chạy được.
- **Kiểm chứng:** test giả chạy nhanh và đạt; notebook ghi baseline ResNet-18 3 epoch, ảnh 128: val 78,18% (chỉ có trong báo cáo).
