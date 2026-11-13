# Lab 1 / train: guide for AI agents

## For AI agents (read first)

| Field | Value |
|---|---|
| Module | `labs/lab1/train/train.py` |
| Owner | Member B (Thư), assigned by the team leader |
| Status | done in `progress_status.md` (Run on the official data, 2026-11-04) |
| Assigned work | see the GitHub Issues and `progress_status.md` of the lab; never assign work to yourself, ask the team leader |
| Files you may edit | `labs/lab1/train/train.py`, `labs/lab1/tests/test_train.py`, this guide |
| Files you must not edit | Member A's modules, raw data, committed split file, other owners' notebook cells |
| Agreed signatures | `train_model(model, train_loader, val_loader, epochs: int, lr: float, device, ckpt_path: str, optimizer_name: str = 'adam', weight_decay: float = 0.0) -> dict`; `save_checkpoint(model, path: str)`; `load_checkpoint(model, path: str, device) -> nn.Module` (same as `project_description.md`) |
| Depends on / used by | `features` loaders, `model` / `evaluate`, `experiment`, `main.py`, `extended_study.py`, notebook |
| Prohibited | using the test loader during training or for checkpoint selection; tests writing to the real `artifacts/models/` (use `tmp_path`); starting a full run (35 runs) without the team leader |
| Checks | Unit: `python -m pytest labs/lab1/tests/test_train.py -q` (fake data, includes overfit-one-batch and save/load). Real run: `main.py`, only when the team agrees |
| Done when | unit tests pass; real run recorded in `progress_status.md`; this guide updated |

## What the brief requires

- Brief requirement: train the network with a loss and an optimizer, and monitor validation.
- Team choices: CrossEntropyLoss; Adam lr 1e-3 (SGD with momentum 0.9 only in experiments); 5 epochs; keep the checkpoint with the **lowest validation loss**.

## How the module works (current code)

- Each epoch: `zero_grad`, forward, loss, `backward`, `step` on the training loader; then validation under `torch.no_grad()` in `eval` mode.
- Losses are weighted by batch size and divided by the number of samples, so the last smaller batch is counted exactly.
- Returns `history` with `train_loss`, `val_loss`, `val_acc` per epoch; saves the state_dict to `ckpt_path` whenever validation loss improves.
- `optimizer_name` other than `adam` or `sgd` raises `ValueError`. `load_checkpoint` uses `weights_only=True` and switches to eval mode.

## Inputs, outputs and pitfalls

- The reported validation accuracy must come from the same epoch as the saved checkpoint (lowest val loss), not from `max(val_acc)`; `experiment` does this.

## How to verify

| Type | Command | Expected |
|---|---|---|
| Unit test | `python -m pytest labs/lab1/tests/test_train.py -q` | all pass |
| Real integration run | `python -m labs.lab1.main` | seed 42, best epoch 5: val loss 0.3018, val accuracy 89.05% (Run, 2026-11-04, official data) |

## Open questions / limitations

- CPU time varies a lot between runs on the same laptop (one 5-epoch run took 1.5 hours on 2026-11-04, cause not found); accuracy does not depend on it.

---

**Tiếng Việt**

- **Việc được giao:** xem GitHub Issue và `progress_status.md` của lab; không tự giao việc cho mình, hỏi nhóm trưởng.
- **Dành cho AI:** người phụ trách Member B (Thư), do nhóm trưởng giao. Được sửa `train.py`, `tests/test_train.py`, file này. Chữ ký 3 hàm khớp `project_description.md`. Cấm dùng tập test khi train hay khi chọn checkpoint, cấm test ghi vào `artifacts/models/` thật, cấm tự chạy toàn bộ 35 lần khi chưa hỏi nhóm trưởng.
- **Đề:** huấn luyện mạng với loss và optimizer, theo dõi validation. Nhóm chọn CrossEntropyLoss, Adam lr 1e-3, 5 epoch, giữ checkpoint có **validation loss thấp nhất**.
- **Cách hoạt động:** vòng lặp chuẩn; loss tính theo trọng số batch; lưu checkpoint mỗi khi val loss giảm; accuracy báo cáo phải lấy đúng epoch của checkpoint.
- **Kiểm chứng:** seed 42, epoch tốt nhất là 5: val loss 0,3018, val accuracy 89,05% (đã chạy 04/11/2026 trên data chính thức).
