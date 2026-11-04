# Lab 2: Pre-trained CNNs and fine-tuning on CIFAR-10 (PyTorch): progress status
_Last updated: 2026-11-04 by Team Leader (Thư)_

| Module | Owner | Status (todo / doing / done / N/A) | Smoke run | Normal run | Notes |
|---|---|---|---|---|---|
| `eda` | Member A (Đại) | done | pass | pass | Merged to main via PR #32; verified with test_eda.py |
| `preprocessing` | Member A (Đại) | done | pass | pass | Merged to main via PR #32; verified with test_preprocessing.py |
| `features` | Member A (Đại) | done | pass | pass | Merged to main via PR #32; verified with test_feature.py |
| `model` | Member A (Đại) | done | pass | pass | Merged to main via PR #32; verified with test_model.py |
| `train` | Member B (Bảo) | done | pass | pass | Merged to main via PR #25; verified with test_train.py |
| `evaluate` | Member B (Bảo) | done | pass | pass | Merged to main via PR #25; verified with test_evaluate.py |
| `experiment` | Member B (Bảo) | done | pass | pass | Merged to main via PR #25; verified with test_experiment.py |
| `main.py`, README, integration | Team leader (Thư) | done | pass | - | README completed; main.py wired end-to-end (EDA -> ... -> results); CLI flags in `python -m labs.lab2.main --help`; verified with test_main_pipeline.py (mocked stages, no CIFAR-10 run) |

## Handoff A -> B (sequential, no interleaving)
Member B starts coding only after ALL of the following are true:
- [x] split JSON committed in artifacts/splits/
- [x] make_loaders returns a batch [B,3,H,W] with labels 0-9
- [x] build_model returns logits [B,10]; describe_model prints parameter counts
- [x] all Member A modules are `done` in this file and `pytest labs/lab2/tests` passes
- [x] Member A's PRs are merged into `main`; Member B branches from the updated `main`
- [x] Member A posts the handoff message (what was done, exact run commands, known problems)
After the handoff Member A only answers questions and reviews PRs; Member A does not edit Member B's modules.

## Audit: what actually runs
- `pytest labs/lab2/tests/test_smoke.py` -> 8 passed
- `pytest labs/lab2/tests/test_eda.py` -> 1 passed
- `pytest labs/lab2/tests/test_preprocessing.py` -> 1 passed
- `pytest labs/lab2/tests/test_feature.py` -> 1 passed
- `pytest labs/lab2/tests/test_model.py` -> 2 passed
- `pytest labs/lab2/tests/test_train.py` -> 4 passed
- `pytest labs/lab2/tests/test_evaluate.py` -> 4 passed
- `pytest labs/lab2/tests/test_experiment.py` -> 2 passed
- `pytest labs/lab2/tests/test_tensorboard_optional.py` -> 6 passed
- `pytest labs/lab2/tests/test_model_unfreezing.py` -> 9 passed
- `pytest labs/lab2/tests/test_checkpoint_consistency.py` -> 8 passed
- `pytest labs/lab2/tests/test_main_pipeline.py` -> 6 passed
- `pytest labs/lab2/tests/` -> 52 passed in 9.74s
- `ruff check` on the four modified modules + four new test files -> All checks passed

## Log (dated, one line per step)
- 2026-10-17 pptthu: scaffold created; interfaces agreed -> verified with `pytest labs/lab2/tests`
- 2026-10-26 Bảo (Member B): implemented train, evaluate, experiment modules on `feat/lab2-member-b`, added synthetic unit tests and updated presentation notebook sections -> verified with `pytest labs/lab2/tests` (18 passed).
- 2026-10-31 Thư (Leader): verified all 23 tests passing (Member A + Member B modules merged on main); audited baseline ResNet-18 training run and evaluation metrics; authored comprehensive bilingual README.md -> verified with `pytest labs/lab2/tests/` (23 passed).
- 2026-11-04 Thư (Leader): fixed four integration issues: (1) TensorBoard import is now lazy/optional in `experiment`; (2) `unfreeze_last_blocks` detects VGG-16's five MaxPool-terminated conv blocks instead of taking the last two modules; (3) `train_model` records `best_epoch`/`best_val_loss`/`best_val_acc` and `run_comparison` reports the checkpoint epoch instead of `max(val_acc)`; (4) `main.py` wired end-to-end with stage validation and a JSON/CSV summary -> verified with `pytest labs/lab2/tests/` (52 passed) and `ruff check` on changed files.

## Open issues
- Full pipeline run on the complete CIFAR-10 dataset (default `python -m labs.lab2.main`) has not been executed yet: it is slow on CPU. Unit/integration tests mock the stages; run the real pipeline (or the documented `--epochs 1 --image-size 64 --no-experiments` quick run) before submission.
- Multi-model hyperparameter grid comparison CSV export: partial runs logged on CPU in `artifacts/runs/`; full grid completion ongoing.

---

**Tiếng Việt**

## Tiếng Việt (cách dùng)
- **Bàn giao A -> B (tuần tự, không xen kẽ):** Member B chỉ bắt đầu code khi Member A đã xong HẾT:
  - file JSON split đã commit trong artifacts/splits/
  - make_loaders trả batch [B,3,H,W], nhãn 0-9
  - build_model trả logits [B,10]; describe_model in số tham số
  - mọi module của A ở trạng thái `done`, `pytest` qua, PR đã merge vào `main`, A đã nhắn tin bàn giao. Sau đó A chỉ trả lời câu hỏi và duyệt PR, không sửa module của B.
- Cập nhật file này sau MỖI bước: đổi trạng thái, ghi kết quả chạy thật (không ghi "chạy được" nếu chưa chạy).
- Mỗi dòng log gồm: ngày, người làm, module, đã thay đổi gì, kiểm chứng bằng cách nào.
- Nếu một module không cần cho lab này, ghi `N/A` kèm lý do.
- Lỗi cần ghi đủ 6 mục: mong đợi, thực tế, thông báo lỗi, môi trường, code liên quan, thay đổi gần đây.
