# Lab 2: Pre-trained CNNs and fine-tuning on CIFAR-10 (PyTorch): progress status
_Last updated: 2026-11-19 by Member A (Đại)_

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
- `pytest labs/lab2/tests/test_model.py` -> 3 passed
- `pytest labs/lab2/tests/test_train.py` -> 6 passed
- `pytest labs/lab2/tests/test_evaluate.py` -> 4 passed
- `pytest labs/lab2/tests/test_experiment.py` -> 2 passed
- `pytest labs/lab2/tests/test_tensorboard_optional.py` -> 6 passed
- `pytest labs/lab2/tests/test_model_unfreezing.py` -> 9 passed
- `pytest labs/lab2/tests/test_checkpoint_consistency.py` -> 8 passed
- `pytest labs/lab2/tests/test_main_pipeline.py` -> 7 passed
- `pytest labs/lab2/tests/` -> 56 passed in 44.69s
- `ruff check` on all modified modules + test files -> All checks passed

## Log (dated, one line per step)
- 2026-10-17 pptthu: scaffold created; interfaces agreed -> verified with `pytest labs/lab2/tests`
- 2026-10-26 Bảo (Member B): implemented train, evaluate, experiment modules on `feat/lab2-member-b`, added synthetic unit tests and updated presentation notebook sections -> verified with `pytest labs/lab2/tests` (18 passed).
- 2026-10-31 Thư (Leader): verified all 23 tests passing (Member A + Member B modules merged on main); audited baseline ResNet-18 training run and evaluation metrics; authored comprehensive bilingual README.md -> verified with `pytest labs/lab2/tests/` (23 passed).
- 2026-11-04 Thư (Leader): fixed four integration issues: (1) TensorBoard import is now lazy/optional in `experiment`; (2) `unfreeze_last_blocks` detects VGG-16's five MaxPool-terminated conv blocks instead of taking the last two modules; (3) `train_model` records `best_epoch`/`best_val_loss`/`best_val_acc` and `run_comparison` reports the checkpoint epoch instead of `max(val_acc)`; (4) `main.py` wired end-to-end with stage validation and a JSON/CSV summary -> verified with `pytest labs/lab2/tests/` (52 passed) and `ruff check` on changed files.
- 2026-11-04 Thư (Leader): added per-module guides `labs/lab2/docs/` (eda, preprocessing, features, model, train, evaluate, experiment, demo); README: `main.py` no longer described as a stub, docs folder added -> docs only, no code change; signatures checked against the code with `inspect`
- 2026-11-18 Thư (Leader): refactored and streamlined presentation notebook (`lab2_presentation.ipynb`) into a clean 12-stage presentation flow, eliminating duplicate experiment comparison cells and empty/fragmented environment shells while preserving all pre-rendered training curves, confusion heatmaps, error analyses, and test evaluation metrics -> verified with notebook JSON parser and `pytest labs/lab2/tests/test_smoke.py` (8 passed).
- 2026-11-18 Thư (Leader): authored comprehensive bilingual laboratory report (`REPORT.md`) matching the real notebook execution outputs: ResNet-18 vs DenseNet-121 (Feature Extraction: 78.94% vs 83.22%, Fine-Tuning: 91.90% vs 92.00%, Test Winner: DenseNet-121 at 91.32%) with statistical significance (SE=0.42%) and oral defense questions -> verified against `lab2_presentation.ipynb` and `finetune_comparison_summary.csv`.
- 2026-11-18 Member A (Đại): rewrote `docs/eda.md`, `docs/preprocessing.md`, `docs/features.md`, `docs/model.md` to match the comprehensive 10-section Demo Guide structure of `docs/train.md` (metadata header, table of contents, math formulas, line-by-line code explanation, mermaid architecture/dataflow diagrams, and oral defense Q&A preparation) -> verified with line-number audits and `pytest labs/lab2/tests/` (53 passed in 30.00s).
- 2026-11-19 Member A (Đại): standardized `docs/eda.md`, `docs/preprocessing.md`, `docs/features.md`, `docs/model.md` to the concise Phase Doc structure (v3 template, matching Lab 1) with exact function signatures, line numbers, I/O paths, pipeline diagrams, gotchas, and success criteria -> verified with `pytest` on Member A tests (22 passed in 19.89s).
- 2026-11-21 Member A (Đại): enhanced data augmentation in `features/features.py` (added `RandomRotation(15)` and `ColorJitter(0.2)`) to mitigate fine-tuning overfitting -> verified with `pytest labs/lab2/tests/test_feature.py` and `ruff check`.
- 2026-11-21 Member B (Bảo): added Learning Rate Scheduler support (`scheduler_name`: `'cosine'`, `'step'`, `'plateau'`) and learning rate tracking in `train/train.py`; added unit test in `test_train.py` -> verified with `pytest labs/lab2/tests/test_train.py` (6 passed) and `ruff check`.
- 2026-11-21 Member A (Đại): added optional `dropout` parameter to `build_model` in `model/model.py` and added unit test in `test_model.py` -> verified with `pytest labs/lab2/tests/test_model.py` (3 passed) and `ruff check`.
- 2026-11-21 Thư (Leader): integrated `--dropout`, `--scheduler`, `--patience` CLI options into `labs/lab2/main.py`; ensured models/checkpoints save to `labs/lab2/checkpoint/` and TensorBoard logs save to `labs/lab2/log/`; updated integration test `test_main_pipeline.py`; updated presentation notebook `lab2_presentation.ipynb` (Cell 2 `lab_dir` import, Cell 21 anti-overfitting notes, Cell 22 hyperparameters & training, Cell 25 evaluation) -> verified with `pytest labs/lab2/tests/` (56 passed) and `ruff check`.

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
