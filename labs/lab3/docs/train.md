# Lab 3 / train: guide for AI agents

## For AI agents (read first)

| Field | Value |
|---|---|
| Module | `labs/lab3/train/train.py` |
| Owner | Member B (ngohongphuc2805ag-hash), assigned by the team leader |
| Status | done in `progress_status.md`, merged in PR #39 (Reported only) |
| Files you may edit | `labs/lab3/train/train.py`, `labs/lab3/tests/test_train.py`, this guide |
| Files you must not edit | Member A's modules, `main.py` (team leader), saved checkpoints, other owners' notebook cells |
| Agreed signatures | `build_training_args(output_dir: str, epochs: int = 2, lr: float = 2e-5, batch_size: int = 16)`; `train_with_trainer(model, tokenized_train, tokenized_val, args, compute_metrics)` (`project_description.md`). The code adds **optional** arguments only: `build_training_args(..., *, max_steps=-1, seed=42, logging_steps=10)` and `train_with_trainer(..., tokenizer=None)` |
| Depends on / used by | `features` output, `model`, `evaluate.compute_metrics` / `evaluate.evaluate_on_test`, notebook |
| Prohibited | passing the test set to the Trainer; selecting the best epoch on test; starting a full fine-tune without the team leader (cost on CPU); reporting to external services; removing the version detection |
| Checks | Unit: `python -m pytest labs/lab3/tests/test_train.py -q` (fake `transformers`). Smoke: tiny fine-tune with `max_steps` > 0 on a few rows. Real run: only at an approved milestone |
| Done when | unit tests pass; smoke or approved real run recorded in `progress_status.md`; this guide updated |

## Current task (edited ONLY by the team leader)

- None.

Module owners and agents never assign work to themselves here; they ask the team leader.

## What the brief requires

- Brief requirement (Exercise 2): define training arguments, create a `Trainer` and fine-tune.
- Team choices: evaluate and save once per epoch, keep the best epoch on **validation F1**, weight decay 0.01, `save_total_limit=1`, `report_to="none"`.

## How the module works (current code)

- `build_training_args` checks `epochs`, `lr`, `batch_size` > 0 (`ValueError`), detects `eval_strategy` vs `evaluation_strategy` for the installed `transformers`, sets `load_best_model_at_end=True`, `metric_for_best_model="f1"`. `max_steps > 0` overrides `epochs` (smoke runs).
- `train_with_trainer` refuses empty train/validation sets, pads per batch with `DataCollatorWithPadding(tokenizer)` (the tokenizer is loaded from `model.name_or_path` if not given), detects `processing_class` vs `tokenizer` for `Trainer`, trains, then `trainer.save_model(args.output_dir)`.
- The test set is never passed here; it is used once in `evaluate.evaluate_on_test`.

## Inputs, outputs and pitfalls

- Output: the trained `Trainer` holding the best validation epoch; model files in `args.output_dir` (an artifacts folder, not committed).
- `transformers` argument names change between versions: keep the detection.
- CPU cost: use small subsets, short `max_length`, few epochs; agree the cost before a full run.

## How to verify

| Type | Command | Expected |
|---|---|---|
| Unit test | `python -m pytest labs/lab3/tests/test_train.py -q` | all pass, no network |
| Smoke test | tiny fine-tune: `build_training_args(tmp_dir, max_steps=4)` on 64 train reviews (recorded run: 4 steps, 64 train / 32 test reviews) | runs without error; recorded tiny run: accuracy 0.4688, F1 0.3200 (only proves the code runs; Reported only) |
| Real integration run | only at an approved milestone | recorded full small run (2026-10-31, CPU, MAX_LEN=128, EPOCHS=1, seed 42): fine-tuned test accuracy/F1 0.8220/0.8195 (Reported only) |

## Open questions / limitations

- The recorded full run used 1 epoch while the default is 2; which setting the report uses must be stated with the result.
- Tests use fake objects; the real-library run is recorded only in the notebook and `progress_status.md`.

---

**Tiếng Việt**

- **Dành cho AI:** người phụ trách Member B (ngohongphuc2805ag-hash), do nhóm trưởng giao. Được sửa `train.py`, `tests/test_train.py`, file này. Chữ ký theo `project_description.md`; code chỉ thêm tham số **tùy chọn** (`max_steps`, `seed`, `logging_steps`, `tokenizer`). Cấm đưa tập test vào Trainer, cấm chọn epoch theo test, cấm tự chạy fine-tune đầy đủ khi chưa hỏi nhóm trưởng, cấm báo cáo ra dịch vụ ngoài, cấm bỏ phần tự dò phiên bản.
- **Task hiện tại:** chỉ nhóm trưởng ghi ở mục "Current task".
- **Đề:** định nghĩa training arguments, tạo Trainer, fine-tune. Nhóm chọn đánh giá và lưu mỗi epoch, giữ epoch tốt nhất theo **F1 trên validation**.
- **Cách hoạt động:** tự dò tên tham số theo phiên bản `transformers`; padding theo batch bằng `DataCollatorWithPadding`; lưu model vào `args.output_dir` (không commit).
- **Kiểm chứng:** fine-tune rất ngắn (`max_steps=4`) chỉ để chứng minh code chạy; lần chạy đầy đủ đã ghi 0,8220/0,8195 (1 epoch, MAX_LEN=128; chỉ có trong báo cáo).
- **Câu hỏi mở:** lần chạy đã ghi dùng 1 epoch, mặc định là 2; test dùng object giả.
