# Lab 3 / features: guide for AI agents

## For AI agents (read first)

| Field | Value |
|---|---|
| Module | `labs/lab3/features/features.py` |
| Owner | Member A (Muội), assigned by the team leader |
| Status | done in `progress_status.md` (Reported only, 2026-10-26) |
| Files you may edit | `labs/lab3/features/features.py`, `labs/lab3/tests/test_features.py`, this guide |
| Files you must not edit | other modules (in particular `train.py`, which does the padding), other owners' notebook cells |
| Agreed signatures | `tokenize_dataset(dataset, tokenizer, max_length: int = 256, text_col: str = 'text')` (same as `project_description.md`) |
| Depends on / used by | `preprocessing` output and the tokenizer from `model.load_model_and_tokenizer` / `train.train_with_trainer`, `evaluate.evaluate_on_test` |
| Prohibited | padding to a fixed length here (decision: padding per batch in `train`); using a tokenizer from a different checkpoint than the model; dropping the `label` column |
| Checks | Unit: `python -m pytest labs/lab3/tests/test_features.py -q` (fake tokenizer). Smoke: tokenize a few real rows with the real tokenizer |
| Done when | unit tests pass; smoke output recorded in `progress_status.md`; this guide updated |

## Current task (edited ONLY by the team leader)

- None.

Module owners and agents never assign work to themselves here; they ask the team leader.

## What the brief requires

- Brief requirement (Exercise 2): preprocess the dataset for the model (tokenization).
- Team choices: truncation to `max_length` (default 256; the recorded full run used 128); no padding here.

## How the module works (current code)

- `dataset.map(..., batched=True)` with `truncation=True, max_length=max_length, return_token_type_ids=False` (DistilBERT does not use token type ids).
- Removes `text_col`; keeps `label` (the Trainer renames it to `labels`).
- Missing `text_col` raises `KeyError`.

## Inputs, outputs and pitfalls

- Output columns: `label`, `input_ids`, `attention_mask`; rows have different lengths until the data collator pads each batch.
- The same `max_length` should be used in `evaluate.predict_texts`.

## How to verify

| Type | Command | Expected |
|---|---|---|
| Unit test | `python -m pytest labs/lab3/tests/test_features.py -q` | all pass |
| Smoke test | tokenize `train.select(range(8))` with the tokenizer of `distilbert-base-uncased` and print `column_names` | `['label', 'input_ids', 'attention_mask']` (Reported only) |
| Real integration run | only at an approved milestone | recorded with `train` |

## Open questions / limitations

- Default `max_length` (256) and the value of the recorded run (128) differ; the guide of `train`/`demo` records which value a result used.

---

**Tiếng Việt**

- **Dành cho AI:** người phụ trách Member A (Muội), do nhóm trưởng giao. Được sửa `features.py`, `tests/test_features.py`, file này. Chữ ký `tokenize_dataset(dataset, tokenizer, max_length=256, text_col='text')`, khớp `project_description.md`. Cấm padding cố định ở đây (đã chốt padding theo batch ở `train`), cấm dùng tokenizer khác checkpoint với model, cấm bỏ cột `label`.
- **Task hiện tại:** chỉ nhóm trưởng ghi ở mục "Current task".
- **Đề:** tiền xử lý dataset cho model. Nhóm chọn cắt tới `max_length` (mặc định 256, lần chạy đã ghi dùng 128), không padding.
- **Cách hoạt động:** `map` theo batch, bỏ cột text, giữ `label`; thiếu cột text báo `KeyError`.
- **Kiểm chứng:** cột sau khi tokenize là `label`, `input_ids`, `attention_mask` (chỉ có trong báo cáo).
