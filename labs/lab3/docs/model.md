# Lab 3 / model: guide for AI agents

## For AI agents (read first)

| Field | Value |
|---|---|
| Module | `labs/lab3/model/model.py` |
| Owner | Member B (ngohongphuc2805ag-hash), assigned by the team leader |
| Status | done in `progress_status.md`, merged in PR #38 (Reported only) |
| Files you may edit | `labs/lab3/model/model.py`, `labs/lab3/tests/test_model.py`, this guide |
| Files you must not edit | Member A's modules, `main.py` (team leader), saved models, other owners' notebook cells |
| Agreed signatures | `load_model_and_tokenizer(model_name: str = 'distilbert-base-uncased', num_labels: int = 2)` (same as `project_description.md`) |
| Depends on / used by | `transformers` / `features` (tokenizer), `train`, `evaluate.predict_texts` |
| Prohibited | loading the tokenizer and the model from different checkpoints; changing the label mapping; adding other model families (out of scope); downloads inside unit tests |
| Checks | Unit: `python -m pytest labs/lab3/tests/test_model.py -q` (fake `transformers`). Smoke: one real load of `distilbert-base-uncased` |
| Done when | unit tests pass; smoke output recorded in `progress_status.md`; this guide updated |

## Current task (edited ONLY by the team leader)

- None.

Module owners and agents never assign work to themselves here; they ask the team leader.

## What the brief requires

- Brief requirement (Exercise 2): load a pre-trained model and its tokenizer.
- Team choice (not confirmed by the lecturer): `distilbert-base-uncased`, 2 labels.

## How the module works (current code)

- `AutoTokenizer` and `AutoModelForSequenceClassification` from the **same** `model_name`.
- For 2 labels: `id2label = {0: NEGATIVE, 1: POSITIVE}`, `label2id` the reverse (same convention as the Exercise 1 pipeline).
- `num_labels` must be an int >= 2, otherwise `ValueError`.
- The classification head is newly initialised; the warning Hugging Face prints about it is expected.

## Inputs, outputs and pitfalls

- Output: `(model, tokenizer)`. The model returns logits of shape `(batch, num_labels)`.
- First call downloads the checkpoint (internet needed).

## How to verify

| Type | Command | Expected |
|---|---|---|
| Unit test | `python -m pytest labs/lab3/tests/test_model.py -q` | all pass, no network |
| Smoke test | `python -c "from labs.lab3.model.model import load_model_and_tokenizer; m,t=load_model_and_tokenizer(); print(m.config.id2label, type(t).__name__)"` | `{0: 'NEGATIVE', 1: 'POSITIVE'}` and a DistilBERT tokenizer |
| Real integration run | only at an approved milestone | recorded with `evaluate` |

## Open questions / limitations

- Model choice not confirmed by the lecturer.

---

**Tiếng Việt**

- **Dành cho AI:** người phụ trách Member B (ngohongphuc2805ag-hash), do nhóm trưởng giao. Được sửa `model.py`, `tests/test_model.py`, file này. Chữ ký `load_model_and_tokenizer(model_name='distilbert-base-uncased', num_labels=2)`, khớp `project_description.md`. Cấm lấy tokenizer và model từ hai checkpoint khác nhau, cấm đổi cách gán nhãn, cấm thêm họ model khác (ngoài phạm vi), cấm tải trong unit test.
- **Task hiện tại:** chỉ nhóm trưởng ghi ở mục "Current task".
- **Đề:** nạp model pre-trained và tokenizer. DistilBERT là lựa chọn của nhóm, chưa được cô xác nhận.
- **Cách hoạt động:** tokenizer và model cùng checkpoint; 2 nhãn có `id2label {0: NEGATIVE, 1: POSITIVE}`; đầu phân loại khởi tạo mới nên có cảnh báo, đó là bình thường.
- **Kiểm chứng:** unit test như bảng trên; smoke test in ra `id2label`.
