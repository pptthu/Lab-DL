# Lab 3 / evaluate: guide for AI agents

## For AI agents (read first)

| Field | Value |
|---|---|
| Module | `labs/lab3/evaluate/evaluate.py` |
| Owner | Member B (ngohongphuc2805ag-hash), assigned by the team leader |
| Status | done in `progress_status.md` (Reported only, 2026-10-31) |
| Files you may edit | `labs/lab3/evaluate/evaluate.py`, `labs/lab3/tests/test_evaluate.py`, this guide |
| Files you must not edit | Member A's modules, `main.py` (team leader), saved models, other owners' notebook cells |
| Agreed signatures | `compute_metrics(eval_pred) -> dict`; `evaluate_on_test(trainer, tokenized_test) -> dict`; `predict_texts(model, tokenizer, texts: list) -> list` (`project_description.md`). The code adds keyword-only **optional** `max_length=256, batch_size=16` to `predict_texts`, and helpers `classification_metrics(labels, predictions)` and `zero_shot_labels(results)` |
| Depends on / used by | `numpy`, `torch` (in `predict_texts`) / `train` (as `compute_metrics`), notebook sections on test results and zero-shot comparison |
| Prohibited | calling `evaluate_on_test` more than once or to choose a model; adding metrics the brief does not ask for as required; network access in unit tests |
| Checks | Unit: `python -m pytest labs/lab3/tests/test_evaluate.py -q` (NumPy only, no network). Smoke: `predict_texts` on 3 sentences with a loaded model |
| Done when | unit tests pass; results recorded in `progress_status.md`; this guide updated |

## Current task (edited ONLY by the team leader)

- None.

Module owners and agents never assign work to themselves here; they ask the team leader.

## What the brief requires

- Brief requirement (Exercise 2): evaluate the fine-tuned model.
- Team choices: accuracy and binary F1 (positive class = 1); compare with the zero-shot pipeline of Exercise 1.

## How the module works (current code)

- `classification_metrics`: accuracy and F1 = 2TP / (2TP + FP + FN); F1 is 0.0 when the denominator is 0. Shape mismatch or empty input raises `ValueError`.
- `compute_metrics`: takes `(logits, labels)` or `EvalPrediction`, uses the argmax of 2-D logits; key `f1` is the one `train` uses to keep the best epoch.
- `evaluate_on_test`: `trainer.evaluate(..., metric_key_prefix="test")`, returns keys without the prefix. Use once, at the end, on the model chosen on validation.
- `predict_texts`: eval mode for the call (previous mode restored), same tokenizer and `max_length` as training, padding per batch, returns `{text, label (0/1), score}`.
- `zero_shot_labels`: maps `POSITIVE`/`NEGATIVE` from `sentiment_demo.analyze_sentences` to 1/0.

## Inputs, outputs and pitfalls

- `predict_texts` raises `TypeError` unless `texts` is a list of `str`; an empty list returns `[]`.
- Test numbers on n=500 have a standard error of about 1.4–1.7 points (recorded in the notebook); small differences are not meaningful.

## How to verify

| Type | Command | Expected |
|---|---|---|
| Unit test | `python -m pytest labs/lab3/tests/test_evaluate.py -q` | all pass; recorded check against scikit-learn on 300 random cases: max difference 0 (Reported only) |
| Smoke test | `predict_texts(model, tokenizer, ['I love this movie', 'The plot was boring', 'Not bad at all'])` | three dicts with `label` 0/1 and `score` in (0, 1) |
| Real integration run | only at an approved milestone | recorded (2026-10-31): test n=500, zero-shot 0.8920/0.8889, fine-tuned 0.8220/0.8195 (Reported only) |

## Open questions / limitations

- Recorded results: the zero-shot SST-2 model scores higher than the 1-epoch fine-tune; any explanation in the report is a hypothesis unless tested.

---

**Tiếng Việt**

- **Dành cho AI:** người phụ trách Member B (ngohongphuc2805ag-hash), do nhóm trưởng giao. Được sửa `evaluate.py`, `tests/test_evaluate.py`, file này. Chữ ký theo `project_description.md`; `predict_texts` có thêm tham số tùy chọn `max_length`, `batch_size`; có thêm hàm phụ `classification_metrics`, `zero_shot_labels`. Cấm gọi `evaluate_on_test` nhiều lần hoặc dùng nó để chọn model, cấm dùng mạng trong unit test.
- **Task hiện tại:** chỉ nhóm trưởng ghi ở mục "Current task".
- **Đề:** đánh giá model đã fine-tune. Nhóm chọn accuracy và F1 nhị phân, so với zero-shot của Bài 1.
- **Cách hoạt động:** F1 = 2TP/(2TP+FP+FN); `compute_metrics` lấy argmax của logits; `evaluate_on_test` chỉ dùng một lần ở cuối; `predict_texts` dùng cùng tokenizer và `max_length` với lúc train.
- **Kiểm chứng:** số đã ghi (31/10): zero-shot 0,8920/0,8889, fine-tune 0,8220/0,8195 trên 500 mẫu test (chỉ có trong báo cáo). Sai số chuẩn khoảng 1,4–1,7 điểm nên chênh lệch nhỏ không có ý nghĩa.
- **Câu hỏi mở:** zero-shot đang cao hơn fine-tune 1 epoch; mọi giải thích trong báo cáo chỉ là giả thuyết nếu chưa kiểm chứng.
