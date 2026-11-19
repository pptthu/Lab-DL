# Lab 3 / evaluate: guide for AI agents

## Purpose

- **Measure** how good the model is: accuracy and F1 (positive class = 1 = POSITIVE).
- Give the Trainer a metric function (`compute_metrics`) so it can pick the best epoch on validation.
- Run the **one final test evaluation**, and predict labels for new raw texts.
- Convert Exercise 1 zero-shot labels (POSITIVE / NEGATIVE) to 1 / 0 so both models can be compared.

## For AI agents (read first)

| Field | Value |
|---|---|
| Module | `labs/lab3/evaluate/evaluate.py` |
| Owner | Member B (ngohongphuc2805ag-hash), assigned by the team leader |
| Status | done in `progress_status.md` (Reported only, 2026-10-31) |
| Assigned work | see the GitHub Issues and `progress_status.md` of the lab; never assign work to yourself, ask the team leader |
| Files you may edit | `labs/lab3/evaluate/evaluate.py`, `labs/lab3/tests/test_evaluate.py`, this guide |
| Files you must not edit | Member A's modules, `main.py` (team leader), saved models, other owners' notebook cells |
| Agreed signatures | `compute_metrics(eval_pred) -> dict`; `evaluate_on_test(trainer, tokenized_test) -> dict`; `predict_texts(model, tokenizer, texts: list) -> list` (`project_description.md`). The code adds keyword-only **optional** `max_length=256, batch_size=16` to `predict_texts`, and helpers `classification_metrics(labels, predictions)` and `zero_shot_labels(results)` |
| Depends on / used by | `numpy`, `torch` (inside `predict_texts`) / `train` (as `compute_metrics`), notebook sections on test results and zero-shot comparison |
| Prohibited | calling `evaluate_on_test` more than once or to choose a model; adding metrics the brief does not ask for as required; network access in unit tests; naming a file or import so that it clashes with the Hugging Face package `evaluate` |
| Checks | Unit: `python -m pytest labs/lab3/tests/test_evaluate.py -q` (NumPy and fake `torch`, no network). Smoke: `predict_texts` on 3 sentences with a loaded model |
| Done when | unit tests pass; results recorded in `progress_status.md`; this guide updated |

## What the brief requires

- Brief requirement (Exercise 2): evaluate the fine-tuned model.
- Team implementation choice: accuracy and binary F1; compare with the zero-shot pipeline of Exercise 1.

## How the module works (current code)

- `classification_metrics`: accuracy, and F1 = 2TP / (2TP + FP + FN); F1 is 0.0 when the denominator is 0. A shape mismatch or empty input raises `ValueError`.
- `compute_metrics`: takes `(logits, labels)` or an `EvalPrediction`, uses the argmax of 2-D logits. The key `f1` is the one `train` uses to keep the best epoch.
- `evaluate_on_test`: calls `trainer.evaluate(..., metric_key_prefix="test")` and returns the keys without the prefix. Use it once, at the end, on the model chosen on validation.
- `predict_texts`:
  - puts the model in eval mode for the call and restores the previous mode.
  - uses the same tokenizer and `max_length` as training, padding per batch.
  - returns a list of `{text, label (0/1), score}`; `score` is the softmax probability of that label.
- `zero_shot_labels`: maps `POSITIVE` / `NEGATIVE` from `sentiment_demo.analyze_sentences` to 1 / 0; any other label raises `ValueError`.
- Metrics use NumPy only (no `evaluate` library, no network).

## Inputs, outputs and pitfalls

- Input:
  1. Model to evaluate: the fine-tuned model held by the `Trainer` (best validation epoch). If it is loaded from disk, it is the folder written by `train` (see "Where the model is saved" in `train.md`).
  2. Tokenized test subset (for `evaluate_on_test`) or a list of raw `str` (for `predict_texts`).
- Output:
  - `evaluate_on_test`: dict with `accuracy`, `f1`, `loss`, ... (no `test_` prefix).
  - `predict_texts`: list of dicts `{text, label, score}`.
- This module saves nothing to disk; `main.py` writes the dict returned by `evaluate_on_test` to `labs/lab3/artifacts/logs/test_metrics.json`.
- Pitfalls:
  - `predict_texts` raises `TypeError` unless `texts` is a list of `str`; an empty list returns `[]`.
  - using the test set more than once to pick settings makes the test score optimistic.
  - test numbers on n=500 have a standard error of about 1.4 to 1.7 points (recorded in the notebook); small differences are not meaningful.

## How to verify

| Type | Command | Expected |
|---|---|---|
| Unit test | `python -m pytest labs/lab3/tests/test_evaluate.py -q` | all pass; recorded check against scikit-learn on 300 random cases: max difference 0 (Reported only) |
| Smoke test | `predict_texts(model, tokenizer, ['I love this movie', 'The plot was boring', 'Not bad at all'])` | three dicts with `label` 0/1 and `score` in (0, 1) |
| Real integration run | only at an approved milestone | recorded (2026-10-31, CPU): test n=500, zero-shot accuracy/F1 0.8920/0.8889, fine-tuned 0.8220/0.8195 (MAX_LEN=128, EPOCHS=1, seed 42; Reported only) |

## Open questions / limitations

- Recorded results: the zero-shot SST-2 model scores higher than the 1-epoch fine-tune. Any explanation in the report is a hypothesis unless it is tested.

---

**Tiếng Việt**

## Mục đích

- **Đo** model tốt đến đâu: accuracy và F1 (lớp dương = 1 = POSITIVE).
- Cung cấp hàm metric (`compute_metrics`) cho Trainer để chọn epoch tốt nhất trên validation.
- Chạy **đúng một lần đánh giá cuối trên test**, và dự đoán nhãn cho văn bản mới (chuỗi thô).
- Đổi nhãn zero-shot của Bài 1 (POSITIVE / NEGATIVE) sang 1 / 0 để so sánh hai model.

## Dành cho AI (đọc trước)

- Người phụ trách: Member B (ngohongphuc2805ag-hash), do nhóm trưởng giao. Trạng thái: xong (chỉ có trong báo cáo, 31/10).
- Việc được giao: xem GitHub Issue và `progress_status.md`; không tự giao việc cho mình, hỏi nhóm trưởng.
- Được sửa: `evaluate.py`, `tests/test_evaluate.py`, file này.
- Cấm sửa: module của Member A, `main.py` (nhóm trưởng), model đã lưu, ô notebook của người khác.
- Chữ ký: theo `project_description.md`; `predict_texts` có thêm tham số tùy chọn `max_length`, `batch_size`; có thêm hàm phụ `classification_metrics`, `zero_shot_labels`.
- Cấm: gọi `evaluate_on_test` nhiều lần hoặc dùng nó để chọn model; thêm metric đề không yêu cầu như bắt buộc; dùng mạng trong unit test; đặt tên file hay import trùng với package `evaluate` của Hugging Face.
- Kiểm tra: unit test `pytest labs/lab3/tests/test_evaluate.py` (NumPy và `torch` giả, không cần mạng); smoke test `predict_texts` trên 3 câu với model đã nạp.
- Xong khi: unit test qua, kết quả ghi vào `progress_status.md`, đã cập nhật file này.

## Đề yêu cầu gì

- Đề (Bài 2): đánh giá model đã fine-tune.
- Lựa chọn của nhóm: accuracy và F1 nhị phân; so với pipeline zero-shot của Bài 1.

## Cách hoạt động

- `classification_metrics`: accuracy, và F1 = 2TP / (2TP + FP + FN); F1 bằng 0.0 khi mẫu số bằng 0. Sai kích thước hoặc đầu vào rỗng thì báo `ValueError`.
- `compute_metrics`: nhận `(logits, labels)` hoặc `EvalPrediction`, lấy argmax của logits 2 chiều. Khóa `f1` là khóa `train` dùng để giữ epoch tốt nhất.
- `evaluate_on_test`: gọi `trainer.evaluate(..., metric_key_prefix="test")` và trả về khóa đã bỏ tiền tố. Chỉ dùng một lần ở cuối, trên model đã chọn bằng validation.
- `predict_texts`:
  - đặt model ở chế độ eval khi gọi, xong trả về chế độ cũ.
  - dùng cùng tokenizer và `max_length` như lúc train, padding theo batch.
  - trả về danh sách `{text, label (0/1), score}`; `score` là xác suất softmax của nhãn đó.
- `zero_shot_labels`: đổi `POSITIVE` / `NEGATIVE` từ `sentiment_demo.analyze_sentences` sang 1 / 0; nhãn khác thì báo `ValueError`.
- Metric chỉ dùng NumPy (không dùng thư viện `evaluate`, không cần mạng).

## Đầu vào, đầu ra và lỗi hay gặp

- Đầu vào:
  1. Model cần đánh giá: model đã fine-tune do `Trainer` giữ (epoch validation tốt nhất). Nếu nạp từ đĩa thì là thư mục `train` đã ghi (xem mục "Model được lưu ở đâu" trong `train.md`).
  2. Tập test đã tokenize (cho `evaluate_on_test`) hoặc danh sách chuỗi thô (cho `predict_texts`).
- Đầu ra:
  - `evaluate_on_test`: dict có `accuracy`, `f1`, `loss`, ... (không có tiền tố `test_`).
  - `predict_texts`: danh sách dict `{text, label, score}`.
- Module này không lưu gì xuống đĩa; `main.py` ghi dict mà `evaluate_on_test` trả về vào `labs/lab3/artifacts/logs/test_metrics.json`.
- Lưu ý:
  - `predict_texts` báo `TypeError` nếu `texts` không phải danh sách `str`; danh sách rỗng trả về `[]`.
  - dùng test nhiều lần để chọn thiết lập làm điểm test bị lạc quan.
  - số liệu test với n=500 có sai số chuẩn khoảng 1,4 đến 1,7 điểm (ghi trong notebook); chênh lệch nhỏ không có ý nghĩa.

## Cách kiểm chứng

- Unit test: `python -m pytest labs/lab3/tests/test_evaluate.py -q`, kỳ vọng tất cả qua; đã đối chiếu với scikit-learn trên 300 trường hợp ngẫu nhiên: chênh lệch lớn nhất bằng 0 (chỉ có trong báo cáo).
- Smoke test: `predict_texts` trên 3 câu mẫu, kỳ vọng 3 dict có `label` 0/1 và `score` trong (0, 1).
- Chạy tích hợp thật: chỉ ở mốc nhóm đã duyệt; đã ghi (31/10, CPU): test n=500, zero-shot 0,8920/0,8889, fine-tune 0,8220/0,8195 (MAX_LEN=128, 1 epoch, seed 42; chỉ có trong báo cáo).

## Câu hỏi mở

- Số đã ghi: model zero-shot SST-2 cao hơn model fine-tune 1 epoch. Mọi giải thích trong báo cáo chỉ là giả thuyết nếu chưa kiểm chứng.
