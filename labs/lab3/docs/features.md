# Lab 3 / features: guide for AI agents

## Purpose

- Turn review text into **numbers the model can read** (token ids and an attention mask). This is tokenization.
- Cut reviews that are too long to `max_length` tokens. Padding is **not** done here; it is done per batch in `train`.

## For AI agents (read first)

| Field | Value |
|---|---|
| Module | `labs/lab3/features/features.py` |
| Owner | Member A (Muội), assigned by the team leader |
| Status | done in `progress_status.md` (Reported only, 2026-10-26) |
| Assigned work | see the GitHub Issues and `progress_status.md` of the lab; never assign work to yourself, ask the team leader |
| Files you may edit | `labs/lab3/features/features.py`, `labs/lab3/tests/test_features.py`, this guide |
| Files you must not edit | other modules (in particular `train.py`, which does the padding), other owners' notebook cells |
| Agreed signatures | `tokenize_dataset(dataset, tokenizer, max_length: int = 256, text_col: str = 'text')` (same as `project_description.md`) |
| Depends on / used by | `preprocessing` output and the tokenizer from `model.load_model_and_tokenizer` / `train.train_with_trainer`, `evaluate.evaluate_on_test` |
| Prohibited | padding to a fixed length here (decision: padding per batch in `train`); using a tokenizer from a different checkpoint than the model; dropping the `label` column |
| Checks | Unit: `python -m pytest labs/lab3/tests/test_features.py -q` (fake tokenizer). Smoke: tokenize a few real rows with the real tokenizer |
| Done when | unit tests pass; smoke output recorded in `progress_status.md`; this guide updated |

## What the brief requires

- Brief requirement (Exercise 2): preprocess the dataset for the model (tokenization).
- Team implementation choice: truncation to `max_length` (default 256; the recorded full run used 128); no padding here.

## How the module works (current code)

- Checks that `text_col` exists; otherwise raises `KeyError`.
- Calls `dataset.map(..., batched=True)` with `truncation=True`, `max_length=max_length`, `return_token_type_ids=False` (DistilBERT does not use token type ids).
- Removes `text_col`; keeps `label` (the Trainer renames it to `labels`).

## Inputs, outputs and pitfalls

- Input: a dataset with `text` and `label`, the tokenizer, `max_length`.
- Output: a dataset with columns `label`, `input_ids`, `attention_mask`.
- Pitfalls:
  - rows have different lengths until the data collator pads each batch; do not expect a rectangular array here.
  - the tokenizer must come from the same checkpoint as the model.
  - the same `max_length` should be used in `evaluate.predict_texts`.
- Nothing is saved to disk by this module.

## How to verify

| Type | Command | Expected |
|---|---|---|
| Unit test | `python -m pytest labs/lab3/tests/test_features.py -q` | all pass |
| Smoke test | tokenize `train.select(range(8))` with the tokenizer of `distilbert-base-uncased` and print `column_names` | `['label', 'input_ids', 'attention_mask']` (Reported only) |
| Real integration run | only at an approved milestone | recorded together with `train` |

## Open questions / limitations

- The default `max_length` (256) and the value of the recorded run (128) differ; the guides of `train` and `demo` state which value a result used.

---

**Tiếng Việt**

## Mục đích

- Biến văn bản review thành **số để model đọc được** (token id và attention mask). Đây là bước tokenize.
- Cắt review quá dài xuống `max_length` token. **Không** padding ở đây; padding làm theo từng batch trong `train`.

## Dành cho AI (đọc trước)

- Người phụ trách: Member A (Muội), do nhóm trưởng giao. Trạng thái: xong (chỉ có trong báo cáo, 26/10).
- Việc được giao: xem GitHub Issue và `progress_status.md`; không tự giao việc cho mình, hỏi nhóm trưởng.
- Được sửa: `features.py`, `tests/test_features.py`, file này.
- Cấm sửa: module khác (nhất là `train.py`, nơi làm padding), ô notebook của người khác.
- Chữ ký: `tokenize_dataset(dataset, tokenizer, max_length=256, text_col='text')`, khớp `project_description.md`.
- Cấm: padding cố định ở đây (đã chốt padding theo batch ở `train`); dùng tokenizer khác checkpoint với model; bỏ cột `label`.
- Kiểm tra: unit test `pytest labs/lab3/tests/test_features.py` (tokenizer giả); smoke test tokenize vài dòng thật bằng tokenizer thật.
- Xong khi: unit test qua, kết quả smoke ghi vào `progress_status.md`, đã cập nhật file này.

## Đề yêu cầu gì

- Đề (Bài 2): tiền xử lý dataset cho model (tokenize).
- Lựa chọn của nhóm: cắt tới `max_length` (mặc định 256, lần chạy đã ghi dùng 128); không padding ở đây.

## Cách hoạt động

- Kiểm tra có cột `text_col`; không có thì báo `KeyError`.
- Gọi `dataset.map(..., batched=True)` với `truncation=True`, `max_length=max_length`, `return_token_type_ids=False` (DistilBERT không dùng token type ids).
- Bỏ cột text; giữ cột `label` (Trainer tự đổi tên thành `labels`).

## Đầu vào, đầu ra và lỗi hay gặp

- Đầu vào: dataset có `text` và `label`, tokenizer, `max_length`.
- Đầu ra: dataset có các cột `label`, `input_ids`, `attention_mask`.
- Lưu ý:
  - các dòng có độ dài khác nhau cho tới khi data collator padding từng batch; đừng mong đây là mảng chữ nhật.
  - tokenizer phải cùng checkpoint với model.
  - nên dùng cùng `max_length` trong `evaluate.predict_texts`.
- Module này không lưu gì xuống đĩa.

## Cách kiểm chứng

- Unit test: `python -m pytest labs/lab3/tests/test_features.py -q`, kỳ vọng tất cả qua.
- Smoke test: tokenize `train.select(range(8))` bằng tokenizer `distilbert-base-uncased`, in `column_names`; kỳ vọng `['label', 'input_ids', 'attention_mask']` (chỉ có trong báo cáo).
- Chạy tích hợp thật: chỉ ở mốc nhóm đã duyệt, ghi cùng với `train`.

## Câu hỏi mở

- `max_length` mặc định (256) khác giá trị lần chạy đã ghi (128); guide của `train` và `demo` ghi rõ kết quả dùng giá trị nào.
