# Lab 3 / model: guide for AI agents

## Purpose

- Load a **pre-trained** DistilBERT and its tokenizer, ready to be fine-tuned for 2-class sentiment (NEGATIVE / POSITIVE).
- This module only **loads** a model. It does not train and does not save anything (saving is in `train.md`).

## For AI agents (read first)

| Field | Value |
|---|---|
| Module | `labs/lab3/model/model.py` |
| Owner | Member B (ngohongphuc2805ag-hash), assigned by the team leader |
| Status | done in `progress_status.md`, merged in PR #38 (Reported only) |
| Assigned work | see the GitHub Issues and `progress_status.md` of the lab; never assign work to yourself, ask the team leader |
| Files you may edit | `labs/lab3/model/model.py`, `labs/lab3/tests/test_model.py`, this guide |
| Files you must not edit | Member A's modules, `main.py` (team leader), saved models, other owners' notebook cells |
| Agreed signatures | `load_model_and_tokenizer(model_name: str = 'distilbert-base-uncased', num_labels: int = 2)` (same as `project_description.md`) |
| Depends on / used by | `transformers` / `features` (tokenizer), `train`, `evaluate.predict_texts` |
| Prohibited | loading the tokenizer and the model from different checkpoints; changing the label mapping; adding other model families (out of scope); downloads inside unit tests |
| Checks | Unit: `python -m pytest labs/lab3/tests/test_model.py -q` (fake `transformers`). Smoke: one real load of `distilbert-base-uncased` |
| Done when | unit tests pass; smoke output recorded in `progress_status.md`; this guide updated |

## What the brief requires

- Brief requirement (Exercise 2): load a pre-trained model and its tokenizer.
- Team implementation choice, not yet confirmed by the lecturer: `distilbert-base-uncased`, 2 labels.

## How the module works (current code)

- Loads the tokenizer with `AutoTokenizer` and the model with `AutoModelForSequenceClassification`, both from the **same** `model_name`.
- For 2 labels it sets `id2label = {0: NEGATIVE, 1: POSITIVE}` and the reverse `label2id` (same convention as the Exercise 1 pipeline).
- `num_labels` must be an int >= 2, otherwise `ValueError`.
- The classification head is newly initialised (random); the Hugging Face warning about it is expected, because the head is learned during fine-tuning.
- `transformers` is imported inside the function, so tests need no network.

## Inputs, outputs and pitfalls

- Input:
  1. Initial model:
     - architecture: DistilBERT encoder with a sequence-classification head, 2 outputs (`AutoModelForSequenceClassification`).
     - source: Hugging Face Hub checkpoint `distilbert-base-uncased` (the `model_name` argument).
     - where the downloaded files go: the default Hugging Face cache; the code sets no `cache_dir` (Implemented; location on the machine not checked).
  2. Tokenizer: same checkpoint, same source.
- Output: `(model, tokenizer)`; the model returns logits of shape `(batch, num_labels)`.
- Saved model: **not produced here**. The fine-tuned model is saved by `train.train_with_trainer`; see "Where the model is saved" in `train.md`.
- Pitfalls:
  - the first call downloads the checkpoint (internet needed).
  - tokenizer and model from different checkpoints give wrong ids without an error message.
  - do not use `distilbert-base-uncased-finetuned-sst-2-english` here: that checkpoint is the Exercise 1 zero-shot baseline (`demo.md`).

## How to verify

| Type | Command | Expected |
|---|---|---|
| Unit test | `python -m pytest labs/lab3/tests/test_model.py -q` | all pass, no network |
| Smoke test | `python -c "from labs.lab3.model.model import load_model_and_tokenizer; m,t=load_model_and_tokenizer(); print(m.config.id2label, type(t).__name__)"` | `{0: 'NEGATIVE', 1: 'POSITIVE'}` and a DistilBERT tokenizer |
| Real integration run | only at an approved milestone | recorded together with `evaluate` |

## Open questions / limitations

- Model choice not confirmed by the lecturer.

---

**Tiếng Việt**

## Mục đích

- Nạp **DistilBERT đã pre-train** và tokenizer của nó, sẵn sàng để fine-tune phân loại cảm xúc 2 lớp (NEGATIVE / POSITIVE).
- Module này chỉ **nạp** model. Nó không train và không lưu gì (phần lưu nằm ở `train.md`).

## Dành cho AI (đọc trước)

- Người phụ trách: Member B (ngohongphuc2805ag-hash), do nhóm trưởng giao. Trạng thái: xong, đã merge PR #38 (chỉ có trong báo cáo).
- Việc được giao: xem GitHub Issue và `progress_status.md`; không tự giao việc cho mình, hỏi nhóm trưởng.
- Được sửa: `model.py`, `tests/test_model.py`, file này.
- Cấm sửa: module của Member A, `main.py` (nhóm trưởng), model đã lưu, ô notebook của người khác.
- Chữ ký: `load_model_and_tokenizer(model_name='distilbert-base-uncased', num_labels=2)`, khớp `project_description.md`.
- Cấm: lấy tokenizer và model từ hai checkpoint khác nhau; đổi cách gán nhãn; thêm họ model khác (ngoài phạm vi); tải về trong unit test.
- Kiểm tra: unit test `pytest labs/lab3/tests/test_model.py` (`transformers` giả); smoke test nạp thật `distilbert-base-uncased` một lần.
- Xong khi: unit test qua, kết quả smoke ghi vào `progress_status.md`, đã cập nhật file này.

## Đề yêu cầu gì

- Đề (Bài 2): nạp model pre-trained và tokenizer.
- Lựa chọn của nhóm, cô chưa xác nhận: `distilbert-base-uncased`, 2 nhãn.

## Cách hoạt động

- Nạp tokenizer bằng `AutoTokenizer` và model bằng `AutoModelForSequenceClassification`, cả hai từ **cùng** `model_name`.
- Với 2 nhãn, đặt `id2label = {0: NEGATIVE, 1: POSITIVE}` và `label2id` ngược lại (cùng quy ước với pipeline ở Bài 1).
- `num_labels` phải là số nguyên >= 2, nếu không báo `ValueError`.
- Đầu phân loại được khởi tạo ngẫu nhiên; cảnh báo của Hugging Face về điều này là bình thường vì đầu này được học khi fine-tune.
- `transformers` được import bên trong hàm nên test không cần mạng.

## Đầu vào, đầu ra và lỗi hay gặp

- Đầu vào:
  1. Model ban đầu:
     - kiến trúc: bộ mã hóa DistilBERT cùng đầu phân loại chuỗi, 2 đầu ra (`AutoModelForSequenceClassification`).
     - nguồn: checkpoint `distilbert-base-uncased` trên Hugging Face Hub (tham số `model_name`).
     - file tải về nằm đâu: cache mặc định của Hugging Face; code không đặt `cache_dir` (đã có trong code; vị trí trên máy chưa kiểm tra).
  2. Tokenizer: cùng checkpoint, cùng nguồn.
- Đầu ra: `(model, tokenizer)`; model trả logits có dạng `(batch, num_labels)`.
- Model đã lưu: **không được tạo ở đây**. Model fine-tune được lưu bởi `train.train_with_trainer`; xem mục "Model được lưu ở đâu" trong `train.md`.
- Lưu ý:
  - lần gọi đầu tải checkpoint về (cần mạng).
  - tokenizer và model khác checkpoint sẽ cho id sai mà không có thông báo lỗi.
  - không dùng `distilbert-base-uncased-finetuned-sst-2-english` ở đây: đó là baseline zero-shot của Bài 1 (`demo.md`).

## Cách kiểm chứng

- Unit test: `python -m pytest labs/lab3/tests/test_model.py -q`, kỳ vọng tất cả qua, không cần mạng.
- Smoke test: lệnh trong bảng tiếng Anh ở trên, kỳ vọng `{0: 'NEGATIVE', 1: 'POSITIVE'}` và tokenizer DistilBERT.
- Chạy tích hợp thật: chỉ ở mốc nhóm đã duyệt, ghi cùng với `evaluate`.

## Câu hỏi mở

- Cô chưa xác nhận việc chọn model.
