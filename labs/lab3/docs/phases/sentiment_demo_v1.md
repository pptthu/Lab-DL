# sentiment_demo — Phân tích cảm xúc bằng model có sẵn (v1)

- **Created**: 2026-11-18T13:00:00+07:00
- **Last Updated**: 2026-11-18T13:00:00+07:00 (Member B)
- **Người phụ trách**: Member A (Muội)
- **Status**: Done

---

## Name
Sentiment demo (Bài 1) — dùng model cảm xúc có sẵn, không train; xem một câu được tách thành token

## Background
Đề Practice 3, Bài 1: dùng model cảm xúc pre-trained từ Hugging Face Hub và tokenize một câu mẫu.
Module này cũng là baseline zero-shot của Bài 2: `evaluate.zero_shot_labels` đổi nhãn của nó sang 0/1 để so với model đã fine-tune.

## Goals / Purpose
- `show_tokens` trả dict `tokens`, `input_ids`; token đầu là `[CLS]` (id 101), token cuối là `[SEP]` (id 102)
- `analyze_sentences` trả danh sách dict `sentence`, `label` (`POSITIVE` hoặc `NEGATIVE`), `score` (float)
- Cả hai hàm dùng cùng checkpoint `distilbert-base-uncased-finetuned-sst-2-english` khi `model_name=None`
- Không làm:
  - Fine-tune hay lưu model — thuộc `train`; module này không ghi gì xuống đĩa
  - Đổi nhãn sang 0/1 — thuộc `evaluate.zero_shot_labels` ([evaluate_v1.md](evaluate_v1.md))
  - Dùng `distilbert-base-uncased` gốc — chưa có đầu phân loại cảm xúc (`sentiment_demo/sentiment_demo.py:12-13`); model gốc được nạp ở [model_v1.md](model_v1.md)

## Input / Output
- **Input:** một câu `str` hoặc danh sách `str`; `model_name` (`None` = checkpoint mặc định); model lấy từ Hugging Face Hub, lần đầu cần mạng
- **Output:** `show_tokens` → `{tokens, input_ids}`; `analyze_sentences` → danh sách `{sentence, label, score}`
  - Module không lưu file. Model tải về nằm trong cache mặc định của Hugging Face (code không đặt `cache_dir`).

## How to do it (general plan)
1. `show_tokens`: nạp tokenizer một lần (lưu trong `_tokenizers`), tokenize câu, đổi id sang token
2. `analyze_sentences`: nạp `pipeline("sentiment-analysis")` một lần (lưu trong `_pipelines`), chạy `batch_size=16`, `truncation=True`; chuỗi đơn tự bọc thành danh sách
3. Test bằng đối tượng giả, không cần mạng
4. Smoke run với 3 câu mẫu (lần đầu cần mạng)
5. Normal run: zero-shot trên 500 review test (notebook mục 10.2)

## Pipeline
```
câu mẫu → show_tokens → tokenizer → tokens + input_ids
câu mẫu → analyze_sentences → pipeline SST-2 → label + score
500 review test → analyze_sentences → evaluate.zero_shot_labels (0/1) → classification_metrics
```

## Detailed plan
- **Checkpoint mặc định:** `distilbert-base-uncased-finetuned-sst-2-english` (`DEFAULT_MODEL`, `sentiment_demo/sentiment_demo.py:14`) — DistilBERT đã fine-tune sẵn trên SST-2 nên không cần train; số zero-shot trong báo cáo phụ thuộc vào checkpoint này, nên không đổi khi chưa hỏi nhóm trưởng
- **Import trong hàm:** `transformers` được import bên trong `_get_tokenizer` (`:24`) và `_get_pipeline` (`:33`) — import module nhẹ, test thay được bằng đối tượng giả
- **Hàm hiện có:** `show_tokens(sentence: str, model_name: str = None) -> dict` (`sentiment_demo/sentiment_demo.py:38`); `analyze_sentences(sentences: list, model_name: str = None) -> list` (`:60`); hàm phụ `_get_tokenizer` (`:20`), `_get_pipeline` (`:29`)
- **Chỗ code chưa khớp plan:** không có
- **Gotchas:**
  - `show_tokens` báo `TypeError` nếu `sentence` không phải `str` (`:53`)
  - Tokenizer của `show_tokens` và model của `analyze_sentences` phải cùng checkpoint — cả hai dùng `DEFAULT_MODEL` khi `model_name=None`
  - Pipeline cắt review dài (`truncation=True`, `:76`); khi so với model fine-tune (cắt ở 128 token) thì hai bên cắt ở độ dài khác nhau (notebook mục 10.2)
  - Chỉ thử 3 câu mẫu, nên việc câu phủ định `Not bad at all` được đọc đúng chưa cho biết model xử lý phủ định tốt nói chung (`../../README.md`)

## Success criteria
- **Test:** `python -m pytest labs/lab3/tests/test_sentiment_demo.py -q` đạt, gồm test hợp đồng `show_tokens`, hợp đồng `analyze_sentences`, nhận chuỗi đơn, và `TypeError` khi không phải `str` (đối tượng giả, không cần mạng)
- **Smoke run:** `python -c "from labs.lab3.sentiment_demo.sentiment_demo import show_tokens, analyze_sentences; s=['I love this movie','The plot was boring','Not bad at all']; [print(show_tokens(x)) for x in s]; print(analyze_sentences(s))"` — token đầu `[CLS]` (101), token cuối `[SEP]` (102); nhãn POSITIVE, NEGATIVE, POSITIVE
- **Normal run:** zero-shot trên 500 review test, accuracy và F1 ghi vào `progress_status.md`

## Kết quả audit (số liệu lần chạy 2026-10-27 và 2026-10-31)
- **Test:** `test_sentiment_demo.py` → 4 passed ✔ (`progress_status.md:6`)
- **Smoke run:** token đầu `[CLS]` (101), token cuối `[SEP]` (102); POSITIVE 0,9999 / NEGATIVE 0,9998 / POSITIVE 0,9991 ✔ (`progress_status.md:29`)
- **Normal run:** zero-shot trên 500 review test: accuracy 0,8920, F1 0,8889 ✔ (notebook ô 27; `progress_status.md:12`)

## Links
- Desc lab: [../lab3_description_v1.md](../lab3_description_v1.md) | Khuôn: [../../../lab1/docs/templates/DESC_TEMPLATE_v3.md](../../../lab1/docs/templates/DESC_TEMPLATE_v3.md)
- Module liên quan: [evaluate_v1.md](evaluate_v1.md), [demo_v1.md](demo_v1.md)
- Tiến độ và bằng chứng chạy: [../../progress_status.md](../../progress_status.md) (dòng 6, 12, 29)
- Code: `labs/lab3/sentiment_demo/sentiment_demo.py` | Test: `labs/lab3/tests/test_sentiment_demo.py`
