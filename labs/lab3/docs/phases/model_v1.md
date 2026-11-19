# model — Nạp DistilBERT pre-trained (v1)

- **Created**: 2026-11-18T13:00:00+07:00
- **Last Updated**: 2026-11-18T13:00:00+07:00 (Member B)
- **Người phụ trách**: Member B (ngohongphuc2805ag-hash)
- **Status**: Done

---

## Name
Model — nạp DistilBERT pre-trained và tokenizer cho phân loại 2 nhãn (NEGATIVE / POSITIVE)

## Background
Đề Practice 3, Bài 2: load một model pre-trained và tokenizer của nó. Nhóm chọn `distilbert-base-uncased` với 2 nhãn; lựa chọn này chưa xác nhận với cô.
Module chỉ **nạp** model. Việc train và lưu model thuộc `train`. Model đã nạp là model ban đầu (Initial model) trong [model card](../models/distilbert_imdb_v1.md).

## Goals / Purpose
- `load_model_and_tokenizer` trả `(model, tokenizer)` nạp từ **cùng một** checkpoint
- Với 2 nhãn, `model.config.id2label` là `{0: 'NEGATIVE', 1: 'POSITIVE'}` — cùng quy ước với pipeline của Bài 1
- Model trả logits dạng `(batch, num_labels)`
- Không làm:
  - Train hay lưu model — thuộc `train` ([train_v1.md](train_v1.md))
  - Dùng `distilbert-base-uncased-finetuned-sst-2-english` — đó là baseline zero-shot của `sentiment_demo` ([sentiment_demo_v1.md](sentiment_demo_v1.md)), đã có sẵn đầu phân loại
  - Thêm họ model khác ngoài DistilBERT — ngoài phạm vi lab

## Input / Output
- **Input:** `model_name` (mặc định `distilbert-base-uncased`), `num_labels` (mặc định 2); checkpoint lấy từ Hugging Face Hub, lần đầu cần mạng
- **Output:** `(model, tokenizer)`; model là `DistilBertForSequenceClassification` với 67,0 triệu tham số (notebook ô 15); chuyển cho `features` (tokenizer), `train` (cả hai) và `evaluate.predict_texts`
  - Module không lưu file. Checkpoint tải về nằm trong cache mặc định của Hugging Face (code không đặt `cache_dir`). Model đã fine-tune được lưu ở `train`, xem mục "Nơi lưu model" trong [train_v1.md](train_v1.md).

## How to do it (general plan)
1. Kiểm `num_labels` là số nguyên từ 2 trở lên
2. `AutoTokenizer.from_pretrained(model_name)`
3. Với 2 nhãn, đặt `id2label = {0: NEGATIVE, 1: POSITIVE}` và `label2id` ngược lại
4. `AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=...)`
5. Smoke run: nạp thật `distilbert-base-uncased` và in `id2label`
6. Normal run: dùng model này cho lần fine-tune ở [train_v1.md](train_v1.md)

## Pipeline
```
model_name → AutoTokenizer + AutoModelForSequenceClassification (cùng checkpoint)
→ encoder nạp từ checkpoint; đầu phân loại khởi tạo mới
→ (model, tokenizer) → features, train, evaluate.predict_texts
```

## Detailed plan
- **Checkpoint:** `distilbert-base-uncased` — model nhỏ (67,0 triệu tham số) chạy được trên CPU, hợp ràng buộc CPU trong `project_description.md`
- **Đầu phân loại khởi tạo ngẫu nhiên:** `classifier.*` và `pre_classifier.*` không có trong checkpoint nên được tạo mới (báo cáo nạp ở notebook ô 15 ghi chúng là `MISSING`). Cảnh báo của Hugging Face là bình thường, vì đầu này được học khi fine-tune
- **`id2label` đặt tay cho 2 nhãn:** khớp với nhãn 0/1 của dataset và với `evaluate.zero_shot_labels`
- **Hàm hiện có:** `load_model_and_tokenizer(model_name: str = "distilbert-base-uncased", num_labels: int = 2)` (`model/model.py:7`)
- **Chỗ code chưa khớp plan:** không có
- **Gotchas:**
  - `num_labels` không phải số nguyên ≥ 2 thì `ValueError` (`model/model.py:23-24`)
  - Lần gọi đầu tải checkpoint nên cần mạng
  - Tokenizer và model từ hai checkpoint khác nhau cho id sai mà không báo lỗi — hàm này nạp cả hai từ `model_name` (`:27`, `:32`)
  - `transformers` được import trong hàm (`:25`) để test không cần mạng

## Success criteria
- **Test:** `python -m pytest labs/lab3/tests/test_model.py -q` đạt, gồm test cùng checkpoint, tên nhãn cho 2 nhãn, 3 nhãn không có tên nhị phân, `num_labels` không hợp lệ (`transformers` giả)
- **Smoke run:** `python -c "from labs.lab3.model.model import load_model_and_tokenizer; m,t=load_model_and_tokenizer(); print(m.config.id2label, type(t).__name__)"` in `{0: 'NEGATIVE', 1: 'POSITIVE'}` và tên lớp tokenizer DistilBERT
- **Normal run:** model nạp được dùng cho lần fine-tune đầy đủ ở [train_v1.md](train_v1.md)

## Kết quả audit (số liệu lần chạy 2026-10-29 và 2026-10-31)
- **Test:** `test_model.py` → 8 passed ✔ (`progress_status.md:10`); đã merge PR #38
- **Smoke run:** notebook ô 15 in `DistilBertForSequenceClassification (67.0M parameters)`, `checkpoint: distilbert-base-uncased | tokenizer: distilbert-base-uncased`, `id2label : {0: 'NEGATIVE', 1: 'POSITIVE'}` ✔
- **Normal run:** model này được fine-tune trong lần chạy 2026-10-31 (xem audit của [train_v1.md](train_v1.md)) ✔

## Links
- Desc lab: [../lab3_description_v1.md](../lab3_description_v1.md) | Khuôn: [../../../lab1/docs/templates/DESC_TEMPLATE_v3.md](../../../lab1/docs/templates/DESC_TEMPLATE_v3.md)
- Module liên quan: [features_v1.md](features_v1.md), [train_v1.md](train_v1.md), [evaluate_v1.md](evaluate_v1.md), [sentiment_demo_v1.md](sentiment_demo_v1.md)
- Model card: [../models/distilbert_imdb_v1.md](../models/distilbert_imdb_v1.md)
- Tiến độ và bằng chứng chạy: [../../progress_status.md](../../progress_status.md) (dòng 10)
- Code: `labs/lab3/model/model.py` | Test: `labs/lab3/tests/test_model.py`
