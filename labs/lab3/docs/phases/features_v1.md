# features — Tokenize review (v1)

- **Created**: 2026-11-18T13:00:00+07:00
- **Last Updated**: 2026-11-18T13:00:00+07:00 (Member B)
- **Người phụ trách**: Member A (Muội)
- **Status**: Done

---

## Name
Features — đổi review thành `input_ids` và `attention_mask`, cắt về `max_length` token, chưa padding

## Background
Đề Practice 3, Bài 2: tiền xử lý dataset cho model (tokenization). Model chỉ đọc số, nên mỗi review được tokenizer đổi thành dãy id.
Review dài hơn `max_length` bị cắt. Padding không làm ở đây mà làm theo từng batch ở `train`.

## Goals / Purpose
- `tokenize_dataset` trả dataset có đúng 3 cột `label`, `input_ids`, `attention_mask`; cột `text` bị bỏ
- Mỗi review có tối đa `max_length` token (cắt, không padding)
- Không làm:
  - Padding cố định — padding theo từng batch bằng `DataCollatorWithPadding` ở [train_v1.md](train_v1.md)
  - Dùng tokenizer của checkpoint khác — tokenizer lấy từ `model.load_model_and_tokenizer` ([model_v1.md](model_v1.md))
  - Chọn `max_length` — module chỉ nhận tham số; `eda` cho số liệu độ dài, giá trị do `main.py` hoặc notebook truyền vào

## Input / Output
- **Input:** dataset có cột `text`, `label` từ [preprocessing_v1.md](preprocessing_v1.md); tokenizer; `max_length`; `text_col`
- **Output:** dataset đã tokenize có cột `label`, `input_ids`, `attention_mask`; các dòng chưa cùng độ dài; chuyển cho `train` và `evaluate`
  - Module không ghi file.

## How to do it (general plan)
1. Kiểm cột `text_col` có trong dataset
2. `dataset.map(batched=True)` gọi tokenizer với `truncation=True`, `max_length`, `return_token_type_ids=False`
3. Bỏ cột `text`, giữ `label` (Trainer tự đổi tên thành `labels`)
4. Smoke run: tokenize vài dòng thật bằng tokenizer của `distilbert-base-uncased`
5. Normal run: tokenize cả ba tập (train, validation, test) cùng một tokenizer và cùng `max_length`

## Pipeline
```
preprocessing (train, val, test) + tokenizer của model
→ tokenize_dataset(max_length) → label, input_ids, attention_mask
→ train.train_with_trainer (padding theo batch) → evaluate.evaluate_on_test
```

## Detailed plan
- **`max_length`:** mặc định 256 (`features/features.py:11`; `main.py:160`); lần fine-tune đã ghi dùng 128 (notebook ô 17) vì giới hạn RAM 7,8 GB và 2 nhân CPU (`../../README.md`)
- **Không padding ở đây:** quyết định của lab — các dòng có độ dài khác nhau, `DataCollatorWithPadding` ở `train` padding theo từng batch để đỡ tốn tính toán
- **`return_token_type_ids=False`:** DistilBERT không dùng `token_type_ids` (`features/features.py:33`)
- **Hàm hiện có:** `tokenize_dataset(dataset, tokenizer, max_length: int = 256, text_col: str = 'text')` (`features/features.py:11`)
- **Chỗ code chưa khớp plan:** mặc định `max_length` (256) khác giá trị của lần chạy đã ghi (128); báo cáo phải nêu kết quả nào dùng giá trị nào
- **Gotchas:**
  - Thiếu cột `text_col` thì `KeyError` (`features/features.py:29-30`)
  - 802 / 2000 review train (40,1%) đạt `max_length=256` token — bị cắt hoặc vừa đủ 256; ảnh hưởng của việc cắt tới accuracy chưa đo (notebook ô 9)
  - Phải dùng cùng `max_length` khi gọi `evaluate.predict_texts` (xem [evaluate_v1.md](evaluate_v1.md))
  - Tokenizer và model phải cùng checkpoint, nếu không id sai mà không báo lỗi

## Success criteria
- **Test:** `python -m pytest labs/lab3/tests/test_features.py -q` đạt (3 test với tokenizer giả: có đúng 3 cột và bỏ cột `text`, cắt đúng `max_length`, thiếu cột báo `KeyError`)
- **Smoke run:** tokenize `train.select(range(8))` bằng tokenizer thật của `distilbert-base-uncased`, `column_names` là `['label', 'input_ids', 'attention_mask']`
- **Normal run:** ba tập 2000 / 500 / 500 dòng đều có 3 cột trên với `max_length` đã chọn

## Kết quả audit (số liệu lần chạy 2026-10-26 và 2026-10-31)
- **Test:** `test_features.py` → 3 passed ✔; cùng `test_smoke.py` → 11 passed ✔ (`progress_status.md:9`, `:25`, `:26`)
- **Smoke run:** notebook ô 9 in `['label', 'input_ids', 'attention_mask']`; 12 id đầu của dòng đầu tiên bắt đầu bằng 101 (`[CLS]`) ✔
- **Normal run:** notebook ô 17 (`max_length=128`) in `columns: ['label', 'input_ids', 'attention_mask'] | sizes (train/val/test): 2000 500 500` ✔

## Links
- Desc lab: [../lab3_description_v1.md](../lab3_description_v1.md) | Khuôn: [../../../lab1/docs/templates/DESC_TEMPLATE_v3.md](../../../lab1/docs/templates/DESC_TEMPLATE_v3.md)
- Module liên quan: [preprocessing_v1.md](preprocessing_v1.md), [model_v1.md](model_v1.md), [train_v1.md](train_v1.md), [eda_v1.md](eda_v1.md)
- Tiến độ và bằng chứng chạy: [../../progress_status.md](../../progress_status.md) (dòng 9, 25, 26)
- Code: `labs/lab3/features/features.py` | Test: `labs/lab3/tests/test_features.py`
