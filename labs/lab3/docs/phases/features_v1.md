# features — tokenize dataset cho model (v1)

- **Created**: 2026-11-20T01:00:14+07:00
- **Last Updated**: 2026-11-20T14:29:08+07:00 (Thư)
- **Người phụ trách**: Thư
- **Status**: Done — audit 2026-11-20

---

## Name
Tiền xử lý cho model (Bài 2.4) — tokenize 3 split, cắt theo `max_length`, padding động theo batch

## Background
Bài 2.4 của đề: chuẩn bị dataset để đưa vào model. Dùng tokenizer cùng checkpoint với model (`model`). Đầu ra cho `train` và `evaluate`.

## Goals / Purpose
- `tokenize_datasets(ds, tokenizer, max_length)` trả `DatasetDict` có `input_ids`, `attention_mask`, `label`; mọi chuỗi ≤ `max_length`
- `make_collator(tokenizer)` trả `DataCollatorWithPadding` (pad theo câu dài nhất trong batch)
- Không làm:
  - Chọn `max_length` — Thư chốt 512 sau EDA (2026-11-20), đọc từ `config.MAX_LENGTH`, truyền vào làm tham số
  - Làm sạch văn bản — thuộc `preprocessing`

## Input / Output
- **Input:** `DatasetDict` từ `preprocessing`; tokenizer từ `model.load_tokenizer()`
- **Output:** `DatasetDict` đã tokenize (trong bộ nhớ; cache của `datasets`); collator

## How to do it (general plan)
1. `ds.map(tokenize, batched=True)` với `truncation=True`, `max_length`, **không** padding cố định
2. Bỏ cột `text` khỏi dữ liệu đưa vào Trainer; giữ `id` riêng để nối lại dự đoán với review khi phân tích lỗi
3. Collator `DataCollatorWithPadding`
4. Test với tokenizer giả
5. Smoke run; normal run trên Kaggle (gọi trong `main.py`)

## Pipeline
```
preprocessing.load_datasets → tokenize_datasets(max_length) → DatasetDict (input_ids, attention_mask, label)
  + make_collator → train (Trainer), evaluate (predict)
```

## Detailed plan
- **Padding động:** pad theo batch thay vì pad mọi câu tới `max_length` — cách làm trong tutorial text classification chính thức của HF; nhanh hơn vì phần lớn batch ngắn hơn `max_length`
- **Cắt từ đầu (head-only):** mặc định của tokenizer; head+tail nằm ngoài phạm vi (plan §7)
- **Gotchas:**
  - Tokenizer phải cùng checkpoint với model (`distilbert-base-uncased`), nếu khác thì id token lệch từ vựng
  - Đổi `max_length` thì cache `datasets` phải tính lại → tên cache gắn với `max_length`

## Triển khai (2026-11-20)
- **Hàm** (`labs/lab3/features/features.py`): `KEEP_COLUMNS = ("uid", "label")` (`:8`); `tokenize_datasets(ds, tokenizer, max_length, text_column="text")` (`:11`); `make_collator(tokenizer)` (`:31`). `max_length` đọc từ `labs/lab3/config.py` (`MAX_LENGTH`)
- **Chỗ code khác plan:** thêm `return_token_type_ids=False` — tokenizer thật của `distilbert-base-uncased` trong transformers 5.22 vẫn trả cột `token_type_ids` (đo trên dữ liệu thật) mà DistilBERT không dùng. Fixture test dùng tokenizer có trả cột này để test bắt được lỗi
- **Kết quả trên dữ liệu thật** (`max_length` 512): 22.499 / 2.501 / 25.000 dòng, cột `uid, label, input_ids, attention_mask`, không dòng nào vượt 512; 2.996 / 323 / 3.086 review dài hơn 512 token bị cắt (sửa khi audit, xem mục Kết quả audit). Một batch 4 review qua collator → model thật: input [4, 173], logits [4, 2], loss 0,7101 (≈ ln 2 khi đầu phân loại chưa train)
- **Test:** `pytest labs/lab3/tests/test_features.py` 3 passed

## Success criteria
- **Test:** `pytest labs/lab3/tests/test_features.py` đạt: có đủ 3 cột, độ dài ≤ `max_length`, token đầu là `[CLS]`, collator pad đúng
- **Smoke run:** 3 split tokenize xong, đưa được 1 batch qua model
- **Normal run:** 22.5k / 2.5k / 25k dòng (theo split thật), không dòng nào vượt `max_length`

## Kết quả audit (2026-11-20)
- **Test:** `pytest labs/lab3/tests/test_features.py` → 3 passed ✔
- **Smoke run** (`run_member_a(smoke=True)`, 38 giây, máy Thư, 40 review mỗi split): 3 split tokenize xong ở `max_length` 512 ✔
- **Normal run:** 22.499 / 2.501 / 25.000 dòng, không dòng nào vượt 512 ✔
- **Sửa khi audit:** số review bị cắt ghi trước đó (3.010 / 324 / 3.097) đã đếm cả review dài đúng 512 token (14 / 1 / 11 review, không mất token nào). Số bị cắt thật (> 512 token, đếm từ `outputs/results/eda_token_lengths.csv`): 2.996 / 323 / 3.086 — khớp số `visualize` dùng (`error_by_group.csv`: 3.086 review test bị cắt)

## Links
- Desc lab: [../lab3_description_v1.md](../lab3_description_v1.md) | Plan: [../plans/detail_plan_lab3_md_v1.md](../plans/detail_plan_lab3_md_v1.md)
- Module liên quan: [preprocessing_v1.md](preprocessing_v1.md), [model_v1.md](model_v1.md), [train_v1.md](train_v1.md)
- Nguồn: HF Text classification https://huggingface.co/docs/transformers/tasks/sequence_classification
- Code: `labs/lab3/features/features.py` | Test: `labs/lab3/tests/test_features.py`
