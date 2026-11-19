# evaluate — Đánh giá accuracy, F1 và so với zero-shot (v1)

- **Created**: 2026-11-18T13:00:00+07:00
- **Last Updated**: 2026-11-18T13:00:00+07:00 (Member B)
- **Người phụ trách**: Member B (ngohongphuc2805ag-hash)
- **Status**: Done

---

## Name
Evaluate — accuracy và F1, đánh giá test một lần, dự đoán văn bản mới, đổi nhãn zero-shot sang 0/1

## Background
Đề Practice 3, Bài 2: đánh giá model đã fine-tune; `project_description.md` nêu metric là accuracy và F1 (lớp dương = 1 = POSITIVE).
So sánh với pipeline zero-shot của Bài 1 là lựa chọn của nhóm. Trainer cần `compute_metrics` để chọn epoch tốt nhất trên validation.

## Goals / Purpose
- `compute_metrics` trả `accuracy`, `f1` từ `(logits, labels)`; `train` dùng khóa `f1` để giữ epoch tốt nhất
- `evaluate_on_test` trả dict `accuracy`, `f1`, `loss`, … (không có tiền tố `test_`) và chỉ gọi **một lần** ở cuối
- `predict_texts` trả danh sách dict `text`, `label` (0/1), `score` cho văn bản thô
- `zero_shot_labels` đổi `POSITIVE` / `NEGATIVE` của `sentiment_demo` thành 1 / 0
- Không làm:
  - Chọn epoch hay cài đặt bằng tập test — việc chọn dùng validation ở `train` ([train_v1.md](train_v1.md))
  - Dùng thư viện `evaluate` của Hugging Face hay thêm metric đề không yêu cầu — metric chỉ dùng NumPy, không cần mạng, tránh trùng tên với package `evaluate`
  - Lưu file — `main.py` ghi dict của `evaluate_on_test` ra file; xem mục Input / Output

## Input / Output
- **Input:**
  - `compute_metrics`: `eval_pred` là `(logits, labels)` hoặc `EvalPrediction`
  - `evaluate_on_test`: `Trainer` từ [train_v1.md](train_v1.md) (giữ epoch tốt nhất trên validation) và `tokenized_test` từ [features_v1.md](features_v1.md)
  - `predict_texts`: `model`, `tokenizer`, danh sách `str`
  - `zero_shot_labels`: kết quả của `analyze_sentences` ([sentiment_demo_v1.md](sentiment_demo_v1.md))
- **Output:** dict metric hoặc danh sách dict; module không lưu file
  - `main.py` ghi dict của `evaluate_on_test` vào `labs/lab3/artifacts/logs/test_metrics.json` (`main.py:135-136`); notebook ghi so sánh zero-shot / fine-tune vào `artifacts/logs/lab3_results.json` (ô 27). Cả hai không commit (`.gitignore:16`).
  - Nếu nạp model từ ổ đĩa, đó là thư mục do `train` ghi (xem mục "Nơi lưu model" trong [train_v1.md](train_v1.md)).

## How to do it (general plan)
1. `classification_metrics(labels, predictions)`: accuracy và F1 = 2TP / (2TP + FP + FN)
2. `compute_metrics(eval_pred)`: lấy `argmax` của logits, gọi `classification_metrics`
3. `evaluate_on_test(trainer, tokenized_test)`: `trainer.evaluate(metric_key_prefix="test")`, bỏ tiền tố `test_`
4. `predict_texts(model, tokenizer, texts, max_length, batch_size)`: tokenize bằng đúng tokenizer của model, cắt `max_length`, padding theo batch, softmax
5. `zero_shot_labels(results)`: ánh xạ nhãn pipeline sang 0/1
6. Smoke run: `predict_texts` trên 3 câu mẫu
7. Normal run: test 500 review một lần, so với zero-shot trên cùng 500 review

## Pipeline
```
train.Trainer (epoch tốt nhất theo validation)
  → evaluate_on_test(tokenized_test) → accuracy, f1, loss      (một lần)
sentiment_demo.analyze_sentences → zero_shot_labels → classification_metrics
  → so với fine-tuned trên cùng 500 review test
```

## Detailed plan
- **F1 nhị phân:** lớp dương = 1; F1 = 0,0 khi mẫu số bằng 0, không chia cho 0 (`evaluate/evaluate.py:43-47`)
- **Sai số lấy mẫu:** với n = 500, sai số chuẩn của accuracy khoảng 1,4 đến 1,7 điểm; chênh lệch dưới khoảng 2,5 đến 3 điểm nằm trong sai số (notebook ô 27)
- **`predict_texts`:** đặt model về chế độ eval khi gọi và trả về chế độ cũ sau đó (`:125-126`, `:139-140`); bỏ `token_type_ids` vì DistilBERT không dùng (`:134`)
- **Hàm hiện có:** `classification_metrics(labels, predictions) -> dict` (`evaluate/evaluate.py:24`); `compute_metrics(eval_pred) -> dict` (`:50`); `evaluate_on_test(trainer, tokenized_test) -> dict` (`:74`); `predict_texts(model, tokenizer, texts: list, *, max_length: int = 256, batch_size: int = 16) -> list` (`:99`); `zero_shot_labels(results: list) -> list` (`:144`)
- **Chỗ code chưa khớp plan:** `predict_texts` có thêm 2 tham số keyword-only `max_length`, `batch_size`, và có 2 hàm phụ `classification_metrics`, `zero_shot_labels` ngoài hợp đồng trong `project_description.md`; đều có mặc định nên lời gọi theo hợp đồng vẫn chạy
- **Gotchas:**
  - `predict_texts` báo `TypeError` nếu `texts` không phải danh sách `str` (`:116-117`); danh sách rỗng trả `[]` (`:121-122`)
  - `compute_metrics` báo `ValueError` nếu logits không phải mảng 2 chiều (`:69-70`)
  - `evaluate_on_test` báo `ValueError` khi tập test rỗng (`:89-90`); dùng test nhiều lần để chọn cài đặt làm điểm test bị lạc quan
  - So sánh zero-shot với fine-tune: pipeline cắt review ở 512 token, model fine-tune cắt ở 128 token — nguyên nhân của chênh lệch chưa được kiểm chứng
  - Dùng cùng `max_length` với lúc train khi gọi `predict_texts`

## Success criteria
- **Test:** `python -m pytest labs/lab3/tests/test_evaluate.py -q` đạt, gồm test metric tính tay, hoàn hảo và sai hết, F1 = 0 không chia cho 0, `EvalPrediction`, `evaluate_on_test` bỏ tiền tố `test_`, `predict_texts` (định dạng, thứ tự, batch, trả về chế độ train, danh sách rỗng, đầu vào sai), `zero_shot_labels` (NumPy và `torch` giả, không cần mạng)
- **Smoke run:** `predict_texts(model, tokenizer, ['I love this movie', 'The plot was boring', 'Not bad at all'])` cho 3 dict có `label` 0/1 và `score` trong (0, 1)
- **Normal run:** test n = 500 chỉ một lần cho model đã fine-tune; bảng accuracy và F1 của zero-shot và fine-tuned trên cùng 500 review. Lab 3 chưa đặt ngưỡng accuracy trong desc lab

## Kết quả audit (số liệu lần chạy 2026-10-30 và 2026-10-31)
- **Test:** `test_evaluate.py` → 16 passed ✔ (`progress_status.md:12`); so với scikit-learn trên 300 trường hợp ngẫu nhiên, chênh lệch tối đa bằng 0 ✔ (`progress_status.md:32`)
- **Smoke run:** notebook ô 22: POSITIVE 0,9000 / NEGATIVE 0,8555 / NEGATIVE 0,6667. Câu phủ định `Not bad at all` bị model fine-tune đoán NEGATIVE, trái với pipeline (POSITIVE 0,9991); chỉ 3 câu nên chưa kết luận chung ✔
- **Normal run:** test n = 500, chạy một lần: loss 0,3803, accuracy 0,8220, F1 0,8195; zero-shot accuracy 0,8920, F1 0,8889; chênh −7,0 điểm accuracy (−6,9 điểm F1), lớn hơn sai số lấy mẫu; một seed, một tập con ✔ (notebook ô 26, ô 27; `progress_status.md:12`). Trên validation: ma trận nhầm lẫn `[[209, 34], [49, 208]]`, 83 sai trên 500 (notebook ô 22)

## Links
- Desc lab: [../lab3_description_v1.md](../lab3_description_v1.md) | Khuôn: [../../../lab1/docs/templates/DESC_TEMPLATE_v3.md](../../../lab1/docs/templates/DESC_TEMPLATE_v3.md)
- Module liên quan: [train_v1.md](train_v1.md), [features_v1.md](features_v1.md), [sentiment_demo_v1.md](sentiment_demo_v1.md), [model_v1.md](model_v1.md)
- Model card: [../models/distilbert_imdb_v1.md](../models/distilbert_imdb_v1.md)
- Tiến độ và bằng chứng chạy: [../../progress_status.md](../../progress_status.md) (dòng 12, 32)
- Code: `labs/lab3/evaluate/evaluate.py` | Test: `labs/lab3/tests/test_evaluate.py`
