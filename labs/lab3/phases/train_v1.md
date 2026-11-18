# train — Fine-tune bằng Trainer (v1)

- **Created**: 2026-11-18T13:00:00+07:00
- **Last Updated**: 2026-11-18T13:00:00+07:00 (Member B)
- **Người phụ trách**: Member B (ngohongphuc2805ag-hash)
- **Status**: Done

---

## Name
Training — `TrainingArguments`, `Trainer`, giữ epoch tốt nhất theo F1 validation, lưu model

## Background
Đề Practice 3, Bài 2: định nghĩa training arguments, tạo Trainer và fine-tune.
Train dùng tập validation để chọn epoch. Tập test không bao giờ được đưa vào đây; nó chỉ dùng một lần ở `evaluate`.

## Goals / Purpose
- `build_training_args` trả `TrainingArguments` đánh giá và lưu mỗi epoch, giữ epoch có `f1` validation cao nhất (`load_best_model_at_end`)
- `train_with_trainer` trả `Trainer` đã train và lưu model cùng tokenizer vào `args.output_dir`
- Padding theo từng batch bằng `DataCollatorWithPadding`
- Không làm:
  - Đưa tập test vào Trainer hoặc chọn epoch theo test — test chỉ dùng ở `evaluate.evaluate_on_test` ([evaluate_v1.md](evaluate_v1.md))
  - Tokenize hay cắt `max_length` — việc của `features` ([features_v1.md](features_v1.md))
  - Tự chọn nơi lưu model — người gọi (`main.py` hoặc notebook) truyền `output_dir`; xem mục Input / Output

## Input / Output
- **Input:** `(model, tokenizer)` từ [model_v1.md](model_v1.md) (model ban đầu là `distilbert-base-uncased`); `tokenized_train`, `tokenized_val` từ [features_v1.md](features_v1.md); `compute_metrics` từ [evaluate_v1.md](evaluate_v1.md); `output_dir`
- **Output:** `Trainer` đã train, giữ epoch tốt nhất trên validation; model lưu trong `output_dir` (trọng số, config, tokenizer)
  - **Nơi lưu model:** `train.py` không tự chọn đường dẫn, chỉ gọi `trainer.save_model(args.output_dir)` (`train/train.py:94`). Hai nơi đang truyền `output_dir`:
    - `main.py --run-training`: `artifact_dir("lab3", "models") / "distilbert_sentiment"` (`main.py:110-111`) → `labs/lab3/artifacts/models/distilbert_sentiment/`; `main.py` in dòng `Model saved to: ...` (`main.py:145`)
    - notebook (ô 17): `artifact_dir('lab3', 'models') / 'distilbert_imdb'` → `labs/lab3/artifacts/models/distilbert_imdb/`; log của Trainer ghi vào `labs/lab3/artifacts/logs/distilbert_imdb_log.json`
  - Cả hai nằm dưới `labs/lab3/artifacts/` do `artifact_dir` tạo (`shared/paths.py`), không commit (`.gitignore:12` cho `models/`, `.gitignore:16` cho `logs/`). Không lưu vào gốc repo, `raw_data/` (chỉ là cache dataset) hay `docs/`.
  - Vì evaluate và save chạy mỗi epoch nên `Trainer` còn ghi thư mục con `checkpoint-<step>` trong `output_dir`, giữ tối đa 1 (`save_total_limit=1`, `train/train.py:44`).
  - Nạp lại: `AutoModelForSequenceClassification.from_pretrained(<thư mục>)` và `AutoTokenizer.from_pretrained(<thư mục>)` (tokenizer được lưu cùng vì Trainer nhận tokenizer).

## How to do it (general plan)
1. `build_training_args(output_dir, epochs, lr, batch_size, ...)`: weight decay 0,01, đánh giá + lưu mỗi epoch, `metric_for_best_model="f1"`, `save_total_limit=1`, `report_to="none"`; tự dò tên `eval_strategy` hay `evaluation_strategy` theo phiên bản `transformers`
2. `train_with_trainer(...)`: từ chối tập train hoặc validation rỗng; lấy tokenizer được truyền vào, nếu không có thì nạp từ `model.name_or_path`; tự dò tên `processing_class` hay `tokenizer` của `Trainer`
3. `trainer.train()`, rồi `trainer.save_model(args.output_dir)`
4. Smoke run: fine-tune rất ngắn bằng `max_steps=4` trên 64 review
5. Normal run: fine-tune nhỏ đầy đủ trên 2000 review train, chọn epoch trên 500 review validation, ghi `trainer.state.log_history`

## Pipeline
```
features (train, val đã tokenize) + model + evaluate.compute_metrics
→ build_training_args → train_with_trainer:
    mỗi epoch: train → đánh giá validation (f1) → lưu checkpoint
    kết thúc: nạp lại epoch tốt nhất → trainer.save_model(output_dir)
→ Trainer cho evaluate.evaluate_on_test
```

## Detailed plan
- **Siêu tham số mặc định:** epochs 2, lr 2e-5, batch 16, seed 42, weight decay 0,01, `logging_steps` 10 (`train/train.py:7-8`, `:38`)
- **Lần chạy đã ghi (2026-10-31):** 1 epoch, lr 2e-5, batch 8, `max_length` 128, seed 42 (notebook ô 17) — dùng cấu hình nhỏ vì RAM 7,8 GB và 2 nhân CPU (`../../README.md`)
- **Chọn epoch theo F1 validation:** `metric_for_best_model="f1"` (`:42`) — khóa `f1` do `evaluate.compute_metrics` trả
- **`max_steps` > 0 thay `epochs`:** dùng cho smoke run rất ngắn (`:46`)
- **Hàm hiện có:** `build_training_args(output_dir, epochs: int = 2, lr: float = 2e-5, batch_size: int = 16, *, max_steps: int = -1, seed: int = 42, logging_steps: int = 10)` (`train/train.py:7`); `train_with_trainer(model, tokenized_train, tokenized_val, args, compute_metrics, tokenizer=None)` (`:53`)
- **Chỗ code chưa khớp plan:**
  - `max_steps`, `seed`, `logging_steps` (ở `build_training_args`) và `tokenizer` (ở `train_with_trainer`) là tham số thêm ngoài hợp đồng trong `project_description.md`; đều có mặc định nên lời gọi theo hợp đồng vẫn chạy
  - Tên thư mục lưu model của `main.py` (`distilbert_sentiment`) khác notebook (`distilbert_imdb`); model train bằng luồng này không được luồng kia tìm thấy [cần chốt: nhóm trưởng chọn một tên]
  - Mặc định của `main.py` (`--epochs 2`, `--max-length 256`, `--batch-size 16`; `main.py:160`, `:166`, `:168`) khác lần chạy đã ghi (1, 128, 8)
- **Gotchas:**
  - Tên tham số của `transformers` đổi theo phiên bản, nên giữ phần tự dò (`train/train.py:30-31`, `:83`)
  - Lần chạy đã ghi chỉ có 1 epoch, nên việc "giữ epoch tốt nhất" chưa được thử qua nhiều epoch
  - Fine-tune tốn CPU: dùng tập con nhỏ, `max_length` ngắn; lần đã ghi mất khoảng 17 phút (`../../README.md`); thống nhất với nhóm trưởng trước khi chạy đầy đủ
  - Thiếu cả `tokenizer` lẫn `model.name_or_path` thì `ValueError` (`train/train.py:80-81`)

## Success criteria
- **Test:** `python -m pytest labs/lab3/tests/test_train.py -q` đạt, gồm test giá trị `TrainingArguments` và tên `eval_strategy` mới / cũ, `max_steps` và kiểm tham số, Trainer train + lưu + padding theo batch, tokenizer lấy từ tên model, thiếu tokenizer và tên model báo lỗi, tập rỗng báo lỗi (`transformers` giả)
- **Smoke run:** fine-tune `max_steps=4` trên 64 review chạy hết không lỗi
- **Normal run:** `output_dir` nạp lại được cả model lẫn tokenizer; `trainer.state.log_history` có `loss` theo bước và `eval_f1` mỗi epoch; epoch được giữ là epoch có `eval_f1` cao nhất

## Kết quả audit (số liệu lần chạy 2026-10-31)
- **Test:** `test_train.py` → 8 passed ✔ (`progress_status.md:11`); đã merge PR #39
- **Smoke run:** 4 bước, 64 review train / 32 review test: accuracy 0,4688, F1 0,3200 — chỉ chứng minh code chạy ✔ (`progress_status.md:12`)
- **Normal run (1 epoch, batch 8, `max_length` 128, seed 42):** training loss 0,6912 → 0,4108 (25 điểm log); validation epoch 1: loss 0,3607, accuracy 0,8340, F1 0,8337 ✔ (notebook ô 18, ô 20); ô 17 nạp lại model và tokenizer từ `artifacts/models/distilbert_imdb/` không cần train lại ✔
- **`main.py --run-training --max-steps 2`:** chưa có lần chạy nào được ghi — chưa kiểm chứng

## Links
- Desc lab: [../lab3_description_v1.md](../lab3_description_v1.md) | Khuôn: [../../../lab1/docs/templates/DESC_TEMPLATE_v3.md](../../../lab1/docs/templates/DESC_TEMPLATE_v3.md)
- Module liên quan: [features_v1.md](features_v1.md), [model_v1.md](model_v1.md), [evaluate_v1.md](evaluate_v1.md), [demo_v1.md](demo_v1.md)
- Model card: [../models/distilbert_imdb_v1.md](../models/distilbert_imdb_v1.md)
- Tiến độ và bằng chứng chạy: [../../progress_status.md](../../progress_status.md) (dòng 11, 12)
- Code: `labs/lab3/train/train.py` | Test: `labs/lab3/tests/test_train.py`
