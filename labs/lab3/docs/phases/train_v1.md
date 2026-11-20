# train — TrainingArguments, Trainer, fine-tune, log, dừng sớm (v1)

- **Created**: 2026-11-20T01:00:14+07:00
- **Last Updated**: 2026-11-20T14:37:54+07:00 (Thư)
- **Người phụ trách**: Phúc (Member B)
- **Status**: Done — audit 2026-11-20

---

## Name
Fine-tune (Bài 2.5–2.6) — tạo TrainingArguments, Trainer, chạy fine-tune, ghi log theo khuôn, chọn checkpoint theo val accuracy

## Background
Bài 2.5 (TrainingArguments) và 2.6 (Trainer + fine-tune) của đề. Plan L8–L11: lr 2e-5, warmup 10%, linear decay, batch 16, fp16, đánh giá mỗi 1/4 epoch, chọn checkpoint theo val accuracy; số epoch chốt sau pilot. Chạy trên Kaggle GPU (L14).

## Goals / Purpose
- `make_training_args(cfg)` trả `TrainingArguments` đúng cấu hình; mọi tham số đọc từ một `cfg`
- `train_model(cfg, model, tokenized, collator)` trả `best_step`, `best_val_acc`, `stopped_step`, `stop_reason`, đường dẫn checkpoint tốt nhất
- Mỗi lần chạy ghi 3 file theo `LOG_TEMPLATE_v1.md`: `.log`, `_history.csv` (mỗi lần đánh giá một dòng), `_config.json`
- Không làm:
  - Đánh giá trên test — thuộc `evaluate`
  - Chạy nhiều seed / nhiều `max_length` — thuộc `experiment` (gọi `train_model` nhiều lần)

## Input / Output
- **Input:** `cfg` (seed, `max_length`, lr, batch, warmup, trần epoch, chế độ pilot/chính, patience, threshold); model (`model`); dataset đã tokenize + collator (`features`); `compute_metrics` (`evaluate`)
- **Output:** `outputs/logs/<ts>_<run>.*` (commit); `outputs/models/<ts>_<run>_best/` (không commit); smoke run: `outputs/_smoke/`

## How to do it (general plan)
1. `make_training_args`: các tham số ở Detailed plan
2. Callback ghi log: header (GPU, phiên bản thư viện, config) → một dòng mỗi lần đánh giá → footer (`best_step`, `stop_reason`, `best_is_last_eval`)
3. Chế độ **pilot**: không dừng sớm. Chế độ **chính**: thêm `EarlyStoppingCallback(patience, threshold)`
4. `trainer.train()`; lưu checkpoint tốt nhất
5. Test với model DistilBERT rất nhỏ và vài chục mẫu giả (CPU)
6. Smoke run trên máy Thư: vài step, `fp16=False`
7. Normal run trên Kaggle: pilot (S2) → chính 3 seed (S3, qua `experiment`)

## Pipeline
```
cfg + model + tokenized + collator + compute_metrics
  → make_training_args → Trainer (+ callback log, + EarlyStopping nếu chế độ chính)
  → train → checkpoint val acc cao nhất → evaluate
  → outputs/logs/<ts>_<run>.log, _history.csv, _config.json
```

## Detailed plan
- **learning_rate 2e-5, AdamW, warmup_ratio 0,1, lr_scheduler_type linear:** Mosbach et al. 2021 (lr 2e-5, tăng tuyến tính 10% bước đầu rồi giảm về 0); BERT gốc chọn lr trong {5e-5 … 2e-5}
- **per_device_train_batch_size 16:** trong {16, 32} của BERT gốc; vừa VRAM T4 với `max_length` 512 khi bật fp16 `[kiểm ở pilot]`
- **weight_decay 0,01 (Thư duyệt 2026-11-20, cố định, không đưa vào thí nghiệm):** Mosbach et al. mô tả công thức fine-tune mặc định theo Devlin et al. là *"weight decay with λ = 0.01"*; tutorial text classification của HF cũng dùng 0,01. Nguồn chưa thống nhất: có bản tái hiện dùng 0 khi fine-tune
- **fp16=True trên GPU:** chỉ để nhanh và ít VRAM; tắt khi chạy CPU
- **eval_strategy / save_strategy = "steps", eval_steps = save_steps ≈ số bước mỗi epoch / 4:** Dodge et al. 2020 (hiệu năng giữa epoch 1 đã tương quan cao với kết quả cuối)
- **load_best_model_at_end=True, metric_for_best_model="accuracy", greater_is_better=True:** plan L8; thiếu `greater_is_better` thì Trainer coi "nhỏ hơn là tốt" (lỗi đã gặp trên HF forum)
- **num_train_epochs (trần), patience, threshold:** `[cần chốt sau pilot]`; quy tắc `best_is_last_eval` (LOG_TEMPLATE §3.5)
- **save_total_limit 2:** giữ checkpoint tốt nhất + mới nhất, tránh đầy ổ Kaggle
- **report_to="none":** không gửi log ra dịch vụ ngoài
- **Gotchas:**
  - Tên tham số đổi theo phiên bản: `eval_strategy` (mới) / `evaluation_strategy` (cũ); `processing_class` / `tokenizer` → ghi phiên bản trong log, chốt phiên bản trong `requirements.txt` theo Kaggle
  - Lịch lr linear giảm theo **trần epoch** → đổi trần là đổi tốc độ giảm lr (lý do pilot và lần chạy chính phải ghi trần riêng)
  - `EarlyStoppingCallback` cần `load_best_model_at_end=True` và `metric_for_best_model`; tên metric phải khớp `eval_accuracy`, nếu không callback tự tắt kèm cảnh báo

## Triển khai (2026-11-20)
- **Hàm** (`labs/lab3/train/train.py`): `default_cfg(**overrides) -> dict` (`:29`); `eval_steps_for(n_train, cfg) -> int` (`:52`); `make_training_args(cfg, output_dir, n_train)` (`:63`); `run_environment() -> dict` (`:122`); `RunLog` (`:150`, ghi `.log`, `_history.csv`, `_config.json`); `make_log_callback(log, t0_holder)` (`:220`); `summarize_history(rows, max_steps_planned, stopped_step) -> dict` (`:251`); `train_model(cfg, model, tokenized, collator, compute_metrics=None, tokenizer=None, logs_dir=None, models_dir=None) -> dict` (`:271`)
- **Chỗ code khác plan:**
  - `train_model` có thêm tham số tuỳ chọn `compute_metrics`, `tokenizer`, `logs_dir`, `models_dir` (để test và smoke ghi vào thư mục riêng); mặc định dùng `evaluate.compute_metrics` và `outputs/`
  - transformers 5.22 bỏ `warmup_ratio` → dùng `warmup_steps=0.1` (số thực < 1 được hiểu là tỉ lệ tổng step, theo docstring `TrainingArguments`)
  - Tham số của phần B đặt trong `labs/lab3/config.py` (LEARNING_RATE … THRESHOLD), không đặt trong `train.py` (RULES_v1 §2: một chỗ)
  - Log ghi thêm `code: <HEAD>+tree:<cây index>` (code chạy chưa commit) và `test_evaluated` trong `_config.json` (cờ cho `evaluate`)
  - Checkpoint trung gian (`<ts>_<run>_ckpt/`) bị xoá sau khi lưu `<ts>_<run>_best/` để không đầy ổ Kaggle; `save_best=False` cho đo thời gian và thí nghiệm `max_length` (không cần checkpoint)
- **Gotcha gặp thật:** Kaggle T4 x2 làm Trainer tự chạy DataParallel (batch thực 32) → kernel đặt `CUDA_VISIBLE_DEVICES=0` (`labs/lab3/kaggle/kernel_template.py`)
- **Kết quả normal run** (Kaggle Tesla T4, fp16; chi tiết `docs/experiments/experiment_log.md` S2–S3):
  - Đo thời gian: 0,356 s/step (300 step); pilot: **5,2 phút/epoch**, 6 epoch trong 1.872 s
  - Pilot (seed 42, không dừng sớm): best val acc 0,9308 ở epoch 4,0; `best_is_last_eval: no`; val loss thấp nhất ở epoch 1 rồi tăng trong khi val acc còn tăng → xác nhận L8
  - Chốt sau pilot (AI theo uỷ quyền của Thư, Thư duyệt 2026-11-20): trần 6 epoch, patience 8, threshold 0
  - 3 seed: val acc 0,9288 / 0,9308 / 0,9312 (mean 0,9303 ± 0,0013), cả 3 `best_is_last_eval: no`; 2 seed dừng sớm, 1 seed chạy hết trần
  - File: `outputs/logs/<ts>_<run>.log`, `_history.csv`, `_config.json`; hình `outputs/figures/<run>_curves.png`
- **Test / smoke:** `pytest labs/lab3/tests/test_train.py` 7 passed (DistilBERT nhỏ từ config, CPU); smoke `python -m labs.lab3.main --smoke` trên máy và kernel CPU trên Kaggle: đủ 3 file log, checkpoint nạp lại được

## Success criteria
- **Test:** `pytest labs/lab3/tests/test_train.py` đạt: args đúng cấu hình, đủ 3 file log, `_history.csv` có đủ cột, checkpoint nạp lại được
- **Smoke run:** `outputs/_smoke/logs/` có đủ 3 file; log có header / thân / footer
- **Normal run:** pilot có đường cong đầy đủ tới trần epoch; lần chạy chính có `best_is_last_eval: no` và `stop_reason` ghi rõ

## Kết quả audit (2026-11-20)
- **Test:** `pytest labs/lab3/tests/test_train.py` → 7 passed ✔
- **Smoke run** (`python -m labs.lab3.main --smoke`, máy Thư, CPU, 2026-11-20): `outputs/_smoke/logs/` có đủ 3 file log cho `pilot_seed42`, `main_seed42`; log có header (phiên bản thư viện, commit code, config), thân, footer (`best_step`, `stop_reason`, `best_is_last_eval`) ✔
- **Normal run:** pilot 24 lần đánh giá (đủ 6 epoch × 4), `stop_reason: max_epochs`, `best_is_last_eval: no`; 3 seed đều `best_is_last_eval: no`, `stop_reason` ghi rõ (seed 42, 44: early stopping; seed 43: max_epochs); 7 file `_config.json` ghi GPU Tesla T4 ✔
- Số dòng code ghi ở mục Triển khai khớp code hiện tại ✔

## Links
- Desc lab: [../lab3_description_v1.md](../lab3_description_v1.md) | Plan: [../plans/detail_plan_lab3_md_v1.md](../plans/detail_plan_lab3_md_v1.md) | Khuôn log: [../templates/LOG_TEMPLATE_v1.md](../templates/LOG_TEMPLATE_v1.md)
- Module liên quan: [model_v1.md](model_v1.md), [features_v1.md](features_v1.md), [evaluate_v1.md](evaluate_v1.md), [experiment_v1.md](experiment_v1.md)
- Nguồn: Mosbach et al. 2021 https://arxiv.org/abs/2006.04884 ; Dodge et al. 2020 https://arxiv.org/abs/2002.06305 ; HF Callbacks https://huggingface.co/docs/transformers/main_classes/callback
- Code: `labs/lab3/train/train.py` | Test: `labs/lab3/tests/test_train.py`
