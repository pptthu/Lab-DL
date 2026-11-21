# experiment — So sánh 5 cấu hình (v1)

- **Created**: 2026-11-04T14:49:03+07:00
- **Last Updated**: 2026-11-14T03:34:08+07:00 (Thư)
- **Người phụ trách**: Member B (Thư)
- **Status**: Done

---

## Name
Experiment — chạy 5 cấu hình MLP, bảng so sánh trên val, chọn cấu hình

## Background
Đề P1: thử network/hyperparameter khác nhau. Desc lab chốt 5 cấu hình × seed 42 (O10).
Bản cũ chạy 7 cấu hình × 5 seed kèm p-value, CI95 — đề không yêu cầu nên đã cắt gọn.

## Goals / Purpose
- Chạy đủ 5 cấu hình, seed 42, cùng quy tắc dừng của `train`
- Bảng so sánh 5 dòng: số tham số, `best_epoch`, `stopped_epoch`, `val_loss`, `val_acc`, `train_acc`, `train_acc − val_acc`, thời gian
- Chọn một cấu hình theo `val_acc` tại `best_epoch` và ghi lý do
- Không làm:
  - Đánh giá trên test — việc của `evaluate`, chạy một lần sau khi chọn
  - Đa seed, kiểm định thống kê, Optuna — ngoài phạm vi (O10)

## Input / Output
- **Input:** 5 config từ `make_configs` (`VARIANTS`, `experiment/experiment.py:29`); loaders từ [features_v1.md](features_v1.md)
- **Output:**
  - `outputs/results/experiment_results.csv` (5 dòng), `epoch_history.csv`
  - Checkpoint và log của từng run (do `train` ghi)
  - Lựa chọn trong `outputs/results/run_summary.json` (`selection`)

## How to do it (general plan)
1. `make_configs()` tạo 5 cấu hình, mỗi cấu hình khác baseline đúng một yếu tố
2. Mỗi cấu hình: `build_model(cfg)` → `train_model` → 3 file log
3. `save_results` gom bảng so sánh; `save_history` gom lịch sử theo epoch
4. `select_config` chọn theo val
5. Smoke run → normal run

## Pipeline
```
make_configs → build_model → train_model (mỗi cấu hình một run, log riêng)
  → bảng so sánh (val) → select_config → evaluate (test 1 lần), visualize
```

## Detailed plan
- **5 cấu hình** (batch 64, seed 42):

| Cấu hình | Đổi so với baseline |
|---|---|
| `baseline` | — (256-128, Adam lr 1e-3) |
| `softmax_regression` | bỏ tầng ẩn |
| `larger_network` | 512-256-128 |
| `sgd_lr_0.01` | SGD momentum 0,9, lr 0,01 |
| `dropout_0.2` | dropout 0,2 |

- **Một yếu tố mỗi lần:** để chênh lệch quy được về đúng yếu tố đó
- **Chọn theo val, không theo test:** tránh để thông tin của test lọt vào lựa chọn
- **Cột `train_acc − val_acc`:** để nhận xét overfit, không làm ngưỡng
- **Hàm:**
  - `make_configs(epochs=30, seed=42)` (`experiment/experiment.py:38`); `build_model(cfg, num_classes=10)` (`:48`); `describe_architecture(cfg)` (`:65`)
  - `run_experiments(configs, loaders, device, ckpt_prefix="exp_run", log_dir=None, models_dir=None, patience=None, min_delta=0.0, max_batches=None, run_info=None, verbose=False)` (`:76`)
  - `save_results(results, csv_path)` (`:198`); `save_history(results, csv_path)` (`:239`); `select_config(results)` (`:274`)
- **Chỗ code chưa khớp plan:** không có
- **Gotchas:**
  - `best_val_acc` lấy ở epoch có val_loss thấp nhất, không lấy `max(val_acc)`
  - Một seed: chênh lệch vài phần mười điểm chỉ ghi nhận, không đủ kết luận cấu hình nào tốt hơn

## Success criteria
- **Test:** `python -m pytest labs/lab1/tests/test_experiment.py -q` đạt
- **Smoke run:** 5 cấu hình chạy hết; bảng có 5 dòng; mỗi cấu hình 3 file log trong `outputs/_smoke/`
- **Normal run:** bảng 5 dòng đủ cột; baseline `val_acc` > `softmax_regression`; cấu hình được chọn có lý do

## Kết quả audit (2026-11-14)
- **Test:** `test_experiment.py` → 10 passed ✔
- **Smoke run:** bảng `outputs/_smoke/results/experiment_results.csv` có 5 dòng, mỗi cấu hình 3 file log ✔
- **Normal run:**

| Cấu hình | Tham số | best / dừng | val_loss | val_acc | train − val | Thời gian |
|---|---:|:---:|---:|---:|---:|---:|
| baseline | 235.146 | 8 / 13 | 0,2943 | 88,97% | 2,59 | 87 s |
| softmax_regression | 7.850 | 10 / 15 | 0,4243 | 85,15% | 1,07 | 56 s |
| larger_network | 567.434 | 7 / 12 | 0,3009 | **89,28%** | 1,70 | 107 s |
| sgd_lr_0.01 | 235.146 | 10 / 15 | 0,2852 | 89,13% | 2,89 | 68 s |
| dropout_0.2 | 235.146 | 6 / 11 | 0,2996 | 89,00% | −0,24 | 66 s |

  - Baseline 88,97% > softmax_regression 85,15% ✔
  - Chọn `larger_network` (val_acc cao nhất, hơn `sgd_lr_0.01` 0,15 điểm) ✔
  - Cả 5 cấu hình dừng sớm ở epoch 11–15

## Links
- Desc lab: [../lab1_description_v1.md](../lab1_description_v1.md)
- Module liên quan: [model_v1.md](model_v1.md), [train_v1.md](train_v1.md), [evaluate_v1.md](evaluate_v1.md), [visualize_v1.md](visualize_v1.md)
- Bằng chứng chạy: [../../progress_status.md](../../progress_status.md) (mục kiểm chứng 2026-11-14)
- Code: `labs/lab1/experiment/experiment.py`, `labs/lab1/main.py` | Test: `labs/lab1/tests/test_experiment.py`
