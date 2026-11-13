# experiment — So sánh 5 cấu hình (v1)

- **Created**: 2026-11-04T14:49:03+07:00
- **Last Updated**: 2026-11-14T01:41:23+07:00 (Thư)
- **Người phụ trách**: Member B (Thư)
- **Status**: Done

---

## Name
Experiment — chạy 5 cấu hình MLP, bảng so sánh trên val, chọn cấu hình

## Background
Đề P1: "thử network/hyperparameter khác nhau". Desc lab chốt 5 cấu hình × seed 42 (O10).
Bản cũ chạy 7 cấu hình × 5 seed (35 lần) kèm p-value, Holm, CI95 — đề không yêu cầu, nên cắt gọn.

## Goals / Purpose
- Chạy đủ 5 cấu hình của desc lab, seed 42, mỗi cấu hình qua `train` với quy tắc dừng chung
- Ghi một bảng so sánh 5 dòng: tên, số param, `best_epoch`, `stopped_epoch`, `val_loss`, `val_acc`, `train_acc`, `train_acc − val_acc`, thời gian train
- Chọn một cấu hình theo `val_acc` tại `best_epoch`, ghi lý do chọn
- Không làm:
  - Đánh giá trên test — việc của `evaluate`, chạy một lần sau khi chọn
  - Đa seed, p-value, Holm, CI95 — đã bỏ (O10)
  - Thêm cấu hình ngoài 5 cấu hình (vd `sgd_same_lr`, `weight_decay_1e-4` của bản cũ) — Thư chưa duyệt
  - Optuna hay tìm siêu tham số tự động — ngoài đề

## Input / Output
- **Input:** 5 config (dict) — nằm trong `VARIANTS` (`experiment/experiment.py:29`), tạo bằng `make_configs`; loaders từ [features_v1.md](features_v1.md)
- **Output:** 5 bộ checkpoint + log (do `train` sinh); bảng so sánh CSV
  - Code ghi `outputs/results/experiment_results.csv`, `epoch_history.csv`; checkpoint `outputs/models/<ts>_<name>_best.pt`, `_last.pt`; lựa chọn ghi trong `outputs/results/run_summary.json` (`selection`).

## How to do it (general plan)
1. Giảm `VARIANTS` còn 5 cấu hình, `SEEDS` còn `(42,)`; mỗi cấu hình khác baseline đúng một yếu tố
2. Mỗi cấu hình: `build_model(cfg)` → `train_model` (tối đa 30 epoch + early stopping) → log 3 file
3. Gom kết quả thành bảng so sánh 5 dòng, thêm cột `train_acc − val_acc`
4. Chọn cấu hình theo val, ghi vào phần Quyết định của desc lab
5. Smoke run: 5 cấu hình × 1 epoch × 2 batch → bảng có đủ 5 dòng
6. Normal run 5 cấu hình

## Pipeline
```
5 config → build_model → train_model (mỗi config một run, log riêng)
  → bảng so sánh (val) → chọn cấu hình → evaluate (test 1 lần) / visualize
```

## Detailed plan
- **5 cấu hình:** `baseline` (256,128), Adam lr 1e-3; `softmax_regression` không hidden; `larger_network` (512,256,128); `sgd_lr_0.01` SGD momentum 0.9 lr 0.01; `dropout_0.2` — batch 64 cho tất cả
- **Một yếu tố mỗi lần:** mỗi cấu hình chỉ đổi một thứ so với baseline — để chênh lệch quy được về đúng yếu tố đó
- **Chọn theo val, không theo test:** tránh rò rỉ thông tin từ test vào lựa chọn
- **Cột `train_acc − val_acc`:** chỉ để báo cáo mức overfit, không làm ngưỡng đạt
- **Hàm hiện có:**
  - `BASELINE_CONFIG`, `VARIANTS` (5 cấu hình) (`experiment/experiment.py:25`, `:29`); `make_configs(epochs=30, seed=42)` (`:38`)
  - `build_model(cfg, num_classes=10) -> nn.Module` (`:48`); `describe_architecture(cfg)` (`:65`)
  - `run_experiments(configs, loaders, device, ckpt_prefix="exp_run", log_dir=None, models_dir=None, patience=None, min_delta=0.0, max_batches=None, run_info=None, verbose=False)` (`:76`)
  - `save_results(results, csv_path)` (`:198`); `save_history(results, csv_path)` (`:239`); `select_config(results)` (`:274`)
  - Đã bỏ `analyze_seed_runs`, `compare_pairs` và `extended_study.py` (O10: bỏ đa seed và kiểm định)
- **Chỗ code chưa khớp plan:** không có (sửa 2026-11-14)
- **Gotchas:**
  - `best_val_acc` phải lấy ở epoch có val_loss thấp nhất, không lấy `max(val_acc)` (`experiment.md` cũ)
  - Một cấu hình thua baseline với 1 seed chưa chứng minh nó kém hơn — chỉ ghi nhận, không kết luận chắc (`experiment.md` cũ: "5 seeds and one split…")
  - Thời gian chạy dao động mạnh: `softmax_regression` seed 3407 từng mất 5.445 giây (`progress_status.md:52`)

## Success criteria
- **Test:** `python -m pytest labs/lab1/tests/test_experiment.py -q` đạt, có test bảng so sánh đủ cột mới
- **Smoke run:** 5 cấu hình chạy hết; bảng so sánh có đúng 5 dòng; mỗi cấu hình có 3 file log trong `outputs/_smoke/`
- **Normal run:** bảng so sánh 5 dòng đủ cột; baseline có `val_acc` cao hơn `softmax_regression` (desc lab); cấu hình được chọn có ghi lý do

## Kết quả audit (2026-11-14)
- **Test:** `pytest labs/lab1/tests/test_experiment.py` → 10 passed (gồm 5 cấu hình đúng desc lab, bảng đủ cột mới, `select_config`) ✔
- **Smoke run:** 5 cấu hình chạy hết, bảng `outputs/_smoke/results/experiment_results.csv` có 5 dòng, mỗi cấu hình 3 file log ✔
- **Normal run**, normal run 2026-11-14 01:30–01:37 (`outputs/results/run_summary.json`):

| Cấu hình | Param | best / dừng | val_loss | val_acc | train_acc | train − val | Thời gian |
|---|---:|:---:|---:|---:|---:|---:|---:|
| baseline | 235.146 | 8 / 13 | 0,2943 | 88,97% | 91,55% | 2,59 | 87 s |
| softmax_regression | 7.850 | 10 / 15 | 0,4243 | 85,15% | 86,22% | 1,07 | 56 s |
| larger_network | 567.434 | 7 / 12 | 0,3009 | **89,28%** | 90,98% | 1,70 | 107 s |
| sgd_lr_0.01 | 235.146 | 10 / 15 | 0,2852 | 89,13% | 92,02% | 2,89 | 68 s |
| dropout_0.2 | 235.146 | 6 / 11 | 0,2996 | 89,00% | 88,76% | −0,24 | 66 s |

  - Baseline val_acc 88,97% > softmax_regression 85,15% ✔
  - Chọn `larger_network`: val_acc cao nhất tại best_epoch, hơn `sgd_lr_0.01` 0,15 điểm — chênh nhỏ, một seed, chỉ ghi nhận (Gotcha 2) ✔
  - Cả 5 cấu hình dừng sớm (`early_stopping`), không cấu hình nào chạm 30 epoch

## Links
- Desc lab: [../lab1_description_v1.md](../lab1_description_v1.md)
- Module liên quan: [model_v1.md](model_v1.md), [train_v1.md](train_v1.md), [evaluate_v1.md](evaluate_v1.md), [visualize_v1.md](visualize_v1.md)
- Tiến độ và bằng chứng chạy: [../../progress_status.md](../../progress_status.md) (mục 2026-11-14 normal run; bản cũ 35 lần chạy ở phần lịch sử)
- Code: `labs/lab1/experiment/experiment.py`, `labs/lab1/main.py` | Test: `labs/lab1/tests/test_experiment.py`
