# experiment — So sánh 5 cấu hình (v1)

- **Created**: 2026-11-04T14:49:03+07:00
- **Last Updated**: 2026-11-14T00:03:08+07:00 (Thư)
- **Người phụ trách**: Member B (Thư)
- **Status**: Done (theo yêu cầu cũ) — chờ sửa theo yêu cầu mới

---

## Name
Experiment — chạy 5 cấu hình MLP, bảng so sánh trên val, chọn cấu hình

## Background
Đề P1: "thử network/hyperparameter khác nhau". Desc lab chốt 5 cấu hình × seed 42 (O10).
Bản cũ chạy 7 cấu hình × 5 seed (35 lần) kèm p-value, Holm, CI95 — đề không yêu cầu, nên cắt gọn.

## Goals / Purpose  [đề xuất]
- Chạy đủ 5 cấu hình của desc lab, seed 42, mỗi cấu hình qua `train` với quy tắc dừng chung
- Ghi một bảng so sánh 5 dòng: tên, số param, `best_epoch`, `stopped_epoch`, `val_loss`, `val_acc`, `train_acc`, `train_acc − val_acc`, thời gian train
- Chọn một cấu hình theo `val_acc` tại `best_epoch`, ghi lý do chọn
- Không làm:
  - Đánh giá trên test — việc của `evaluate`, chạy một lần sau khi chọn
  - Đa seed, p-value, Holm, CI95 — đã bỏ (O10)
  - Thêm cấu hình ngoài 5 cấu hình (vd `sgd_same_lr`, `weight_decay_1e-4` của bản cũ) — leader chưa duyệt
  - Optuna hay tìm siêu tham số tự động — ngoài đề

## Input / Output
- **Input:** 5 config (dict) — hiện nằm trong `VARIANTS` (`main.py:37-45`); loaders từ [features_v1.md](features_v1.md)
- **Output:** 5 bộ checkpoint + log (do `train` sinh); bảng so sánh CSV
  - Hiện code ghi `artifacts/runs/experiment_results.csv`, `epoch_history.csv` (`main.py:167-169`), checkpoint `artifacts/models/exp_run_<i>.pt` (`experiment/experiment.py:77`). Sẽ chuyển sang `outputs/results/`, `outputs/models/` ở giai đoạn O5.

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
  - `build_model(cfg: Dict[str, Any], num_classes: int = 10) -> nn.Module` (`experiment/experiment.py:26`)
  - `run_experiments(configs, loaders, device: str, ckpt_prefix: str = "exp_run") -> List[Dict[str, Any]]` (`:43-48`)
  - `save_results(results, csv_path: str) -> None` (`:118`); `save_history(results, csv_path: str) -> None` (`:151`)
  - `analyze_seed_runs(...)` (`:181`), `compare_pairs(...)` (`:273`) — phục vụ đa seed và kiểm định, sẽ không dùng nữa
- **Chỗ code chưa khớp plan:** `VARIANTS` có 7 cấu hình (`main.py:37-45`), `SEEDS` có 5 seed (`main.py:33`); bảng kết quả chưa có `train_acc` và `stopped_epoch`; `analyze_seed_runs` còn đánh giá test cho từng seed (`main.py:175`)
- **Gotchas:**
  - `best_val_acc` phải lấy ở epoch có val_loss thấp nhất, không lấy `max(val_acc)` (`experiment.md` cũ)
  - Một cấu hình thua baseline với 1 seed chưa chứng minh nó kém hơn — chỉ ghi nhận, không kết luận chắc (`experiment.md` cũ: "5 seeds and one split…")
  - Thời gian chạy dao động mạnh: `softmax_regression` seed 3407 từng mất 5.445 giây (`progress_status.md:52`)

## Success criteria  [đề xuất]
- **Test:** `python -m pytest labs/lab1/tests/test_experiment.py -q` đạt, có test bảng so sánh đủ cột mới
- **Smoke run:** 5 cấu hình chạy hết; bảng so sánh có đúng 5 dòng; mỗi cấu hình có 3 file log trong `outputs/_smoke/`
- **Normal run:** bảng so sánh 5 dòng đủ cột; baseline có `val_acc` cao hơn `softmax_regression` (desc lab); cấu hình được chọn có ghi lý do

## Links
- Desc lab: [../lab1_description_v1.md](../lab1_description_v1.md)
- Module liên quan: [model_v1.md](model_v1.md), [train_v1.md](train_v1.md), [evaluate_v1.md](evaluate_v1.md), [visualize_v1.md](visualize_v1.md)
- Tiến độ và bằng chứng chạy: [../../progress_status.md](../../progress_status.md) (bản cũ 35 lần chạy, 2026-11-04, dòng 50–59)
- Code: `labs/lab1/experiment/experiment.py`, `labs/lab1/main.py` | Test: `labs/lab1/tests/test_experiment.py`
