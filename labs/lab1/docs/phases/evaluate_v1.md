# evaluate — Đánh giá model (v1)

- **Created**: 2026-11-04T14:49:03+07:00
- **Last Updated**: 2026-11-14T01:41:23+07:00 (Thư)
- **Người phụ trách**: Member B (Thư)
- **Status**: Done

---

## Name
Evaluation — loss/accuracy, confusion matrix, báo cáo từng lớp, test một lần

## Background
Đề P1: "đánh giá accuracy" của model đã train. Cô hỏi: "đánh giá như nào là đúng, output là gì".
Module này trả lời câu đó: đo gì, trên tập nào, mấy lần, ghi ra file nào.

## Goals / Purpose
- `evaluate` trả `(loss, accuracy)` của `_best.pt` trên val và trên test 10.000 ảnh
- Ghi bảng precision, recall, specificity, F1 của 10 lớp ra CSV và confusion matrix `[10,10]`
- Test chạy **một lần cho mỗi model**: baseline và cấu hình được chọn trên val; baseline phải đạt mức ≥ 87% của desc lab
- Không làm:
  - Chọn cấu hình hay checkpoint theo kết quả test — chọn trên val ở `experiment` và `train`
  - Vẽ confusion matrix — việc của `visualize`; `evaluate` chỉ trả mảng
  - p-value, CI95, đa seed — đã bỏ (O10)
  - Test trên ảnh ngoài dataset, đo tốc độ inference — ngoài phạm vi desc lab

## Input / Output
- **Input:** `_best.pt` từ [train_v1.md](train_v1.md) nạp bằng `load_checkpoint`; `val_loader`, `test_loader` từ [features_v1.md](features_v1.md) (không shuffle)
- **Output:** `(loss, acc)` val và test; mảng confusion matrix; CSV báo cáo từng lớp
  - Code ghi `outputs/results/test_results.csv` (val + test của baseline và cấu hình chọn) và `classification_report_val_<tên>.csv`.

## How to do it (general plan)
1. Nạp `_best.pt` của baseline và của cấu hình đã chọn vào model mới cùng kiến trúc, `model.eval()`
2. `evaluate` trên val → `(val_loss, val_acc)`; phải khớp số ở `best_epoch` trong `_history.csv`
3. `confusion_matrix` + `save_classification_report` trên val để phân tích lỗi theo lớp
4. `evaluate` trên test, mỗi model một lần → `(test_loss, test_acc)`; ghi vào `outputs/results/` cùng tên run
5. Smoke run trên checkpoint của smoke → kiểm các hàm chạy hết, file CSV có 10 dòng
6. Normal run sau khi `experiment` chọn xong cấu hình

## Pipeline
```
_best.pt + model mới → load_checkpoint → evaluate(val) → confusion_matrix(val)
  → save_classification_report(csv) → evaluate(test, 1 lần) → results → visualize / báo cáo
```

## Detailed plan
- **Metric chính:** accuracy trên test chính thức 10.000 ảnh — đề P1 yêu cầu accuracy; dữ liệu cân bằng 1.000 ảnh mỗi lớp nên accuracy dùng được
- **Metric phụ:** loss; precision, recall, specificity, F1 từng lớp — để thấy model yếu ở lớp nào
- **Confusion matrix:** hàng = lớp thật, cột = lớp dự đoán (`evaluate/evaluate.py:69`)
- **Hàm hiện có:**
  - `evaluate(model, loader, device) -> Tuple[float, float]` (`evaluate/evaluate.py:22`)
  - `confusion_matrix(model, loader, device, num_classes=10) -> np.ndarray` (`:63`); `per_class_accuracy(cm)` (`:94`)
  - `save_classification_report(cm, class_names, csv_path) -> pd.DataFrame` (`:120`)
  - `TEST_ACC_THRESHOLD = 0.87` (`:19`); `save_test_results(rows, csv_path, threshold=0.87) -> pd.DataFrame` (`:161`)
- **Chỗ code chưa khớp plan:** không có (sửa 2026-11-14: test ghi ra `outputs/results/test_results.csv`; `main.py` nhận cấu hình đã chọn từ `select_config`)
- **Test cho cả hai model (2026-11-14):** baseline để kiểm mốc ≥ 87% đã chốt trước; cấu hình được chọn trên val là kết quả cuối. Mỗi model chạy test đúng một lần; nếu cấu hình được chọn chính là baseline thì chỉ chạy một lần
- **Gotchas:**
  - Tập test đã được xem nhiều lần trong lúc phát triển, dù không lựa chọn nào dựa vào nó (`evaluate.md` cũ)
  - Val loader và test loader không shuffle để dự đoán khớp thứ tự ảnh (`features/features.py:107-108`)
  - Lớp không có mẫu thì `per_class_accuracy` trả 0, không báo lỗi (`evaluate.md` cũ)

## Success criteria
- **Test:** `python -m pytest labs/lab1/tests/test_evaluate.py -q` đạt (5 test)
- **Smoke run:** chạy hết trên checkpoint smoke; CSV báo cáo có đúng 10 dòng lớp; confusion matrix có tổng = số ảnh val
- **Normal run:** `val_acc` từ `evaluate` khớp `val_acc` ở `best_epoch` trong `_history.csv`; baseline test accuracy ≥ 87% trên 10.000 ảnh (mức của desc lab); có kết quả test của cả baseline và cấu hình được chọn, ghi ra file trong `outputs/results/`

## Kết quả audit (2026-11-14)
- **Test:** `pytest labs/lab1/tests/test_evaluate.py` → 6 passed (5 test cũ + `save_test_results`) ✔
- **Smoke run:** chạy hết trên checkpoint smoke; báo cáo lớp có 10 dòng ✔
- **Normal run**, normal run 2026-11-14 01:30–01:37 (`outputs/results/run_summary.json`) (`outputs/results/test_results.csv`):
  - val_acc tính lại từ `_best.pt` khớp log: baseline 88,97% = 88,97%, larger_network 89,28% = 89,28% ✔
  - **Baseline test 88,60%** (loss 0,3239) ≥ 87% ✔; cấu hình chọn `larger_network` test 88,46% (loss 0,3315) — mỗi model test đúng một lần ✔
  - Theo lớp (val, larger_network): Shirt recall 68,67% thấp nhất; Trouser 98,67%, Bag 97,50%

## Links
- Desc lab: [../lab1_description_v1.md](../lab1_description_v1.md)
- Module liên quan: [train_v1.md](train_v1.md), [experiment_v1.md](experiment_v1.md), [visualize_v1.md](visualize_v1.md)
- Tiến độ và bằng chứng chạy: [../../progress_status.md](../../progress_status.md) (mục 2026-11-14 normal run; bản cũ ở phần lịch sử)
- Code: `labs/lab1/evaluate/evaluate.py` | Test: `labs/lab1/tests/test_evaluate.py`
