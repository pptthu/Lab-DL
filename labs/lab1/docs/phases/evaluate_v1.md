# evaluate — Đánh giá model (v1)

- **Created**: 2026-11-04T14:49:03+07:00
- **Last Updated**: 2026-11-14T03:34:08+07:00 (Thư)
- **Người phụ trách**: Member B (Thư)
- **Status**: Done

---

## Name
Evaluation — loss/accuracy, confusion matrix, báo cáo từng lớp, test một lần

## Background
Đề P1: đánh giá accuracy của model đã train. Câu hỏi của cô: "đánh giá như nào là đúng, output là gì".
Module này trả lời: đo gì, trên tập nào, mấy lần, ghi ra file nào.

## Goals / Purpose
- `evaluate` trả `(loss, accuracy)` của `_best.pt` trên val và trên test 10.000 ảnh
- Confusion matrix `[10,10]` và bảng precision, recall, specificity, F1 của 10 lớp (CSV)
- Test chạy **một lần cho mỗi model**: baseline và cấu hình được chọn trên val; baseline phải đạt ≥ 87%
- Không làm:
  - Chọn cấu hình hay checkpoint theo test — chọn trên val ở `experiment`, `train`
  - Vẽ confusion matrix — việc của `visualize`
  - p-value, CI95, đa seed — ngoài phạm vi (O10)

## Input / Output
- **Input:** `_best.pt` từ [train_v1.md](train_v1.md); `val_loader`, `test_loader` từ [features_v1.md](features_v1.md) (không xáo)
- **Output:**
  - `outputs/results/test_results.csv`: val và test của baseline và cấu hình được chọn, cột `threshold`, `meets_threshold`
  - `outputs/results/classification_report_val_<tên>.csv`: chỉ số 10 lớp trên val

## How to do it (general plan)
1. Nạp `_best.pt` vào model mới cùng kiến trúc, `model.eval()`
2. `evaluate` trên val → phải khớp `val_acc` ở `best_epoch` trong `_history.csv`
3. `confusion_matrix` + `save_classification_report` trên val để phân tích lỗi theo lớp
4. `evaluate` trên test, mỗi model một lần → `save_test_results`
5. Smoke run → normal run sau khi `experiment` chọn xong

## Pipeline
```
_best.pt → load_checkpoint → evaluate(val) → confusion_matrix(val) → save_classification_report
  → evaluate(test, 1 lần) → save_test_results → visualize, báo cáo
```

## Detailed plan
- **Metric chính:** accuracy trên test 10.000 ảnh — đề yêu cầu accuracy; dữ liệu cân bằng (1.000 ảnh mỗi lớp) nên accuracy phản ánh đúng
- **Metric phụ:** loss; precision, recall, specificity, F1 từng lớp — thấy model yếu ở lớp nào
- **Mức đạt 87%:** MLP tốt nhất trong bài báo gốc Fashion-MNIST (Xiao et al. 2017: 0,871), chốt trước khi chạy
- **Vì sao test cả hai model:** baseline để kiểm mức 87% đã chốt; cấu hình được chọn là kết quả cuối. Nếu cấu hình được chọn là baseline thì chỉ chạy test một lần
- **Confusion matrix:** hàng = lớp thật, cột = lớp dự đoán
- **Hàm:**
  - `evaluate(model, loader, device) -> Tuple[float, float]` (`evaluate/evaluate.py:22`)
  - `confusion_matrix(model, loader, device, num_classes=10) -> np.ndarray` (`:63`); `per_class_accuracy(cm)` (`:94`)
  - `save_classification_report(cm, class_names, csv_path) -> pd.DataFrame` (`:120`)
  - `TEST_ACC_THRESHOLD = 0.87` (`:19`); `save_test_results(rows, csv_path, threshold=0.87) -> pd.DataFrame` (`:161`)
- **Chỗ code chưa khớp plan:** không có
- **Gotchas:**
  - Val loader và test loader không xáo để dự đoán khớp thứ tự ảnh (`features/features.py:107-108`)
  - Lớp không có mẫu thì `per_class_accuracy` trả 0, không báo lỗi

## Success criteria
- **Test:** `python -m pytest labs/lab1/tests/test_evaluate.py -q` đạt
- **Smoke run:** chạy hết trên checkpoint smoke; báo cáo lớp có 10 dòng; tổng confusion matrix = số ảnh val
- **Normal run:** `val_acc` tính lại từ `_best.pt` khớp log; baseline test ≥ 87%; có kết quả test của baseline và cấu hình được chọn trong `outputs/results/`

## Kết quả audit (2026-11-14)
- **Test:** `test_evaluate.py` → 6 passed ✔
- **Smoke run:** báo cáo lớp có 10 dòng ✔
- **Normal run** (`outputs/results/test_results.csv`):
  - val_acc từ checkpoint khớp log: baseline 88,97%, larger_network 89,28% ✔
  - **Baseline test 88,60%** (loss 0,3239) ≥ 87% ✔; larger_network test 88,46% (loss 0,3315)
  - Theo lớp (val, larger_network): Shirt recall 68,67% thấp nhất; Trouser 98,67%, Bag 97,50% cao nhất

## Links
- Desc lab: [../lab1_description_v1.md](../lab1_description_v1.md)
- Module liên quan: [train_v1.md](train_v1.md), [experiment_v1.md](experiment_v1.md), [visualize_v1.md](visualize_v1.md)
- Bằng chứng chạy: [../../progress_status.md](../../progress_status.md) (mục kiểm chứng 2026-11-14)
- Code: `labs/lab1/evaluate/evaluate.py` | Test: `labs/lab1/tests/test_evaluate.py`
