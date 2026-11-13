# evaluate — Đánh giá model (v1)

- **Created**: 2026-11-04T14:49:03+07:00
- **Last Updated**: 2026-11-14T00:55:48+07:00 (Thư)
- **Người phụ trách**: Member B (Thư)
- **Status**: Done (theo yêu cầu cũ) — chờ sửa theo yêu cầu mới

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
  - Hiện code ghi `artifacts/runs/classification_report.csv` (`main.py:137-138`), test in ra màn hình (`main.py:141-142`), chưa ghi file. Sẽ chuyển sang `outputs/results/` ở giai đoạn O5.

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
- **Confusion matrix:** hàng = lớp thật, cột = lớp dự đoán (`evaluate/evaluate.py:66`)
- **Hàm hiện có:**
  - `evaluate(model: nn.Module, loader, device: str | torch.device) -> Tuple[float, float]` (`evaluate/evaluate.py:19`)
  - `confusion_matrix(model: nn.Module, loader, device: str | torch.device, num_classes: int = 10) -> np.ndarray` (`:60-62`)
  - `per_class_accuracy(cm: np.ndarray) -> Dict[int, float]` (`:91`)
  - `save_classification_report(cm: np.ndarray, class_names: Sequence[str], csv_path: str) -> pd.DataFrame` (`:117`)
- **Chỗ code chưa khớp plan:** test chỉ in ra màn hình, chưa lưu file; `main.py` đánh giá baseline cố định, chưa nhận cấu hình đã chọn từ `experiment`
- **Test cho cả hai model (2026-11-14):** baseline để kiểm mốc ≥ 87% đã chốt trước; cấu hình được chọn trên val là kết quả cuối. Mỗi model chạy test đúng một lần; nếu cấu hình được chọn chính là baseline thì chỉ chạy một lần
- **Gotchas:**
  - Tập test đã được xem nhiều lần trong lúc phát triển, dù không lựa chọn nào dựa vào nó (`evaluate.md` cũ)
  - Val loader và test loader không shuffle để dự đoán khớp thứ tự ảnh (`features/features.py:107-108`)
  - Lớp không có mẫu thì `per_class_accuracy` trả 0, không báo lỗi (`evaluate.md` cũ)

## Success criteria
- **Test:** `python -m pytest labs/lab1/tests/test_evaluate.py -q` đạt (5 test)
- **Smoke run:** chạy hết trên checkpoint smoke; CSV báo cáo có đúng 10 dòng lớp; confusion matrix có tổng = số ảnh val
- **Normal run:** `val_acc` từ `evaluate` khớp `val_acc` ở `best_epoch` trong `_history.csv`; baseline test accuracy ≥ 87% trên 10.000 ảnh (mức của desc lab); có kết quả test của cả baseline và cấu hình được chọn, ghi ra file trong `outputs/results/`

## Links
- Desc lab: [../lab1_description_v1.md](../lab1_description_v1.md)
- Module liên quan: [train_v1.md](train_v1.md), [experiment_v1.md](experiment_v1.md), [visualize_v1.md](visualize_v1.md)
- Tiến độ và bằng chứng chạy: [../../progress_status.md](../../progress_status.md) (bản cũ 5 epoch, seed 42: val 89,05%, test 88,36%, dòng 50)
- Code: `labs/lab1/evaluate/evaluate.py` | Test: `labs/lab1/tests/test_evaluate.py`
