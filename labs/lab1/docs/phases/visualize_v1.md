# visualize — Biểu đồ từ log và ảnh dự đoán (v1)

- **Created**: 2026-11-04T14:49:03+07:00
- **Last Updated**: 2026-11-14T03:34:08+07:00 (Thư)
- **Người phụ trách**: Member B (Thư)
- **Status**: Done

---

## Name
Visualize — loss/accuracy curve từ `_history.csv`, ảnh dự đoán, confusion matrix

## Background
Đề P1: vẽ loss, hiển thị ảnh dự đoán so với nhãn thật. Yêu cầu của cô: biểu đồ phải dựng **từ training log**.
Bản cũ vẽ loss từ dict trong bộ nhớ nên không dựng lại được sau khi chạy.

## Goals / Purpose
- Từ `_history.csv` của mỗi run vẽ loss curve và accuracy curve (train + val)
- Ảnh dự đoán 16 ảnh **test** bằng `_best.pt`: nhãn dự đoán và nhãn thật, ảnh sai tô đỏ
- Heatmap confusion matrix từ mảng của `evaluate`
- Không làm:
  - Tính metric — việc của `evaluate`
  - Train lại để lấy số vẽ — mọi số đọc từ file log
  - Sơ đồ kiến trúc — nằm trong model card, xem [model_v1.md](model_v1.md)

## Input / Output
- **Input:** `_history.csv` từ [train_v1.md](train_v1.md); `_best.pt` + `test_loader`; confusion matrix từ [evaluate_v1.md](evaluate_v1.md)
- **Output** (`outputs/figures/`):
  - `<ts>_<run>_loss.png`, `<ts>_<run>_acc.png` cho 5 run
  - `<ts>_predictions_test_<tên>.png`, `<ts>_confusion_matrix_val_<tên>.png`

## How to do it (general plan)
1. `plot_loss_curves`, `plot_accuracy_curves` đọc `_history.csv`
2. `show_predictions` trên `test_loader`
3. `plot_confusion_matrix` từ mảng của `evaluate`
4. Smoke run → normal run sau khi `experiment` và `evaluate` xong

## Pipeline
```
_history.csv → plot_loss_curves / plot_accuracy_curves → PNG
_best.pt + test_loader → show_predictions → PNG
confusion matrix (evaluate) → plot_confusion_matrix → PNG
```

## Detailed plan
- **Vẽ từ log:** hình dựng lại được chỉ từ file log, không chạy lại model
- **Ảnh dự đoán trên test:** test loader không xáo nên lần nào cũng là 16 ảnh đầu
- **Hàm:**
  - `load_history(csv_path) -> dict` (`visualize/visualize.py:21`)
  - `plot_loss_curves(history: dict | str, out_path, title=...)` (`:38`); `plot_accuracy_curves(history: dict | str, out_path, title=...)` (`:74`)
  - `show_predictions(model, loader, class_names, device, out_path, n=16)` (`:103`)
  - `plot_confusion_matrix(cm, class_names, out_path)` (`:191`); `demo_predictions(...)` (`:247`, xem [demo_v1.md](demo_v1.md))
- **Chỗ code chưa khớp plan:** không có
- **Gotchas:** `class_names` phải đủ 10 tên, nếu không báo `ValueError`

## Success criteria
- **Test:** `python -m pytest labs/lab1/tests/test_visualize.py -q` đạt, có test vẽ từ `_history.csv` giả
- **Smoke run:** đủ loss và accuracy PNG trong `outputs/_smoke/figures/`
- **Normal run:** đủ loss curve, accuracy curve, ảnh dự đoán 16 ảnh test, confusion matrix; Thư đã xem từng hình

## Kết quả audit (2026-11-14)
- **Test:** `test_visualize.py` → 5 passed ✔
- **Smoke run:** `outputs/_smoke/figures/` có 14 hình ✔
- **Normal run:** `outputs/figures/` có 14 hình (2 EDA, 10 loss/accuracy, ảnh dự đoán, confusion matrix); Thư đã xem từng hình ✔

## Links
- Desc lab: [../lab1_description_v1.md](../lab1_description_v1.md) | Khuôn log: [../templates/LOG_TEMPLATE_v1.md](../templates/LOG_TEMPLATE_v1.md)
- Module liên quan: [train_v1.md](train_v1.md), [evaluate_v1.md](evaluate_v1.md), [demo_v1.md](demo_v1.md)
- Bằng chứng chạy: [../../progress_status.md](../../progress_status.md) (mục kiểm chứng 2026-11-14)
- Code: `labs/lab1/visualize/visualize.py` | Test: `labs/lab1/tests/test_visualize.py`
