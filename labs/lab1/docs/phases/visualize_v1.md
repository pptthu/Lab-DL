# visualize — Biểu đồ từ log và ảnh dự đoán (v1)

- **Created**: 2026-11-04T14:49:03+07:00
- **Last Updated**: 2026-11-14T01:40:57+07:00 (Thư)
- **Người phụ trách**: Member B (Thư)
- **Status**: Done

---

## Name
Visualize — loss/accuracy curve từ `_history.csv`, ảnh dự đoán, confusion matrix

## Background
Đề P1: "vẽ loss, hiển thị ảnh dự đoán vs thực tế". Cô yêu cầu biểu đồ phải dựng **từ training logs**.
Bản cũ vẽ loss từ dict `history` trong bộ nhớ, ảnh dự đoán lấy trên val.

## Goals / Purpose
- Từ `_history.csv` của mỗi run vẽ được loss curve (train + val) và accuracy curve (train + val)
- Ảnh dự đoán 16 ảnh **test** bằng `_best.pt`: nhãn dự đoán và nhãn thật, sai tô đỏ
- Confusion matrix dạng heatmap từ mảng của `evaluate`
- Không làm:
  - Tính metric — việc của `evaluate`; `visualize` chỉ vẽ từ số đã có
  - Train lại hay chạy lại model để lấy số vẽ — mọi số lấy từ log/file
  - Commit hình lên git
  - Vẽ sơ đồ kiến trúc — nằm trong model card (mermaid), xem [model_v1.md](model_v1.md)

## Input / Output
- **Input:** `_history.csv` từ [train_v1.md](train_v1.md); `_best.pt` + `test_loader`; confusion matrix từ [evaluate_v1.md](evaluate_v1.md)
- **Output:** `<ts>_<run>_loss.png`, `<ts>_<run>_acc.png`, `predictions_grid.png`, `confusion_matrix.png`
  - Code ghi `outputs/figures/<ts>_<run>_loss.png`, `_acc.png` (5 run), `<ts>_predictions_test_<tên>.png`, `<ts>_confusion_matrix_val_<tên>.png`, cùng `class_counts.png`, `sample_grid.png` của EDA.

## How to do it (general plan)
1. Đổi `plot_loss_curves` đọc từ `_history.csv` thay vì dict trong bộ nhớ
2. Thêm accuracy curve (train_acc, val_acc) từ cùng file
3. `show_predictions` dùng `test_loader` thay cho `val_loader`
4. `plot_confusion_matrix` giữ nguyên
5. Smoke run: vẽ từ `_history.csv` của smoke, hình ghi vào `outputs/_smoke/`
6. Normal run sau khi `experiment` và `evaluate` xong

## Pipeline
```
_history.csv → plot_loss_curves / plot_acc_curves → PNG
_best.pt + test_loader → show_predictions → predictions_grid.png
confusion matrix (evaluate) → plot_confusion_matrix → confusion_matrix.png
```

## Detailed plan
- **Vẽ từ log:** biểu đồ phải dựng lại được chỉ từ file log, không cần chạy lại — đúng yêu cầu của cô
- **Ảnh dự đoán trên test:** desc lab ghi "ảnh dự đoán trên test"; test loader không shuffle nên lần nào cũng 16 ảnh đầu giống nhau
- **Hàm hiện có:**
  - `load_history(csv_path) -> dict` (`visualize/visualize.py:21`)
  - `plot_loss_curves(history: dict | str, out_path, title=...)` (`:38`) — nhận dict hoặc đường dẫn `_history.csv`
  - `plot_accuracy_curves(history: dict | str, out_path, title=...)` (`:74`)
  - `show_predictions(model, loader, class_names, device, out_path, n=16)` (`:103`) — `main.py` truyền `test_loader`
  - `plot_confusion_matrix(cm, class_names, out_path)` (`:191`); `demo_predictions(...)` (`:247`, xem [demo_v1.md](demo_v1.md))
- **Chỗ code chưa khớp plan:** không có (sửa 2026-11-14)
- **Gotchas:**
  - `class_names` phải đủ 10 tên, nếu không `ValueError` (`visualize/visualize.py:73`)
  - Hình của lần chạy 2026-11-04 vẫn cần người xem lại (`visualize.md` cũ)

## Success criteria
- **Test:** `python -m pytest labs/lab1/tests/test_visualize.py -q` đạt, có test vẽ từ file CSV giả
- **Smoke run:** từ `_history.csv` của smoke ra đủ loss và accuracy PNG trong `outputs/_smoke/`
- **Normal run:** đủ loss curve, accuracy curve, ảnh dự đoán 16 ảnh test, confusion matrix; người trong nhóm đã xem từng hình

## Kết quả audit (2026-11-14)
- **Test:** `pytest labs/lab1/tests/test_visualize.py` → 5 passed (gồm vẽ từ `_history.csv` giả và demo) ✔
- **Smoke run:** `outputs/_smoke/figures/` có 14 hình (2 EDA + 10 loss/acc + ảnh dự đoán + confusion matrix) ✔
- **Normal run**, normal run 2026-11-14 01:30–01:37 (`outputs/results/run_summary.json`): `outputs/figures/` có đủ 14 hình; notebook Stage 7 vẽ lại loss/acc của 5 run từ `_history.csv` ✔. Còn chờ: Thư xem lại từng hình trước buổi trình bày

## Links
- Desc lab: [../lab1_description_v1.md](../lab1_description_v1.md) | Khuôn log: [../templates/LOG_TEMPLATE_v1.md](../templates/LOG_TEMPLATE_v1.md)
- Module liên quan: [train_v1.md](train_v1.md), [evaluate_v1.md](evaluate_v1.md), [model_v1.md](model_v1.md), [demo_v1.md](demo_v1.md)
- Tiến độ và bằng chứng chạy: [../../progress_status.md](../../progress_status.md) (mục 2026-11-14 normal run)
- Code: `labs/lab1/visualize/visualize.py` | Test: `labs/lab1/tests/test_visualize.py`
