# demo — Nạp model đã lưu và dự đoán ảnh test (v1)

- **Created**: 2026-11-04T14:49:03+07:00
- **Last Updated**: 2026-11-14T03:34:08+07:00 (Thư)
- **Người phụ trách**: Member B (Thư)
- **Status**: Done

---

## Name
Demo — nạp `_best.pt` không train lại, dự đoán ảnh test, lưu vào `outputs/demo/`; chạy toàn bộ pipeline bằng `main.py`

## Background
Đề P1: save/load model và hiển thị ảnh dự đoán so với nhãn thật. Demo chứng minh model đã lưu dùng lại được mà không chạy lại pipeline (O8: code đặt trong `visualize/`).

## Goals / Purpose
- Nạp `_best.pt` của cấu hình được chọn, dự đoán 16 ảnh test, không gọi `train_model`
- Lưu `outputs/demo/<ts>_predictions.png` (ảnh, nhãn dự đoán, nhãn thật, xác suất) và `<ts>_predictions.csv` (`index, true, pred, prob, correct`)
- `python -m labs.lab1.main` chạy toàn bộ pipeline; `--smoke` chạy thử
- Không làm:
  - Ảnh ngoài dataset, giao diện web, đo tốc độ inference — ngoài phạm vi desc lab
  - Tính accuracy toàn tập test — việc của `evaluate`

## Input / Output
- **Input:** `_best.pt` từ [train_v1.md](train_v1.md); config của cấu hình từ `make_configs`; `test_loader` từ [features_v1.md](features_v1.md)
- **Output:** PNG + CSV trong `outputs/demo/`; `outputs/results/run_summary.json` cho notebook

## How to do it (general plan)
1. `build_model(cfg)` dựng lại kiến trúc của cấu hình được chọn
2. `load_checkpoint(model, "<ts>_<run>_best.pt", device)` → `model.eval()`
3. Lấy 16 ảnh đầu của `test_loader`, softmax trên logits, ghi PNG và CSV
4. Smoke run → normal run

## Pipeline
```
make_configs → build_model → load_checkpoint(_best.pt) → test_loader (16 ảnh)
  → logits → softmax → predictions.png + predictions.csv
```

## Detailed plan
- **Không train lại:** demo chỉ đọc file đã lưu — chứng minh phần save/load của đề
- **Softmax chỉ ở demo:** model trả logits; demo đổi sang xác suất cho người xem
- **Dựng lại kiến trúc:** dùng config của `experiment` (`build_model(cfg)`); `_config.json` của run lưu cùng `hidden_sizes`, `dropout` để đối chiếu
- **Thứ tự `main.py`:** EDA → split → chuẩn hoá + loader → train 5 cấu hình → chọn trên val → test baseline và cấu hình chọn → hình → demo → `run_summary.json` (`main.py:77`); `--smoke`: 1 epoch, 2 batch, ghi vào `outputs/_smoke/`
- **Hàm:** `demo_predictions(model, loader, class_names, device, out_png, out_csv, n=16) -> pd.DataFrame` (`visualize/visualize.py:247`); dùng `load_checkpoint` (`train/train.py:39`), `build_model` (`experiment/experiment.py:48`)
- **Chỗ code chưa khớp plan:** không có
- **Gotchas:**
  - Kiến trúc dựng lại phải khớp lúc lưu, nếu không `load_state_dict` báo lỗi (`train/train.py:52`)
  - Windows: đặt `PYTHONIOENCODING=utf-8` khi chạy script in tiếng Việt

## Success criteria
- **Test:** `test_visualize.py` có test demo với model giả ghi PNG và CSV 16 dòng
- **Smoke run:** demo trên `_best.pt` của smoke có PNG và CSV trong `outputs/_smoke/demo/`
- **Normal run:** `outputs/demo/` có PNG và CSV 16 dòng; nạp lại checkpoint vào model mới cho dự đoán trùng CSV

## Kết quả audit (2026-11-14)
- **Test:** `test_demo_predictions_writes_png_and_csv` → đạt ✔
- **Smoke run:** PNG và CSV trong `outputs/_smoke/demo/` ✔
- **Normal run:** `outputs/demo/20261114_013036_predictions.png`, `.csv` 16 dòng, đúng 15/16 (sai: Sneaker đoán thành Sandal); notebook Stage 10b nạp lại `_best.pt`, dự đoán trùng CSV ✔

## Links
- Desc lab: [../lab1_description_v1.md](../lab1_description_v1.md)
- Module liên quan: [train_v1.md](train_v1.md), [experiment_v1.md](experiment_v1.md), [visualize_v1.md](visualize_v1.md)
- Bằng chứng chạy: [../../progress_status.md](../../progress_status.md) (mục kiểm chứng 2026-11-14)
- Code: `labs/lab1/visualize/visualize.py`, `labs/lab1/main.py` | Test: `labs/lab1/tests/test_visualize.py`
