# demo — Nạp model đã lưu và dự đoán ảnh test (v1)

- **Created**: 2026-11-04T14:49:03+07:00
- **Last Updated**: 2026-11-14T01:41:23+07:00 (Thư)
- **Người phụ trách**: Member B (Thư)
- **Status**: Done

---

## Name
Demo — nạp `_best.pt` không cần train lại, dự đoán ảnh test, lưu kết quả vào `outputs/demo/`

## Background
Đề P1: "save/load model" và "hiển thị ảnh dự đoán vs thực tế". Demo chứng minh model đã lưu dùng lại được
mà không chạy lại pipeline (O8: demo = nạp best checkpoint → dự đoán ảnh test → `outputs/demo/`, code đặt trong `visualize/`).
Bản cũ (`demo.md`) chỉ mô tả cách chạy `main.py`, notebook, README; chưa có bước demo riêng.

## Goals / Purpose
- Một lệnh nạp `_best.pt` của cấu hình đã chọn, dự đoán N ảnh test, không gọi `train_model`
- Lưu `outputs/demo/<ts>_predictions.png` (ảnh, nhãn dự đoán, nhãn thật, xác suất) và `<ts>_predictions.csv` (chỉ số ảnh, nhãn thật, nhãn dự đoán, xác suất)
- Không làm:
  - Ảnh ngoài dataset, upload ảnh, giao diện web — ngoài phạm vi desc lab
  - Đo tốc độ inference — ngoài phạm vi desc lab
  - Tính accuracy toàn tập test — việc của `evaluate`

## Input / Output
- **Input:** `_best.pt` từ [train_v1.md](train_v1.md); config của run (đọc từ `_config.json` để dựng lại đúng kiến trúc); `test_loader` từ [features_v1.md](features_v1.md)
- **Output:** PNG + CSV trong `outputs/demo/`
  - Code ghi `outputs/demo/<ts>_predictions.png` và `<ts>_predictions.csv` (bước 7 của `main.py`).

## How to do it (general plan)
1. Đọc `_config.json` của run đã chọn → `build_model(cfg)` dựng lại kiến trúc
2. `load_checkpoint(model, "<ts>_<run>_best.pt", device)` → `model.eval()`
3. Lấy N = 16 ảnh đầu của `test_loader` (không shuffle nên lần nào cũng giống nhau)
4. Dự đoán, tính xác suất bằng softmax trên logits, ghi PNG và CSV
5. Smoke run: dùng `_best.pt` của smoke run, ghi vào `outputs/_smoke/`
6. Normal run sau khi `experiment` chọn cấu hình

## Pipeline
```
_config.json → build_model → load_checkpoint(_best.pt) → test_loader (16 ảnh)
  → logits → softmax → predictions.png + predictions.csv (outputs/demo/)
```

## Detailed plan
- **Không train lại:** demo chỉ đọc file đã lưu — để chứng minh phần save/load của đề
- **Softmax chỉ ở demo:** model trả logits; demo đổi sang xác suất để người xem dễ hiểu
- **Hàm dùng lại:** `load_checkpoint` (`train/train.py:39`), `build_model` (`experiment/experiment.py:48`), `show_predictions` (`visualize/visualize.py:103`)
- **Chạy toàn bộ pipeline (`main.py`):** thứ tự EDA → split → chuẩn hoá + loader → train 5 cấu hình → chọn trên val → test baseline và cấu hình chọn → hình → demo → `run_summary.json` (`main.py:77`, hàm `main`); `--smoke` ghi mọi thứ vào `outputs/_smoke/`
- **Hàm demo:** `demo_predictions(model, loader, class_names, device, out_png, out_csv, n=16) -> pd.DataFrame` (`visualize/visualize.py:247`)
- **Chỗ code chưa khớp plan:** dựng lại kiến trúc bằng config của `experiment` (`build_model(cfg)`), không đọc `_config.json`; `_config.json` có cùng `hidden_sizes`, `dropout` nên kết quả như nhau. Giữ cách này vì `main.py` và notebook đã có config trong bộ nhớ
- **Gotchas:**
  - Kiến trúc dựng lại phải khớp lúc lưu, nếu không `load_state_dict` báo lỗi (`train/train.py:52`)
  - Windows + đường dẫn tiếng Việt: đặt `PYTHONIOENCODING=utf-8` khi chạy script (`demo.md` cũ)
  - Thời gian chạy `main.py` dao động mạnh giữa các lần (19 phút và 2 giờ 3 phút cho cùng kết quả) (`demo.md` cũ)

## Success criteria
- **Test:** test mới trong `labs/lab1/tests/test_visualize.py`: demo với model giả ghi đủ PNG và CSV 16 dòng vào `tmp_path`
- **Smoke run:** demo trên `_best.pt` của smoke chạy hết, có PNG và CSV trong `outputs/_smoke/`
- **Normal run:** `outputs/demo/` có PNG và CSV 16 dòng; nhãn trong CSV khớp nhãn thật của 16 ảnh test đầu

## Kết quả audit (2026-11-14)
- **Test:** `test_demo_predictions_writes_png_and_csv` trong `labs/lab1/tests/test_visualize.py` → đạt (16 dòng, PNG + CSV, prob trong (0,1]) ✔
- **Smoke run:** demo trên `_best.pt` của smoke chạy hết, có PNG và CSV trong `outputs/_smoke/demo/` ✔
- **Normal run**, normal run 2026-11-14 01:30–01:37 (`outputs/results/run_summary.json`): `outputs/demo/20261114_013036_predictions.png` và `.csv` 16 dòng, 15/16 đúng (sai 1: Sneaker đoán thành Sandal 96%); notebook Stage 10 nạp lại `_best.pt` vào model mới, dự đoán trùng CSV ✔

## Links
- Desc lab: [../lab1_description_v1.md](../lab1_description_v1.md)
- Module liên quan: [train_v1.md](train_v1.md), [experiment_v1.md](experiment_v1.md), [visualize_v1.md](visualize_v1.md)
- Tiến độ và bằng chứng chạy: [../../progress_status.md](../../progress_status.md) (mục 2026-11-14 normal run)
- Code: `labs/lab1/visualize/visualize.py` (sẽ thêm hàm demo) | Test: `labs/lab1/tests/test_visualize.py`
