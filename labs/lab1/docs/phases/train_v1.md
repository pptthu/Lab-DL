# train — Huấn luyện MLP (v1)

- **Created**: 2026-11-04T14:49:03+07:00
- **Last Updated**: 2026-11-14T01:40:57+07:00 (Thư)
- **Người phụ trách**: Member B (Thư)
- **Status**: Done

---

## Name
Training — vòng lặp train, log theo lần chạy, quy tắc dừng

## Background
Đề P1: định nghĩa loss, optimizer, viết training loop, theo dõi validation, save/load model.
Module này là vòng lặp train dùng chung cho baseline (`main.py`) và 5 cấu hình (`experiment`).
Bản cũ chạy cố định 5 epoch, không có file log theo lần chạy; cần sửa theo `templates/LOG_TEMPLATE_v1.md`.

## Goals / Purpose
- `train_model` train một cấu hình tới khi dừng theo quy tắc ở Detailed plan và trả `best_epoch`, `stopped_epoch`, `stop_reason`
- Mỗi lần chạy sinh đủ 3 file log theo `templates/LOG_TEMPLATE_v1.md`; `_history.csv` có đủ 7 cột, mỗi epoch một dòng
- Sinh `_best.pt` (val_loss thấp nhất) và `_last.pt`, cả hai nạp lại được bằng `load_checkpoint`
- Không làm:
  - Dùng tập test trong lúc train hay khi chọn checkpoint — "loss dùng để đánh giá trong quá trình training" là **val loss**; test chỉ dùng ở `evaluate`
  - Chạy và so sánh 5 cấu hình — việc của `experiment`
  - Vẽ loss curve — việc của `visualize`; `train` chỉ ghi `_history.csv`
  - LR scheduler — log có cột `lr` nhưng lr cố định trong cả lần chạy; đề P1 không yêu cầu

## Input / Output
- **Input:** `train_loader`, `val_loader` từ [features_v1.md](features_v1.md) (batch `[64,1,28,28]` float32, nhãn int64, 54000 train / 6000 val); model từ [model_v1.md](model_v1.md)
- **Output:**
  - Checkpoint `<ts>_<run>_best.pt` (val_loss thấp nhất) và `<ts>_<run>_last.pt`
  - Log `<ts>_<run>.log`, `<ts>_<run>_history.csv`, `<ts>_<run>_config.json`
  - Code lưu checkpoint ở `outputs/models/`, log ở `outputs/logs/` (`main.py`, `_out_dirs`); smoke run ghi vào `outputs/_smoke/models|logs/`.

## How to do it (general plan)
1. Loss: `nn.CrossEntropyLoss()` trên logits (model không có softmax)
2. Optimizer: Adam lr 1e-3 cho baseline; SGD momentum 0.9 lr 0.01 cho cấu hình `sgd_lr_0.01`
3. Mỗi epoch: `model.train()` → zero_grad → forward → loss → backward → step; rồi `model.eval()` + `torch.no_grad()` đo val
4. Ghi một dòng vào `_history.csv`: `epoch, train_loss, train_acc, val_loss, val_acc, lr, time_s`
5. Kiểm quy tắc dừng; val_loss tốt hơn thì lưu `_best.pt`; hết vòng lặp lưu `_last.pt`
6. Smoke run (1 epoch, 2 batch, ghi vào `outputs/_smoke/`) → kiểm log đủ 3 phần, `_best.pt` nạp lại dự đoán được
7. Normal run baseline → `experiment` chạy 5 cấu hình

## Pipeline
```
features.make_loaders + model.MLP → train_model:
  ghi đầu log (config, seed, kiến trúc, số param)
  mỗi epoch: train step → val step → ghi history → kiểm dừng → lưu _best.pt
  ghi cuối log (best_epoch, stopped_epoch, stop_reason) → lưu _last.pt
→ _best.pt cho evaluate / demo; _history.csv cho visualize
```

## Detailed plan
- **Quy tắc dừng (training xong khi nào):** tối đa 30 epoch cho mọi cấu hình; theo dõi `val_loss`; giảm ít nhất `1e-4` so với best thì lưu `_best.pt` và reset bộ đếm, không thì bộ đếm +1; bộ đếm = 5 thì dừng (`early_stopping`), hết 30 epoch thì dừng (`max_epochs`)
- **Training loss để làm gì:** cho biết model có đang học không (train loss phải giảm). Đặt cạnh val loss để thấy overfit: train loss vẫn giảm mà val loss tăng nghĩa là model đang học thuộc tập train. Chênh `train_acc − val_acc` chỉ báo cáo, không làm ngưỡng
- **Vì sao theo dõi `val_loss` chứ không phải acc:** `val_loss` nhạy hơn; `val_acc` vẫn log và báo cáo
- **Hàm hiện có:** `train_model(model, train_loader, val_loader, epochs, lr, device, ckpt_path, optimizer_name="adam", weight_decay=0.0, patience=None, min_delta=0.0, last_ckpt_path=None, log_dir=None, run_name="run", run_info=None, max_batches=None, verbose=False) -> Dict[str, Any]` (`train/train.py:98`); `save_checkpoint` (`:22`); `load_checkpoint` (`:39`). Các tham số mới đều có mặc định nên lời gọi cũ vẫn chạy
- **Chỗ code chưa khớp plan:** không có (sửa 2026-11-14: history đủ 7 cột, `min_delta`, early stopping, `_last.pt`, 3 file log; `max_batches` cho smoke run)
- **Gotchas:**
  - Val acc báo cáo phải lấy ở epoch của `_best.pt`, không lấy `max(val_acc)` (`train.md` cũ)
  - Đổi key của `history` phải giữ `train_loss`, `val_loss`, `val_acc` vì `experiment` đang đọc (`experiment/experiment.py:161-163`)
  - `ckpt_path` bị ghi đè mỗi lần val_loss giảm; tên có `<ts>` để không mất checkpoint cũ (`train/train.py:265`; tên đặt ở `experiment/experiment.py`)
  - `load_checkpoint` cần model cùng kiến trúc; dùng `weights_only=True` (`train/train.py:51`)
  - Thời gian CPU dao động mạnh, có lần 5 epoch mất 1,5 giờ, chưa rõ nguyên nhân (`train.md` cũ; `progress_status.md:52`)

## Success criteria
- **Test:** `python -m pytest labs/lab1/tests/test_train.py -q` đạt, gồm 4 test hiện có và test mới cho: dừng đúng sau 5 epoch val_loss không giảm (dữ liệu giả); đủ 3 file log; có `_last.pt`
- **Smoke run:** trong `outputs/_smoke/` có đủ 3 file; `.log` có dòng `=== RUN START`, ít nhất 1 dòng epoch, dòng `=== RUN END`; `_best.pt` nạp lại, dự đoán 1 batch ra logits `[B,10]`
- **Normal run (baseline):** `stopped_epoch` ≤ 30; `stop_reason` là `early_stopping` hoặc `max_epochs`; `best_epoch` ≤ `stopped_epoch`; số dòng `_history.csv` = `stopped_epoch`
- Không đặt ngưỡng accuracy ở `train`: mức test ≥ 87% nằm ở `lab1_description_v1.md` và được kiểm ở `evaluate`

## Kết quả audit (2026-11-14)
- **Test:** `pytest labs/lab1/tests/test_train.py` → 7 passed (4 test cũ + dừng sớm đúng ở epoch 6 khi val_loss đứng yên, đủ 3 file log, `max_batches`) ✔
- **Smoke run** (`python -m labs.lab1.main --smoke`, 21 giây): `outputs/_smoke/logs/` có 15 file (5 run × 3); `.log` có `=== RUN START`, dòng epoch, `=== RUN END`; `_best.pt` nạp lại và dự đoán được ở bước evaluate, demo ✔
- **Normal run (baseline)**, normal run 2026-11-14 01:30–01:37 (`outputs/results/run_summary.json`): `stopped_epoch` 13 ≤ 30, `stop_reason` early_stopping, `best_epoch` 8 ≤ 13, `_history.csv` có 13 dòng epoch ✔ — log `outputs/logs/20261114_013050_baseline.log`

## Links
- Desc lab: [../lab1_description_v1.md](../lab1_description_v1.md) | Khuôn log: [../templates/LOG_TEMPLATE_v1.md](../templates/LOG_TEMPLATE_v1.md)
- Module liên quan: [features_v1.md](features_v1.md), [model_v1.md](model_v1.md), [evaluate_v1.md](evaluate_v1.md), [experiment_v1.md](experiment_v1.md)
- Tiến độ và bằng chứng chạy: [../../progress_status.md](../../progress_status.md) (mục 2026-11-14 normal run; bản cũ 5 epoch ở phần lịch sử)
- Code: `labs/lab1/train/train.py` | Test: `labs/lab1/tests/test_train.py`
