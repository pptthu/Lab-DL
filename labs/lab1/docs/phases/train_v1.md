# train — Huấn luyện MLP (v1)

- **Created**: 2026-11-04T14:49:03+07:00
- **Last Updated**: 2026-11-14T03:34:08+07:00 (Thư)
- **Người phụ trách**: Member B (Thư)
- **Status**: Done

---

## Name
Training — vòng lặp train, log theo lần chạy, quy tắc dừng

## Background
Đề P1: định nghĩa loss, optimizer, viết training loop, theo dõi validation, save/load model.
Vòng lặp train dùng chung cho 5 cấu hình của `experiment`. Nhận xét của cô: bản cũ chạy cố định 5 epoch, không có log theo lần chạy, không rõ khi nào train xong.

## Goals / Purpose
- `train_model` train tới khi dừng theo quy tắc dừng và trả `best_epoch`, `stopped_epoch`, `stop_reason`
- Mỗi lần chạy ghi 3 file log theo [`LOG_TEMPLATE_v1.md`](../templates/LOG_TEMPLATE_v1.md); `_history.csv` mỗi epoch một dòng, 7 cột
- Lưu `_best.pt` (val_loss thấp nhất) và `_last.pt`, cả hai nạp lại được bằng `load_checkpoint`
- Không làm:
  - Dùng tập test khi train hay khi chọn checkpoint — loss theo dõi trong lúc train là **val loss**; test chỉ dùng ở `evaluate`
  - So sánh cấu hình (`experiment`), vẽ curve (`visualize` đọc `_history.csv`)
  - LR scheduler — đề không yêu cầu; cột `lr` trong log giữ cố định

## Input / Output
- **Input:** `train_loader`, `val_loader` từ [features_v1.md](features_v1.md) (batch `[64,1,28,28]`, 54.000 train / 6.000 val); model từ [model_v1.md](model_v1.md)
- **Output:**
  - `outputs/models/<ts>_<run>_best.pt`, `<ts>_<run>_last.pt`
  - `outputs/logs/<ts>_<run>.log`, `_history.csv`, `_config.json`
  - Smoke run ghi vào `outputs/_smoke/models|logs/`

## How to do it (general plan)
1. Loss `nn.CrossEntropyLoss()` trên logits; optimizer theo config (Adam lr 1e-3, hoặc SGD momentum 0,9 lr 0,01)
2. Mỗi epoch: `model.train()` → zero_grad → forward → loss → backward → step; rồi `model.eval()` + `torch.no_grad()` đo val
3. Ghi một dòng `epoch, train_loss, train_acc, val_loss, val_acc, lr, time_s` vào `_history.csv`
4. Kiểm quy tắc dừng; val_loss tốt hơn thì lưu `_best.pt`; kết thúc thì lưu `_last.pt` và ghi phần cuối log
5. Smoke run (1 epoch, 2 batch) → normal run

## Pipeline
```
features.make_loaders + model.MLP → train_model:
  ghi đầu log (config, seed, kiến trúc, số tham số)
  mỗi epoch: train → val → ghi history → kiểm dừng → lưu _best.pt
  ghi cuối log (best_epoch, stopped_epoch, stop_reason) → lưu _last.pt
→ _best.pt cho evaluate, demo; _history.csv cho visualize
```

## Detailed plan
- **Quy tắc dừng (train xong khi nào):** tối đa 30 epoch; theo dõi `val_loss`; giảm ít nhất `1e-4` so với best thì lưu `_best.pt` và đặt lại bộ đếm, không thì bộ đếm +1; bộ đếm = 5 thì dừng (`early_stopping`); hết 30 epoch thì dừng (`max_epochs`)
- **Training loss để làm gì:** cho biết model có đang học không (train loss phải giảm); đặt cạnh val loss để thấy overfit — train loss vẫn giảm mà val loss tăng nghĩa là model đang học thuộc tập train
- **Vì sao theo dõi `val_loss` chứ không phải `val_acc`:** loss thay đổi liên tục theo độ tự tin của model nên nhạy hơn; `val_acc` vẫn ghi và báo cáo
- **Hàm:** `train_model(model, train_loader, val_loader, epochs, lr, device, ckpt_path, optimizer_name="adam", weight_decay=0.0, patience=None, min_delta=0.0, last_ckpt_path=None, log_dir=None, run_name="run", run_info=None, max_batches=None, verbose=False) -> Dict[str, Any]` (`train/train.py:98`); `save_checkpoint` (`:22`); `load_checkpoint` (`:39`). Tham số mới đều có mặc định nên lời gọi cũ vẫn chạy
- **Chỗ code chưa khớp plan:** không có
- **Gotchas:**
  - Val acc báo cáo lấy ở epoch của `_best.pt`, không lấy `max(val_acc)`
  - `experiment` đọc các key `train_loss`, `val_loss`, `val_acc` của history (`experiment/experiment.py:161-163`)
  - Tên file có `<ts>` để lần chạy sau không ghi đè checkpoint cũ
  - `load_checkpoint` cần model cùng kiến trúc; dùng `weights_only=True` (`train/train.py:51`)

## Success criteria
- **Test:** `python -m pytest labs/lab1/tests/test_train.py -q` đạt, có test dừng sớm, đủ 3 file log, có `_last.pt`
- **Smoke run:** `outputs/_smoke/logs/` có đủ 3 file mỗi run; `.log` có `=== RUN START`, dòng epoch, `=== RUN END`; `_best.pt` nạp lại dự đoán ra logits `[B,10]`
- **Normal run (baseline):** `stopped_epoch` ≤ 30; `stop_reason` là `early_stopping` hoặc `max_epochs`; `best_epoch` ≤ `stopped_epoch`; số dòng `_history.csv` = `stopped_epoch`

## Kết quả audit (2026-11-14)
- **Test:** `test_train.py` → 7 passed ✔
- **Smoke run** (21 giây): 5 run × 3 file log trong `outputs/_smoke/logs/`; `_best.pt` nạp lại dự đoán được ✔
- **Normal run (baseline):** dừng ở epoch 13 (`early_stopping`), best_epoch 8, `_history.csv` 13 dòng ✔ — `outputs/logs/20261114_013050_baseline.log`

## Links
- Desc lab: [../lab1_description_v1.md](../lab1_description_v1.md) | Khuôn log: [../templates/LOG_TEMPLATE_v1.md](../templates/LOG_TEMPLATE_v1.md)
- Module liên quan: [features_v1.md](features_v1.md), [model_v1.md](model_v1.md), [evaluate_v1.md](evaluate_v1.md), [experiment_v1.md](experiment_v1.md)
- Bằng chứng chạy: [../../progress_status.md](../../progress_status.md) (mục kiểm chứng 2026-11-14)
- Code: `labs/lab1/train/train.py` | Test: `labs/lab1/tests/test_train.py`
