# Khuôn log và quy tắc dừng (v1)

- Tác giả: Thư | Tạo: 2026-11-13T11:22:06+07:00 | Cập nhật: Thư, 2026-11-14T01:18:05+07:00
- Dùng cho: mọi lần chạy train của Lab 1 (smoke run và normal run).

## 1. File sinh ra mỗi lần chạy

Lưu trong `labs/lab1/outputs/logs/` (smoke run: `outputs/_smoke/`), `<ts>` = `YYYYMMDD_HHMMSS` lúc bắt đầu:

| File | Nội dung |
|---|---|
| `<ts>_<run>.log` | log đọc được bằng mắt, khuôn ở §2 |
| `<ts>_<run>_history.csv` | mỗi epoch một dòng, cùng cột với phần epoch trong `.log`; biểu đồ loss/acc vẽ từ file này |
| `<ts>_<run>_config.json` | phần đầu log: config, seed, device, early stopping, kiến trúc, số param, đường dẫn split |

## 2. Khuôn `.log`

    === RUN START 2026-11-13T10:15:00+07:00 ===
    run: baseline | seed: 42 | device: cpu
    config: hidden_sizes=[256,128] lr=0.001 optimizer=adam dropout=0.0 weight_decay=0.0 batch_size=64 max_epochs=30
    early_stopping: monitor=val_loss patience=5 min_delta=1e-4
    architecture: MLP 784-256-128-10 (ReLU) | params: 235,146
    data: train=54000 val=6000 | split: labs/lab1/data/splits/train_val_split.json
    epoch,train_loss,train_acc,val_loss,val_acc,lr,time_s
    1,...
    === RUN END 2026-11-13T10:17:30+07:00 ===
    best_epoch: <n> | stopped_epoch: <n> | stop_reason: early_stopping | max_epochs
    best_ckpt: outputs/models/<ts>_baseline_best.pt | last_ckpt: outputs/models/<ts>_baseline_last.pt

Nguồn số trong ví dụ:
- `params: 235,146`: cột `n_params` của run baseline trong `labs/lab1/outputs/results/experiment_results.csv`.
- `train=54000 val=6000`: `labs/lab1/data/splits/train_val_split.json`.
- `batch_size=64`: mặc định của `make_loaders` trong `labs/lab1/features/features.py`.
- Giờ chạy chỉ là ví dụ.

## 3. Quy tắc dừng

1. Tối đa 30 epoch, giống nhau cho mọi cấu hình.
2. Sau mỗi epoch tính `val_loss`. Nếu giảm ít nhất `1e-4` so với best: lưu `_best.pt`, reset bộ đếm về 0. Nếu không: tăng bộ đếm thêm 1.
3. Dừng khi bộ đếm = 5 (`early_stopping`) hoặc khi hết 30 epoch (`max_epochs`).
4. Luôn lưu `_last.pt`. Đánh giá và demo dùng `_best.pt`.
5. Val tách từ train. Test chạy một lần, sau khi đã chọn cấu hình trên val.
6. Theo dõi `val_loss` thay vì "acc" như note của cô vì `val_loss` nhạy hơn; `val_acc` vẫn được log và báo cáo.
