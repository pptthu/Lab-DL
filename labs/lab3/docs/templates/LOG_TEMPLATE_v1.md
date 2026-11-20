# Khuôn log và quy tắc dừng (v1)

- Tác giả: Thư | Tạo: 2026-11-19T17:00:38+07:00 | Cập nhật: Thư, 2026-11-19T17:00:38+07:00
- Dùng cho: mọi lần chạy fine-tune của Lab 3 (smoke run, pilot, lần chạy chính, thí nghiệm `max_length`).
- Nguồn: theo `labs/lab1/docs/templates/LOG_TEMPLATE_v1.md`. Đổi so với Lab 1: log theo **mỗi lần đánh giá** (mỗi 1/4 epoch) thay vì mỗi epoch; thêm GPU và phiên bản thư viện; chọn checkpoint theo **val accuracy**; quy tắc dừng chốt sau pilot (plan v1, L8, L10, L11).

## 1. File sinh ra mỗi lần chạy

Lưu trong `labs/lab3/outputs/logs/` (smoke run: `outputs/_smoke/logs/`), `<ts>` = `YYYYMMDD_HHMMSS` lúc bắt đầu, `<run>` = tên lần chạy (vd `main_seed42`, `pilot`, `maxlen128`):

| File | Nội dung |
|---|---|
| `<ts>_<run>.log` | log đọc được bằng mắt, khuôn ở §2 |
| `<ts>_<run>_history.csv` | mỗi lần đánh giá một dòng, cùng cột với phần thân `.log`; đường loss/acc vẽ từ file này |
| `<ts>_<run>_config.json` | phần đầu log: config, seed, GPU, phiên bản thư viện, quy tắc dừng, model, số param, split |

## 2. Khuôn `.log`

    === RUN START <YYYY-MM-DDTHH:MM:SS+07:00> ===
    run: <run> | seed: <seed> | device: <cuda/cpu> | gpu: <tên GPU, vd Tesla T4> | where: <kaggle/local>
    versions: python=<…> torch=<…> transformers=<…> datasets=<…> evaluate=<…>
    model: distilbert-base-uncased + classification head (2 lớp) | params: <tổng> (trainable <…>)
    config: max_length=<…> lr=2e-5 warmup_ratio=0.1 scheduler=linear batch_size=16 fp16=<true/false> weight_decay=<…> max_epochs=<…>
    eval: every <n> steps (~1/4 epoch) | select: val accuracy (cao nhất) | monitor thêm: val_loss
    early_stopping: patience=<n lần đánh giá> threshold=<…>  (pilot: none)
    data: train=<n> val=<n> test=<n> | split: labs/lab3/data/splits/train_val_split.json
    step,epoch,train_loss,val_loss,val_acc,val_f1,lr,time_s
    <…>
    === RUN END <YYYY-MM-DDTHH:MM:SS+07:00> ===
    best_step: <n> (epoch <x.xx>) | best_val_acc: <…> | stopped_step: <n> | stop_reason: early_stopping | max_epochs
    best_is_last_eval: <yes/no>   # yes → trần epoch quá thấp, xem §3.5
    best_ckpt: outputs/models/<ts>_<run>_best/ | train_time_s: <…>

Mọi `<…>` lấy từ lần chạy thật; khuôn không chứa số mẫu.

## 3. Quy tắc chọn checkpoint và dừng

1. Val tách từ train (plan L3). Test 25k chỉ chạy một lần cho mỗi model, sau khi đã chọn xong trên val.
2. Đánh giá val mỗi khoảng 1/4 epoch (L11); mỗi lần ghi một dòng vào `_history.csv`.
3. Checkpoint tốt nhất = lần đánh giá có **val accuracy cao nhất** (L8). `val_loss` chỉ ghi để theo dõi overfit / quá tự tin.
4. **Pilot:** không dừng sớm, chạy hết trần epoch để có đường cong đầy đủ. **Lần chạy chính và thí nghiệm:** dừng khi val accuracy không cải thiện quá `threshold` trong `patience` lần đánh giá liên tiếp (`stop_reason: early_stopping`), hoặc khi hết trần (`max_epochs`). Trần epoch, `patience`, `threshold` `[cần chốt sau pilot]`.
5. Nếu `best_is_last_eval: yes` → trần epoch quá thấp: tăng trần và chạy lại, không báo cáo lần chạy đó.
6. Đánh giá test và demo dùng checkpoint tốt nhất.
