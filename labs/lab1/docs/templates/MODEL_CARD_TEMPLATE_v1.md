# Khuôn model card (v1)

- Tác giả: Thư | Tạo: 2026-11-08T00:00:00+07:00 | Cập nhật: Thư, 2026-11-08T00:00:00+07:00
- Dùng cho: mỗi model đã train của Lab 1. Bản đã điền lưu ở `docs/models/<tên>_v<N>.md`.

## Cách dùng
- Phần **Initial model** điền khi viết xong code model, trước khi train.
- Phần **Trained model** điền sau normal run, lấy số từ log và file kết quả, ghi đường dẫn.
- Ví dụ trong `<...>` lấy theo baseline hiện tại (`labs/lab1/model/model.py`: Linear → ReLU, Dropout sau ReLU nếu dropout > 0, tầng cuối trả logits).

## Khuôn

    # Model card — <tên>  (v<N>)
    - Tác giả, ngày tạo | Người cập nhật, ngày cập nhật
    ## Purpose
    <1–2 câu: model này dùng để làm gì, so với model nào>
    ## Initial model
    - Architecture: <vd 784→256→128→10, ReLU, dropout p> | Sơ đồ: outputs/figures/<model>_architecture.png
    - Xây từ đâu / backbone: <tự viết, không dùng pretrained>
    - Số param: <…> | Input: [B,1,28,28] (chuẩn hóa theo mean/std train) | Output: logits [B,10]
    - Source: labs/lab1/model/model.py
    ## Trained model
    - Người train | Ngày train xong | Log/config: outputs/logs/<ts>_<run>.*
    - Checkpoint: outputs/models/<ts>_<run>_best.pt (best_epoch, stopped_epoch)
    - Kết quả: val_acc, test_acc
    - Ghi chú
