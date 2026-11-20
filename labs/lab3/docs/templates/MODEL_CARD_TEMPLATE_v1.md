# Khuôn model card (v1)

- Tác giả: Thư | Tạo: 2026-11-19T17:00:38+07:00 | Cập nhật: Thư, 2026-11-19T17:00:38+07:00
- Dùng cho: model fine-tune của Lab 3 (và pipeline zero-shot của Bài 1 làm baseline). Bản đã điền lưu ở `docs/models/<tên>_v<N>.md`.
- Nguồn: theo `labs/lab1/docs/templates/MODEL_CARD_TEMPLATE_v1.md`. Đổi so với Lab 1: Initial model là backbone pre-trained + đầu phân loại 2 lớp (Lab 1: MLP tự viết); Trained model báo trung bình ± std của 3 seed và so với zero-shot.

## Cách dùng
- Phần **Initial model** điền khi viết xong code `model/`, trước khi train. Số param lấy từ code (đếm bằng hàm), không chép từ nguồn ngoài.
- Phần **Trained model** điền sau lần chạy chính trên Kaggle, lấy số từ log và file kết quả, ghi đường dẫn.

## Khuôn

    # Model card — <tên>  (v<N>)
    - Tác giả, ngày tạo | Người cập nhật, ngày cập nhật
    ## Purpose
    <1–2 câu: phân loại cảm xúc review IMDB (positive / negative); so với pipeline zero-shot của Bài 1>
    ## Initial model
    - Backbone pre-trained: `distilbert-base-uncased` (Hugging Face Hub; Sanh et al. 2019) — không train từ đầu
    - Đầu phân loại: <các lớp thêm vào, vd lớp ẩn + Linear → 2 logits>, khởi tạo ngẫu nhiên | `id2label`: {0: NEGATIVE, 1: POSITIVE}
    - Sơ đồ: khối mermaid trong file này (tokenizer → backbone → đầu phân loại → logits)
    - Số param: tổng <…>, backbone <…>, đầu phân loại <…> (đếm từ code)
    - Input: `input_ids`, `attention_mask` [B, max_length=<…>] (tokenizer cùng checkpoint) | Output: logits [B, 2]
    - Source: labs/lab3/model/model.py
    ## Trained model
    - Người train | Ngày train xong | Nơi chạy: Kaggle <GPU> | Log/config: outputs/logs/<ts>_<run>.*
    - Cấu hình: max_length, lr, batch, warmup, trần epoch, patience (trỏ tới plan và log)
    - Checkpoint: outputs/models/<ts>_<run>_best/ (best_step, stopped_step) — không commit
    - Kết quả 3 seed: val acc, test acc, test F1 — từng seed và trung bình ± std (outputs/results/<…>.csv)
    - So sánh: pipeline zero-shot (Bài 1) trên cùng 25k test: acc, F1
    - Giới hạn đã biết: <vd review dài hơn max_length bị cắt — số liệu từ EDA>
    - Ghi chú
