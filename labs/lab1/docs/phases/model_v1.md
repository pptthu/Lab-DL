# model — Kiến trúc MLP (v1)

- **Created**: 2026-11-04T14:49:03+07:00
- **Last Updated**: 2026-11-14T00:55:48+07:00 (Thư)
- **Người phụ trách**: Member A (Toàn)
- **Status**: Done (theo yêu cầu cũ) — chờ sửa theo yêu cầu mới

---

## Name
Model — kiến trúc MLP, model builder, sơ đồ kiến trúc, model card

## Background
Đề P1: "build NN" cho Fashion-MNIST bằng PyTorch. Nhóm chốt chỉ dùng MLP (2026-11-04).
Cô yêu cầu model phải vẽ kiến trúc và có model card ghi rõ xây từ đâu, số lớp, số param, I/O, checkpoint, người train.
Code model đã có; còn thiếu sơ đồ kiến trúc và model card.

## Goals / Purpose
- `MLP(hidden_sizes, num_classes, dropout)` tạo được đủ 5 kiến trúc của desc lab; `forward` nhận `[B,1,28,28]`, trả logits `[B,10]`
- `count_parameters` trả đúng số param: baseline 235,146; larger_network 567,434; softmax_regression 7,850
- Model card `docs/models/mlp_baseline_v1.md` theo `templates/MODEL_CARD_TEMPLATE_v1.md`, có sơ đồ kiến trúc vẽ bằng mermaid và bảng số param từng lớp
- Không làm:
  - CNN hay kiến trúc khác — nhóm chốt MLP only (2026-11-04)
  - Pretrained / backbone có sẵn — model tự viết, khởi tạo ngẫu nhiên theo seed
  - Softmax trong `forward` — `CrossEntropyLoss` ở `train` cần logits
  - Chọn kiến trúc tốt nhất — việc của `experiment`

## Input / Output
- **Input:** batch ảnh `[B,1,28,28]` float32 đã chuẩn hoá từ [features_v1.md](features_v1.md); config (`hidden_sizes`, `dropout`) từ [experiment_v1.md](experiment_v1.md)
- **Output:** đối tượng `nn.Module` chưa train; logits `[B,10]`; model card có sơ đồ kiến trúc (mermaid)
  - Hiện chưa có model card; sẽ lưu ở `docs/models/`.

## How to do it (general plan)
1. Kiểm `MLP` với 3 bộ `hidden_sizes`: `(256,128)`, `(512,256,128)`, `()`; in số param từng bộ
2. Vẽ sơ đồ kiến trúc baseline bằng mermaid ngay trong model card (784 → 256 → 128 → 10, ReLU), kèm bảng số param từng lớp — không cần code vẽ, `model.py` giữ nguyên
3. Điền phần Initial model của model card baseline
4. Smoke run cùng `train` (1 epoch, 2 batch) → `_best.pt` nạp lại vào `MLP()` mới, dự đoán ra `[B,10]`
5. Sau normal run: điền phần Trained model (người train, ngày xong, log, checkpoint, val_acc, test_acc)

## Pipeline
```
config (experiment) → build_model(cfg) → MLP(hidden_sizes, dropout)
  → Flatten [B,784] → [Linear → ReLU (→ Dropout)] × số hidden → Linear → logits [B,10]
→ train / evaluate / visualize
```

## Detailed plan
- **Kiến trúc baseline:** 784 → 256 → 128 → 10, ReLU — 3 lớp Linear (2 hidden + 1 output)
- **5 kiến trúc trong desc lab:** baseline (256,128); softmax_regression `()` — 1 lớp Linear, làm mốc tuyến tính; larger_network (512,256,128) — 4 lớp Linear; `sgd_lr_0.01` và `dropout_0.2` dùng kiến trúc baseline (dropout 0.2 sau mỗi ReLU)
- **Model builder (xây từ đâu):** tự viết bằng `nn.Sequential`, không backbone, không pretrained. `experiment.build_model(cfg)` đọc `hidden_sizes`, `dropout` từ config rồi gọi `MLP` (`experiment/experiment.py:26-40`)
- **Hàm hiện có:** `MLP(hidden_sizes=(256, 128), num_classes: int = 10, dropout: float = 0.0)` (`model/model.py:28`); `forward(x)` (`:45`); `count_parameters(model) -> int` (`:49`)
- **Số param:** cột `n_params` trong `artifacts/runs/experiment_results.csv` (235146 / 567434 / 7850)
- **Sơ đồ kiến trúc (2026-11-14):** mermaid trong model card — GitHub và VS Code tự hiển thị, không cần code; notebook ô 9 (Stage 4) có cùng sơ đồ
- **Chỗ code chưa khớp plan:** không có; chỉ còn thiếu model card (tài liệu)
- **Gotchas:**
  - MLP bỏ qua cấu trúc 2 chiều của ảnh — giới hạn của model, không phải lỗi (`model.md` cũ)
  - Đổi kích thước mặc định của `MLP` thì checkpoint cũ không nạp được nữa (`model.md` cũ, mục Prohibited)
  - `count_parameters` chỉ đếm param có `requires_grad=True` (`model/model.py:59`)

## Success criteria
- **Test:** `python -m pytest labs/lab1/tests/test_model.py -q` đạt (6 test: shape, logits, số param, dropout, overfit 1 batch)
- **Smoke run:** `count_parameters(MLP())` = 235146; `_best.pt` của smoke run nạp lại được vào `MLP()` mới
- **Normal run:** model card baseline có sơ đồ mermaid hiển thị được trên GitHub, điền đủ cả Initial và Trained model, số liệu có đường dẫn log

## Links
- Desc lab: [../lab1_description_v1.md](../lab1_description_v1.md) | Khuôn model card: [../templates/MODEL_CARD_TEMPLATE_v1.md](../templates/MODEL_CARD_TEMPLATE_v1.md)
- Module liên quan: [train_v1.md](train_v1.md), [experiment_v1.md](experiment_v1.md), [visualize_v1.md](visualize_v1.md)
- Tiến độ và bằng chứng chạy: [../../progress_status.md](../../progress_status.md) (MLP forward trên batch thật: logits `[64,10]`, dòng 89 — lần chạy cũ trên CSV Kaggle)
- Code: `labs/lab1/model/model.py` | Test: `labs/lab1/tests/test_model.py`
