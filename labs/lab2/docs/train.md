# train — Huấn luyện mô hình pre-trained (v2)

- **Created**: 2026-11-04 · **Last Updated**: 2026-11-20 (Bảo) — viết lại theo code hiện tại
- **Người phụ trách**: Member B (Bảo)
- **Status**: Done
- **Code**: [`labs/lab2/train/train.py`](../train/train.py)

---

## Name
Training — vòng lặp train/validation, log TensorBoard tuỳ chọn, giữ checkpoint có validation loss nhỏ nhất

## Background
Lab 2 dùng lại CNN pre-trained (ResNet-18, DenseNet-121, VGG-16 của Member A) cho CIFAR-10 10 lớp.
`train` nhận model đã dựng sẵn (`build_model`, có thể đã `unfreeze_last_blocks`), nhận 3 DataLoader từ
`features`, huấn luyện và để lại checkpoint cho `evaluate`; `experiment` gọi lại chính `train_model`
cho từng cấu hình nên module này là nơi duy nhất chứa vòng lặp train.

## Goals / Purpose
- `train_model` chạy tối đa `epochs` epoch: mỗi epoch train toàn bộ `train_loader` rồi validate trên `val_loader`.
- Giữ checkpoint tốt nhất theo **validation loss nhỏ nhất**; trả `history` để `main.py`/`experiment` báo cáo.
- Ghi 4 scalar TensorBoard mỗi epoch khi được truyền `writer`.
- Không làm:
  - Không có LR scheduler (không StepLR/Cosine) — `lr` giữ nguyên suốt quá trình; đề không yêu cầu.
  - Không lưu optimizer state vào checkpoint — `save_checkpoint` chỉ lưu `model.state_dict()`.
  - Không đụng tập test, không chọn model theo test (việc của `evaluate` dùng một lần cho cấu hình thắng).

## Input / Output
- **Input:** `model` (`nn.Module`, forward trả logits `[B,10]`); `train_loader`, `val_loader` (batch `[B,3,H,W]`, nhãn 0–9);
  `epochs`, `lr`, `device`; `ckpt_path`; `writer` (`SummaryWriter` hoặc `None`); `optimizer_name` (`'adam'`/`'sgd'`);
  `weight_decay`; `patience` (mặc định `None` = tắt early stopping).
- **Output:** `history` dict
  - `train_loss`, `train_acc`, `val_loss`, `val_acc`: list theo từng epoch (accuracy trong `[0,1]`)
  - `best_epoch`, `best_val_loss`, `best_val_acc`: đúng epoch đã lưu checkpoint; `stopped_epoch`: epoch dừng thực tế
  - File `ckpt_path` (state_dict) và log TensorBoard `Loss/train`, `Accuracy/train`, `Loss/val`, `Accuracy/val`

## How to do it (general plan)
1. Kiểm tra `epochs > 0`, `lr > 0`; chuyển `model.to(device)`.
2. Loss `nn.CrossEntropyLoss()` trên logits; optimizer chỉ nhận tham số `requires_grad=True`
   (Adam hoặc SGD momentum 0,9; tên khác → `ValueError`; không có tham số trainable → `ValueError`).
3. Mỗi epoch: `model.train()` → `zero_grad` → forward → loss → `backward` → `step`; cộng dồn loss và số dự đoán đúng **theo số mẫu**.
4. `model.eval()` + `torch.no_grad()` trên `val_loader` → `val_loss`, `val_acc` cùng cách tính.
5. Ghi history và TensorBoard; nếu `val_loss` thấp hơn best (hoặc epoch đầu) thì lưu checkpoint.
6. Tăng bộ đếm không cải thiện; bằng `patience` (nếu có) thì dừng và in best epoch.

## Pipeline
```
features.make_loaders ────┐
model.build_model ────────┼─→ train_model(...)
model.unfreeze_last_blocks┘     mỗi epoch: train → validate → log → checkpoint nếu val_loss tốt hơn
                                → history + ckpt_path → evaluate / experiment
main.py: artifacts/models/<model>_baseline_best.pt, artifacts/runs/<model>_baseline_history.csv
```

## Detailed plan
- **Loss / optimizer:** `nn.CrossEntropyLoss()` nhận logits thô (không softmax ở head).
  `Adam(lr, weight_decay)` hoặc `SGD(lr, momentum=0.9, weight_decay)`; tên khác bị từ chối rõ ràng.
- **Tiêu chí checkpoint:** `val_loss` nhỏ nhất; epoch đầu luôn được lưu (`best_epoch == 0` bảo đảm có file kể cả loss NaN).
  `best_val_acc` được ghi cùng epoch đó, nên `history["best_val_acc"]` khớp file `.pt` — không dùng `max(val_acc)`.
- **Early stopping:** chỉ bật khi truyền `patience` (`None` mặc định); đếm số epoch liên tiếp `val_loss` không giảm.
- **Hàm hiện có:** `train_model(...) -> Dict[str, Any]` (`train.py:18`); `save_checkpoint(model, path)` (`:182`, tự tạo thư mục cha);
  `load_checkpoint(model, path, device)` (`:198`, trả model ở `eval` mode, `weights_only=True`).
- **Chỗ code chưa khớp plan:** không có.
- **Gotchas:**
  - `load_checkpoint` cần **cùng kiến trúc** (`name`, `num_classes`) như lúc train; `freeze_backbone` không đổi cấu trúc
    `state_dict`, nhưng `main.py` vẫn dựng lại cùng giá trị và dùng `pretrained=False` rồi nạp checkpoint.
  - Không lưu optimizer state → muốn train tiếp phải tạo optimizer mới, `train_model` không hỗ trợ resume.
  - Chọn checkpoint theo loss, không theo accuracy: accuracy cao nhất có thể ở epoch khác.
  - Reproducible chỉ khi gọi `shared.seed.set_seed` trước (như `main.py`); tắt seed thì DataLoader shuffle khác nhau mỗi lần chạy.

## Success criteria
- **Test:** `python -m pytest labs/lab2/tests/test_train.py labs/lab2/tests/test_checkpoint_consistency.py -q`
  kiểm history đủ key, SGD/uneven batch, round-trip `save`/`load`, early stopping, input sai bị từ chối.
- **Smoke run:** 1–2 epoch trên loader nhỏ → `history["val_loss"]` giảm hoặc ổn định, file `.pt` tồn tại và nạp lại được.
- **Normal run:** `python -m labs.lab2.main --epochs 1 --image-size 64 --no-experiments` in được
  `Best checkpoint: epoch ... (val_loss=..., val_acc=...)` và ghi history CSV dưới `artifacts/runs/`.

## Links
- Desc lab: [../project_description.md](../project_description.md) | Model: [model.md](model.md)
- Module liên quan: [evaluate.md](evaluate.md), [experiment.md](experiment.md), [preprocessing.md](preprocessing.md)
- Tiến độ và bằng chứng chạy: [../progress_status.md](../progress_status.md)
- Test: `labs/lab2/tests/test_train.py`, `labs/lab2/tests/test_checkpoint_consistency.py`
