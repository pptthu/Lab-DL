# evaluate — Đánh giá mô hình trên tập validation/test (v2)

- **Created**: 2026-11-04 · **Last Updated**: 2026-11-20 (Bảo) — viết lại theo code hiện tại
- **Người phụ trách**: Member B (Bảo)
- **Status**: Done
- **Code**: [`labs/lab2/evaluate/evaluate.py`](../evaluate/evaluate.py)

---

## Name
Evaluation — loss trung bình theo số mẫu, accuracy top-1, confusion matrix, top nhầm lẫn

## Background
Sau khi `train` để lại checkpoint, cần đo chất lượng trên dữ liệu **không dùng để cập nhật trọng số**.
`evaluate` chỉ đọc: nhận một `nn.Module` đã nạp trọng số, không train, không ghi file.
Validation dùng để chọn cấu hình; test chỉ được gọi một lần cho cấu hình thắng.

## Goals / Purpose
- `evaluate` trả `(avg_loss, accuracy)` trên một loader bất kỳ (`val_loader` hoặc `test_loader`).
- `confusion_matrix` đếm ma trận `[C,C]` (hàng = nhãn thật, cột = nhãn dự đoán).
- `top_confusions` lấy `k` cặp nhầm lẫn nhiều nhất để phân tích lỗi.
- Không làm:
  - Không có `classification_report`/precision–recall (không dùng scikit-learn); phân tích lỗi dựa trên confusion matrix + `top_confusions`.
  - Không tự nạp checkpoint, không tự chọn model — người gọi phải `load_checkpoint` trước.
  - Không ghi file: `pipeline_summary.json` do `main.py` ghi, hình/bảng do notebook vẽ.

## Input / Output
- **Input:**
  - `evaluate(model, loader, device)`
  - `confusion_matrix(model, loader, device, num_classes=10)` — `num_classes > 0`
  - `top_confusions(cm, class_names, k=5)` — `cm` khớp `len(class_names)`
- **Output:**
  - `(avg_loss: float, accuracy: float)` với accuracy trong `[0,1]`
  - `np.ndarray [C,C]` kiểu `int64`; tổng mọi ô = số mẫu đã đánh giá
  - `list[(true_class, predicted_class, count)]`, sắp giảm dần, tối đa `k` phần tử

## How to do it (general plan)
1. `device = torch.device(device)`, `model.to(device)`, `model.eval()` → BatchNorm dùng running stats, tắt Dropout.
2. Tạo `nn.CrossEntropyLoss()` bên trong hàm để loss cùng công thức với lúc train.
3. Duyệt loader trong `torch.no_grad()`: forward, cộng `loss.item() * batch_size`, đếm `argmax == target`.
4. Chia cho **tổng số mẫu** (không chia theo số batch) → `avg_loss`, `accuracy`; loader rỗng → `ValueError`.
5. `confusion_matrix`: `cm[t, p] += 1` từng mẫu; nhãn ngoài `[0, C-1]` → `ValueError`.
6. `top_confusions`: bỏ đường chéo và ô 0, sort giảm dần theo số lần, cắt còn `k`.

## Pipeline
```
features.make_loaders → val_loader / test_loader (eval_tf, shuffle=False)
train.load_checkpoint → model đã nạp trọng số
   → evaluate(model, test_loader, device) → (test_loss, test_acc)
   → confusion_matrix(...) → cm [10,10]
   → top_confusions(cm, CIFAR10_CLASSES, k=5) → 5 cặp nhầm lẫn
main.py ghi val_*/test_*/top_confusions vào artifacts/runs/pipeline_summary.json
```

## Detailed plan
- **Trung bình theo số mẫu:** `loss.item()` là trung bình của batch, nên phải nhân lại `batch_size` rồi chia tổng số mẫu;
  nhờ vậy `avg_loss` của `evaluate` **cùng đại lượng** với `val_loss` trong `train_model` (so sánh trực tiếp được).
- **Không gradient, không cập nhật:** `torch.no_grad()` + `model.eval()`; điểm số không phụ thuộc thứ tự batch.
- **Chiều ma trận:** hàng = nhãn thật, cột = nhãn dự đoán; đường chéo = dự đoán đúng.
  `top_confusions` đọc `cm[i, j]` với `i != j` nên cần `class_names` đúng thứ tự lớp của dataset.
- **Hàm hiện có:** `evaluate` (`evaluate.py:18`), `confusion_matrix` (`:59`), `top_confusions` (`:98`).
- **Chỗ code chưa khớp plan:** không có.
- **Gotchas:**
  - `evaluate` để model ở `eval` mode sau khi xong; `train_model` tự gọi `model.train()` ở mỗi epoch nên không ảnh hưởng.
  - Loader phải không shuffle và không augment khi đánh giá; `make_loaders` đã đặt `shuffle=False` cho val/test.
  - `num_classes` phải khớp số lớp thật: nhãn vượt `num_classes-1` → `ValueError`; nếu truyền lớn hơn số lớp thật
    thì không báo lỗi nhưng ma trận có thêm hàng/cột 0.
  - Kết quả tái lập được vì không có yếu tố ngẫu nhiên; model/loader phải do người gọi cố định seed trước.

## Success criteria
- **Test:** `python -m pytest labs/lab2/tests/test_evaluate.py -q` — 4 test (accuracy/loss hợp lệ, loader rỗng raise,
  shape và tổng của confusion matrix, `top_confusions` sắp xếp và bỏ đường chéo).
- **Smoke run:** đánh giá trên loader giả 1–2 batch → `0 ≤ accuracy ≤ 1`, `avg_loss ≥ 0`, `cm.shape == (10,10)`.
- **Normal run:** `main.py` in `Validation: loss=..., accuracy=...`, `Test: loss=..., accuracy=...` và 5 dòng `top confusion`.

## Links
- Desc lab: [../project_description.md](../project_description.md) | Model: [model.md](model.md)
- Module liên quan: [train.md](train.md), [experiment.md](experiment.md), [features.md](features.md)
- Tiến độ và bằng chứng chạy: [../progress_status.md](../progress_status.md)
- Test: `labs/lab2/tests/test_evaluate.py`
