# features — Chuẩn hoá và DataLoader (v1)

- **Created**: 2026-11-04T14:49:03+07:00
- **Last Updated**: 2026-11-19T13:35:00+07:00 (Đại)
- **Người phụ trách**: Member A (Đại)
- **Status**: Done

---

## Name
Features — transforms chuẩn hoá ImageNet, augmentation, Dataset wrapper, DataLoaders

## Background
Đề Practice 2: Sử dụng các mạng CNN tiền huấn luyện (pre-trained trên ImageNet). Cần đưa ảnh CIFAR-10 về cùng định dạng chuẩn mà backbone yêu cầu: resize kích thước phù hợp, chuyển tensor và chuẩn hoá theo phân phối ImageNet.
Module tạo các DataLoader độc lập cho train (có shuffle, tuỳ chọn augmentation), val và test (không shuffle, không augmentation).

## Goals / Purpose
- `build_transforms(image_size, augment)` trả `(train_transform, eval_transform)` với chuẩn hoá ImageNet mean `[0.485, 0.456, 0.406]` và std `[0.229, 0.224, 0.225]`
- `make_loaders` trả 3 DataLoader: `train_loader` (45.000 mẫu), `val_loader` (5.000 mẫu), `test_loader` (10.000 mẫu)
- Lấy một batch từ `train_loader` trả ảnh `[B, 3, image_size, image_size]` kiểu `float32` và nhãn `[B]` kiểu `int64` trong đoạn 0–9
- Không làm:
  - Data augmentation trên tập validation hoặc test — tránh làm sai lệch việc đánh giá mô hình
  - Dùng mean/std riêng của CIFAR-10 để đưa vào backbone pre-trained — backbone được huấn luyện trên phân phối ImageNet nên bắt buộc dùng ImageNet stats
  - Đặt `num_workers > 0` trên môi trường Windows để tránh deadlock tiến trình con

## Input / Output
- **Input:** `train_set`, `test_set` (từ `preprocessing.load_raw_dataset`); `train_idx`, `val_idx` (từ `preprocessing.load_split`); `image_size` (mặc định 224 hoặc 128/64/32), `augment` (bool), `batch_size` (64)
- **Output:** `(train_transform, eval_transform)`; `(train_loader, val_loader, test_loader)` (đối tượng trong bộ nhớ, không ghi file)
  - Code gọi `build_transforms` và `make_loaders` trong `main.py:208-217`.

## How to do it (general plan)
1. `build_transforms`: Tạo chuỗi transform ImageNet; nếu `augment=True` thêm `RandomResizedCrop` và `RandomHorizontalFlip` cho train
2. Xây dựng hai lớp wrapper `TransformedSubset` (cho train_idx, val_idx) và `TransformedDataset` (cho test) để transform on-the-fly khi nạp từng mẫu
3. `make_loaders`: Tạo DataLoader với `batch_size=64`, `num_workers=0`, chỉ shuffle train
4. Smoke run: Lấy 1 batch từ `train_loader`, kiểm tra shape `[64, 3, 32, 32]`, kiểu `float32`, nhãn trong `[0, 9]`
5. Normal run trong pipeline `main.py`

## Pipeline
```
train_set, test_set, train_idx, val_idx
  → build_transforms(image_size, augment) → train_tf, eval_tf
  → TransformedSubset × 2 + TransformedDataset × 1
  → make_loaders(batch_size=64) → train_loader, val_loader, test_loader
  → model / train / evaluate
```

## Detailed plan
- **Chuẩn hoá ImageNet:** Bắt buộc dùng `IMAGENET_MEAN = [0.485, 0.456, 0.406]` và `IMAGENET_STD = [0.229, 0.224, 0.225]` để tương thích với trọng số tiền huấn luyện của torchvision
- **Không sao chép mảng dữ liệu:** `TransformedSubset` chỉ lưu tham chiếu đến base dataset và mảng chỉ số, chỉ biến đổi khi truy cập `__getitem__`
- **`num_workers=0`:** An toàn tuyệt đối trên Windows, tránh lỗi `RuntimeError` khi spawn tiến trình đa luồng
- **Hàm hiện có:**
  - `build_transforms(image_size: int = 224, augment: bool = False) -> tuple` (`features/features.py:60`)
  - `make_loaders(train_set, test_set, train_idx: list[int], val_idx: list[int], train_tf, eval_tf, batch_size: int = 64) -> tuple[DataLoader, DataLoader, DataLoader]` (`:97`)
  - `class TransformedSubset(Dataset)` (`:20`); `class TransformedDataset(Dataset)` (`:41`)
- **Chỗ code chưa khớp plan:** không có
- **Gotchas:**
  - Kích thước ảnh 224x224 chuẩn ImageNet chạy rất chậm trên CPU; hỗ trợ linh hoạt tham số `image_size` (ví dụ 64 hoặc 128) để chạy nhanh mà không đổi kiến trúc pipeline (`features/features.py:61`)
  - Shuffle phải là `False` cho `val_loader` và `test_loader` để đảm bảo thứ tự nhãn nhất quán khi tính ma trận nhầm lẫn (confusion matrix)

## Success criteria
- **Test:** `python -m pytest labs/lab2/tests/test_feature.py -q` đạt (1 test)
- **Smoke run:** 1 batch huấn luyện có shape `[64, 3, H, W]` float32, nhãn int64 từ 0–9
- **Normal run:** 704 batch train, 79 batch val, 157 batch test (với batch_size=64); hoàn thành Milestone M1 bàn giao cho Member B

## Links
- Desc lab: [../project_description.md](../project_description.md)
- Module liên quan: [preprocessing.md](preprocessing.md), [model.md](model.md), [train.md](train.md)
- Tiến độ và bằng chứng chạy: [../progress_status.md](../progress_status.md) (mục Handoff A -> B dòng 18, Audit dòng 29)
- Code: `labs/lab2/features/features.py` | Test: `labs/lab2/tests/test_feature.py`
