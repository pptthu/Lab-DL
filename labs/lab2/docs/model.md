# model — Kiến trúc CNN tiền huấn luyện (v1)

- **Created**: 2026-11-04T14:49:03+07:00
- **Last Updated**: 2026-11-19T13:35:00+07:00 (Đại)
- **Người phụ trách**: Member A (Đại)
- **Status**: Done

---

## Name
Model — nạp mạng CNN pre-trained, thay classification head, đóng băng/mở khóa khối sâu

## Background
Đề Practice 2: Tái sử dụng các mạng CNN tiền huấn luyện của `torchvision.models` (ResNet-18, DenseNet-121, VGG-16) cho bài toán phân loại ảnh 10 lớp CIFAR-10.
Cần hỗ trợ 2 chế độ:
1. **Feature Extraction (Transfer Learning):** Đóng băng toàn bộ backbone (`requires_grad = False`), chỉ huấn luyện classifier head mới.
2. **Fine-Tuning:** Mở khóa thêm các khối tích chập sâu ngữ nghĩa cao cuối cùng (`unfreeze_last_blocks`) để thích nghi đặc trưng với CIFAR-10.

## Goals / Purpose
- `build_model(name, num_classes, freeze_backbone, pretrained)` hỗ trợ đủ 3 kiến trúc: `resnet18`, `densenet121`, `vgg16`; thay thế lớp cuối thành `num_classes=10`; trả mô hình sẵn sàng xuất logits `[B, 10]`
- `describe_model` thống kê đúng số tham số:
  - ResNet-18: tổng 11.181.642 params; trainable 5.130 (khi đóng băng backbone, chỉ lớp `fc`)
  - Fine-tuning mở block layer4: trainable params tăng lên 8.403.466
- `unfreeze_last_blocks(model, n_blocks)` mở khóa chính xác khối sâu tương ứng:
  - ResNet: các khối tầng `layer4`, `layer3`, `layer2`, `layer1`
  - VGG-16: tách theo 5 khối tích chập phân cách bởi `MaxPool2d` (không bị ảnh hưởng bởi cấu trúc flat Sequential)
  - DenseNet-121: các khối `denseblock` và `transition` trong `features`
- Không làm:
  - Thêm Softmax vào cuối mạng — hàm `CrossEntropyLoss` trong `train` nhận logits thô
  - Dùng tham số cũ đã deprecated `pretrained=True` của torchvision — dùng chuẩn mới `weights=...Weights.DEFAULT`
  - Chạy vòng lặp huấn luyện — việc của `train`

## Input / Output
- **Input:** Batch ảnh tensor `[B, 3, H, W]` float32 đã chuẩn hoá từ `features`; cấu hình mô hình (`name`, `num_classes`, `freeze_backbone`, `pretrained`) từ `main.py` / `experiment`
- **Output:** Đối tượng `nn.Module` (PyTorch model); forward pass trả tensor logits `[B, 10]`; dict thống kê tham số từ `describe_model`
  - Code gọi `build_model`, `unfreeze_last_blocks`, `describe_model` trong `main.py:226-239`.

## How to do it (general plan)
1. `build_model`: Nạp model từ `torchvision.models` với API `weights=DEFAULT`, đóng băng `requires_grad=False` nếu `freeze_backbone=True`
2. Thay thế lớp phân loại cuối: ResNet-18 (`model.fc`), DenseNet-121 (`model.classifier`), VGG-16 (`model.classifier[6]`) thành `nn.Linear(in_features, 10)`
3. `describe_model`: Đếm `total_params`, `trainable_params` và lấy danh sách tên layer con
4. `unfreeze_last_blocks`: Xác định khối ngữ nghĩa theo kiến trúc, mở khóa `requires_grad=True` cho $N$ khối cuối cùng kèm classifier head
5. Smoke run: Forward pass 1 batch giả định `[64, 3, 32, 32]`, kiểm tra shape logits `[64, 10]`, không chứa NaN
6. Normal run trong pipeline `main.py`

## Pipeline
```
config (name, num_classes=10, freeze, pretrained)
  → build_model → nn.Module (thay head 10 lớp)
  → (tuỳ chọn) unfreeze_last_blocks(model, n_blocks)
  → describe_model → in số tham số
  → forward pass [B, 3, H, W] → logits [B, 10]
  → train / evaluate / experiment
```

## Detailed plan
- **API torchvision hiện đại:** Dùng `ResNet18_Weights.DEFAULT`, `DenseNet121_Weights.DEFAULT`, `VGG16_Weights.DEFAULT` thay vì cờ `pretrained=True` cũ (`model/model.py:33-41`)
- **Tách khối VGG-16 chuẩn xác:** Vì `model.features` của VGG là 1 danh sách phẳng (`nn.Sequential`), nếu lấy 2 phần tử cuối sẽ chỉ trúng ReLU và MaxPool2d (0 tham số). Hàm `_split_vgg_conv_blocks` gom nhóm theo mốc `MaxPool2d` thành đúng 5 khối conv (`:94-114`)
- **Tự động kích hoạt gradient cho head:** Khi tạo mới `nn.Linear(in_features, num_classes)`, PyTorch mặc định đặt `requires_grad=True` cho layer mới này
- **Hàm hiện có:**
  - `build_model(name: str, num_classes: int = 10, freeze_backbone: bool = True, pretrained: bool = True) -> nn.Module` (`model/model.py:15`)
  - `describe_model(model: nn.Module) -> dict` (`:64`)
  - `unfreeze_last_blocks(model: nn.Module, n_blocks: int) -> None` (`:117`)
  - Hàm phụ: `_unfreeze(module)` (`:85`), `_split_vgg_conv_blocks(features)` (`:94`)
- **Chỗ code chưa khớp plan:** không có
- **Gotchas:**
  - `n_blocks` phải là int >= 1, nếu truyền bool hoặc < 1 sẽ raise `TypeError` hoặc `ValueError` (`model/model.py:146-149`)
  - Tên mô hình không thuộc danh sách hỗ trợ sẽ raise `ValueError` (`:43`)
  - Sau khi unfreeze, cần truyền đúng danh sách tham số có `requires_grad=True` vào optimizer (`filter(lambda p: p.requires_grad, model.parameters())`)

## Success criteria
- **Test:** `python -m pytest labs/lab2/tests/test_model.py labs/lab2/tests/test_model_unfreezing.py -q` đạt (11 test)
- **Smoke run:** `build_model('resnet18')` trả total_params = 11.181.642, trainable_params = 5.130; forward pass nhận `[64, 3, 32, 32]` ra logits `[64, 10]`
- **Normal run:** Hỗ trợ đầy đủ ResNet-18, DenseNet-121, VGG-16 cho cả baseline (đóng băng) và fine-tuning (mở khóa khối cuối); hoàn thành Milestone M2 bàn giao cho Member B

## Links
- Desc lab: [../project_description.md](../project_description.md)
- Module liên quan: [features.md](features.md), [train.md](train.md), [experiment.md](experiment.md)
- Tiến độ và bằng chứng chạy: [../progress_status.md](../progress_status.md) (mục Handoff A -> B dòng 19, Audit dòng 30, 35)
- Code: `labs/lab2/model/model.py` | Test: `labs/lab2/tests/test_model.py`, `labs/lab2/tests/test_model_unfreezing.py`
