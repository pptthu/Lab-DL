# Model Module Documentation (`labs/lab2/model/model.py`)

## 1. Purpose
- Load pre-trained CNN architectures from `torchvision.models` (ResNet-18, DenseNet-121, VGG-16) using modern `weights=` API.
- Replace the final classification head with a new linear layer adapted for 10-class CIFAR-10 classification.
- Support Feature Extraction (Transfer Learning) by freezing backbone parameters (`param.requires_grad = False`).
- Support Fine-Tuning by selectively unfreezing deep residual/dense blocks (`unfreeze_last_blocks`).
- Inspect model parameter counts (total vs. trainable) and architectural layers.

## 2. Input - Output
- **Input:**
  - `name` (`str`): Architecture name (`"resnet18"`, `"densenet121"`, `"vgg16"`).
  - `num_classes` (`int`): Target classification classes (default: `10` for CIFAR-10).
  - `freeze_backbone` (`bool`): Whether to freeze backbone weights (default: `True`).
  - `pretrained` (`bool`): Whether to download/load ImageNet pre-trained weights (default: `True`).
  - `n_blocks` (`int`): Number of deep blocks to unfreeze for fine-tuning.
  - Forward pass tensor input: Mini-batch tensor of shape `[B, 3, H, W]` (e.g. `[64, 3, 224, 224]` or `[64, 3, 32, 32]`).
  - Model checkpoint output path (managed downstream by `train`):
    - **Standard path:** `labs/lab2/artifacts/models/<model_name>_best.pt` (resolved via `shared.paths.artifact_dir("lab2", "models")`).
- **Output:**
  - Instantiated `nn.Module` producing output logits tensor of shape `[B, 10]` with dtype `torch.float32`.
  - Architecture description dictionary from `describe_model`:
    - `total_params` (`int`): Total parameter count (e.g., 11,181,642 for ResNet-18).
    - `trainable_params` (`int`): Trainable parameter count (5,130 when frozen; 8,398,858 when unfreezing `layer4`).
    - `layer_names` (`list[str]`): List of top-level children modules.

## 3. Success
- Model loads using valid modern torchvision weights (`ResNet18_Weights.DEFAULT`, etc.) without deprecation warnings.
- Output layer correctly outputs logits of shape `[B, 10]` matching the target class count.
- Freezing works accurately: only the new classification head has `requires_grad=True` (5,130 trainable parameters for ResNet-18).
- Fine-tuning unfreezing works accurately: calling `unfreeze_last_blocks(model, n_blocks=1)` unfreezes `layer4` alongside `fc`.
- Forward pass produces clean logits without any `NaN` or `Inf` values.

## 4. How to do
Execute the model initialization and configuration pipeline:
1. **Build base model (Transfer Learning):**
   - Call `model = build_model("resnet18", num_classes=10, freeze_backbone=True, pretrained=True)`.
   - The backbone weights are frozen, and `model.fc` is replaced by `nn.Linear(512, 10)`.
2. **Inspect architecture & parameters:**
   - Call `stats = describe_model(model)`.
   - Verify `stats["trainable_params"] == 5130` for ResNet-18 feature extraction.
3. **Configure Fine-Tuning (optional):**
   - Call `unfreeze_last_blocks(model, n_blocks=1)` to unfreeze `layer4` convolutional filters.
   - Verify parameter count increases to enable fine-tuning.
4. **Handoff to training loop:**
   - Pass `model` and its trainable parameters directly to the optimizer in `train_model()`.

---

**Tiếng Việt**

# Tài liệu Module Mô Hình (`labs/lab2/model/model.py`)

## 1. Purpose (Mục đích)
- Nạp các kiến trúc mạng CNN tiền huấn luyện từ `torchvision.models` (ResNet-18, DenseNet-121, VGG-16) sử dụng API hiện đại `weights=`.
- Thay thế lớp phân loại cuối (classification head) bằng lớp Linear mới phù hợp với 10 lớp của CIFAR-10.
- Hỗ trợ chế độ Transfer Learning (Feature Extraction) bằng cách đóng băng toàn bộ backbone (`param.requires_grad = False`).
- Hỗ trợ chế độ Fine-Tuning bằng cách mở khóa có chọn lọc các khối tích chập sâu cuối cùng (`unfreeze_last_blocks`).
- Kiểm tra số lượng tham số mô hình (tổng tham số và tham số cần huấn luyện) cùng cấu trúc các tầng.

## 2. Input - Output (Đầu vào - Đầu ra)
- **Input (Đầu vào):**
  - `name` (`str`): Tên kiến trúc mô hình (`"resnet18"`, `"densenet121"`, `"vgg16"`).
  - `num_classes` (`int`): Số lượng nhãn phân loại (mặc định: `10` cho CIFAR-10).
  - `freeze_backbone` (`bool`): Đóng băng trọng số backbone hay không (mặc định: `True`).
  - `pretrained` (`bool`): Có tải và nạp trọng số ImageNet hay không (mặc định: `True`).
  - `n_blocks` (`int`): Số lượng khối tích chập sâu cần mở khóa để fine-tuning.
  - Tensor đầu vào cho hàm forward: Batch ảnh có kích thước `[B, 3, H, W]` (ví dụ: `[64, 3, 224, 224]` hoặc `[64, 3, 32, 32]`).
  - Đường dẫn lưu file checkpoint trọng số (do module `train` quản lý):
    - **Đường dẫn chuẩn:** `labs/lab2/artifacts/models/<tên_model>_best.pt` (truy xuất qua `shared.paths.artifact_dir("lab2", "models")`).
- **Output (Đầu ra):**
  - Thể hiện mô hình `nn.Module` xuất ra tensor logits kích thước `[B, 10]` kiểu `torch.float32`.
  - Dictionary mô tả cấu trúc từ hàm `describe_model`:
    - `total_params` (`int`): Tổng số tham số (11.181.642 tham số cho ResNet-18).
    - `trainable_params` (`int`): Số tham số huấn luyện (5.130 khi đóng băng backbone; tăng lên 8.398.858 khi mở khóa `layer4`).
    - `layer_names` (`list[str]`): Danh sách tên các module con cấp 1.

## 3. Success (Tiêu chuẩn đạt yêu cầu)
- Nạp thành công mô hình với trọng số ImageNet chuẩn (`ResNet18_Weights.DEFAULT`...) không có cảnh báo deprecation.
- Lớp phân loại cuối xuất ra logits đúng kích thước `[B, 10]`, khớp với số lớp của CIFAR-10.
- Đóng băng chính xác: chỉ có lớp Linear phân loại mới là có `requires_grad=True` (5.130 tham số huấn luyện cho ResNet-18).
- Mở khóa fine-tuning chính xác: gọi `unfreeze_last_blocks(model, n_blocks=1)` sẽ kích hoạt gradient cho toàn bộ `layer4`.
- Quá trình forward pass hoạt động trơn tru, không sinh ra giá trị `NaN` hay `Inf`.

## 4. How to do (Cách thực hiện)
Quy trình khởi tạo và cấu hình mô hình gồm các bước:
1. **Khởi tạo mô hình cơ sở (Transfer Learning):**
   - Gọi `model = build_model("resnet18", num_classes=10, freeze_backbone=True, pretrained=True)`.
   - Các trọng số backbone được đóng băng, và `model.fc` được thay mới bằng `nn.Linear(512, 10)`.
2. **Kiểm tra thông số kiến trúc:**
   - Gọi `stats = describe_model(model)`.
   - Xác nhận `stats["trainable_params"] == 5130` cho chế độ feature extraction của ResNet-18.
3. **Cấu hình Fine-Tuning (tùy chọn):**
   - Gọi `unfreeze_last_blocks(model, n_blocks=1)` để mở khóa thêm khối tích chập `layer4`.
   - Kiểm tra số lượng tham số trainable tăng lên sẵn sàng cho fine-tuning.
4. **Bàn giao sang module huấn luyện:**
   - Truyền `model` và danh sách tham số trainable sang cho optimizer trong hàm `train_model()`.
