# Model Module Documentation (`labs/lab2/model/model.py`)

## 1. Overview
The `model` module handles the instantiation of pre-trained convolutional neural networks from `torchvision.models` (ResNet-18, DenseNet-121, VGG-16), adapting their final classification head for CIFAR-10 (10 classes), freezing backbone parameters for transfer learning (feature extraction), and selective layer unfreezing for fine-tuning.

- **File location:** `labs/lab2/model/model.py`
- **Assigned owner:** Member A
- **Dependencies:** `torch`, `torch.nn`, `torchvision.models`

---

## 2. Functions & API Contract

### `build_model(name: str, num_classes: int = 10, freeze_backbone: bool = True, pretrained: bool = True) -> nn.Module`
Builds a transfer learning model by loading a pre-trained backbone, freezing weights if requested, and replacing the final fully connected classification layer.

#### Arguments:
- `name` (`str`): Architecture name. Supported options: `"resnet18"`, `"densenet121"`, `"vgg16"` (case-insensitive).
- `num_classes` (`int`): Number of target output classes (default: `10` for CIFAR-10).
- `freeze_backbone` (`bool`): If `True`, freezes all backbone parameters (`param.requires_grad = False`) before attaching the new classification head.
- `pretrained` (`bool`): If `True`, initializes the backbone with pre-trained ImageNet weights (default: `True`).

#### Classification Head Adaptations:
- **ResNet-18:**
  - Original: `model.fc` with 512 input features.
  - Replacement: `model.fc = nn.Linear(512, num_classes)`
- **DenseNet-121:**
  - Original: `model.classifier` with 1024 input features.
  - Replacement: `model.classifier = nn.Linear(1024, num_classes)`
- **VGG-16:**
  - Original: `model.classifier[6]` with 4096 input features.
  - Replacement: `model.classifier[6] = nn.Linear(4096, num_classes)`

#### Returns:
- `nn.Module`: The instantiated PyTorch model producing logits with shape `[Batch_Size, num_classes]`.

---

### `describe_model(model: nn.Module) -> dict`
Inspects model parameters and structural modules.

#### Arguments:
- `model` (`nn.Module`): The model instance to inspect.

#### Returns:
A dictionary containing parameter counts and layer names:
- `total_params` (`int`): Total count of all parameters.
- `trainable_params` (`int`): Count of parameters with `requires_grad=True`.
- `layer_names` (`list[str]`): List of top-level children module names.

---

### `unfreeze_last_blocks(model: nn.Module, n_blocks: int) -> None`
Unfreezes parameters in the deepest `n_blocks` convolutional stages for fine-tuning.

#### Arguments:
- `model` (`nn.Module`): The model modified in-place.
- `n_blocks` (`int`): Number of end blocks to unfreeze (from output backwards).

#### Architecture Support:
- **ResNet:** Traverses `[model.layer4, model.layer3, model.layer2, model.layer1][:n_blocks]` and sets `param.requires_grad = True`.
- **DenseNet:** Unfreezes `model.classifier` and the last `n_blocks * 2` child blocks inside `model.features`.
- **Generic Fallback:** Unfreezes the last `n_blocks` top-level children modules in `model.children()`.

---

## 3. Key Implementation Details & Defense Q&A
- **Q: Why use `weights=models.ResNet18_Weights.DEFAULT` instead of `pretrained=True`?**
  - **A:** `pretrained=True` has been deprecated in modern torchvision. Passing `weights=..._Weights.DEFAULT` ensures forward compatibility and explicit versioning of pre-trained checkpoint weights.
- **Q: Why do we freeze the backbone before replacing the classification layer?**
  - **A:** By freezing the entire model first and then creating `model.fc = nn.Linear(...)`, the new linear layer is automatically created with `requires_grad=True` while all existing convolutional weights remain frozen. This avoids accidentally freezing the new head.
- **Q: What is the difference between Feature Extraction and Fine-Tuning?**
  - **A:** In Feature Extraction (Transfer Learning), the backbone weights are completely frozen, acting as a fixed feature extractor; only the new head is trained. In Fine-Tuning, one or more later convolutional blocks are unfrozen alongside the head and trained with a smaller learning rate to adapt domain-specific visual features.
- **Q: Why unfreeze the last layers instead of the first layers?**
  - **A:** Early convolutional layers capture generic features (edges, corners, color blobs) applicable across all vision tasks. Later layers capture task-specific high-level semantics. Unfreezing later layers allows adaptation to the target domain without destroying low-level feature extraction capabilities.

---

**Tiếng Việt**

# Tài liệu Module Mô Hình (`labs/lab2/model/model.py`)

## 1. Tổng quan
Module `model` phụ trách nạp các mô hình mạng nơ-ron tích chập (CNN) đã được huấn luyện trước từ `torchvision.models` (ResNet-18, DenseNet-121, VGG-16), thay thế lớp phân loại cuối cùng (classification head) cho bài toán phân loại 10 lớp của CIFAR-10, đóng băng (freeze) các trọng số backbone cho chế độ Transfer Learning và mở khóa (unfreeze) một số khối cuối cho chế độ Fine-tuning.

- **Vị trí file:** `labs/lab2/model/model.py`
- **Người phụ trách:** Member A
- **Thư viện phụ thuộc:** `torch`, `torch.nn`, `torchvision.models`

---

## 2. Các hàm & Giao diện lập trình (API)

### `build_model(name: str, num_classes: int = 10, freeze_backbone: bool = True, pretrained: bool = True) -> nn.Module`
Khởi tạo mô hình mạng theo tên kiến trúc, nạp trọng số tiền huấn luyện và thay thế lớp phân loại cuối.

#### Tham số:
- `name` (`str`): Tên kiến trúc mô hình (`"resnet18"`, `"densenet121"`, `"vgg16"`).
- `num_classes` (`int`): Số lớp phân loại mục tiêu (10 cho CIFAR-10).
- `freeze_backbone` (`bool`): Đóng băng các tầng backbone (đặt `param.requires_grad = False`).
- `pretrained` (`bool`): Có nạp trọng số ImageNet hay không.

#### Cách thay thế lớp phân loại:
- **ResNet-18:** Thay thế `model.fc = nn.Linear(512, num_classes)`.
- **DenseNet-121:** Thay thế `model.classifier = nn.Linear(1024, num_classes)`.
- **VGG-16:** Thay thế lớp thứ 6 trong bộ classifier `model.classifier[6] = nn.Linear(4096, num_classes)`.

---

### `describe_model(model: nn.Module) -> dict`
Kiểm tra cấu trúc và đếm số lượng tham số của mô hình.

#### Đầu ra:
- `total_params`: Tổng số lượng tham số trong mô hình.
- `trainable_params`: Số lượng tham số cần cập nhật gradient (`requires_grad=True`).
- `layer_names`: Danh sách tên các khối tầng con cấp 1 của mô hình.

---

### `unfreeze_last_blocks(model: nn.Module, n_blocks: int) -> None`
Mở khóa các khối tích chập ở các tầng sâu cuối cùng để tiến hành Fine-tuning.

- Với ResNet: Mở khóa các tầng `layer4`, `layer3` tùy theo tham số `n_blocks`.
- Với DenseNet: Mở khóa các khối con trong `model.features`.

---

## 3. Chi tiết kỹ thuật & Câu hỏi vấn đáp thường gặp
- **Hỏi: Phân biệt Transfer Learning (Feature Extraction) và Fine-tuning?**
  - **Trả lời:** Trong Transfer Learning (Feature Extraction), toàn bộ các tầng tích chập của mạng đều bị đóng băng, chỉ huấn luyện lớp phân loại mới. Trong Fine-tuning, ngoài lớp phân loại mới, một hoặc nhiều khối tích chập cuối cùng cũng được mở khóa để cùng cập nhật trọng số với learning rate nhỏ.
- **Hỏi: Tại sao nên mở khóa từ các tầng sau cùng thay vì các tầng đầu tiên?**
  - **Trả lời:** Các tầng đầu tiên của mạng CNN học các đặc trưng thị giác cơ bản (cạnh, góc, màu sắc, hoa văn) có tính tổng quát cao cho mọi bộ dữ liệu. Các tầng càng sâu về sau càng học các đặc trưng ngữ nghĩa phức tạp và đặc thù cho từng bài toán, do đó mở khóa tầng sâu giúp mô hình thích nghi tốt nhất với dữ liệu mới mà không làm hỏng các bộ lọc đặc trưng cơ bản.
