"""Module `model` (Model) of Lab 1: Fashion-MNIST classification with an MLP (PyTorch).

Owner: Member A (see progress_status.md). Do not rename functions or change signatures
without telling the team leader: other modules depend on them.

VI: Module `model` (Mô hình). Người phụ trách: Member A. Không đổi tên hàm hoặc chữ ký
khi chưa báo nhóm trưởng vì các module khác phụ thuộc vào chúng.
"""
# Input / Output of this module
#   Input : batches [B,1,28,28] float32 from features.make_loaders
#   Output: 1) an untrained MLP object (in memory) -> train.train_model (Member B)
#           2) forward -> logits [B,10]; count_parameters -> 235,146
#   Saved by train (not by this file): best checkpoint (state_dict) labs/lab1/artifacts/models/mlp_baseline.pt;
#           experiment checkpoints labs/lab1/artifacts/models/exp_run_*.pt; per-epoch history in
#           labs/lab1/artifacts/runs/; none of them committed (*.pt in .gitignore)
# VI: Input / Output của module
#   Vào : batch [B,1,28,28] float32 từ features.make_loaders
#   Ra  : 1) object MLP chưa train (trong bộ nhớ) -> train.train_model (Member B)
#         2) forward -> logits [B,10]; count_parameters -> 235.146
#   Do train lưu (không phải file này): checkpoint tốt nhất (state_dict) labs/lab1/artifacts/models/mlp_baseline.pt;
#         checkpoint thí nghiệm labs/lab1/artifacts/models/exp_run_*.pt; lịch sử từng epoch trong
#         labs/lab1/artifacts/runs/; đều không commit (*.pt nằm trong .gitignore)

# Type hints are not evaluated at import time (works on older Python versions too).
# VI: Không tính type hint lúc import (chạy được cả trên Python cũ).
from __future__ import annotations

# torch: tensors; nn: layer classes (Linear, ReLU, Dropout, Flatten, Sequential).
# VI: torch: tensor; nn: các lớp mạng (Linear, ReLU, Dropout, Flatten, Sequential).
import torch
from torch import nn

# 28 x 28 = 784 input values once the image is flattened.
# VI: 28 x 28 = 784 giá trị đầu vào sau khi làm phẳng ảnh.
INPUT_SIZE = 28 * 28


# Subclass of nn.Module: parameters are registered automatically, so .to(device),
# .train(), .eval(), .parameters() and state_dict() work.
# VI: Kế thừa nn.Module: tham số được đăng ký tự động nên dùng được .to(device),
# .train(), .eval(), .parameters() và state_dict().
class MLP(nn.Module):
    """Fully connected network (subclass of torch.nn.Module).

    Input : __init__(hidden_sizes=(256,128), num_classes=10, dropout=0.0); forward(x: tensor [B,1,28,28])
    Output: forward returns logits [B,10] (NO softmax: CrossEntropyLoss expects logits)

    VI: Mạng kết nối đầy đủ (kế thừa torch.nn.Module).
    Vào : __init__(hidden_sizes=(256,128), num_classes=10, dropout=0.0); forward(x: tensor [B,1,28,28])
    Ra  : forward trả logits [B,10] (KHÔNG softmax: CrossEntropyLoss cần logits)
    """

    # Defaults = the baseline 784 -> 256 -> 128 -> 10; experiments pass other sizes.
    # VI: Giá trị mặc định = baseline 784 -> 256 -> 128 -> 10; thí nghiệm truyền kích thước khác.
    def __init__(self, hidden_sizes=(256, 128), num_classes: int = 10, dropout: float = 0.0) -> None:
        # Must run first so nn.Module can track the sub-layers assigned below.
        # VI: Phải gọi đầu tiên để nn.Module theo dõi được các lớp con gán bên dưới.
        super().__init__()
        layers = [nn.Flatten()]  # [B,1,28,28] -> [B,784]
        # Input width of the next Linear layer; starts at 784.
        # VI: Độ rộng đầu vào của lớp Linear tiếp theo; bắt đầu là 784.
        in_features = INPUT_SIZE
        # One block per hidden layer: Linear -> ReLU (-> Dropout when dropout > 0).
        # VI: Mỗi tầng ẩn là một khối: Linear -> ReLU (-> Dropout nếu dropout > 0).
        for hidden in hidden_sizes:
            # Fully connected: in_features x hidden weights + hidden biases (784 x 256 + 256 = 200,960).
            # VI: Kết nối đầy đủ: in_features x hidden trọng số + hidden bias (784 x 256 + 256 = 200.960).
            layers.append(nn.Linear(in_features, hidden))
            # Non-linearity; without it, stacked Linear layers collapse into one linear map.
            # VI: Phi tuyến; thiếu nó thì nhiều lớp Linear gộp lại chỉ bằng một phép tuyến tính.
            layers.append(nn.ReLU())
            # Dropout only when asked (regularization); the baseline has none.
            # VI: Chỉ thêm Dropout khi được yêu cầu (chống overfit); baseline không có.
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            # The next layer's input is this layer's output.
            # VI: Đầu vào lớp sau là đầu ra lớp này.
            in_features = hidden
        # Last layer has no activation: it returns raw logits.
        # VI: Tầng cuối không có hàm kích hoạt: trả về logits thô.
        layers.append(nn.Linear(in_features, num_classes))
        # Run the layers in order; nn.Sequential is registered as one sub-module.
        # VI: Chạy các lớp theo thứ tự; nn.Sequential được đăng ký là một module con.
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B,1,28,28] -> logits [B,10]; prediction = argmax over the 10 logits.
        # VI: [B,1,28,28] -> logits [B,10]; dự đoán = argmax của 10 logits.
        return self.net(x)


def count_parameters(model) -> int:
    """Count trainable parameters.

    Input : model
    Output: int

    VI: Đếm số tham số huấn luyện được.
    Vào : model
    Ra  : int
    """
    # numel() = number of values in one tensor; frozen tensors (requires_grad=False) are skipped.
    # Baseline: 200,960 + 32,896 + 1,290 = 235,146.
    # VI: numel() = số phần tử của một tensor; tensor bị đóng băng (requires_grad=False) bị bỏ qua.
    # Baseline: 200.960 + 32.896 + 1.290 = 235.146.
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
