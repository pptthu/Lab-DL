"""Module `evaluate` (Evaluation) of Lab 2: Pre-trained CNNs and fine-tuning on CIFAR-10 (PyTorch).

Owner: Member B (see progress_status.md). Do not rename functions or change signatures
without telling the team leader: other modules depend on them.

VI: Module `evaluate` (Đánh giá). Người phụ trách: Member B. Không đổi tên hàm hoặc chữ ký
khi chưa báo nhóm trưởng vì các module khác phụ thuộc vào chúng.
"""
from __future__ import annotations

from typing import Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate(model: nn.Module, loader: DataLoader, device: Any) -> Tuple[float, float]:
    """Evaluate without gradients.

    Input : model, loader, device
    Output: (avg_loss, accuracy)

    VI: Đánh giá không tính gradient.
    Vào : model, loader, device
    Ra  : (avg_loss, accuracy)
    """
    device = torch.device(device)
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Loader yielded 0 samples for evaluation.")

    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples
    return float(avg_loss), float(accuracy)


def confusion_matrix(model: nn.Module, loader: DataLoader, device: Any, num_classes: int = 10) -> np.ndarray:
    """Confusion matrix.

    Input : model, loader, device, num_classes
    Output: array [C,C] (rows = true, columns = predicted)

    VI: Ma trận nhầm lẫn.
    Vào : model, loader, device, num_classes
    Ra  : mảng [C,C] (hàng = thật, cột = dự đoán)
    """
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}")

    device = torch.device(device)
    model.to(device)
    model.eval()

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            t_np = targets.cpu().numpy()
            p_np = preds.cpu().numpy()

            for t, p in zip(t_np, p_np):
                if 0 <= t < num_classes and 0 <= p < num_classes:
                    cm[t, p] += 1
                else:
                    raise ValueError(f"Target {t} or pred {p} out of range [0, {num_classes - 1}]")

    return cm


def top_confusions(cm: np.ndarray, class_names: List[str], k: int = 5) -> List[Tuple[str, str, int]]:
    """Most frequent confusions.

    Input : cm, class_names, k
    Output: list of (true_class, predicted_class, count)

    VI: Các cặp nhầm lẫn nhiều nhất.
    Vào : cm, class_names, k
    Ra  : danh sách (lớp thật, lớp dự đoán, số lần)
    """
    if not isinstance(cm, np.ndarray):
        cm = np.array(cm)

    num_classes = len(class_names)
    if cm.shape != (num_classes, num_classes):
        raise ValueError(f"cm shape {cm.shape} does not match len(class_names)={num_classes}")

    confusions: List[Tuple[str, str, int]] = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j:  # Exclude main diagonal (correct predictions)
                count = int(cm[i, j])
                if count > 0:
                    confusions.append((class_names[i], class_names[j], count))

    # Sort descending by error count
    confusions.sort(key=lambda x: x[2], reverse=True)
    return confusions[:k]
