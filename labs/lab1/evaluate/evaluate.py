"""Module `evaluate` (Evaluation) of Lab 1: Fashion-MNIST classification with an MLP (PyTorch).

Owner: Member B (see progress_status.md). Do not rename functions or change signatures
without telling the team leader: other modules depend on them.

VI: Module `evaluate` (Đánh giá). Người phụ trách: Member B. Không đổi tên hàm hoặc chữ ký
khi chưa báo nhóm trưởng vì các module khác phụ thuộc vào chúng.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import pandas as pd
import torch
from torch import nn

# Mức test của baseline theo desc lab: MLP tốt nhất trong bài báo gốc Fashion-MNIST (Xiao et al. 2017, Bảng 3: 0,871).
TEST_ACC_THRESHOLD = 0.87


def evaluate(model: nn.Module, loader, device: str | torch.device) -> Tuple[float, float]:
    """Evaluate without gradients (model.eval + torch.no_grad).

    Input : model, loader, device
    Output: (avg_loss: float, accuracy: float in 0-1)

    VI: Đánh giá không tính gradient (model.eval + torch.no_grad).
    Vào : model, loader, device
    Ra  : (avg_loss: float, accuracy: float trong 0-1)
    """
    model.to(device)
    # Eval mode (dropout off) and no gradients: we only measure, we do not train.
    # VI: Chế độ eval (tắt dropout) và không tính gradient: chỉ đo, không huấn luyện.
    model.eval()
    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            batch_size = x_batch.size(0)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)

            # Weight by batch size so the mean loss is exact per sample.
            # VI: Nhân với số ảnh của batch để loss trung bình đúng theo từng mẫu.
            running_loss += loss.item() * batch_size
            preds = torch.argmax(logits, dim=1)
            running_correct += (preds == y_batch).sum().item()
            total_samples += batch_size

    avg_loss = float(running_loss / total_samples) if total_samples > 0 else 0.0
    accuracy = float(running_correct / total_samples) if total_samples > 0 else 0.0
    return avg_loss, accuracy


def confusion_matrix(
    model: nn.Module, loader, device: str | torch.device, num_classes: int = 10
) -> np.ndarray:
    """Build the confusion matrix.

    Input : model, loader, device, num_classes
    Output: array [num_classes, num_classes]; rows = true class, columns = predicted class

    VI: Tạo ma trận nhầm lẫn.
    Vào : model, loader, device, num_classes
    Ra  : mảng [num_classes, num_classes]; hàng = lớp thật, cột = lớp dự đoán
    """
    model.to(device)
    model.eval()

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            targets = y_batch.cpu().numpy()

            # Add 1 at [true class, predicted class]: rows = true, columns = predicted.
            # VI: Cộng 1 vào ô [lớp thật, lớp dự đoán]: hàng = lớp thật, cột = lớp dự đoán.
            np.add.at(cm, (targets, preds), 1)

    return cm


def per_class_accuracy(cm: np.ndarray) -> Dict[int, float]:
    """Accuracy of each class.

    Input : cm: confusion matrix
    Output: dict: class index -> accuracy

    VI: Độ chính xác từng lớp.
    Vào : cm: ma trận nhầm lẫn
    Ra  : dict: chỉ số lớp -> độ chính xác
    """
    cm_arr = np.asarray(cm)
    num_classes = cm_arr.shape[0]
    accs: Dict[int, float] = {}

    for i in range(num_classes):
        row_sum = cm_arr[i].sum()
        if row_sum > 0:
            # Diagonal / row sum = correct images of class i / all images of class i (recall).
            # VI: Đường chéo / tổng hàng = số ảnh lớp i đoán đúng / tổng ảnh lớp i (recall).
            accs[i] = float(cm_arr[i, i] / row_sum)
        else:
            accs[i] = 0.0

    return accs


def save_classification_report(cm: np.ndarray, class_names: Sequence[str], csv_path: str) -> pd.DataFrame:
    """Per-class precision, recall, specificity and F1 (values in 0-1), saved as CSV.

    Input : cm: confusion matrix; class_names; csv_path
    Output: DataFrame with one row per class; CSV file at csv_path

    VI: Precision, recall, specificity và F1 của từng lớp (giá trị trong 0-1), lưu ra CSV.
    Vào : cm: ma trận nhầm lẫn; class_names; csv_path
    Ra  : DataFrame mỗi lớp một dòng; file CSV tại csv_path
    """
    cm_arr = np.asarray(cm)
    rows = []
    for i, name in enumerate(class_names):
        # TP: class i predicted as i; FN: class i predicted as another; FP: another class predicted as i; TN: the rest.
        # VI: TP: lớp i đoán đúng là i; FN: lớp i bị đoán sang lớp khác; FP: lớp khác bị đoán thành i; TN: phần còn lại.
        tp = int(cm_arr[i, i])
        fn = int(cm_arr[i].sum()) - tp
        fp = int(cm_arr[:, i].sum()) - tp
        tn = int(cm_arr.sum()) - tp - fp - fn
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        specificity = tn / (tn + fp) if tn + fp > 0 else 0.0
        # F1 = harmonic mean of precision and recall.
        # VI: F1 = trung bình điều hòa của precision và recall.
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        rows.append({
            "class": name,
            "support": tp + fn,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": f1,
        })

    df = pd.DataFrame(rows)
    p = Path(csv_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return df


def save_test_results(rows: List[Dict[str, Any]], csv_path: str, threshold: float = TEST_ACC_THRESHOLD) -> pd.DataFrame:
    """Ghi kết quả val và test của các model đã chọn (mỗi model chạy test đúng một lần).

    Vào : rows: list dict có name, role ('baseline' / 'selected' / 'baseline+selected'), run_name, ckpt_path,
          val_loss, val_acc, test_loss, test_acc; csv_path; threshold: mức test của baseline
    Ra  : DataFrame (thêm cột meets_threshold, chỉ tính cho dòng có baseline); file CSV tại csv_path
    """
    df = pd.DataFrame(rows)
    df["threshold"] = threshold
    df["meets_threshold"] = [
        bool(acc >= threshold) if "baseline" in str(role) else None
        for acc, role in zip(df["test_acc"], df["role"])
    ]
    p = Path(csv_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return df
