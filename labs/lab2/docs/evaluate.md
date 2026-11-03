# Evaluate Module Documentation (`labs/lab2/evaluate/evaluate.py`)

## 1. Overview
The `evaluate` module handles post-training evaluation, computing final loss and top-1 accuracy on unseen test data, constructing the multi-class confusion matrix, and identifying the top confusing class pairs.

- **File location:** `labs/lab2/evaluate/evaluate.py`
- **Assigned owner:** Member B
- **Dependencies:** `torch`, `torch.nn`, `numpy`, `torch.utils.data.DataLoader`

---

## 2. Functions & API Contract

### `evaluate(model: nn.Module, loader: DataLoader, device: Any) -> Tuple[float, float]`
Evaluates model performance over a given DataLoader in evaluation mode with gradients disabled.

#### Arguments:
- `model` (`nn.Module`): Model to evaluate.
- `loader` (`DataLoader`): Data provider (typically `test_loader`).
- `device` (`Any`): Execution hardware device (`"cuda"` or `"cpu"`).

#### Returns:
- `(avg_loss, accuracy)`:
  - `avg_loss` (`float`): Cross-entropy loss averaged across all samples.
  - `accuracy` (`float`): Ratio of correct predictions over total samples (`correct / total_samples`).

---

### `confusion_matrix(model: nn.Module, loader: DataLoader, device: Any, num_classes: int = 10) -> np.ndarray`
Computes the full confusion matrix for multi-class classification.

#### Arguments:
- `model` (`nn.Module`): Model to evaluate.
- `loader` (`DataLoader`): Data provider.
- `device` (`Any`): Execution device.
- `num_classes` (`int`): Total number of distinct categories (default: `10`).

#### Returns:
- `cm` (`np.ndarray`): A 2D array of shape `(num_classes, num_classes)` and dtype `int64`.
  - **Rows:** Ground-truth class index `t`.
  - **Columns:** Model predicted class index `p`.
  - `cm[t, p]` represents the count of samples belonging to true class `t` that were classified as `p`.

---

### `top_confusions(cm: np.ndarray, class_names: List[str], k: int = 5) -> List[Tuple[str, str, int]]`
Finds the top `k` most frequent misclassifications from the confusion matrix.

#### Arguments:
- `cm` (`np.ndarray`): Confusion matrix of shape `(C, C)`.
- `class_names` (`List[str]`): List of human-readable class names with length `C`.
- `k` (`int`): Maximum number of top confusion pairs to return (default: `5`).

#### Logic:
- Filters out correct predictions along the main diagonal (`i == j`).
- Collects off-diagonal elements where `cm[i, j] > 0`.
- Sorts pairs in descending order based on misclassification count.
- Returns the top `k` entries as tuples: `(true_class_name, predicted_class_name, count)`.

---

## 3. Key Implementation Details & Defense Q&A
- **Q: Why use `with torch.no_grad():` during evaluation?**
  - **A:** Disabling autograd's calculation and tracking of computational graphs drastically cuts memory consumption and speeds up the inference pass.
- **Q: Why set `model.eval()` before evaluation?**
  - **A:** Many CNN components behave differently during training versus inference:
    - **Dropout:** Drops neurons randomly during training, but scales activations and keeps all neurons active during evaluation.
    - **BatchNorm:** Computes batch statistics during training, but uses frozen running mean and variance during evaluation. Failing to call `model.eval()` would cause predictions to vary depending on batch compositions.
- **Q: How to read a confusion matrix row vs. column?**
  - **A:** By convention in machine learning and this codebase:
    - Row $i$: All samples whose actual ground truth is class $i$.
    - Diagonal element $cm[i, i]$: True positives for class $i$.
    - Off-diagonal element $cm[i, j]$ (where $i \neq j$): Actual class $i$ incorrectly predicted as class $j$.

---

**Tiếng Việt**

# Tài liệu Module Đánh Giá (`labs/lab2/evaluate/evaluate.py`)

## 1. Tổng quan
Module `evaluate` chịu trách nhiệm đánh giá mô hình sau huấn luyện trên tập dữ liệu kiểm thử (test set), tính toán hàm mất mát (loss), độ chính xác (accuracy), dựng ma trận nhầm lẫn (confusion matrix), và lọc ra các cặp nhãn hay bị dự đoán nhầm lẫn nhất.

- **Vị trí file:** `labs/lab2/evaluate/evaluate.py`
- **Người phụ trách:** Member B
- **Thư viện phụ thuộc:** `torch`, `torch.nn`, `numpy`, `torch.utils.data.DataLoader`

---

## 2. Các hàm & Giao diện lập trình (API)

### `evaluate(model: nn.Module, loader: DataLoader, device: Any) -> Tuple[float, float]`
Đánh giá độ mất mát và độ chính xác phân loại trên tập dữ liệu được cung cấp bởi `loader`.

#### Đầu vào:
- `model`: Mô hình cần kiểm thử.
- `loader`: DataLoader tập kiểm thử (test set).
- `device`: Thiết bị chạy tính toán (`"cuda"` hoặc `"cpu"`).

#### Đầu ra:
- `(avg_loss, accuracy)`: Bộ đôi gồm giá trị loss trung bình và tỷ lệ dự đoán chính xác (`correct / total_samples`).

---

### `confusion_matrix(model, loader, device, num_classes=10) -> np.ndarray`
Xây dựng ma trận nhầm lẫn (Confusion Matrix) kích thước `(C, C)` với $C = 10$.

- **Quy ước:**
  - Hàng biểu diễn nhãn thực tế (True label $t$).
  - Cột biểu diễn nhãn dự đoán (Predicted label $p$).
  - `cm[t, p]` là số lần một mẫu thuộc lớp $t$ bị mô hình gán nhãn là $p$.

---

### `top_confusions(cm, class_names, k=5) -> List[Tuple[str, str, int]]`
Trích xuất `k` cặp lớp bị mô hình nhầm lẫn nhiều nhất.

- Bỏ qua đường chéo chính (nơi dự đoán đúng $i = j$).
- Sắp xếp các lỗi giảm dần theo số lượng mẫu bị nhầm.
- Trả về danh sách gồm các bộ ba: `(tên_lớp_thật, tên_lớp_dự_đoán, số_lần)`.

---

## 3. Chi tiết kỹ thuật & Câu hỏi vấn đáp thường gặp
- **Hỏi: Lệnh `model.eval()` và `torch.no_grad()` khác nhau thế nào?**
  - **Trả lời:**
    - `model.eval()` thay đổi chế độ hoạt động của các lớp đặc biệt như `Dropout` (tắt ngắt kết nối ngẫu nhiên) và `BatchNorm` (sử dụng running mean/variance đã học thay vì tính trên batch hiện tại).
    - `torch.no_grad()` là context manager tắt cơ chế autograd của PyTorch, không lưu trữ đồ thị tính toán đạo hàm, giúp tiết kiệm bộ nhớ và tăng tốc độ suy diễn (inference).
- **Hỏi: Ma trận nhầm lẫn giúp ích gì so với chỉ nhìn vào chỉ số Accuracy?**
  - **Trả lời:** Accuracy chỉ cho biết tỷ lệ đoán đúng tổng thể nhưng không chỉ ra được điểm yếu cụ thể của mô hình. Confusion Matrix cho thấy rõ mô hình hay nhầm lẫn giữa các lớp nào (ví dụ: hay nhầm giữa `cat` và `dog`, hoặc giữa `automobile` và `truck`), từ đó định hướng cải thiện kiến trúc hoặc bổ sung dữ liệu augmentation phù hợp.
