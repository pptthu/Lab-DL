# Lab 1: Fashion-MNIST Classification with PyTorch MLP

## Technical Report

### Problem Formulation
Multi-class image classification on Fashion-MNIST (10 apparel classes) using a PyTorch Multi-Layer Perceptron (MLP), following strict software engineering modularity, automated testing, and reproducibility constraints.

### Dataset & Preprocessing
The dataset comprises Kaggle Fashion-MNIST CSVs (60,000 train, 10,000 test; 28x28 grayscale pixels in [0, 255]). A stratified 90/10 split (seed 42) yielded 54,000 train and 6,000 validation samples (exactly 600 per class). Normalization statistics ($\mu = 0.2860, \sigma = 0.3527$) were computed strictly from the training split to avoid data leakage.

### Architecture & Training
- **Model**: Fully connected MLP: $784 \to 256 \to 128 \to 10$ ($235,146$ trainable parameters) with ReLU activations and unscaled logits.
- **Objective & Optimization**: CrossEntropyLoss averaged per sample; Adam optimizer ($\text{lr} = 10^{-3}$, weight decay $= 0.0$); batch size $= 64$.
- **Model Selection**: Checkpoint selection governed exclusively by validation loss.
- **Ablation Studies**: 4 controlled configurations evaluated on validation: baseline vs. SGD ($\text{lr} = 0.01$, momentum $= 0.9$), larger network ($512 \to 256 \to 128$, $567,434$ params), and regularization (Dropout $0.2$, weight decay $10^{-4}$).

### Empirical Results & Analysis
- **Validation**: Baseline achieved highest validation accuracy ($89.05\%$, loss $0.3031$) in 5 epochs ($21.26\text{s}$ on CPU).
- **Test Set**: Evaluated **once** on 10,000 test samples, yielding **$88.99\%$ accuracy** (loss $0.3013$, 8,899/10,000 correct).
- **Error Analysis**: Class accuracy spanned from $98.00\%$ (Trouser) down to $68.33\%$ (Shirt). The confusion matrix confirms primary misclassifications occurred between visually similar upper-body garments (Shirt confused with T-shirt/top 87 times, Pullover 49 times, Coat 36 times).
- **Ablations**: Expanding network size doubled parameter count without improving 5-epoch validation accuracy ($88.10\%$, $34.5\text{s}$), indicating slight underfitting within short training budgets.

### Conclusion
The modular architecture (`eda`, `preprocessing`, `features`, `model`, `train`, `evaluate`, `visualize`, `experiment`) verified through 42 automated unit tests achieves reproducible, production-grade training and evaluation on Fashion-MNIST.

---

**Tiếng Việt**

# Báo cáo Kỹ thuật: Phân loại Fashion-MNIST bằng MLP (PyTorch)

### Bài toán
Phân loại ảnh 10 lớp thời trang trên tập dữ liệu Fashion-MNIST bằng mạng nơ-ron truyền thẳng (MLP) trong PyTorch, tuân thủ kiến trúc module hóa độc lập, kiểm thử đơn vị tự động và tính tái lập thực nghiệm.

### Dữ liệu & Tiền xử lý
Tập dữ liệu Fashion-MNIST gồm 60.000 mẫu train và 10.000 mẫu test (ảnh xám 28x28, 784 điểm ảnh thang [0, 255]). Dữ liệu được phân chia phân tầng (stratified) 90/10 với seed 42 tạo ra 54.000 mẫu train và 6.000 mẫu validation (đúng 600 mẫu/lớp). Tham số chuẩn hóa ($\mu = 0.2860, \sigma = 0.3527$) được tính toán độc quyền trên tập train để chống rò rỉ dữ liệu (data leakage).

### Kiến trúc & Huấn luyện
- **Kiến trúc**: Mạng MLP kết nối đầy đủ: $784 \to 256 \to 128 \to 10$ ($235.146$ tham số huấn luyện), kích hoạt ReLU, đầu ra logits thô.
- **Hàm mất mát & Tối ưu hóa**: CrossEntropyLoss lấy trung bình theo mẫu; bộ tối ưu Adam ($\text{lr} = 10^{-3}$), batch size 64. Trọng số checkpoint tốt nhất được lưu dựa trên validation loss thấp nhất.
- **Thử nghiệm đối sánh (Ablation)**: 4 cấu hình so sánh trên tập validation: baseline, SGD ($\text{lr} = 0.01$), mạng lớn hơn (567k tham số), và cơ chế điều hòa (Dropout $0.2$, weight decay $10^{-4}$).

### Kết quả Thực nghiệm & Đánh giá
- **Validation**: Cấu hình baseline đạt độ chính xác validation cao nhất: **$89.05\%$** (loss $0.3031$) sau 5 epochs ($21.26\text{s}$ trên CPU).
- **Test Set**: Được đánh giá **đúng 1 lần duy nhất** trên 10.000 mẫu test, đạt **$88.99\%$** độ chính xác (loss $0.3013$, 8.899/10.000 ảnh đúng).
- **Phân tích sai số**: Phân loại tốt nhất ở lớp Quần (Trouser: $98.00\%$) và thấp nhất ở lớp Áo sơ mi (Shirt: $68.33\%$). Ma trận nhầm lẫn chỉ ra sự nhầm lẫn tập trung giữa các loại áo có hình dạng tương tự (Shirt bị nhầm sang T-shirt 87 lần, Pullover 49 lần, Coat 36 lần).
- **Thí nghiệm đối sánh**: Mạng lớn hơn tăng gấp đôi tham số nhưng trong 5 epochs đạt $88.10\%$ và mất $34.5\text{s}$, cho thấy dung lượng mạng lớn cần thời gian hội tụ dài hơn.

### Kết luận
Toàn bộ hệ thống gồm 8 module độc lập vượt qua 42 unit test tự động (dữ liệu giả lập), tái lập kết quả ổn định và đạt hiệu năng cân bằng tối ưu giữa độ chính xác, số lượng tham số và tốc độ tính toán.

---

## Cấu trúc thư mục & Cách chạy

```bash
# 1. Chạy toàn bộ 42 unit tests
pytest labs/lab1/tests -v

# 2. Chạy toàn bộ pipeline tích hợp từ đầu đến cuối
python -m labs.lab1.main
```
