# Lab 1: Fashion-MNIST Classification with PyTorch MLP

## Technical Report

### Problem & Dataset
Multi-class image classification on Fashion-MNIST (10 apparel categories) using a PyTorch Multi-Layer Perceptron (MLP). The dataset contains Kaggle CSVs (60,000 train, 10,000 test; 28x28 grayscale). A stratified 90/10 split (seed 42) generated 54,000 train and 6,000 validation samples (600/class). Normalization ($\mu = 0.2860, \sigma = 0.3527$) was computed strictly from the training split to prevent data leakage.

### Architecture & Training
- **Model**: Fully connected MLP ($784 \to 256 \to 128 \to 10$, $235,146$ trainable parameters) with ReLU activations and unscaled logits.
- **Optimization**: CrossEntropyLoss averaged per sample; Adam optimizer ($\text{lr} = 10^{-3}$); batch size 64; 5 epochs. Best checkpoint selected via lowest validation loss.
- **Ablation Studies**: Compared baseline against SGD ($\text{lr} = 0.01$), larger network ($512 \to 256 \to 128$, $567\text{k}$ params), and Dropout ($0.2$) with weight decay ($10^{-4}$).

### Empirical Results
- **Validation**: Baseline achieved $88.77\%$ accuracy (loss $0.3194$); in ablation benchmarks, baseline scored $89.05\%$.
- **Test Set**: Evaluated **once** on 10,000 unseen test samples, achieving **$88.32\%$ accuracy** (loss $0.3212$, 8,832/10,000 correct).
- **Diagnostics**: Trouser reached highest accuracy ($98.5\%$) while Shirt was lowest ($71.0\%$) due to silhouette similarity with T-shirt and Pullover. Macro ROC-AUC reached $0.9913$.
- **Ablation Insight**: The larger network doubled parameter count ($567\text{k}$) and inference latency without accuracy gain ($88.10\%$), confirming the baseline as the well-balanced choice.

---

**Tiếng Việt**

# Báo cáo Kỹ thuật: Phân loại Fashion-MNIST bằng MLP (PyTorch)

### Bài toán & Dữ liệu
Phân loại ảnh 10 lớp thời trang Fashion-MNIST bằng mạng MLP trong PyTorch. Dữ liệu gồm 60.000 mẫu train và 10.000 mẫu test (ảnh xám 28x28). Dữ liệu được chia phân tầng 90/10 (seed 42) thành 54.000 mẫu train và 6.000 mẫu validation (600 mẫu/lớp). Tham số chuẩn hóa ($\mu = 0.2860, \sigma = 0.3527$) được tính riêng trên tập train nhằm ngăn chặn rò rỉ phân phối dữ liệu.

### Kiến trúc & Huấn luyện
- **Mô hình**: MLP kết nối đầy đủ ($784 \to 256 \to 128 \to 10$, $235.146$ tham số), hàm kích hoạt ReLU, đầu ra logits thô.
- **Tối ưu hóa**: CrossEntropyLoss tính trung bình theo mẫu; bộ tối ưu Adam ($\text{lr} = 10^{-3}$); batch size 64; 5 epochs. Checkpoint tối ưu được lưu theo validation loss thấp nhất.
- **Thực nghiệm đối sánh**: So sánh baseline với SGD ($\text{lr} = 0.01$), mạng lớn hơn (567k tham số), và điều hòa Dropout (0.2).

### Kết quả Thực nghiệm
- **Validation**: Baseline đạt độ chính xác $88.77\%$ (loss $0.3194$); trong bảng đối sánh đạt $89.05\%$.
- **Tập Test**: Đánh giá **duy nhất 1 lần** trên 10.000 mẫu test, đạt **$88.32\%$ độ chính xác** (loss $0.3212$, 8.832/10.000 ảnh đúng).
- **Chẩn đoán**: Lớp Quần (Trouser) đạt cao nhất ($98.5\%$), lớp Áo sơ mi (Shirt) thấp nhất ($71.0\%$) do nhầm lẫn hình thái với T-shirt và Pullover. Macro ROC-AUC đạt $0.9913$.
- **Đánh giá thực nghiệm**: Mạng lớn hơn tăng gấp đôi tham số và tăng độ trễ nhưng độ chính xác không tăng ($88.10\%$), khẳng định baseline là cấu hình cân bằng hợp lý nhất.

---

## Cấu trúc thư mục & Cách chạy / Directory Structure & Execution

```text
labs/lab1/
├── eda/            # Stage 1: Phân tích khám phá dữ liệu (EDA)
├── preprocessing/  # Stage 2: Đọc CSV & Stratified Split
├── features/       # Stage 3: Train-only Normalization & DataLoaders
├── model/          # Stage 4: Kiến trúc mạng MLP & đếm tham số
├── train/          # Stage 5: Huấn luyện có trọng số & Checkpointing
├── evaluate/       # Stage 6: Đánh giá, Ma trận nhầm lẫn & Báo cáo phân loại
├── visualize/      # Stage 7: Biểu đồ Loss, dự đoán mẫu & Heatmap
├── experiment/     # Stage 8: Thí nghiệm đối sánh (Ablations) & lưu CSV
├── main.py         # Điểm chạy tích hợp toàn bộ pipeline
├── notebook/       # lab1_presentation.ipynb (Notebook trình bày hoàn chỉnh)
└── tests/          # 42 automated unit tests
```

```bash
# 1. Chạy 42 unit tests của Lab 1 / Run all 42 automated tests
pytest labs/lab1/tests -v

# 2. Chạy toàn bộ pipeline tích hợp / Run end-to-end pipeline
python -m labs.lab1.main
```
