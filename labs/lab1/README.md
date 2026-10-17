# Lab 1: Fashion-MNIST Classification with PyTorch MLP

## Technical Report

### Problem & Data Split
Multi-class image classification on Fashion-MNIST (10 categories) using a PyTorch Multi-Layer Perceptron (MLP). The dataset consists of Kaggle CSVs (60,000 train, 10,000 test; 28x28 grayscale). A stratified 90/10 split (seed 42) created 54,000 train and 6,000 validation samples (600/class). Normalization ($\mu = 0.2860, \sigma = 0.3527$) was computed strictly on training data.

### Architecture & Training
The model is an MLP ( \to 256 \to 128 \to 10$, ,146$ parameters) with ReLU activations and unscaled logits. It was trained for 5 epochs using Adam ($	ext{lr} = 10^{-3}$, batch size 64) with sample-weighted CrossEntropyLoss. The checkpoint was saved at lowest validation loss.

### Empirical Results & Visualizations
- **Performance**: Baseline achieved .77\%$ validation accuracy (loss .3194$) and .32\%$ test accuracy (loss .3212$, 8,832/10,000 correct).
- **Ablations**: A 4-configuration ablation (baseline .05\%$, SGD .57\%$, larger network .10\%$, dropout .47\%$) showed that variations under $\approx 0.5-1.0$ points reflect run-to-run noise rather than proven superiority.
- **Artifacts**: Loss curves are saved at rtifacts/figures/loss_curves.png, prediction samples at rtifacts/figures/predictions_grid.png, and the confusion matrix at rtifacts/figures/confusion_matrix.png.

### How to Run
`ash
python -m labs.lab1.main                              # Run end-to-end pipeline
jupyter notebook labs/lab1/notebook/lab1_presentation.ipynb  # Interactive presentation
`

---

**Tiếng Việt**

# Báo cáo Kỹ thuật: Phân loại Fashion-MNIST bằng MLP (PyTorch)

### Bài toán & Phân chia Dữ liệu
Phân loại ảnh 10 lớp trang phục Fashion-MNIST bằng mạng MLP trong PyTorch. Dữ liệu gồm các file CSV từ Kaggle (60.000 train, 10.000 test; 28x28). Phân chia phân tầng 90/10 (seed 42) tạo ra 54.000 mẫu train và 6.000 mẫu validation (600 mẫu/lớp). Tham số chuẩn hóa ($\mu = 0.2860, \sigma = 0.3527$) được tính riêng trên tập train.

### Kiến trúc & Huấn luyện
Mạng MLP ( \to 256 \to 128 \to 10$, .146$ tham số) dùng kích hoạt ReLU và logits thô. Huấn luyện 5 epochs bằng Adam ($	ext{lr} = 10^{-3}$, batch size 64) với CrossEntropyLoss theo trọng số mẫu. Checkpoint tối ưu lưu theo validation loss thấp nhất.

### Kết quả Thực nghiệm & Đồ thị
- **Hiệu năng**: Baseline đạt độ chính xác validation .77\%$ (loss .3194$) và test .32\%$ (loss .3212$, 8.832/10.000 ảnh đúng).
- **Đối sánh**: Thử nghiệm 4 cấu hình (baseline .05\%$, SGD .57\%$, mạng lớn .10\%$, dropout .47\%$) cho thấy dao động dưới $\approx 0.5-1.0$ điểm phản ánh sai số ngẫu nhiên giữa các lần chạy.
- **Đồ thị**: Lưu tại rtifacts/figures/loss_curves.png, rtifacts/figures/predictions_grid.png, và rtifacts/figures/confusion_matrix.png.

### Cách chạy
`ash
python -m labs.lab1.main                              # Chạy toàn bộ pipeline
jupyter notebook labs/lab1/notebook/lab1_presentation.ipynb  # Mở notebook trình bày
`
