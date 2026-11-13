# Lab 1 — Phân loại Fashion-MNIST bằng MLP  (v1)
- Tác giả: Thư | Tạo: 2026-11-13T11:22:06+07:00 | Cập nhật: Thư, 2026-11-14T00:55:48+07:00
- Trạng thái: To Do (train lại theo log và quy tắc dừng mới)
## Background
Đề P1: load Fashion-MNIST, build NN, train, đánh giá accuracy, save/load, thử network/hyperparameter, vẽ loss, hiện ảnh dự đoán. Bản cũ chạy 5 epoch cố định, không có log theo lần chạy, lưu model trong `artifacts/`, nên train lại theo luật mới.
## Purpose
- Làm: train 5 cấu hình MLP, seed 42, tối đa 30 epoch + early stopping (`templates/LOG_TEMPLATE_v1.md`). Batch 64, các cấu hình khác baseline đúng một yếu tố:
  1. `baseline`: hidden (256,128), Adam, lr 1e-3
  2. `softmax_regression`: không hidden layer
  3. `larger_network`: hidden (512,256,128)
  4. `sgd_lr_0.01`: SGD momentum 0.9, lr 0.01
  5. `dropout_0.2`: dropout 0.2 sau mỗi hidden layer
- Không làm: CNN, Optuna, đa seed, p-value/CI95, test trên ảnh ngoài dataset, đo tốc độ inference.
## Input / Output
- Input: `data/raw/fashion-mnist_train.csv` (60000 dòng), `fashion-mnist_test.csv` (10000 dòng); mỗi dòng label + 784 pixel. Split: `data/splits/train_val_split.json` (54000 train / 6000 val).
- Output: `outputs/{models,logs,figures,results,demo}/` (xem `rules/FOLDER_STRUCTURE_v1.md`); model card ở `docs/models/`.
- Code: `labs/lab1/main.py` gọi 8 module `labs/lab1/<module>/<module>.py`.
## Success criteria
- Mọi run có đủ 3 file log và `_best.pt` nạp lại dự đoán được.
- Baseline đạt test ≥ 87% trên tập test chính thức 10k — mốc MLP tốt nhất trong bài báo gốc Fashion-MNIST (Xiao et al. 2017, Bảng 3: 0,871). Bản cũ của nhóm: 88,00 ± 0,39% (`artifacts/runs/seed_results.csv`).
- Baseline có val_acc cao hơn softmax_regression (bài báo gốc: MLP 0,871 so với LogisticRegression 0,842).
- Có đủ: sơ đồ kiến trúc trong model card, loss curve train+val vẽ từ `_history.csv`, ảnh dự đoán trên test, confusion matrix, bảng so sánh 5 cấu hình (có cột train_acc − val_acc, chỉ để báo cáo).
## How-to-do
1. Load data, kiểm tra split (`preprocessing`), xem phân bố lớp (`eda`).
2. Chuẩn hoá theo mean/std của train, tạo loader (`features`).
3. Build MLP, điền phần Initial model và sơ đồ kiến trúc (mermaid) trong model card (`model`).
4. Thêm log 3 file và quy tắc dừng vào train (`train`), smoke run.
5. Normal run 5 cấu hình (`experiment`), chọn cấu hình trên val.
6. Đánh giá test một lần cho baseline và cho cấu hình được chọn (`evaluate`); vẽ biểu đồ từ log và demo bằng `_best.pt` (`visualize`).
7. Điền Trained model trong model card, audit code so với desc này.
## Quyết định (người ghi)
- 2026-11-04: chỉ dùng MLP, không CNN.
- 2026-11-12: dừng theo `val_loss`, patience 5, `min_delta` 1e-4, tối đa 30 epoch — `val_loss` nhạy hơn acc; luôn giữ best checkpoint.
- 2026-11-13: còn 5 cấu hình × seed 42, bỏ đa seed và kiểm định — đề không yêu cầu, giữ desc ngắn.
- 2026-11-14: mức test ≥ 87% theo bài báo gốc Fashion-MNIST; test chạy cho cả baseline và cấu hình được chọn; sơ đồ kiến trúc vẽ bằng mermaid trong model card.
