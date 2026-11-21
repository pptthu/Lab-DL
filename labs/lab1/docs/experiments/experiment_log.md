# Sổ ghi thí nghiệm

- **Created**: 2026-11-01T02:08:09+07:00 (pptthu)
- **Last Updated**: 2026-11-14T01:43:54+07:00 (pptthu)

Mỗi mục ghi một thí nghiệm: đổi gì, vì sao, kết quả ra sao. Số liệu lấy từ `outputs/results/`. Các mục trước 14/11 lấy từ `artifacts/runs/` cũ, đã xoá ở O5 — đổi thư mục `artifacts/`, `raw_data/` sang `data/`, `outputs/`.

## 14/11/2026: 5 cấu hình, seed 42, early stopping (desc mới)
- **Vì sao:** cô nhận xét bản cũ chạy cố định 5 epoch, không có log theo lần chạy, không rõ khi nào train xong. Desc mới `docs/lab1_description_v1.md` thay thí nghiệm 7 cấu hình × 5 seed bằng 5 cấu hình × seed 42, có quy tắc dừng.
- **Thay đổi:** tối đa 30 epoch, dừng khi `val_loss` không giảm ít nhất 1e-4 trong 5 epoch, giữ `_best.pt`; mỗi run ghi `.log`, `_history.csv`, `_config.json`. Bỏ `sgd_same_lr`, `weight_decay_1e-4`, đa seed và kiểm định (O10). Cùng data, cùng split, cùng 4 thread CPU.
- **Kết quả** (`outputs/results/experiment_results.csv`, `test_results.csv`): cả 5 cấu hình dừng sớm ở epoch 11–15. Val_acc tại best_epoch: baseline 88,97%, softmax_regression 85,15%, larger_network 89,28%, sgd_lr_0.01 89,13%, dropout_0.2 89,00%. Chọn larger_network theo val; test: baseline 88,60% (≥ 87% ✔), larger_network 88,46%. Lớp ẩn vẫn là yếu tố rõ nhất (+3,8 điểm so với mô hình tuyến tính), giống kết luận cũ; các cấu hình còn lại chênh baseline ≤ 0,31 điểm với một seed nên chưa kết luận. Dropout đưa train − val về −0,24 điểm (baseline 2,59).

## 06/11/2026: lưu lịch sử theo epoch
- **Vì sao:** trước đây mỗi lần chạy chỉ giữ giá trị tốt nhất và cuối cùng, nên các giải thích như "SGD lr 1e-3 vẫn đang học" hay "baseline overfit sau epoch 7" không có đường cong làm bằng chứng.
- **Thay đổi:** `run_experiments` giữ lịch sử của từng lần chạy; `save_history` ghi `epoch_history.csv` (35 lần chạy × 5 epoch) và `extended_epoch_history.csv` (10 lần chạy × 20 epoch). Data, cấu hình, seed và quy tắc giữ nguyên.
- **Kết quả:** 12 file CSV kết quả giống hệt lần chạy 04/11 (trừ cột thời gian); lịch sử khớp epoch tốt nhất, giá trị tốt nhất và cuối của từng lần chạy (0 sai lệch trên 45 lần chạy). Đường cong cho thấy: mô hình tuyến tính đứng quanh 84% từ epoch 2; SGD lr 1e-3 vẫn đang tăng ở epoch 5; baseline 20 epoch có val loss trung bình thấp nhất ở epoch 7 (0,3028) rồi tăng lên 0,3972 trong khi train loss giảm xuống 0,1240; dropout giữ val loss trong khoảng 0,2984 đến 0,3100 từ epoch 7 đến 20.

## 04/11/2026: chạy lại trên data chính thức của Zalando
- **Vì sao:** CSV Kaggle khác file chính thức (xem README); các mục ngày 01/11 bên dưới đo trên CSV đó.
- **Thay đổi:** chỉ đổi data (file chính thức, cùng định dạng CSV, cùng quy tắc chia và seed). Code, cấu hình, seed và quy tắc giữ nguyên.
- **Kết quả:** các tác động rõ giữ nguyên: bỏ lớp ẩn -3,80, SGD so với Adam ở lr 1e-3 -3,45, SGD lr 0,01 so với 1e-3 +3,14 điểm. Mạng lớn hơn -0,52, dropout -0,46, weight decay -0,29: chưa rõ (không qua quy tắc; p sau hiệu chỉnh Holm 0,084; 0,084; 0,059). Baseline test 88,00 ± 0,39%. 20 epoch: baseline overfit (val loss thấp nhất 0,2988, cuối 0,3972), test 88,62 ± 0,19%; dropout giữ val loss cuối thấp hơn (0,3095, ở cả 5 seed) nhưng chưa tăng accuracy rõ (+0,36 điểm, p Holm 0,15).

## 01/11/2026: bảy cấu hình, năm seed
- **Vì sao:** bảng cũ mỗi cấu hình chỉ một seed, và hai cấu hình đổi cùng lúc hai yếu tố.
- **Thay đổi:** mỗi cặp chỉ khác một yếu tố. Seed: 42, 7, 123, 2024, 3407. 5 epoch, cùng split, 4 thread CPU.
- **Quy tắc, chốt trước khi đọc kết quả:** một cặp có khác biệt rõ khi cả 5 seed cùng chiều và |chênh lệch trung bình| ≥ 2 std.

| So sánh | Yếu tố | Chênh lệch trung bình (val, điểm) | Std | Số seed cao hơn | Kết luận |
|---|---|---|---|---|---|
| `softmax_regression` so với `baseline` | Lớp ẩn | -3,75 | 0,45 | 0/5 | Rõ |
| `sgd_same_lr` so với `baseline` | Optimizer | -3,33 | 0,34 | 0/5 | Rõ |
| `sgd_lr_0.01` so với `sgd_same_lr` | Learning rate | +2,81 | 0,38 | 5/5 | Rõ |
| `larger_network` so với `baseline` | Kích thước mạng | -0,34 | 0,48 | 1/5 | Chưa rõ |
| `dropout_0.2` so với `baseline` | Dropout | -0,68 | 0,46 | 0/5 | Chưa rõ |
| `weight_decay_1e-4` so với `baseline` | Weight decay | -0,30 | 0,37 | 1/5 | Chưa rõ |

- **Accuracy test của baseline:** 88,42 ± 0,37% qua 5 seed. Baseline được chốt trước nên tập test không dùng để chọn gì.
- **Giới hạn:** 5 epoch có thể thiên về cấu hình học nhanh. Quy tắc mang tính thực hành, không phải kiểm định thống kê.
- **Bước tiếp:** kiểm chứng giả thuyết về Shirt, ví dụ bằng CNN nhỏ trên cùng split và seed.

## 01/11/2026: kiểm định và 20 epoch
- **Vì sao:** quy tắc 5 epoch chỉ mang tính thực hành, và 5 epoch có thể che mất overfit hoặc tác động muộn của dropout.
- **Thay đổi:** kiểm định t theo cặp có hiệu chỉnh Holm và khoảng tin cậy 95% cho 6 cặp; baseline và dropout chạy 20 epoch, 5 seed. Bỏ CNN vì thiếu thời gian.
- **Kết quả:** vẫn là 3 cặp khác biệt rõ. Dropout ở 5 epoch: p = 0,031, sau hiệu chỉnh 0,093. Ở 20 epoch baseline overfit (val loss thấp nhất 0,3055, cuối 0,4020; test 89,11 ± 0,17%). Dropout: val accuracy +0,08 điểm, p = 0,71, val loss cuối 0,3101.
- **Tái lập:** chạy `main.py` lần hai ra trọng số baseline y hệt và kết quả y hệt ở cả 35 lần chạy.
