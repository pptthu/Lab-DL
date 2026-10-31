# Experiment log

One entry per experiment: what changed, why, and the result. Numbers come from `artifacts/runs/`.

## 2026-11-01: seven configurations, five seeds
- **Why:** the earlier table had one seed per configuration, and two configurations changed two factors at once.
- **Change:** each pair differs in one factor. Seeds: 42, 7, 123, 2024, 3407. Epochs: 5. Same split. 4 CPU threads.
- **Rule, fixed before reading results:** a pair shows a clear difference if all 5 seeds point the same way and |mean difference| >= 2 std.

| Compared | Factor | Mean difference (val, points) | Std | Seeds higher | Result |
|---|---|---|---|---|---|
| `softmax_regression` vs `baseline` | Hidden layers | -3.75 | 0.45 | 0/5 | Clear |
| `sgd_same_lr` vs `baseline` | Optimizer | -3.33 | 0.34 | 0/5 | Clear |
| `sgd_lr_0.01` vs `sgd_same_lr` | Learning rate | +2.81 | 0.38 | 5/5 | Clear |
| `larger_network` vs `baseline` | Network size | -0.34 | 0.48 | 1/5 | Not clear |
| `dropout_0.2` vs `baseline` | Dropout | -0.68 | 0.46 | 0/5 | Not clear |
| `weight_decay_1e-4` vs `baseline` | Weight decay | -0.30 | 0.37 | 1/5 | Not clear |

- **Baseline test accuracy:** 88.42 +/- 0.37% across 5 seeds. It was fixed beforehand, so the test set chose nothing.
- **Limits:** 5 epochs may favour configurations that learn fast. The rule is practical, not a statistical test.
- **Next:** test the Shirt hypotheses, for example with a small CNN on the same split and seeds.

## 2026-11-01: statistical check and 20 epochs
- **Why:** the 5-epoch rule was only practical, and 5 epochs could hide overfitting or a late effect of dropout.
- **Change:** paired t-test with Holm correction and 95% CI on the six pairs; baseline and dropout at 20 epochs, 5 seeds. Planned CNN dropped for time.
- **Result:** the same three pairs stay clear. Dropout at 5 epochs: p = 0.031, 0.093 after correction. At 20 epochs the baseline overfits (lowest val loss 0.3055, final 0.4020; test 89.11 +/- 0.17%). Dropout: val accuracy +0.08 points, p = 0.71, final val loss 0.3101.
- **Reproducibility:** a second `main.py` run gave identical baseline weights and identical results for all 35 runs.

---

**Tiếng Việt**

# Sổ ghi thí nghiệm

Mỗi mục ghi một thí nghiệm: đổi gì, vì sao, kết quả ra sao. Số liệu lấy từ `artifacts/runs/`.

## 2026-11-01: bảy cấu hình, năm seed
- **Vì sao:** bảng cũ mỗi cấu hình chỉ một seed, và hai cấu hình đổi cùng lúc hai yếu tố.
- **Thay đổi:** mỗi cặp chỉ khác một yếu tố. Seed: 42, 7, 123, 2024, 3407. 5 epoch, cùng split, 4 thread CPU.
- **Quy tắc, chốt trước khi đọc kết quả:** một cặp có khác biệt rõ khi cả 5 seed cùng chiều và |chênh lệch trung bình| >= 2 std.
- **Kết quả:** rõ ở lớp ẩn (-3,75 điểm), optimizer (-3,33) và learning rate (+2,81). Chưa rõ ở kích thước mạng, dropout và weight decay (xem bảng trên).
- **Accuracy test của baseline:** 88,42 +/- 0,37% qua 5 seed. Baseline được chốt trước nên tập test không dùng để chọn gì.
- **Giới hạn:** 5 epoch có thể thiên về cấu hình học nhanh. Quy tắc mang tính thực hành, không phải kiểm định thống kê.
- **Bước tiếp:** kiểm chứng giả thuyết về Shirt, ví dụ bằng CNN nhỏ trên cùng split và seed.

## 2026-11-01: kiểm định và 20 epoch
- **Vì sao:** quy tắc 5 epoch chỉ mang tính thực hành, và 5 epoch có thể che mất overfit hoặc tác động muộn của dropout.
- **Thay đổi:** kiểm định t theo cặp có hiệu chỉnh Holm và khoảng tin cậy 95% cho 6 cặp; baseline và dropout chạy 20 epoch, 5 seed. Bỏ CNN vì thiếu thời gian.
- **Kết quả:** vẫn là 3 cặp khác biệt rõ. Dropout ở 5 epoch: p = 0,031, sau hiệu chỉnh 0,093. Ở 20 epoch baseline overfit (val loss thấp nhất 0,3055, cuối 0,4020; test 89,11 +/- 0,17%). Dropout: val accuracy +0,08 điểm, p = 0,71, val loss cuối 0,3101.
- **Tái lập:** chạy `main.py` lần hai ra trọng số baseline y hệt và kết quả y hệt ở cả 35 lần chạy.
