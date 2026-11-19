# eda — Khám phá dữ liệu CIFAR-10 (v1)

- **Created**: 2026-11-04T14:49:03+07:00
- **Last Updated**: 2026-11-19T13:35:00+07:00 (Đại)
- **Người phụ trách**: Member A (Đại)
- **Status**: Done

---

## Name
EDA — kiểm tra tập CIFAR-10, phân bố 10 lớp, ảnh mẫu, thống kê kênh màu

## Background
Đề Practice 2: "Pre-trained CNNs and fine-tuning on CIFAR-10". Trước khi nạp mô hình và chia tập huấn luyện, cần khảo sát bộ dữ liệu gốc: số lượng mẫu, kích thước ảnh, độ cân bằng giữa các lớp và giá trị thống kê kênh màu (RGB).
EDA chỉ đọc dữ liệu thô nguyên bản (`transform=None`), không can thiệp tiền xử lý hay thay đổi cấu trúc pixel.

## Goals / Purpose
- `run_eda` trả dict thống kê: `n_train` (50.000), `n_test` (10.000), `class_counts` (5.000 mẫu/lớp trên train), `image_size` ((32, 32, 3)), `channel_mean` (~[0.4914, 0.4822, 0.4465]), `channel_std` (~[0.2470, 0.2435, 0.2616])
- Lưu 2 hình PNG: `cifar10_class_distribution.png` (biểu đồ cột phân bố lớp) và `cifar10_sample_grid.png` (lưới ảnh 10 lớp × 3 mẫu)
- Pipeline `main.py` ghi nhận kết quả ra file `artifacts/figures/eda_summary.json`
- Không làm:
  - Áp dụng transforms, chuẩn hoá hay data augmentation — việc của `features`
  - Chia tập train/val — việc của `preprocessing`
  - Sửa đổi hoặc lưu lại dữ liệu ảnh — module chỉ đọc (read-only)

## Input / Output
- **Input:** `data_root` (đường dẫn thư mục chứa CIFAR-10 thô từ torchvision), `out_dir` (thư mục xuất hình ảnh)
- **Output:** dict thống kê; 2 file PNG trực quan hoá
  - Code đọc qua `torchvision.datasets.CIFAR10` (`eda/eda.py:47-48`), ghi hình vào `artifacts/figures/` (`main.py:174`, `main.py:182-184`).

## How to do it (general plan)
1. Tải và nạp raw CIFAR-10 (`train=True`, `train=False`) không kèm transform bằng `torchvision.datasets.CIFAR10`
2. Đếm số lượng ảnh, kiểm tra tính cân bằng 10 lớp từ `train_raw.targets`, xác định kích thước ảnh gốc `(32, 32, 3)`
3. Tính mean và std trên 3 kênh màu RGB trên thang `[0, 1]` của tập train
4. Vẽ lưới ảnh 10 hàng × 3 cột hiển thị mẫu ngẫu nhiên cho từng lớp (`cifar10_sample_grid.png`)
5. Vẽ biểu đồ cột phân bố tần suất 10 lớp trên tập train (`cifar10_class_distribution.png`)
6. Smoke run: chạy `run_eda` với thư mục tạm, kiểm tra dict trả về và 2 file PNG hợp lệ
7. Normal run trong pipeline `main.py`: lưu biểu đồ vào `artifacts/figures/` và ghi `eda_summary.json`

## Pipeline
```
data_root → CIFAR10(raw) → đếm lớp + tính mean/std (dict)
  → vẽ bar chart → cifar10_class_distribution.png
  → vẽ lưới ảnh mẫu → cifar10_sample_grid.png
  → main.py lưu artifacts/figures/eda_summary.json
```

## Detailed plan
- **Chỉ tính thống kê trên tập train:** Mọi đặc trưng phân phối kênh màu được tính trên tập train; tập test giữ nguyên cho `evaluate` để tránh rò rỉ thông tin
- **Backend vẽ không tương tác (Agg):** Ép dùng `matplotlib.use("Agg")` khi chạy headless/script ngoài Jupyter để tránh lỗi giao diện GUI (`eda/eda.py:19-20`)
- **Tự động tạo thư mục:** Dùng `Path(out_dir).mkdir(parents=True, exist_ok=True)` đảm bảo thư mục đích luôn sẵn sàng
- **Hàm hiện có:** `run_eda(data_root: str, out_dir: str) -> dict` (`eda/eda.py:33`); hằng số `CIFAR10_CLASSES` (`:27-30`)
- **Chỗ code chưa khớp plan:** không có (mọi hàm và đầu ra đều khớp hợp đồng giao diện)
- **Gotchas:**
  - Định dạng mảng ảnh của `train_raw.data` là `(50000, 32, 32, 3)` dạng `uint8`, tính mean/std cần ép kiểu `np.float32 / 255.0` và tính trung bình theo `axis=(0, 1, 2)` (`eda/eda.py:62-64`)
  - Giá trị mean/std tính từ CIFAR-10 dùng để quan sát đặc tính dữ liệu; khi đưa vào backbone pre-trained bắt buộc dùng ImageNet mean/std ở module `features`

## Success criteria
- **Test:** `python -m pytest labs/lab2/tests/test_eda.py -q` đạt (1 test)
- **Smoke run:** `run_eda` sinh đủ 2 file PNG (`cifar10_class_distribution.png`, `cifar10_sample_grid.png`) có kích thước > 0 bytes
- **Normal run:** `n_train` = 50000, `n_test` = 10000, 10 lớp mỗi lớp đúng 5000 ảnh, `image_size` = (32, 32, 3), channel_mean ~ [0.4914, 0.4822, 0.4465], channel_std ~ [0.2470, 0.2435, 0.2616]

## Links
- Desc lab: [../project_description.md](../project_description.md)
- Module liên quan: [preprocessing.md](preprocessing.md), [features.md](features.md)
- Tiến độ và bằng chứng chạy: [../progress_status.md](../progress_status.md) (mục Audit dòng 27, Log dòng 46)
- Code: `labs/lab2/eda/eda.py` | Test: `labs/lab2/tests/test_eda.py`
