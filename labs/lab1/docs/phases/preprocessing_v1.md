# preprocessing — Tải dữ liệu và chia train/val (v1)

- **Created**: 2026-11-13T23:36:00+07:00
- **Last Updated**: 2026-11-13T23:36:00+07:00 (Thư)
- **Người phụ trách**: Member A (Toàn)
- **Status**: Done (chờ đổi đường dẫn ở giai đoạn O5)

---

## Name
Preprocessing — tải Fashion-MNIST chính thức, đọc CSV, chia train/val cố định

## Background
Đề P1: "load FashionMNIST". Nhóm dùng file chính thức của Zalando (kiểm MD5) thay cho CSV Kaggle.
Val phải tách từ train, test giữ nguyên 10.000 ảnh để đánh giá cuối.

## Goals / Purpose  [đề xuất]
- Có 2 file CSV: train 60.000 dòng (6.000 mỗi lớp), test 10.000 dòng (1.000 mỗi lớp), cột `label` + `pixel1..pixel784`
- `load_csv` trả ảnh `[N,28,28]` uint8 và nhãn `[N]` int64
- File split `train_val_split.json` cố định: 54.000 train / 6.000 val (600 mỗi lớp), seed 42
- Không làm:
  - Lấy val từ file test — test chỉ dùng ở `evaluate`
  - Chuẩn hoá ảnh — việc của `features`
  - Đổi `val_ratio` hay `seed` — mọi số liệu đã báo cáo phụ thuộc vào file split này
  - Commit file CSV hay IDX lên git

## Input / Output
- **Input:** 4 file IDX gốc tải bằng `torchvision.datasets.FashionMNIST(download=True)`
- **Output:** `fashion-mnist_train.csv`, `fashion-mnist_test.csv`; `train_val_split.json` (`train_idx`, `val_idx`)
  - Hiện code ghi CSV vào `raw_data/` (`shared/paths.py:16-18`), split vào `artifacts/splits/train_val_split.json` (`main.py:67`, `:83`). Sẽ chuyển sang `data/raw/`, `data/splits/` ở giai đoạn O5.

## How to do it (general plan)
1. Chạy `python -m labs.lab1.preprocessing.download_official` để tải và ghi 2 CSV, script tự đọc lại để so từng pixel
2. `load_csv` đọc CSV, kiểm cột và miền giá trị 0–255
3. `make_split(labels, val_ratio=0.1, seed=42)` chia phân tầng theo lớp
4. `save_split` ghi JSON; `main.py` dùng lại file đã commit nếu có (`main.py:84-88`)
5. Smoke run: tạo lại split từ CSV và so với file đã commit
6. Normal run trong pipeline `main.py`

## Pipeline
```
torchvision FashionMNIST (MD5) → download_official → 2 CSV
  → load_csv → make_split (phân tầng, seed 42) → save_split → train_val_split.json
  → features
```

## Detailed plan
- **Chia phân tầng 90/10:** mỗi lớp lấy 10% làm val — val cân bằng 600 ảnh mỗi lớp
- **Seed 42 và file split commit:** để mọi người, mọi lần chạy dùng đúng một cách chia
- **Hàm hiện có:**
  - `load_csv(csv_path: str) -> tuple` (`preprocessing/preprocessing.py:25`)
  - `make_split(labels, val_ratio: float = 0.1, seed: int = 42) -> tuple` (`:50`)
  - `save_split(train_idx, val_idx, path: str) -> None` (`:77`); `load_split(path: str) -> tuple` (`:99`)
- **Chỗ code chưa khớp plan:** chỉ còn đường dẫn, chờ O5
- **Gotchas:**
  - Kiểm miền 0–255 **trước** khi ép kiểu `uint8`, nếu không giá trị sai bị cắt im lặng (`preprocessing/preprocessing.py:44`; `preprocessing.md` cũ)
  - Windows + đường dẫn có tiếng Việt: đặt `PYTHONIOENCODING=utf-8`, nếu không lệnh `print` cuối có thể lỗi `UnicodeEncodeError` (file vẫn đã ghi) (`preprocessing.md` cũ)
  - Không tự viết đường dẫn dữ liệu, dùng `raw_data_dir("lab1")` (`preprocessing.md` cũ)
  - Chỉ có một cách chia; kết quả trên cách chia khác chưa đo (`preprocessing.md` cũ)

## Success criteria  [đề xuất]
- **Test:** `python -m pytest labs/lab1/tests/test_preprocessing.py -q` đạt (6 test, CSV giả, không tải mạng)
- **Smoke run:** `make_split(labels, 0.1, 42)` trên CSV vừa tạo cho ra đúng `train_val_split.json` đã commit
- **Normal run:** MD5 CSV train `8ff2cced1c624159b21dac4f735b4215`, test `78e18be2098f1256a07ce6f91a4d436e`; split 54.000 / 6.000

## Links
- Desc lab: [../lab1_description_v1.md](../lab1_description_v1.md)
- Module liên quan: [eda_v1.md](eda_v1.md), [features_v1.md](features_v1.md)
- Tiến độ và bằng chứng chạy: [../../progress_status.md](../../progress_status.md) (tải và chia lại 2026-11-02, dòng 39–41, 49)
- Code: `labs/lab1/preprocessing/preprocessing.py`, `labs/lab1/preprocessing/download_official.py` | Test: `labs/lab1/tests/test_preprocessing.py`
