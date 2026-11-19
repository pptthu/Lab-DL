# eda — Khám phá dữ liệu train (v1)

- **Created**: 2026-11-18T13:00:00+07:00
- **Last Updated**: 2026-11-18T13:00:00+07:00 (Member B)
- **Người phụ trách**: Member A (Muội)
- **Status**: Done

---

## Name
EDA — kiểm tra cân bằng nhãn và độ dài review của tập train

## Background
Slide Practice 3 không nêu bước EDA, và nhóm chưa xác nhận với cô là có bắt buộc hay không. Nhóm vẫn làm để biết hai nhãn có cân bằng không và có số liệu độ dài review khi chọn `max_length` cho `features`.
EDA chỉ chạy trên tập train để không đưa thông tin của validation và test vào quyết định.

## Goals / Purpose
- `run_eda` trả dict `n_samples`, `label_counts`, `length_stats` (`mean`, `std`, `min`, `median`, `p95`, `max`; đơn vị là số từ)
- Lưu 3 hình vào `out_dir`: `label_distribution.png`, `length_distribution.png` (có đường median và p95), `length_by_label.png` (boxplot thêm)
- Không làm:
  - Sửa dataset — chỉ đọc
  - Dùng validation hoặc test để ra quyết định — tránh rò rỉ thông tin
  - Tokenize và chọn `max_length` — việc của `features`; độ dài ở đây tính theo từ, không phải theo token, nên p95 chỉ là gợi ý

## Input / Output
- **Input:** `dataset` là tập **train** có cột `text`, `label` từ [preprocessing_v1.md](preprocessing_v1.md); `out_dir`; tên cột (`text_col`, `label_col`)
- **Output:** dict thống kê; 3 file PNG trong `out_dir` (tự tạo nếu chưa có)
  - `main.py --run-training` ghi hình vào `labs/lab3/artifacts/figures/` (`main.py:85-86`), không commit (`.gitignore:14`). Test và smoke run chỉ dùng thư mục tạm.

## How to do it (general plan)
1. Đọc cột text và cột nhãn; dataset rỗng thì báo lỗi
2. Đếm số mẫu mỗi nhãn
3. Đo độ dài mỗi review theo số từ (tách theo khoảng trắng)
4. Tính `length_stats`
5. Vẽ và lưu 3 hình bằng matplotlib backend `Agg`
6. Smoke run trên tập train thật, ghi hình vào thư mục tạm → normal run trong `main.py`

## Pipeline
```
train (text, label) → đếm nhãn → label_counts
                    → đo độ dài (số từ) → length_stats
→ label_distribution.png, length_distribution.png, length_by_label.png
→ dict {n_samples, label_counts, length_stats}
```

## Detailed plan
- **Chỉ tập train:** mọi quyết định về dữ liệu dựa trên train; test để dành cho `evaluate`
- **Độ dài theo số từ:** `len(str(t).split())` (`eda/eda.py:47`) — không cần tokenizer; số token nhiều hơn số từ
- **Backend `Agg`:** không mở cửa sổ, chỉ lưu file (`:16`)
- **Hàm hiện có:** `run_eda(dataset, out_dir: str, text_col: str = 'text', label_col: str = 'label') -> dict` (`eda/eda.py:21`)
- **Chỗ code chưa khớp plan:** không có
- **Gotchas:**
  - Dataset rỗng thì `ValueError` (`eda/eda.py:40-41`)
  - `boxplot` có nhánh dự phòng cho matplotlib cũ (`labels=`) và mới (`tick_labels=`) (`:79-82`)
  - Test phải ghi hình vào `tmp_path`, không ghi vào `artifacts/figures/` thật

## Success criteria
- **Test:** `python -m pytest labs/lab3/tests/test_eda.py -q` đạt (3 test: hợp đồng đầu ra + 3 file PNG, tên cột tùy chỉnh, dataset rỗng báo lỗi)
- **Smoke run:** `run_eda` trên tập train thật ghi đủ 3 PNG vào thư mục tạm: `python -c "import tempfile; from labs.lab3.preprocessing.preprocessing import load_text_dataset; from labs.lab3.eda.eda import run_eda; tr,_,_=load_text_dataset(); print(run_eda(tr, tempfile.mkdtemp()))"`
- **Normal run:** `n_samples` = 2000, `label_counts` = {0: 1000, 1: 1000}; `length_stats` lệch phải (mean lớn hơn median)

## Kết quả audit (số liệu lần chạy 2026-10-25)
- **Test:** `test_eda.py` → 3 passed ✔ (`progress_status.md:7`)
- **Smoke run:** `run_eda` ghi 3 PNG, `n_samples` = 2000 ✔ (`progress_status.md:7`)
- **Normal run:** notebook ô 7: 2000 mẫu, nhãn {0: 1000, 1: 1000}; số từ mean 226,2, std 167,0, min 12, median 170, p95 585,05, max 999 — lệch phải ✔. Tính theo token, 802 trên 2000 review train (40,1%) đạt `max_length=256` (notebook ô 9, xem [features_v1.md](features_v1.md))

## Links
- Desc lab: [../lab3_description_v1.md](../lab3_description_v1.md) | Khuôn: [../../../lab1/docs/templates/DESC_TEMPLATE_v3.md](../../../lab1/docs/templates/DESC_TEMPLATE_v3.md)
- Module liên quan: [preprocessing_v1.md](preprocessing_v1.md), [features_v1.md](features_v1.md)
- Tiến độ và bằng chứng chạy: [../../progress_status.md](../../progress_status.md) (dòng 7)
- Code: `labs/lab3/eda/eda.py` | Test: `labs/lab3/tests/test_eda.py`
