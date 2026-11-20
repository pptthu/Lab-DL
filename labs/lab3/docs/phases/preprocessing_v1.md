# preprocessing — tải IMDB gốc, làm sạch, chia train/val theo phim (v1)

- **Created**: 2026-11-20T01:00:14+07:00
- **Last Updated**: 2026-11-20T02:32:51+07:00 (Thư)
- **Người phụ trách**: Thư
- **Status**: In Progress — code, test, smoke và normal run xong (2026-11-20); chờ Thư audit

---

## Name
Nạp dữ liệu (Bài 2.2) — tải `aclImdb_v1.tar.gz`, đọc review + nhãn + ID phim, làm sạch, chia val 10% theo phim

## Background
Bài 2.2 của đề: nạp dataset phân loại văn bản nhị phân. Plan L1–L4: IMDB 25k train / 25k test chính thức; val 10% tách từ train **theo phim** vì README của bộ dữ liệu ghi train và test là hai tập phim không trùng nhau. Đầu ra dùng cho `eda`, `features`, `evaluate`.

## Goals / Purpose
- `download_imdb()` tải và giải nén vào `data/raw/aclImdb/` (bỏ qua nếu đã có)
- `load_split(split)` trả bảng `id`, `text`, `label` (pos = 1, neg = 0), `rating`, `movie_id` cho `train` / `test`
- `make_train_val_split(seed=42)` ghi `data/splits/train_val_split.json`: val ≈ 2.500 review, cân bằng nhãn, **không có phim nào và không có nội dung review trùng nào nằm ở cả train và val**
- `load_datasets()` trả `DatasetDict` `train` / `validation` / `test` (thư viện `datasets`) đã làm sạch
- Không làm:
  - Thống kê, vẽ hình — thuộc `eda`
  - Tokenize — thuộc `features`
  - Dùng 50k review không nhãn (`unsup/`) — ngoài phạm vi (plan §7)

## Input / Output
- **Input:** `https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz` (lần đầu cần mạng)
- **Output:** `data/raw/aclImdb/` (không commit); `data/splits/train_val_split.json` (commit); `DatasetDict` trong bộ nhớ

## How to do it (general plan)
1. Tải + giải nén; kiểm số file: `train/pos`, `train/neg`, `test/pos`, `test/neg` mỗi thư mục 12.500
2. Đọc review: tên file `<id>_<rating>.txt` → `id`, `rating`; nhãn theo thư mục
3. Đọc ID phim từ `urls_pos.txt` / `urls_neg.txt`: dòng thứ i là URL trang review của phim chứa review id i (URL dạng `.../title/tt<số>/...`). **Kiểm**: số dòng = số review, mọi URL có mã `tt…`; sai thì **dừng và báo Thư**, không tự đổi cách chia
4. Làm sạch: thay thẻ `<br />` bằng khoảng trắng, gộp khoảng trắng thừa; không bỏ stopword, không hạ chữ thường (tokenizer uncased tự làm)
5. Nhóm khi chia = phim; các review **trùng nội dung** (băm văn bản đã làm sạch) được gộp vào cùng một nhóm để cả cặp nằm cùng một phía
6. Chia val: `StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)` (scikit-learn), lấy 1 fold làm val
7. Ghi split JSON; tạo `DatasetDict`
8. Test với vài file giả trong thư mục tạm
9. Smoke run → lấy vài chục review mỗi split; normal run trên full data (máy Thư)

## Pipeline
```
aclImdb_v1.tar.gz → download_imdb → data/raw/aclImdb/
  → load_split (+ movie_id từ urls_*.txt) → làm sạch
  → make_train_val_split (theo phim, cân bằng nhãn) → data/splits/train_val_split.json
  → load_datasets → DatasetDict → eda, features, evaluate
```

## Detailed plan
- **Nguồn dữ liệu:** bản gốc Stanford thay vì `load_dataset("imdb")` — cần ID phim cho bước chia theo phim (plan L4)
- **Val 10%, theo phim:** plan L3; `StratifiedGroupKFold` giữ cân bằng nhãn và không để một phim rơi vào cả hai phía. 10 fold → mỗi fold ≈ 10%
- **Làm sạch tối thiểu (Thư duyệt 2026-11-20):** chỉ bỏ `<br />` — là thẻ HTML, tokenizer tách thành nhiều token (`<`, `br`, `/`, `>`) làm tốn chỗ trong `max_length` (`eda` đo số token tốn). Không bỏ stopword: danh sách stopword phổ biến chứa `not`, `no`, `nor` → bỏ là xoá phủ định. Tổng quan arXiv 2501.00174 trích Kurniasih & Manik (2022): tiền xử lý ảnh hưởng không đáng kể với model dùng BERT (trích lại, chưa đọc bản gốc)
- **Review trùng nội dung:** README chỉ đảm bảo train/test khác phim, không đảm bảo khác nội dung → gộp nhóm khi chia train/val (tránh lộ dữ liệu sang val); test giữ nguyên (xem `evaluate`)
- **Ánh xạ dòng ↔ id:** README aclImdb (qua IEEE DataPort, Visual Studio Magazine, textacy): *"a review with unique id 200 will have its URL on line 200"*; URL trỏ tới trang review của phim, không phải review riêng
- **Gotchas:**
  - Mỗi phim tối đa 30 review (README aclImdb) → các fold theo phim có kích thước gần bằng nhau, nhưng không đúng chính xác 2.500
  - File review có thể chứa ký tự không phải ASCII → đọc bằng `encoding="utf-8"`

## Triển khai (2026-11-20)
- **Hàm** (`labs/lab3/preprocessing/preprocessing.py`): `download_imdb(root=None, url=config.IMDB_URL) -> Path` (`:37`); `read_movie_ids(urls_file) -> list[str]` (`:60`); `load_split(split, root=None) -> DataFrame` (`:75`); `clean_text(text) -> str` (`:112`); `text_hash(text) -> str` (`:121`); `split_groups(df) -> Series` (`:127`); `make_train_val_split(train, seed=42, n_folds=10) -> dict` (`:158`); `save_split` / `read_split` (`:189`, `:197`); `load_datasets(root=None, split_path=None, smoke_n=None) -> DatasetDict` (`:202`). Chạy: `python -m labs.lab3.preprocessing.preprocessing [--force]`
- **Chỗ code khác plan:** (1) `load_split` lưu bộ đệm `data/raw/imdb_<split>.parquet` (git bỏ qua) — lần đọc đầu 50k file nhỏ mất 260 s + 237 s trên máy Thư (Windows Defender quét từng file), từ lần sau 2 s; (2) thêm `sample_per_label` cho smoke run; (3) val = 2.501 chứ không đúng 2.500 vì chia theo nhóm
- **Kết quả normal run:** tải 84.125.825 byte; 4 thư mục × 12.500 file, số dòng URL khớp. Train 3.456 phim, test 3.581 phim, tối đa 30 review/phim (đúng README). 1 phim ở cả train và test (`tt0406816`, 2 review train) — ghi ở `docs/experiments/experiment_log.md` S1. Split (`data/splits/train_val_split.json`): train 22.499 / val 2.501, tỉ lệ dương ở val 0,5002, 346 phim ở val, `groups_in_both` = 0, 0 review trùng nội dung giữa train và val; 18 phim được gộp nhóm vì có review trùng
- **Test / smoke:** `pytest labs/lab3/tests/test_preprocessing.py` 9 passed (cây aclImdb giả)

## Success criteria
- **Test:** `pytest labs/lab3/tests/test_preprocessing.py` đạt: đọc đúng id/rating/nhãn, bỏ hết `<br />`, không phim nào và không review trùng nào ở cả train và val, val gần 10%, số dòng URL sai thì báo lỗi
- **Smoke run:** `DatasetDict` có 3 split, đủ cột
- **Normal run:** train + val = 25.000, test = 25.000; val ≈ 2.500 (cân bằng nhãn ± 1%); giao giữa `movie_id` train và val = rỗng; split JSON tồn tại

## Links
- Desc lab: [../lab3_description_v1.md](../lab3_description_v1.md) | Plan: [../plans/detail_plan_lab3_md_v1.md](../plans/detail_plan_lab3_md_v1.md)
- Module liên quan: [eda_v1.md](eda_v1.md), [features_v1.md](features_v1.md)
- Nguồn: trang Stanford https://ai.stanford.edu/~amaas/data/sentiment/ ; README aclImdb (trong file nén)
- Code: `labs/lab3/preprocessing/preprocessing.py` | Test: `labs/lab3/tests/test_preprocessing.py`
