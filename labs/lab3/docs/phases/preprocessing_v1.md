# preprocessing — Nạp IMDB và làm sạch văn bản (v1)

- **Created**: 2026-11-18T13:00:00+07:00
- **Last Updated**: 2026-11-18T13:00:00+07:00 (Member B)
- **Người phụ trách**: Member A (Muội)
- **Status**: Done

---

## Name
Preprocessing — nạp IMDB, cắt tập con train / validation / test tái lập được, làm sạch văn bản

## Background
Đề Practice 3, Bài 2: load một dataset văn bản nhị phân đơn giản và tiền xử lý cho model.
Nhóm chọn IMDB (Maas et al., ACL 2011) từ Hugging Face Hub (`stanfordnlp/imdb`) và dùng tập con nhỏ 2000 / 500 / 500 để chạy được trên CPU. Hai lựa chọn này chưa xác nhận với cô: danh sách link dataset do một nhóm khác tổng hợp, không phải của cô.

## Goals / Purpose
- `load_text_dataset` trả `(train, val, test)` gồm 2000 / 500 / 500 dòng với cột `text`, `label` (0/1), seed 42; ba tập con không trùng nhau
- `clean_text` bỏ thẻ HTML như `<br />`, giải mã ký tự HTML, gộp khoảng trắng; giữ nguyên chữ hoa và dấu câu
- Tập train cân bằng 1000 / 1000; cả hai nhãn xuất hiện trong mỗi tập
- Không làm:
  - Lấy validation từ test — validation cắt từ train gốc, test lấy từ test gốc
  - Hạ chữ thường, bỏ dấu câu hay stop word — tokenizer uncased của DistilBERT tự hạ chữ thường, và dấu như `!` mang thông tin cảm xúc
  - Tokenize — thuộc `features` ([features_v1.md](features_v1.md))
  - Commit dataset — `labs/lab3/raw_data/` bị git bỏ qua (`.gitignore:10`)

## Input / Output
- **Input:** `name='imdb'`, `n_train`, `n_val`, `n_test`, `seed`; dataset tải từ Hugging Face Hub, lần đầu cần mạng
- **Output:** `(train, val, test)` là ba `datasets.Dataset` có cột `text`, `label`; chuyển cho [eda_v1.md](eda_v1.md) và [features_v1.md](features_v1.md)
  - Cache dataset tải về nằm ở `labs/lab3/raw_data/` (`raw_data_dir("lab3")`, `preprocessing/preprocessing.py:57`), không commit. Code không ghi file chia tập; việc tái lập dựa vào `seed`.

## How to do it (general plan)
1. `load_dataset("stanfordnlp/imdb", cache_dir=labs/lab3/raw_data)`
2. Xáo train gốc và test gốc theo `seed` **trước khi** chọn (nguồn xếp theo nhãn)
3. Train = `[0, n_train)` và validation = `[n_train, n_train + n_val)` từ train gốc; test = `[0, n_test)` từ test gốc
4. Kiểm có cột `text`, `label` và nhãn chỉ gồm 0 và 1; làm sạch từng tập bằng `clean_text`
5. Smoke run bằng test dùng dataset giả → normal run với IMDB thật

## Pipeline
```
stanfordnlp/imdb → shuffle(seed) → select
  (train, val từ train gốc; test từ test gốc)
  → kiểm cột và nhãn → clean_text → (train, val, test) → eda, features
```

## Detailed plan
- **Xáo trước khi chọn:** `shuffle(seed=seed)` rồi mới `select` (`preprocessing/preprocessing.py:61-62`) — IMDB trên Hub xếp theo nhãn, chọn trước khi xáo sẽ cho tập con chỉ có một nhãn
- **Chia tập:** train `range(n_train)`, validation `range(n_train, n_train + n_val)`, test `range(n_test)` (`:69-71`)
- **Làm sạch tối giản:** chỉ nhận thẻ HTML thật (`<br />`, `<i>`…), giữ nguyên `<3` hoặc `a < b` (`:21`)
- **Hàm hiện có:** `load_text_dataset(name: str = 'imdb', n_train: int = 2000, n_val: int = 500, n_test: int = 500, seed: int = 42) -> tuple` (`preprocessing/preprocessing.py:38`); `clean_text(text: str) -> str` (`:84`); hàm phụ `label_counts(dataset, label_col: str = "label") -> dict` (`:25`, ngoài hợp đồng trong `project_description.md`)
- **Chỗ code chưa khớp plan:** `README.md` ghi "put the dataset in `labs/lab3/raw_data/`" nhưng code tự tải dataset về thư mục này làm cache (`:57-59`); `project_description.md` có thư mục `artifacts/splits/` nhưng chưa có code ghi vào đó
- **Gotchas:**
  - `n_train + n_val` lớn hơn train gốc, hoặc `n_test` lớn hơn test gốc, thì `ValueError` (`:63-66`)
  - Thiếu cột `text` / `label`, hoặc nhãn ngoài {0, 1}, thì `ValueError` (`:74-80`)
  - `clean_text(None)` trả chuỗi rỗng (`:101-102`)
  - Lần gọi đầu cần mạng; các lần sau dùng cache

## Success criteria
- **Test:** `python -m pytest labs/lab3/tests/test_preprocessing.py -q` đạt, gồm test làm sạch HTML và khoảng trắng, giữ chữ hoa / dấu câu / dấu `<`, `None` và chuỗi trống, kích thước + cột + cân bằng, tái lập + không trùng, từ chối tập con quá lớn (dataset giả, không cần mạng)
- **Smoke run:** `python -c "from labs.lab3.preprocessing.preprocessing import load_text_dataset, label_counts; tr,va,te=load_text_dataset(); print(len(tr), label_counts(tr), len(va), label_counts(va), len(te), label_counts(te), '<br' in ' '.join(tr['text']))"` in `2000 {0: 1000, 1: 1000} 500 {0: 243, 1: 257} 500 {0: 254, 1: 246} False`
- **Normal run:** notebook mục 4 in cùng số dòng và số nhãn

## Kết quả audit (số liệu lần chạy 2026-10-25)
- **Test:** `pytest labs/lab3/tests` → 24 passed (16 test của Member A + 8 test `test_smoke.py`) ✔ (`progress_status.md:8`, `:27`)
- **Smoke run:** `2000 {0: 1000, 1: 1000} 500 {0: 243, 1: 257} 500 {0: 254, 1: 246} False` ✔ (`progress_status.md:28`)
- **Normal run:** notebook ô 6 in train 2000 `{0: 1000, 1: 1000}`, val 500 `{0: 243, 1: 257}`, test 500 `{0: 254, 1: 246}` ✔; ô 7 cho `clean_text` đổi `Great movie!<br /><br />Loved   it &amp; more.` thành `Great movie! Loved it & more.` ✔

## Links
- Desc lab: [../lab3_description_v1.md](../lab3_description_v1.md) | Khuôn: [../../../lab1/docs/templates/DESC_TEMPLATE_v3.md](../../../lab1/docs/templates/DESC_TEMPLATE_v3.md)
- Module liên quan: [eda_v1.md](eda_v1.md), [features_v1.md](features_v1.md)
- Tiến độ và bằng chứng chạy: [../../progress_status.md](../../progress_status.md) (dòng 8, 27, 28)
- Code: `labs/lab3/preprocessing/preprocessing.py` | Test: `labs/lab3/tests/test_preprocessing.py`
