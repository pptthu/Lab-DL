# Tiến độ Lab 3

- **Created**: 2026-11-20T01:25:13+07:00 (pptthu)
- **Last Updated**: 2026-11-20T02:41:59+07:00 (pptthu)

Cập nhật file này sau MỖI bước.
Bảng dưới đây là nguồn thông tin duy nhất về việc ai đang làm gì. Bài làm lại độc lập trên nhánh `feat/lab3-redo` (plan: `docs/plans/detail_plan_lab3_md_v1.md`).

## Phân công (Thư chốt 2026-11-20)
- **Member A — Thư:** `sentiment_demo` (Bài 1), `preprocessing`, `eda`, `features`, `model` (Bài 2.1–2.4); tích hợp `main.py`, README, notebook trình bày
- **Member B — Phúc (ngohongphuc2805ag-hash):** `train`, `evaluate`, `experiment`, `visualize` (Bài 2.5–2.7, mở rộng); notebook Kaggle và các lần chạy trên Kaggle
- Theo cách chia của Lab 1 và Lab 3 cũ (A: dữ liệu → model; B: train → đánh giá)

| Module | Phụ trách | Trạng thái (chưa làm / đang làm / xong) | Test (đạt / lỗi / chưa chạy) | Chạy trên dữ liệu thật (đạt / lỗi / chưa chạy) | Ghi chú |
|---|---|---|---|---|---|
| `sentiment_demo` | Member A (Thư) | xong, chờ audit | đạt (4 test) | đạt (2026-11-20) | Bài 1: 5/5 câu kiểm soát có nhãn đúng (kể cả phủ định, "but"); câu trung tính → NEGATIVE 0,9955; `outputs/demo/sentiment_demo*.csv` |
| `preprocessing` | Member A (Thư) | xong, chờ audit | đạt (9 test) | đạt (2026-11-20) | aclImdb gốc 25k/25k; split train 22.499 / val 2.501 theo phim + review trùng, 0 nhóm ở cả hai phía; `data/splits/train_val_split.json` |
| `eda` | Member A (Thư) | xong, chờ audit | đạt (5 test) | đạt (2026-11-20, 51 s) | > 256 token: 41%, > 512: 13%; `<br />` 58% review; 123 review test trùng train; `max_length` 512 (Thư chốt, `docs/experiments/experiment_log.md` S1) |
| `features` | Member A (Thư) | xong, chờ audit | đạt (3 test) | đạt (tokenize full data ở 512) | Bỏ `token_type_ids` (tokenizer thật vẫn trả); 1 batch qua model thật: logits [4, 2], loss 0,7101 |
| `model` | Member A (Thư) | xong, chờ audit | đạt (3 test) | đạt (nạp thật) | 66.955.010 param (đầu phân loại 592.130); `outputs/results/model_summary.json`; model card phần Initial |
| `train` | Member B (Phúc) | chưa làm | chưa chạy | chưa chạy | Chạy trên Kaggle GPU |
| `evaluate` | Member B (Phúc) | chưa làm | chưa chạy | chưa chạy | |
| `experiment` | Member B (Phúc) | chưa làm | chưa chạy | chưa chạy | Pilot → 3 seed → `max_length` |
| `visualize` | Member B (Phúc) | chưa làm | chưa chạy | chưa chạy | |
| Notebook Kaggle, các lần chạy Kaggle | Member B (Phúc) | chưa làm | – | chưa chạy | |
| `main.py`, README, notebook trình bày | Member A (Thư) | đang làm | – | notebook chạy 0 lỗi (2026-11-20) | Notebook: Stage 0–5 (phần A) có kết quả; Stage 6–10 đọc file của B, chưa có thì in "chưa có kết quả". `main.py`, README làm sau khi B có kết quả |

## Handoff A → B (làm tuần tự)
Phúc bắt đầu code khi TẤT CẢ các mục sau đúng:
- [x] Dataset đã tokenize có `input_ids`, `attention_mask`, `label` (3 split) — kiểm trên dữ liệu thật 2026-11-20
- [x] `max_length` đã chốt sau EDA: **512** (Thư, 2026-11-20), ghi trong `docs/experiments/experiment_log.md`
- [x] `model.build_model(seed)` và `model.load_tokenizer()` chạy được (nạp thật, forward 1 batch)
- [x] `pytest labs/lab3/tests` đạt cho các module của Member A (24 passed)
- [ ] Code của Member A đã push lên `feat/lab3-redo`; Thư gửi tin bàn giao — code đang ở nhánh `lab3/member-a` (chưa push), chờ Thư duyệt

**Cách B dùng phần A** (sau khi bàn giao):
```python
from labs.lab3 import config
from labs.lab3.preprocessing.preprocessing import load_datasets
from labs.lab3.features.features import tokenize_datasets, make_collator
from labs.lab3.model.model import load_tokenizer, build_model
ds = load_datasets()                     # train 22.499 / validation 2.501 / test 25.000 (lần đầu tự tải dữ liệu)
tok = load_tokenizer()
td = tokenize_datasets(ds, tok, config.MAX_LENGTH)
collator = make_collator(tok)
model = build_model(seed=42)             # đặt seed trước khi tạo model
```
Trên Kaggle: lần đầu `load_datasets()` tự tải `aclImdb_v1.tar.gz` (cần bật Internet); split đọc từ `data/splits/train_val_split.json` đã commit nên val giống hệt máy Thư.

Thư duyệt ở các cổng: smoke run, pilot (chốt trần epoch, patience), audit.

## Nhật ký (mỗi bước một dòng, có ngày)
- 2026-11-20 Thư (AI soạn, nhánh `lab3/member-a`): xong code + test + chạy thật 5 module của A; EDA full data; split train/val; notebook Stage 0–5; đề xuất `max_length` 512.
- 2026-11-20 Thư: chốt `max_length` 512; giữ split 22.499 / 2.501. `pytest labs/lab3/tests` 24 passed.
- 2026-11-20 Thư: chia việc Member A (Thư) / Member B (Phúc); `eda` chuyển về Member A.
- 2026-11-20 Thư: xong tài liệu (plan, rules, folder structure, templates, desc lab, mục lục, 9 file `phases/`).
- 2026-11-19 Thư: tạo nhánh `feat/lab3-redo` từ `main` (`a648057`), bỏ bài Lab 3 cũ trên nhánh này để làm lại độc lập.
