# Tiến độ Lab 3

- **Created**: 2026-11-20T01:25:13+07:00 (pptthu)
- **Last Updated**: 2026-11-20T13:26:21+07:00 (Phúc, AI soạn)

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
| `train` | Member B (Phúc) | xong, chờ audit | đạt (7 test) | đạt (Kaggle T4, 2026-11-20) | Pilot best val acc 0,9308 (epoch 4,0); 3 seed val acc 0,9303 ± 0,0013; 5,2 phút/epoch; trần 6, patience 8, threshold 0 (AI chốt theo uỷ quyền, Thư duyệt 2026-11-20) |
| `evaluate` | Member B (Phúc) | xong, chờ audit | đạt (7 test) | đạt (Kaggle T4, 2026-11-20) | **Test acc 0,9315 ± 0,0020** (≥ 91%: đạt), F1 0,9323 ± 0,0021; zero-shot 0,8922; test dùng 1 lần/model (`test_evaluated: true`); fine-tuned sai 2/2 câu phủ định kiểm soát |
| `experiment` | Member B (Phúc) | xong, chờ audit | đạt (6 test) | đạt (Kaggle T4, 2026-11-20) | timing → pilot → 3 seed → `max_length` (seed 42): val acc 128: 0,8729; 256: 0,9152; 512: 0,9288 (1 seed/giá trị vì std ≤ 0,3 điểm; Thư duyệt) |
| `visualize` | Member B (Phúc) | xong, chờ audit | đạt (4 test) | đạt (máy, đọc `outputs/`) | Đường huấn luyện 6 run, `maxlen_val_acc.png`, confusion matrix; lỗi: bị cắt 10,05% vs 6,14%, rating 4: 17,2%, rating 7: 11,7% |
| Notebook Kaggle, các lần chạy Kaggle | Member B (Phúc) | xong, chờ audit | – | đạt: 8 kernel (`pptt1103/lab3-*`) | `labs/lab3/kaggle/` (CLI, dataset code private, 1 GPU); chạy trên tài khoản Kaggle của Thư (`pptt1103`) |
| `main.py`, README, notebook trình bày | Member A (Thư) | đang làm | – | notebook chạy 0 lỗi (2026-11-20; Stage 6–10 có kết quả + nhận xét của Phúc) | Notebook: Stage 0–5 (phần A) có kết quả; Stage 6–10 đọc file của B, chưa có thì in "chưa có kết quả". `main.py`, README làm sau khi B có kết quả |

## Handoff A → B (làm tuần tự)
Phúc bắt đầu code khi TẤT CẢ các mục sau đúng:
- [x] Dataset đã tokenize có `input_ids`, `attention_mask`, `label` (3 split) — kiểm trên dữ liệu thật 2026-11-20
- [x] `max_length` đã chốt sau EDA: **512** (Thư, 2026-11-20), ghi trong `docs/experiments/experiment_log.md`
- [x] `model.build_model(seed)` và `model.load_tokenizer()` chạy được (nạp thật, forward 1 batch)
- [x] `pytest labs/lab3/tests` đạt cho các module của Member A (24 passed)
- [x] Code của Member A đã push lên `feat/lab3-redo` (`8932b58`, 2026-11-20); Thư gửi tin bàn giao cho Phúc (2026-11-20)

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
- 2026-11-20 Phúc (AI soạn, theo phản hồi của Thư): S4 so cùng seed 42 (512: 0,9288; 512 hơn 256 1,36 điểm, trung bình 3 seed để tham khảo); S3 ghi chú seed 43 sát trần 6 epoch (không chạy lại); ghi đúng tài khoản Kaggle của Thư (`pptt1103`); notebook Stage 6–10 thêm ô "Nhận xét" + cập nhật Tóm tắt, chạy lại toàn bộ bằng kernel python313: 0 lỗi; Thư duyệt quy tắc dừng.
- 2026-11-20 Phúc (AI chạy): S5 test 25k một lần trên Kaggle: fine-tuned 0,9315 ± 0,0020, zero-shot 0,8922 → đạt ngưỡng 91%; confusion matrix, phân tích lỗi, câu kiểm soát. Tài liệu: `experiment_log.md` S2–S5, mục "Triển khai" của 4 phase, model card phần Trained model. Commit đứng tên Phúc theo quyết định của Thư.
- 2026-11-20 Phúc (AI chạy): S4 `max_length` 128 / 256 (1 seed/giá trị) — val acc 0,8729 / 0,9152 so với 0,9303 ở 512.
- 2026-11-20 Phúc (AI chạy): S3 3 seed (42, 43, 44) — val acc 0,9303 ± 0,0013; std ≤ 0,3 điểm → 1 seed/giá trị cho S4.
- 2026-11-20 Phúc (AI chạy): S2 đo thời gian (5,2 phút/epoch thật) + pilot 6 epoch (best val acc 0,9308 ở epoch 4,0); AI chốt trần 6, patience 8, threshold 0 theo uỷ quyền của Thư (Thư duyệt 2026-11-20).
- 2026-11-20 Phúc (AI soạn): code 4 module B + test (`pytest labs/lab3/tests` 48 passed); smoke trên máy và trên Kaggle (CPU) chạy hết; công cụ Kaggle `labs/lab3/kaggle/`.
- 2026-11-20 Thư: code phần A lên `feat/lab3-redo` (`8932b58`); PR #79 đưa bài cũ về lại `main`; bàn giao phần B cho Phúc (Phúc làm trên nhánh `lab3-redo/member-b`, PR vào `feat/lab3-redo`).
- 2026-11-20 Thư (AI soạn, nhánh `lab3/member-a`): xong code + test + chạy thật 5 module của A; EDA full data; split train/val; notebook Stage 0–5; đề xuất `max_length` 512.
- 2026-11-20 Thư: chốt `max_length` 512; giữ split 22.499 / 2.501. `pytest labs/lab3/tests` 24 passed.
- 2026-11-20 Thư: chia việc Member A (Thư) / Member B (Phúc); `eda` chuyển về Member A.
- 2026-11-20 Thư: xong tài liệu (plan, rules, folder structure, templates, desc lab, mục lục, 9 file `phases/`).
- 2026-11-19 Thư: tạo nhánh `feat/lab3-redo` từ `main` (`a648057`), bỏ bài Lab 3 cũ trên nhánh này để làm lại độc lập.
