# demo — Chạy và trình bày Lab 3 (v1)

- **Created**: 2026-11-18T13:00:00+07:00
- **Last Updated**: 2026-11-18T13:00:00+07:00 (Member B)
- **Người phụ trách**: Nhóm trưởng (`main.py`, `README.md`, notebook mục 3, 11, 12); Member B (các mục notebook của B)
- **Status**: In Progress

---

## Name
Demo — chạy cả lab bằng `main.py` và notebook, nơi lưu từng kết quả, nội dung cần trình bày khi vấn đáp

## Background
Đề Practice 3 có hai bài: Bài 1 (model cảm xúc có sẵn) và Bài 2 (fine-tune). Bài nộp theo nhóm trưởng chốt ngày 2026-11-04: toàn bộ source code của ba lab và một báo cáo PDF viết sau.
File này ghi hôm nay chạy được gì, còn thiếu gì, kết quả nằm ở đâu và khi vấn đáp cần cho cô xem gì. Phần Bài 1 của `sentiment_demo` ở [sentiment_demo_v1.md](sentiment_demo_v1.md).

## Goals / Purpose
- `python -m labs.lab3.main --run-demo` in token và dự đoán của 3 câu mẫu (Bài 1)
- `python -m labs.lab3.main --run-training` chạy Bài 2 theo thứ tự: nạp dữ liệu → EDA (chỉ tập train) → nạp model → tokenize 3 tập → train → test một lần → ghi `test_metrics.json`
- Notebook `labs/lab3/notebook/lab3_presentation.ipynb` Run All được từ kernel mới và số liệu khớp `progress_status.md`
- Không làm:
  - So sánh zero-shot với fine-tune trong `main.py` — hiện chỉ có trong notebook mục 10.2
  - Tự tải dữ liệu hoặc tự train khi không truyền cờ — không có cờ thì `main.py` chỉ in phần trợ giúp (`main.py:183-185`)
  - Báo cáo PDF — nhóm trưởng viết sau khi có số liệu

## Input / Output
- **Input:** `pip install -r labs/lab3/requirements.txt`; mạng ở lần chạy đầu (tải dataset và model); các cờ `--seed`, `--model-name`, `--max-length`, `--n-train`, `--n-val`, `--n-test`, `--epochs`, `--learning-rate`, `--batch-size`, `--max-steps`
- **Output:** file trong `labs/lab3/`; mỗi thứ ghi ở đâu:

| Nơi | Nội dung | Có commit? |
|---|---|---|
| `raw_data/` | cache dataset do `preprocessing` tải | không (`.gitignore:10`) |
| `artifacts/figures/` | 3 hình EDA do `main.py --run-training` ghi | không (`.gitignore:14`) |
| `artifacts/models/distilbert_sentiment/` | model fine-tune do `main.py --run-training` ghi | không (`.gitignore:12`) |
| `artifacts/logs/test_metrics.json` | metric test cuối do `main.py` ghi | không (`.gitignore:16`) |
| `artifacts/models/distilbert_imdb/` | model do notebook lưu và nạp lại | không (`.gitignore:12`) |
| `artifacts/logs/distilbert_imdb_log.json`, `lab3_results.json` | log train và bảng so sánh zero-shot / fine-tune của notebook | không (`.gitignore:16`) |
| `artifacts/models/distilbert_imdb_len<ALT_LEN>/` | lần chạy tùy chọn ở notebook mục 9 với `max_length` khác (tắt mặc định) | không (`.gitignore:12`) |
| `artifacts/runs/`, `artifacts/splits/` | chưa có code nào ghi vào | `runs/` không (`.gitignore:18`); `splits/` không có quy tắc bỏ qua |
| `docs/`, `notebook/` | tài liệu và notebook | có |

## How to do it (general plan)
1. Cài đặt: `pip install -r labs/lab3/requirements.txt`
2. Test: `python -m pytest labs/lab3/tests -q`
3. Smoke run: `python -m labs.lab3.main --run-training --max-steps 2`
4. Bài 1: `python -m labs.lab3.main --run-demo`
5. Normal run (thiết lập tương đương lần chạy đã ghi): `python -m labs.lab3.main --run-training --max-length 128 --epochs 1 --batch-size 8`
6. Mở notebook và Run All từ kernel mới; đối chiếu số với `progress_status.md`

## Pipeline
```
main.py --run-training:
  load_text_dataset → run_eda (train) → load_model_and_tokenizer
  → tokenize_dataset ×3 → build_training_args → train_with_trainer
  → evaluate_on_test (một lần) → artifacts/logs/test_metrics.json
main.py --run-demo:
  show_tokens + analyze_sentences (3 câu mẫu)
```

## Detailed plan
- **Thứ tự trong `run_training`** (`main.py:52`): dataset (`:70`), EDA (`:85-86`), model (`:91`), tokenize 3 tập (`:98-104`), thư mục lưu model (`:110-111`), train (`:113-129`), test một lần (`:133`), ghi metric (`:135-136`)
- **Mặc định `main.py` so với lần chạy đã ghi (notebook ô 17):**

| Tham số | Mặc định `main.py` | Lần chạy đã ghi |
|---|---|---|
| `--max-length` | 256 (`main.py:160`) | 128 |
| `--epochs` | 2 (`main.py:166`) | 1 |
| `--batch-size` | 16 (`main.py:168`) | 8 |
| `--learning-rate` | 2e-5 (`main.py:167`) | 2e-5 |
| `--seed` | 42 (`main.py:158`) | 42 |

- **Notebook** (29 ô, mục 1 đến 12): mục 3 là setup (nhóm trưởng); mục 4 và 5.1 của Member A; mục 5.2 đến 10 của Member B (mục 9 tùy chọn, tắt mặc định); mục 11 (Tổng kết) và 12 (Phụ lục) của nhóm trưởng
- **Hàm hiện có:** `run_sentiment_demo() -> None` (`main.py:28`); `run_training(args: argparse.Namespace) -> None` (`main.py:52`); `build_parser()` (`main.py:148`); `main()` (`main.py:179`)
- **Chỗ code chưa khớp plan:**
  - `STAGES` (`main.py:17`) không được dùng ở luồng nào
  - `progress_status.md:13` vẫn ghi `main.py` là `todo`
  - Notebook mục 11 và 12 mới là chỗ giữ chỗ, chưa viết
  - Notebook lưu model vào `distilbert_imdb`, `main.py` lưu vào `distilbert_sentiment` [cần chốt: nhóm trưởng chọn một tên]
- **Gotchas:**
  - Phần lớn test dùng đối tượng giả, nên không chứng minh thư viện thật chạy đúng
  - Phân tích 2026-11-02 thấy notebook không Run All được từ kernel mới (thiếu setup / import); PR #47 sửa notebook sau đó, cần kiểm lại trước khi dựa vào
  - Lần chạy đầu cần mạng để tải dataset và model

## Success criteria
- **Test:** `python -m pytest labs/lab3/tests -q` đạt, gồm `test_smoke.py` (kiểm tên hàm của các module và import được `main`)
- **Smoke run:** `main.py --run-training --max-steps 2` in metric test, `Metrics saved to: .../test_metrics.json` và `Model saved to: .../distilbert_sentiment`
- **Normal run:** `main.py --run-training --max-length 128 --epochs 1 --batch-size 8` in accuracy và F1 test, ghi kết quả vào `progress_status.md`

## Kết quả audit (đối chiếu số liệu đã ghi đến 2026-10-31)
- **Test:** `pytest labs/lab3/tests` → 56 passed (8 model + 8 train + 16 evaluate + 24 của Member A và `test_smoke.py`) ✔ (`progress_status.md:31`)
- **Smoke run của `main.py`:** chưa có lần chạy nào được ghi — chưa kiểm chứng
- **Normal run của `main.py`:** chưa có lần chạy nào được ghi — số liệu đã ghi lấy từ notebook (test n = 500: fine-tuned accuracy / F1 0,8220 / 0,8195; zero-shot 0,8920 / 0,8889), không phải từ `main.py` ✔ cho notebook (`progress_status.md:12`)
- **Notebook Run All từ kernel mới:** chưa xác nhận sau PR #47

## Khi vấn đáp cần cho cô xem
- **Bài 1:** token của một câu (`[CLS]` … `[SEP]`) và 3 dự đoán mẫu, kể cả câu phủ định
- **Bài 2:**
  - Các tập con dữ liệu (2000 / 500 / 500, seed 42) và vì sao phải xáo trước khi chọn
  - Tokenize: cắt bớt (`max_length`), padding theo batch
  - `TrainingArguments` và epoch tốt nhất chọn theo F1 validation
  - Model fine-tune được lưu ở đâu
  - Đánh giá test đúng một lần, và so sánh zero-shot với fine-tune trên cùng 500 review

## Links
- Desc lab: [../lab3_description_v1.md](../lab3_description_v1.md) | Khuôn: [../../../lab1/docs/templates/DESC_TEMPLATE_v3.md](../../../lab1/docs/templates/DESC_TEMPLATE_v3.md)
- Module liên quan: [sentiment_demo_v1.md](sentiment_demo_v1.md), [train_v1.md](train_v1.md), [evaluate_v1.md](evaluate_v1.md)
- Tiến độ và bằng chứng chạy: [../../progress_status.md](../../progress_status.md) (dòng 12, 13, 31)
- Code: `labs/lab3/main.py`, `labs/lab3/notebook/lab3_presentation.ipynb` | Test: `labs/lab3/tests/test_smoke.py`
