# Lab 3 / demo: how to run and show the lab, and module `sentiment_demo` (Exercise 1)

This guide has two parts: (A) the module `sentiment_demo` (Exercise 1), and (B) how to run and present the whole lab, including `main.py`.

## Purpose

- Part A: use a **ready-made** sentiment model (no training) to classify sentences, and show how a sentence is split into tokens.
- Part B: explain **what runs today**, what is still missing, where files are saved, and what to show in the oral exam.

## For AI agents (read first): module `sentiment_demo`

| Field | Value |
|---|---|
| Module | `labs/lab3/sentiment_demo/sentiment_demo.py` |
| Owner | Member A (Muội), assigned by the team leader |
| Status | done in `progress_status.md` (Reported only, 2026-10-27) |
| Assigned work | see the GitHub Issues and `progress_status.md` of the lab; never assign work to yourself, ask the team leader |
| Files you may edit | `labs/lab3/sentiment_demo/sentiment_demo.py`, `labs/lab3/tests/test_sentiment_demo.py`, this guide |
| Files you must not edit | other modules, `main.py` (team leader), raw data, saved models, other owners' notebook cells |
| Agreed signatures | `show_tokens(sentence: str, model_name: str = None) -> dict`; `analyze_sentences(sentences: list, model_name: str = None) -> list` (same as `project_description.md`) |
| Depends on / used by | `transformers` only / notebook Exercise 1; `evaluate.zero_shot_labels` maps its labels to 0/1 for the zero-shot baseline |
| Prohibited | changing the default checkpoint without the team leader (the zero-shot numbers depend on it); loading the pipeline at import time; fine-tuning in this module |
| Checks | Unit: `python -m pytest labs/lab3/tests/test_sentiment_demo.py -q` (fake objects, no network). Smoke: the 3-sentence call in "How to verify" (needs internet the first time) |
| Done when | unit tests pass; smoke output recorded in `progress_status.md`; this guide updated |

## A. Module `sentiment_demo`

### What the brief requires

- Brief requirement (Practice 3, Exercise 1): use a pre-trained sentiment model from the Hugging Face Hub and tokenize a sample sentence.
- Team implementation choice: checkpoint `distilbert-base-uncased-finetuned-sst-2-english` (`DEFAULT_MODEL`), used when `model_name=None` by both functions.

### How the module works (current code)

- `show_tokens` loads the tokenizer once (cached in `_tokenizers`) and returns the tokens and ids, including `[CLS]` and `[SEP]`.
- `analyze_sentences` loads a `pipeline("sentiment-analysis")` once (cached in `_pipelines`) and returns `{sentence, label, score}` per sentence. Long texts are truncated; batch size 16. A single string is accepted and wrapped in a list.
- `transformers` is imported inside the helpers, so importing the module is cheap and tests can replace the helpers with fakes.

### Inputs, outputs and pitfalls

- Input:
  1. Initial model: the SST-2 DistilBERT checkpoint above, taken from the Hugging Face Hub (already fine-tuned, so no training is needed).
  2. Sentences as `str` or a list of `str`.
- Output:
  - `show_tokens`: `{tokens, input_ids}`.
  - `analyze_sentences`: list of `{sentence, label, score}`; labels are the strings `POSITIVE` / `NEGATIVE`, `score` is a Python float.
- This module saves nothing; the downloaded model goes to the default Hugging Face cache.
- Pitfalls:
  - `show_tokens` raises `TypeError` if `sentence` is not a `str`.
  - use `evaluate.zero_shot_labels` to get 0/1 from the labels.
  - the tokenizer of `show_tokens` and the model of `analyze_sentences` must be the same checkpoint (both use `DEFAULT_MODEL` when `model_name=None`).
  - the first call downloads the model (internet needed).

## B. How to run and show the lab

### Current (what runs today)

| Step | Command (repository root) | State |
|---|---|---|
| Install | `pip install -r labs/lab3/requirements.txt` | root `requirements.txt` plus `transformers`, `datasets`, `evaluate`, `accelerate` |
| Unit and contract tests | `python -m pytest labs/lab3/tests -q` | most tests use fake objects; they do not prove the real libraries work |
| Notebook | `labs/lab3/notebook/lab3_presentation.ipynb` | holds the recorded runs; the 2026-11-02 analysis found it could not Run All from a fresh kernel (missing setup/imports). PR #47 changed it afterwards: re-check before relying on it |
| Entry point | `python -m labs.lab3.main --run-demo` (Exercise 1) and `python -m labs.lab3.main --run-training` (Exercise 2; add `--max-steps 2` for a smoke run) | Implemented (`main.py` is wired, commit `f4659f9`). No recorded run of `main.py` was seen (Pending). `progress_status.md` still lists `main.py` as `todo`; the team leader updates it |

- Recorded results (Reported only, `progress_status.md`, 2026-10-31, CPU): test n=500; zero-shot accuracy/F1 0.8920/0.8889; fine-tuned 0.8220/0.8195 (MAX_LEN=128, EPOCHS=1, seed=42).

### Where files go (folders of `labs/lab3/`)

| Folder | What goes there | Committed? |
|---|---|---|
| `raw_data/` | dataset cache downloaded by `preprocessing` | no (ignored by git, stated in `shared/paths.py`) |
| `artifacts/figures/` | EDA plots from `eda`, written by `main.py --run-training` | intended no |
| `artifacts/models/distilbert_sentiment/` | the fine-tuned model, written by `train` when `main.py --run-training` runs (Implemented) | no (Verified: ignored by `.gitignore` line 12) |
| `artifacts/logs/test_metrics.json` | final test metrics written by `main.py` | intended no |
| `artifacts/models/distilbert_imdb/` | the model saved and re-loaded by the notebook (Implemented in the notebook; same ignore rule as above) | no (Verified: `.gitignore` line 12) |
| `artifacts/logs/distilbert_imdb_log.json` | the notebook's log file | no (per the notebook comment; not checked) |
| `artifacts/models/distilbert_imdb_len<ALT_LEN>/` | optional run of the notebook with another `max_length` | no (Verified: `.gitignore` line 12) |
| `artifacts/runs/`, `artifacts/splits/` | not written by any code seen in this guide | intended no |
| `docs/` | these guides | yes |
| `notebook/` | `lab3_presentation.ipynb` | yes |

- "intended no" is the intent shown by the `.gitkeep` placeholders; only the `models/` rule is verified (`git check-ignore -v`); `figures/`, `logs/`, `runs/`, `splits/` are not yet checked.

### Intended (what the brief needs)

- Implemented in `main.py`:
  - `--run-demo` runs Exercise 1; `--run-training` runs Exercise 2. With neither flag it only prints the help, so no download or training starts by accident.
  - `--run-training` order: load the dataset, EDA on the train split only, load model and tokenizer, tokenize the three splits, train (validation used for model selection), evaluate on test once, then write `artifacts/logs/test_metrics.json`. The `STAGES` list in the file is not used by this code path.
  - options: `--seed`, `--model-name`, `--max-length`, `--n-train`, `--n-val`, `--n-test`, `--epochs`, `--learning-rate`, `--batch-size`, `--max-steps`.
  - defaults are `--max-length 256` and `--epochs 2`; the recorded run used 128 and 1, so use `--max-length 128 --epochs 1` for equivalent settings.
- Not in `main.py`: the zero-shot vs fine-tuned comparison (it is only in the notebook).
- Notebook: Run All works from a fresh kernel, and its numbers match `progress_status.md`.
- Deliverables (team leader, 2026-11-04): the full source code of the three labs and one written report (PDF, written later).

### What to show in the oral exam

- Exercise 1: the tokens of one sentence (`[CLS]` ... `[SEP]`) and the 3 sample predictions, including the negation sentence.
- Exercise 2:
  - data subsets and why the data is shuffled before selecting.
  - tokenization: truncation, padding per batch.
  - `TrainingArguments` and the best epoch chosen on validation F1.
  - where the fine-tuned model is saved.
  - one final test evaluation, and the zero-shot vs fine-tuned comparison.

## How to verify

| Type | Command | Expected |
|---|---|---|
| Unit test | `python -m pytest labs/lab3/tests/test_sentiment_demo.py -q` | all pass, no network |
| Smoke test | `python -c "from labs.lab3.sentiment_demo.sentiment_demo import show_tokens, analyze_sentences; s=['I love this movie','The plot was boring','Not bad at all']; [print(show_tokens(x)) for x in s]; print(analyze_sentences(s))"` | tokens start with `[CLS]` (101), end with `[SEP]` (102); labels POSITIVE / NEGATIVE / POSITIVE (recorded scores 0.9999 / 0.9998 / 0.9991, Reported only) |
| Real integration run | `main.py` once it is wired, only at an approved milestone | metrics and command recorded in `progress_status.md` |

## Open questions / limitations

- No recorded run of `main.py --run-training` yet (Pending); `progress_status.md` still says `todo` for `main.py`.
- Notebook Run All from a fresh kernel is not confirmed after PR #47.
- IMDB and DistilBERT are team choices; the dataset link list came from another group, not from the lecturer.
- The notebook saves the model to `artifacts/models/distilbert_imdb` and its log to `artifacts/logs/distilbert_imdb_log.json`, while `main.py` uses `artifacts/models/distilbert_sentiment` and `artifacts/logs/test_metrics.json` (see `train.md`); the team leader must agree on one name.

---

**Tiếng Việt**

## Mục đích

- Phần A: dùng model cảm xúc **có sẵn** (không train) để phân loại câu, và cho thấy một câu được tách thành token như thế nào.
- Phần B: giải thích **hiện tại chạy được gì**, còn thiếu gì, file được lưu ở đâu, và khi vấn đáp cần trình bày gì.

## Dành cho AI (đọc trước): module `sentiment_demo`

- Người phụ trách: Member A (Muội), do nhóm trưởng giao. Trạng thái: xong (chỉ có trong báo cáo, 27/10).
- Việc được giao: xem GitHub Issue và `progress_status.md`; không tự giao việc cho mình, hỏi nhóm trưởng.
- Được sửa: `sentiment_demo.py`, `tests/test_sentiment_demo.py`, file này.
- Cấm sửa: module khác, `main.py` (nhóm trưởng), data gốc, model đã lưu, ô notebook của người khác.
- Chữ ký: `show_tokens(sentence, model_name=None) -> dict`, `analyze_sentences(sentences, model_name=None) -> list`, khớp `project_description.md`.
- Cấm: đổi checkpoint mặc định khi chưa hỏi nhóm trưởng (số zero-shot phụ thuộc vào nó); nạp pipeline lúc import; fine-tune trong module này.
- Kiểm tra: unit test `pytest labs/lab3/tests/test_sentiment_demo.py` (không cần mạng); smoke test với 3 câu mẫu (lần đầu cần mạng).
- Xong khi: unit test qua, kết quả smoke ghi vào `progress_status.md`, đã cập nhật file này.

## A. Module `sentiment_demo`

### Đề yêu cầu gì

- Đề (Practice 3, Bài 1): dùng model cảm xúc pre-trained từ Hugging Face Hub và tokenize một câu mẫu.
- Lựa chọn của nhóm: checkpoint `distilbert-base-uncased-finetuned-sst-2-english` (`DEFAULT_MODEL`), cả hai hàm dùng khi `model_name=None`.

### Cách hoạt động

- `show_tokens` nạp tokenizer một lần (lưu trong `_tokenizers`) và trả về token cùng id, có cả `[CLS]` và `[SEP]`.
- `analyze_sentences` nạp `pipeline("sentiment-analysis")` một lần (lưu trong `_pipelines`) và trả `{sentence, label, score}` cho mỗi câu. Văn bản dài bị cắt; batch size 16. Nhận một chuỗi đơn và tự bọc thành danh sách.
- `transformers` import bên trong hàm nên import module rất nhẹ và test có thể thay bằng đối tượng giả.

### Đầu vào, đầu ra và lỗi hay gặp

- Đầu vào:
  1. Model ban đầu: checkpoint DistilBERT SST-2 ở trên, lấy từ Hugging Face Hub (đã fine-tune sẵn nên không cần train).
  2. Câu là `str` hoặc danh sách `str`.
- Đầu ra:
  - `show_tokens`: `{tokens, input_ids}`.
  - `analyze_sentences`: danh sách `{sentence, label, score}`; nhãn là chuỗi `POSITIVE` / `NEGATIVE`, `score` là float của Python.
- Module này không lưu gì; model tải về nằm trong cache mặc định của Hugging Face.
- Lưu ý:
  - `show_tokens` báo `TypeError` nếu `sentence` không phải `str`.
  - muốn nhãn 0/1 thì dùng `evaluate.zero_shot_labels`.
  - tokenizer của `show_tokens` và model của `analyze_sentences` phải cùng checkpoint (cả hai dùng `DEFAULT_MODEL` khi `model_name=None`).
  - lần gọi đầu tải model về (cần mạng).

## B. Cách chạy và trình bày lab

### Hiện tại (đang chạy được gì)

- Cài đặt: `pip install -r labs/lab3/requirements.txt`.
- Test: `python -m pytest labs/lab3/tests -q`. Phần lớn test dùng object giả, không chứng minh thư viện thật chạy đúng.
- Notebook: `labs/lab3/notebook/lab3_presentation.ipynb` chứa các lần chạy đã ghi. Bản phân tích 02/11 thấy notebook không Run All được từ kernel mới (thiếu setup/import). PR #47 đã sửa notebook sau đó, cần kiểm tra lại trước khi dựa vào.
- Điểm vào: `python -m labs.lab3.main --run-demo` (Bài 1) và `python -m labs.lab3.main --run-training` (Bài 2; thêm `--max-steps 2` để chạy thử). `main.py` đã được nối đủ (commit `f4659f9`). Chưa thấy lần chạy nào của `main.py` được ghi lại. `progress_status.md` vẫn ghi `main.py` là `todo`; nhóm trưởng cập nhật.
- Số đã ghi (chỉ có trong báo cáo, 31/10, CPU): test n=500; zero-shot accuracy/F1 0,8920/0,8889; fine-tune 0,8220/0,8195 (MAX_LEN=128, 1 epoch, seed 42).

### File nằm ở đâu (các thư mục trong `labs/lab3/`)

- `raw_data/`: cache dataset do `preprocessing` tải về; không commit (git bỏ qua, ghi trong `shared/paths.py`).
- `artifacts/figures/`: biểu đồ EDA do `eda` tạo, `main.py --run-training` ghi vào; ý định là không commit.
- `artifacts/models/distilbert_sentiment/`: model fine-tune do `train` ghi khi chạy `main.py --run-training`; không commit (đã kiểm chứng: `.gitignore` dòng 12 bỏ qua).
- `artifacts/logs/test_metrics.json`: số liệu test cuối cùng do `main.py` ghi; ý định là không commit.
- `artifacts/models/distilbert_imdb/`: model do notebook lưu và nạp lại (có trong notebook; cùng quy tắc bỏ qua ở trên); không commit (đã kiểm chứng: `.gitignore` dòng 12).
- `artifacts/logs/distilbert_imdb_log.json`: file log của notebook; không commit (theo chú thích trong notebook, chưa kiểm tra).
- `artifacts/models/distilbert_imdb_len<ALT_LEN>/`: lần chạy tùy chọn của notebook với `max_length` khác; không commit (đã kiểm chứng: `.gitignore` dòng 12).
- `artifacts/runs/`, `artifacts/splits/`: không có code nào đã xem ghi vào; ý định là không commit.
- `docs/`: các guide này; có commit.
- `notebook/`: `lab3_presentation.ipynb`; có commit.
- "Ý định là không commit" thể hiện qua các file `.gitkeep`; mới kiểm chứng quy tắc của `models/` (`git check-ignore -v`); `figures/`, `logs/`, `runs/`, `splits/` chưa kiểm tra.

### Dự kiến (đề cần gì)

- Đã có trong `main.py`:
  - `--run-demo` chạy Bài 1; `--run-training` chạy Bài 2. Không truyền cờ nào thì chỉ in phần trợ giúp, nên không tự tải dữ liệu hay tự train.
  - thứ tự của `--run-training`: nạp dataset, EDA chỉ trên tập train, nạp model và tokenizer, tokenize ba tập, train (dùng validation để chọn model), đánh giá test một lần, rồi ghi `artifacts/logs/test_metrics.json`. Danh sách `STAGES` trong file không được dùng ở luồng này.
  - tùy chọn: `--seed`, `--model-name`, `--max-length`, `--n-train`, `--n-val`, `--n-test`, `--epochs`, `--learning-rate`, `--batch-size`, `--max-steps`.
  - mặc định là `--max-length 256` và `--epochs 2`; lần chạy đã ghi dùng 128 và 1, nên dùng `--max-length 128 --epochs 1` để có thiết lập tương đương.
- Chưa có trong `main.py`: so sánh zero-shot với fine-tune (chỉ có trong notebook).
- Notebook: Run All chạy được từ kernel mới, số liệu khớp `progress_status.md`.
- Bài nộp (nhóm trưởng, 04/11): toàn bộ source code của ba lab và một bài báo cáo (PDF, viết sau).

### Khi vấn đáp cần trình bày

- Bài 1: token của một câu (`[CLS]` ... `[SEP]`) và 3 câu dự đoán mẫu, có câu phủ định.
- Bài 2:
  - các tập con dữ liệu và vì sao phải xáo trước khi chọn.
  - tokenize: cắt bớt (truncation), padding theo batch.
  - `TrainingArguments` và epoch tốt nhất chọn theo F1 trên validation.
  - model fine-tune được lưu ở đâu.
  - đánh giá test một lần, và so sánh zero-shot với fine-tune.

## Cách kiểm chứng

- Unit test: `python -m pytest labs/lab3/tests/test_sentiment_demo.py -q`, kỳ vọng tất cả qua, không cần mạng.
- Smoke test: lệnh trong bảng tiếng Anh ở trên; kỳ vọng token bắt đầu bằng `[CLS]` (101), kết thúc bằng `[SEP]` (102); nhãn POSITIVE / NEGATIVE / POSITIVE (điểm đã ghi 0,9999 / 0,9998 / 0,9991, chỉ có trong báo cáo).
- Chạy tích hợp thật: `main.py` sau khi được nối đủ, chỉ ở mốc đã duyệt; ghi số liệu và lệnh vào `progress_status.md`.

## Câu hỏi mở

- Chưa có lần chạy `main.py --run-training` nào được ghi lại; `progress_status.md` vẫn ghi `main.py` là `todo`.
- Notebook sau PR #47 chưa được xác nhận Run All từ kernel mới.
- IMDB và DistilBERT là lựa chọn của nhóm; danh sách link dataset do một nhóm khác tổng hợp, không phải của cô.
- Notebook lưu model vào `artifacts/models/distilbert_imdb` và log vào `artifacts/logs/distilbert_imdb_log.json`, còn `main.py` dùng `artifacts/models/distilbert_sentiment` và `artifacts/logs/test_metrics.json` (xem `train.md`); nhóm trưởng phải chốt một tên.
