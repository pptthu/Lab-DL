# Lab 3 / demo: how to run and show the lab, and module `sentiment_demo` (Exercise 1)

This guide has two parts: (A) the module `sentiment_demo` (Exercise 1), and (B) how to run and present the whole lab, including `main.py`.

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
- Team choice: checkpoint `distilbert-base-uncased-finetuned-sst-2-english` (`DEFAULT_MODEL`), used when `model_name=None` by both functions.

### How the module works (current code)

- `show_tokens` loads the tokenizer of the checkpoint once (cached in `_tokenizers`) and returns the tokens and ids, including `[CLS]` and `[SEP]`.
- `analyze_sentences` loads a `pipeline("sentiment-analysis")` once (cached in `_pipelines`) and returns `{sentence, label, score}` per sentence; long texts are truncated, batch size 16. A single string is accepted and wrapped in a list.
- `transformers` is imported inside the helpers, so importing the module is cheap and tests can replace the helpers with fakes.

### Inputs, outputs and pitfalls

- `show_tokens` raises `TypeError` if `sentence` is not a `str`.
- Labels are the strings `POSITIVE` / `NEGATIVE`; `score` is a Python float. Use `evaluate.zero_shot_labels` to get 0/1.
- The tokenizer of `show_tokens` and the model of `analyze_sentences` must be the same checkpoint; both use `DEFAULT_MODEL` when `model_name=None`.
- First call downloads the model (internet needed).

## B. How to run and show the lab

### Current (what runs today)

| Step | Command (repository root) | State |
|---|---|---|
| Install | `pip install -r labs/lab3/requirements.txt` | root `requirements.txt` plus `transformers`, `datasets`, `evaluate`, `accelerate` |
| Unit and contract tests | `python -m pytest labs/lab3/tests -q` | most tests use fake objects; they do not prove the real libraries work |
| Notebook | `labs/lab3/notebook/lab3_presentation.ipynb` | holds the recorded runs; the 2026-11-02 analysis found it could not Run All from a fresh kernel (missing setup/imports). PR #47 changed it afterwards: re-check before relying on it |
| Entry point | `python -m labs.lab3.main` | **Blocker:** `main.py` is a stub. It prints `[todo]` for each stage and raises `NotImplementedError`. Do not describe it as a working pipeline |

Recorded results (Reported only, `progress_status.md`, 2026-10-31, CPU): test n=500; zero-shot accuracy/F1 0.8920/0.8889; fine-tuned 0.8220/0.8195 (MAX_LEN=128, EPOCHS=1, seed=42).

### Intended (what the brief needs)

- `main.py` runs Exercise 1 (`sentiment_demo`) and Exercise 2 in the order `eda -> preprocessing -> model (load) -> features (tokenize) -> train (Trainer) -> evaluate` (`STAGES` in `main.py`). It must not start a long training run or a download without an explicit option.
- Notebook: Run All works from a fresh kernel, and its numbers match `progress_status.md`.
- Deliverables (team leader, 2026-11-04): the full source code of the three labs and one written report (PDF, written later).

### What to show in the oral exam

- Exercise 1: tokens of one sentence (`[CLS]` ... `[SEP]`) and the 3 sample predictions, including the negation sentence.
- Exercise 2: data subsets, tokenization (truncation, padding per batch), `TrainingArguments`, best epoch chosen on validation F1, one final test evaluation, zero-shot vs fine-tuned comparison.

## How to verify

| Type | Command | Expected |
|---|---|---|
| Unit test | `python -m pytest labs/lab3/tests/test_sentiment_demo.py -q` | all pass, no network |
| Smoke test | `python -c "from labs.lab3.sentiment_demo.sentiment_demo import show_tokens, analyze_sentences; s=['I love this movie','The plot was boring','Not bad at all']; [print(show_tokens(x)) for x in s]; print(analyze_sentences(s))"` | tokens start with `[CLS]` (101), end with `[SEP]` (102); labels POSITIVE / NEGATIVE / POSITIVE (recorded scores 0.9999 / 0.9998 / 0.9991, Reported only) |
| Real integration run | `main.py` once it is wired, only at an approved milestone | metrics and command recorded in `progress_status.md` |

## Open questions / limitations

- `main.py` stub (blocker for the entry point).
- Notebook Run All from a fresh kernel not confirmed after PR #47.
- Dataset and model choices are team choices (IMDB, DistilBERT); the dataset link list came from another group, not from the lecturer.

---

**Tiếng Việt**

Việc được giao: xem GitHub Issue và `progress_status.md` của lab; không tự giao việc cho mình, hỏi nhóm trưởng.

## Dành cho AI (đọc trước): module `sentiment_demo`

- Người phụ trách: Member A (Muội), do nhóm trưởng giao. Trạng thái: xong (chỉ có trong báo cáo, 27/10).
- Được sửa: `sentiment_demo.py`, `tests/test_sentiment_demo.py`, file này. Cấm sửa: module khác, `main.py` (nhóm trưởng), data, model đã lưu, ô notebook của người khác.
- Chữ ký: `show_tokens(sentence, model_name=None) -> dict`, `analyze_sentences(sentences, model_name=None) -> list`, khớp `project_description.md`.
- Cấm: đổi checkpoint mặc định khi chưa hỏi nhóm trưởng (số zero-shot phụ thuộc vào nó), nạp pipeline lúc import, fine-tune trong module này.
- Kiểm tra: unit test `pytest labs/lab3/tests/test_sentiment_demo.py` (không cần mạng); smoke test với 3 câu mẫu (lần đầu cần mạng).

## A. Module `sentiment_demo`

Đề yêu cầu (Bài 1): dùng model cảm xúc có sẵn trên Hub và tokenize một câu. Nhóm chọn checkpoint `distilbert-base-uncased-finetuned-sst-2-english`, dùng chung cho cả hai hàm khi `model_name=None`. Tokenizer và pipeline được nạp một lần rồi dùng lại; `transformers` import bên trong hàm. Nhãn là chuỗi `POSITIVE`/`NEGATIVE`; muốn đổi sang 0/1 thì dùng `evaluate.zero_shot_labels`.

## B. Cách chạy và trình bày lab

- **Hiện tại:** test chạy bằng `pytest labs/lab3/tests` (phần lớn dùng object giả, không chứng minh thư viện thật chạy đúng). Notebook có kết quả đã chạy; bản phân tích 02/11 thấy notebook không Run All được từ kernel mới, PR #47 đã sửa notebook sau đó nên cần kiểm tra lại. **`main.py` đang là stub** (in `[todo]` rồi báo `NotImplementedError`): đây là blocker, không được mô tả như pipeline đã chạy.
- **Số đã ghi** (chỉ có trong báo cáo, 31/10): test 500 mẫu; zero-shot 0,8920/0,8889; fine-tune 0,8220/0,8195 (MAX_LEN=128, 1 epoch, seed 42).
- **Dự kiến:** `main.py` chạy Bài 1 và Bài 2 theo thứ tự `STAGES`, không tự train dài hay tự tải; notebook Run All được; bài nộp gồm toàn bộ source của 3 lab và một bài báo cáo (PDF, viết sau).
- **Khi vấn đáp:** token của một câu, 3 câu dự đoán (có câu phủ định), tập con, tokenize, `TrainingArguments`, chọn epoch theo F1 trên validation, đánh giá test một lần, so sánh zero-shot và fine-tune.

## Câu hỏi mở

`main.py` còn là stub; notebook sau PR #47 chưa được xác nhận Run All; IMDB và DistilBERT là lựa chọn của nhóm, danh sách link data do một nhóm khác tổng hợp, không phải của cô.
