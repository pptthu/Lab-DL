# Lab 3: Hugging Face sentiment analysis and fine-tuning: progress status
_Last updated: 2026-10-27 by Muội_

| Module | Owner | Status (todo / doing / done / N/A) | Smoke run | Normal run | Notes |
|---|---|---|---|---|---|
| `sentiment_demo` | Member A (name TBD) | todo | - | - | |
| `eda` | Member A (Muội) | done | `pytest labs/lab3/tests/test_eda.py`: 3 passed | `run_eda`: n=2000, 3 PNG saved | Figures go to `artifacts/figures/` (not committed) |
| `preprocessing` | Member A (Muội) | done | `pytest labs/lab3/tests`: 24 passed | real IMDB run: 2000/500/500, both labels in each split, no `<br` left | Shuffle(seed) before select; val cut from train, test from test. |
| `features` | Member A (Muội) | todo | - | - | |
| `model` | Member B (name TBD) | todo | - | - | |
| `train` | Member B (name TBD) | todo | - | - | |
| `evaluate` | Member B (name TBD) | todo | - | - | |
| `main.py`, README, integration | Team leader | todo | - | - | |

## Handoff A -> B (sequential, no interleaving)
Member B starts coding only after ALL of the following are true:
- [ ] tokenized dataset has input_ids, attention_mask, label
- [ ] Exercise 1 (sentiment_demo) prints tokens and sentiment for 3 sample sentences
- [ ] all Member A modules are `done` in this file and `pytest labs/lab3/tests` passes
- [ ] Member A's PRs are merged into `main`; Member B branches from the updated `main`
- [ ] Member A posts the handoff message (what was done, exact run commands, known problems)
After the handoff Member A only answers questions and reviews PRs; Member A does not edit Member B's modules.

## Audit: what actually runs
- `pytest labs/lab3/tests` -> 24 passed (16 Member A tests + 8 `test_smoke.py`)
- `python -c "from labs.lab3.preprocessing.preprocessing import load_text_dataset, label_counts; tr, va, te = load_text_dataset(); print(len(tr), label_counts(tr), len(va), label_counts(va), len(te), label_counts(te), '<br' in ' '.join(tr['text']))"` -> 2000 {0: 1000, 1: 1000} 500 {0: 243, 1: 257} 500 {0: 254, 1: 246} False
- Environment: Python 3.12.6, transformers 5.20.2, torch 2.14.1+cpu, matplotlib 3.11.2, datasets 5.1.0, pytest 9.1.1

## Log (dated, one line per step)
- <date> <name>: scaffold created; interfaces agreed -> verified with `pytest labs/lab3/tests`
- 2026-10-25 Muội: preprocessing implemented (load_text_dataset, clean_text), verified with real IMDB and 24 tests passing

## Open issues
- (issue): expected / actual / error / env / relevant code / recent changes (use docs/templates/bug_report.md)

---

**Tiếng Việt**

## Tiếng Việt (cách dùng)
- **Bàn giao A -> B (tuần tự, không xen kẽ):** Member B chỉ bắt đầu code khi Member A đã xong HẾT:
  - dataset đã tokenize có input_ids, attention_mask, label
  - Bài 1 (sentiment_demo) in token và cảm xúc của 3 câu mẫu
  - mọi module của A ở trạng thái `done`, `pytest` qua, PR đã merge vào `main`, A đã nhắn tin bàn giao. Sau đó A chỉ trả lời câu hỏi và duyệt PR, không sửa module của B.
- Cập nhật file này sau MỖI bước: đổi trạng thái, ghi kết quả chạy thật (không ghi "chạy được" nếu chưa chạy).
- Mỗi dòng log gồm: ngày, người làm, module, đã thay đổi gì, kiểm chứng bằng cách nào.
- Nếu một module không cần cho lab này, ghi `N/A` kèm lý do.
- Lỗi cần ghi đủ 6 mục: mong đợi, thực tế, thông báo lỗi, môi trường, code liên quan, thay đổi gần đây.
