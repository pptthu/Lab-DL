# Lab 3: Hugging Face sentiment analysis and fine-tuning: progress status
_Last updated: 2026-10-27 by Muội_

| Module | Owner | Status (todo / doing / done / N/A) | Smoke run | Normal run | Notes |
|---|---|---|---|---|---|
| `sentiment_demo` | Member A (name TBD) | todo | - | - | |
| `eda` | Member A (name TBD) | todo | - | - | |
| `preprocessing` | Member A (name TBD) | todo | - | - | |
| `features` | Member A (Muội) | done | `pytest labs/lab3/tests/test_features.py`: 3 passed | `tokenize_dataset`: columns label, input_ids, attention_mask | Truncation only (max_length=256), NO fixed padding: Member B must use DataCollatorWithPadding. |
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
- pytest labs/lab3/tests/test_features.py -v -> PASS: 3 tests passed.
- pytest labs/lab3/tests/test_features.py labs/lab3/tests/test_smoke.py -v -> 11 tests passed in 0.46s.

## Log (dated, one line per step)
- <date> <name>: scaffold created; interfaces agreed -> verified with `pytest labs/lab3/tests`
- 2026-10-26 Muội: implemented `features.tokenize_dataset`; verified with fake-tokenizer tests and the real tokenizer.
- 2026-10-27 Muội: verified `features.tokenize_dataset` with 3 unit tests and 8 smoke tests; all 11 tests passed.

## Open issues
- (issue): expected / actual / error / env / relevant code / recent changes (use docs/templates/bug_report.md)
- None reported for the `features` module.
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
