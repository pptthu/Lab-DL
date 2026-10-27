# Lab 3: Hugging Face sentiment analysis and fine-tuning: progress status
_Last updated: <date> by <name>_

| Module | Owner | Status (todo / doing / done / N/A) | Smoke run | Normal run | Notes |
|---|---|---|---|---|---|
| `sentiment_demo` | Member A (Muội) | done | `pytest labs/lab3/tests/test_sentiment_demo.py -v`: 4 passed | `show_tokens` + `analyze_sentences`: 3 samples OK (POSITIVE / NEGATIVE / POSITIVE) | `model_name=None` = DistilBERT SST-2 for BOTH functions. Needs internet for the first model download. |
| `eda` | Member A (name TBD) | todo | - | - | |
| `preprocessing` | Member A (name TBD) | todo | - | - | |
| `features` | Member A (name TBD) | todo | - | - | |
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
- (command) -> (result: pass/fail + key output)

## Log (dated, one line per step)
- 2026-10-27 Muội: implemented sentiment_demo; verified 4 unit tests and 12 tests with smoke tests passed.
- <date> <name>: scaffold created; interfaces agreed -> verified with `pytest labs/lab3/tests`

## Open issues
- (issue): expected / actual / error / env / relevant code / recent changes (use docs/templates/bug_report.md)

---

**Tiáº¿ng Viá»‡t**

## Tiáº¿ng Viá»‡t (cÃ¡ch dÃ¹ng)
- **BÃ n giao A -> B (tuáº§n tá»±, khÃ´ng xen káº½):** Member B chá»‰ báº¯t Ä‘áº§u code khi Member A Ä‘Ã£ xong Háº¾T:
  - dataset Ä‘Ã£ tokenize cÃ³ input_ids, attention_mask, label
  - BÃ i 1 (sentiment_demo) in token vÃ  cáº£m xÃºc cá»§a 3 cÃ¢u máº«u
  - má»i module cá»§a A á»Ÿ tráº¡ng thÃ¡i `done`, `pytest` qua, PR Ä‘Ã£ merge vÃ o `main`, A Ä‘Ã£ nháº¯n tin bÃ n giao. Sau Ä‘Ã³ A chá»‰ tráº£ lá»i cÃ¢u há»i vÃ  duyá»‡t PR, khÃ´ng sá»­a module cá»§a B.
- Cáº­p nháº­t file nÃ y sau Má»–I bÆ°á»›c: Ä‘á»•i tráº¡ng thÃ¡i, ghi káº¿t quáº£ cháº¡y tháº­t (khÃ´ng ghi "cháº¡y Ä‘Æ°á»£c" náº¿u chÆ°a cháº¡y).
- Má»—i dÃ²ng log gá»“m: ngÃ y, ngÆ°á»i lÃ m, module, Ä‘Ã£ thay Ä‘á»•i gÃ¬, kiá»ƒm chá»©ng báº±ng cÃ¡ch nÃ o.
- Náº¿u má»™t module khÃ´ng cáº§n cho lab nÃ y, ghi `N/A` kÃ¨m lÃ½ do.
- Lá»—i cáº§n ghi Ä‘á»§ 6 má»¥c: mong Ä‘á»£i, thá»±c táº¿, thÃ´ng bÃ¡o lá»—i, mÃ´i trÆ°á»ng, code liÃªn quan, thay Ä‘á»•i gáº§n Ä‘Ã¢y.


