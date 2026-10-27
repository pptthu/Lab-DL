# Lab 3: Hugging Face sentiment analysis and fine-tuning: progress status
_Last updated: 2026-10-27 by Muội_

| Module | Owner | Status (todo / doing / done / N/A) | Smoke run | Normal run | Notes |
|---|---|---|---|---|---|
| `sentiment_demo` | Member A (Muội) | done | `pytest labs/lab3/tests/test_sentiment_demo.py -v`: 4 passed | `show_tokens` + `analyze_sentences`: 3 samples OK (POSITIVE / NEGATIVE / POSITIVE) | `model_name=None` = DistilBERT SST-2 for BOTH functions. Needs internet for the first model download. |
| `eda` | Member A (Muội) | todo | - | - | |
| `preprocessing` | Member A (Muội) | done | `pytest labs/lab3/tests`: 24 passed | real IMDB run: 2000/500/500, both labels in each split, no `<br` left | Shuffle(seed) before select; val cut from train, test from test. |
| `features` | Member A (Muội) | done | `pytest labs/lab3/tests/test_features.py -v`: 3 passed | `tokenize_dataset`: columns label, input_ids, attention_mask | Truncation only (max_length=256), NO fixed padding: Member B must use DataCollatorWithPadding. |
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
- `python -c "from labs.lab3.sentiment_demo.sentiment_demo import show_tokens, analyze_sentences; s = ['I love this movie', 'The plot was boring', 'Not bad at all']; [print(show_tokens(x)) for x in s]; print(analyze_sentences(s))"` -> tokens start with `[CLS]` (101) and end with `[SEP]` (102); POSITIVE 0.9999 / NEGATIVE 0.9998 / POSITIVE 0.9991 (positive, negative, sentence with negation).
- Environment: Python 3.12.6, transformers 5.20.2, torch 2.14.1+cpu, matplotlib 3.11.2, datasets 5.1.0, pytest 9.1.1

## Log (dated, one line per step)
- 2026-10-27 Muội: implemented sentiment_demo; verified 4 unit tests and 12 tests with smoke tests passed.
- <date> <name>: scaffold created; interfaces agreed -> verified with `pytest labs/lab3/tests`
- 2026-10-25 Muội: preprocessing implemented (load_text_dataset, clean_text), verified with real IMDB and 24 tests passing

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


