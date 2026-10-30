# Lab 3: Hugging Face sentiment analysis and fine-tuning: progress status
_Last updated: 2026-10-30 by Member B (ngohongphuc2805ag-hash)_

| Module | Owner | Status (todo / doing / done / N/A) | Smoke run | Normal run | Notes |
|---|---|---|---|---|---|
| `sentiment_demo` | Member A (Muội) | done | `pytest labs/lab3/tests/test_sentiment_demo.py -v`: 4 passed | `show_tokens` + `analyze_sentences`: 3 samples OK (POSITIVE / NEGATIVE / POSITIVE) | `model_name=None` = DistilBERT SST-2 for BOTH functions. Needs internet for the first model download. |
| `eda` | Member A (Muội) | done | `pytest labs/lab3/tests/test_eda.py`: 3 passed | `run_eda`: n=2000, 3 PNG saved | Figures go to `artifacts/figures/` (not committed) |
| `preprocessing` | Member A (Muội) | done | `pytest labs/lab3/tests`: 24 passed | real IMDB run: 2000/500/500, both labels in each split, no `<br` left | Shuffle(seed) before select; val cut from train, test from test. |
| `features` | Member A (Muội) | done | `pytest labs/lab3/tests/test_features.py -v`: 3 passed | `tokenize_dataset`: columns label, input_ids, attention_mask | Truncation only (max_length=256), NO fixed padding: Member B must use DataCollatorWithPadding. |
| `model` | Member B (ngohongphuc2805ag-hash) | done | `pytest labs/lab3/tests/test_model.py`: 8 passed | real load of `distilbert-base-uncased`: see the `evaluate` row (same run) | Merged in PR #38. Tokenizer and model from the SAME checkpoint; `id2label` {0: NEGATIVE, 1: POSITIVE}. The warning about a new classifier head is expected. |
| `train` | Member B (ngohongphuc2805ag-hash) | done | `pytest labs/lab3/tests/test_train.py`: 8 passed | tiny fine-tune + small full run: see the `evaluate` row (same run) | Merged in PR #39. Padding per batch (DataCollatorWithPadding); extra OPTIONAL argument `tokenizer`; `eval_strategy` / `processing_class` names are detected for the installed transformers. |
| `evaluate` | Member B (ngohongphuc2805ag-hash) | doing (set to done after the TODO(B) below) | `pytest labs/lab3/tests/test_evaluate.py`: 16 passed | TODO(B): run the notebook locally (sections 5.2 to 10.2) and paste here the line printed at the end of section 10.2, plus the result of the tiny fine-tune (4 steps, command in the PR description) | Metrics are NumPy only (no network). `predict_texts` has 2 extra keyword-only optional arguments (`max_length`, `batch_size`); helpers `classification_metrics` and `zero_shot_labels` added. Test set is used only in `evaluate_on_test` and section 10. |
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
- `pytest labs/lab3/tests` -> 24 passed (16 Member A tests + 8 `test_smoke.py`)
- `python -c "from labs.lab3.preprocessing.preprocessing import load_text_dataset, label_counts; tr, va, te = load_text_dataset(); print(len(tr), label_counts(tr), len(va), label_counts(va), len(te), label_counts(te), '<br' in ' '.join(tr['text']))"` -> 2000 {0: 1000, 1: 1000} 500 {0: 243, 1: 257} 500 {0: 254, 1: 246} False
- `python -c "from labs.lab3.sentiment_demo.sentiment_demo import show_tokens, analyze_sentences; s = ['I love this movie', 'The plot was boring', 'Not bad at all']; [print(show_tokens(x)) for x in s]; print(analyze_sentences(s))"` -> tokens start with `[CLS]` (101) and end with `[SEP]` (102); POSITIVE 0.9999 / NEGATIVE 0.9998 / POSITIVE 0.9991 (positive, negative, sentence with negation).
- Environment: Python 3.12.6, transformers 5.20.2, torch 2.14.1+cpu, matplotlib 3.11.2, datasets 5.1.0, pytest 9.1.1
- Member B (2026-10-30): `pytest labs/lab3/tests` -> 56 passed (8 model + 8 train + 16 evaluate + 24 from Member A and `test_smoke.py`); with `tests/` of the repo root: 60 passed.
- Member B: `compute_metrics` accuracy and F1 compared with scikit-learn on 300 random cases (random sizes 1 to 599) -> maximum absolute difference 0.
- Member B: the code of the notebook cells of sections 5.2, 6.2, 7, 8, 9, 10 was executed against fake objects only (both branches: train and load the saved model). A real model run (no Hugging Face access in the drafting environment) is still to be recorded in the `evaluate` row above.
## Log (dated, one line per step)
- 2026-10-29 Member B: `model` (PR #38) and `train` (PR #39) merged into main; fake-`transformers` tests pass (8 + 8).
- 2026-10-30 Member B: implemented `evaluate` (`compute_metrics`, `evaluate_on_test`, `predict_texts`, helpers `classification_metrics`, `zero_shot_labels`); 16 tests; `pytest labs/lab3/tests` -> 56 passed.
- 2026-10-30 Member B: wrote notebook sections 5.2, 6.2, 7, 8, 9 (optional, off by default), 10.1, 10.2; only Member B's cells were changed.
- 2026-10-27 Muội: implemented sentiment_demo; verified 4 unit tests and 12 tests with smoke tests passed.
- <date> <name>: scaffold created; interfaces agreed -> verified with `pytest labs/lab3/tests`
- 2026-10-25 Muội: preprocessing implemented (load_text_dataset, clean_text), verified with real IMDB and 24 tests passing
- 2026-10-26 Muội: implemented `features.tokenize_dataset`; verified with fake-tokenizer tests and the real tokenizer.
- 2026-10-27 Muội: verified `features.tokenize_dataset` with 3 unit tests and 8 smoke tests; all 11 tests passed.

## Open issues
- (issue): expected / actual / error / env / relevant code / recent changes (use docs/templates/bug_report.md)
- None reported for the `features` module
- Member B: real-model run (tiny fine-tune, full small run, notebook top-to-bottom with saved outputs) not yet recorded: see the `evaluate` row.
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