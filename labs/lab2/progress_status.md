# Lab 2: Pre-trained CNNs and fine-tuning on CIFAR-10 (PyTorch): progress status
_Last updated: 2026-10-31 by Team Leader (Thư)_

| Module | Owner | Status (todo / doing / done / N/A) | Smoke run | Normal run | Notes |
|---|---|---|---|---|---|
| `eda` | Member A (Đại) | done | pass | pass | Merged to main via PR #32; verified with test_eda.py |
| `preprocessing` | Member A (Đại) | done | pass | pass | Merged to main via PR #32; verified with test_preprocessing.py |
| `features` | Member A (Đại) | done | pass | pass | Merged to main via PR #32; verified with test_feature.py |
| `model` | Member A (Đại) | done | pass | pass | Merged to main via PR #32; verified with test_model.py |
| `train` | Member B (Bảo) | done | pass | pass | Merged to main via PR #25; verified with test_train.py |
| `evaluate` | Member B (Bảo) | done | pass | pass | Merged to main via PR #25; verified with test_evaluate.py |
| `experiment` | Member B (Bảo) | done | pass | pass | Merged to main via PR #25; verified with test_experiment.py |
| `main.py`, README, integration | Team leader (Thư) | doing | pass | - | README completed; main.py end-to-end wiring pending |

## Handoff A -> B (sequential, no interleaving)
Member B starts coding only after ALL of the following are true:
- [x] split JSON committed in artifacts/splits/
- [x] make_loaders returns a batch [B,3,H,W] with labels 0-9
- [x] build_model returns logits [B,10]; describe_model prints parameter counts
- [x] all Member A modules are `done` in this file and `pytest labs/lab2/tests` passes
- [x] Member A's PRs are merged into `main`; Member B branches from the updated `main`
- [x] Member A posts the handoff message (what was done, exact run commands, known problems)
After the handoff Member A only answers questions and reviews PRs; Member A does not edit Member B's modules.

## Audit: what actually runs
- `pytest labs/lab2/tests/test_smoke.py` -> 8 passed
- `pytest labs/lab2/tests/test_eda.py` -> 1 passed
- `pytest labs/lab2/tests/test_preprocessing.py` -> 1 passed
- `pytest labs/lab2/tests/test_feature.py` -> 1 passed
- `pytest labs/lab2/tests/test_model.py` -> 2 passed
- `pytest labs/lab2/tests/test_train.py` -> 4 passed
- `pytest labs/lab2/tests/test_evaluate.py` -> 4 passed
- `pytest labs/lab2/tests/test_experiment.py` -> 2 passed
- `pytest labs/lab2/tests/` -> 23 passed in 12.20s

## Log (dated, one line per step)
- 2026-10-17 pptthu: scaffold created; interfaces agreed -> verified with `pytest labs/lab2/tests`
- 2026-10-26 Bảo (Member B): implemented train, evaluate, experiment modules on `feat/lab2-member-b`, added synthetic unit tests and updated presentation notebook sections -> verified with `pytest labs/lab2/tests` (18 passed).
- 2026-10-31 Thư (Leader): verified all 23 tests passing (Member A + Member B modules merged on main); audited baseline ResNet-18 training run and evaluation metrics; authored comprehensive bilingual README.md -> verified with `pytest labs/lab2/tests/` (23 passed).

## Open issues
- `labs/lab2/main.py`: end-to-end CLI orchestration pipeline remains to be wired by the team leader (currently raises NotImplementedError).
- Multi-model hyperparameter grid comparison CSV export: partial runs logged on CPU in `artifacts/runs/`; full grid completion ongoing.

---

**Tiếng Việt**

## Tiếng Việt (cách dùng)
- **Bàn giao A -> B (tuần tự, không xen kẽ):** Member B chỉ bắt đầu code khi Member A đã xong HẾT:
  - file JSON split đã commit trong artifacts/splits/
  - make_loaders trả batch [B,3,H,W], nhãn 0-9
  - build_model trả logits [B,10]; describe_model in số tham số
  - mọi module của A ở trạng thái `done`, `pytest` qua, PR đã merge vào `main`, A đã nhắn tin bàn giao. Sau đó A chỉ trả lời câu hỏi và duyệt PR, không sửa module của B.
- Cập nhật file này sau MỖI bước: đổi trạng thái, ghi kết quả chạy thật (không ghi "chạy được" nếu chưa chạy).
- Mỗi dòng log gồm: ngày, người làm, module, đã thay đổi gì, kiểm chứng bằng cách nào.
- Nếu một module không cần cho lab này, ghi `N/A` kèm lý do.
- Lỗi cần ghi đủ 6 mục: mong đợi, thực tế, thông báo lỗi, môi trường, code liên quan, thay đổi gần đây.
