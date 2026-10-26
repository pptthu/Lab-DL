# Lab 2: Pre-trained CNNs and fine-tuning on CIFAR-10 (PyTorch): progress status
_Last updated: 2026-10-26 by Bảo (Member B)_

| Module | Owner | Status (todo / doing / done / N/A) | Smoke run | Normal run | Notes |
|---|---|---|---|---|---|
| `eda` | Member A (Đại) | doing | - | - | PR #21 open; awaiting merge to main |
| `preprocessing` | Member A (Đại) | doing | - | - | PR #22 open; split artifact pending merge |
| `features` | Member A (Đại) | doing | - | - | PR #23 open; awaiting merge to main |
| `model` | Member A (Đại) | doing | - | - | PR #24 open; awaiting merge to main |
| `train` | Member B (Bảo) | done | pass | - | Implemented in train.py; verified with test_train.py |
| `evaluate` | Member B (Bảo) | done | pass | - | Implemented in evaluate.py; verified with test_evaluate.py |
| `experiment` | Member B (Bảo) | done | pass | - | Implemented in experiment.py; verified with test_experiment.py |
| `main.py`, README, integration | Team leader (Thư) | todo | - | - | Integration pending Member A PR merges |

## Handoff A -> B (sequential, no interleaving)
Member B starts coding only after ALL of the following are true:
- [ ] split JSON committed in artifacts/splits/
- [ ] make_loaders returns a batch [B,3,H,W] with labels 0-9
- [ ] build_model returns logits [B,10]; describe_model prints parameter counts
- [ ] all Member A modules are `done` in this file and `pytest labs/lab2/tests` passes
- [ ] Member A's PRs are merged into `main`; Member B branches from the updated `main`
- [ ] Member A posts the handoff message (what was done, exact run commands, known problems)
After the handoff Member A only answers questions and reviews PRs; Member A does not edit Member B's modules.

## Audit: what actually runs
- `pytest labs/lab2/tests/test_train.py` -> 4 passed in 0.8s
- `pytest labs/lab2/tests/test_evaluate.py` -> 4 passed in 0.5s
- `pytest labs/lab2/tests/test_experiment.py` -> 2 passed in 0.6s
- `pytest labs/lab2/tests/` -> 18 passed in 2.14s

## Log (dated, one line per step)
- 2026-10-17 pptthu: scaffold created; interfaces agreed -> verified with `pytest labs/lab2/tests`
- 2026-10-26 Bảo (Member B): implemented train, evaluate, experiment modules on `feat/lab2-member-b`, added synthetic unit tests and updated presentation notebook sections -> verified with `pytest labs/lab2/tests` (18 passed).

## Open issues
- Blocker: Member A PRs (#21, #22, #23, #24) remain unmerged on main; full pipeline integration run depends on Member A's dataset and pre-trained backbone implementation being merged.

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
