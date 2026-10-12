# Lab 2: Pre-trained CNNs and fine-tuning on CIFAR-10 (PyTorch): progress status
_Last updated: <date> by <name>_

| Module | Owner | Status (todo / doing / done / N/A) | Smoke run | Normal run | Notes |
|---|---|---|---|---|---|
| `eda` | Member A (name TBD) | todo | - | - | |
| `preprocessing` | Member A (name TBD) | todo | - | - | |
| `features` | Member A (name TBD) | todo | - | - | |
| `model` | Member A (name TBD) | todo | - | - | |
| `train` | Member B (name TBD) | todo | - | - | |
| `evaluate` | Member B (name TBD) | todo | - | - | |
| `experiment` | Member B (name TBD) | todo | - | - | |
| `main.py`, README, integration | Team leader | todo | - | - | |

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
- (command) -> (result: pass/fail + key output)

## Log (dated, one line per step)
- <date> <name>: scaffold created; interfaces agreed -> verified with `pytest labs/lab2/tests`

## Open issues
- (issue): expected / actual / error / env / relevant code / recent changes (use docs/templates/bug_report.md)

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
