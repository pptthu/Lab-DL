# Contributing (Git rules)

- Branch from `main`, one branch per module: `lab1/features`, `lab2/model`, `lab3/train`. Never commit directly to `main`.
- One commit per logical step. Message format: `<lab>/<module>: <what changed>` (e.g. `lab1/features: compute train mean/std`).
- Stage files explicitly: `git add <file>`. Check `git status` and `git diff --staged` before committing.
- Never commit data, model files, outputs, secrets (`.gitignore` covers most of them). Never force-push shared branches.
- Open a Pull Request into `main` using the template; write `Closes #<issue number>`; the team leader reviews and merges.
- Before opening the PR: run `pytest labs/<lab>/tests`, update `progress_status.md`, and add your AI use to `AI_usage.md`.
- Do not rename functions or change signatures without telling the team leader (the interface is a contract).
- Add a bug report (docs/templates/bug_report.md) in the lab's `logbook/` when you fix a non-trivial bug.
- Handoff A -> B is sequential: Member B starts only after Member A's modules are merged into `main` and the handoff checklist in `progress_status.md` is complete.

---

**Tiếng Việt**

- Tạo nhánh từ `main`, mỗi module một nhánh: `lab1/features`, `lab2/model`... Không commit trực tiếp vào `main`.
- Mỗi bước logic một commit. Tin nhắn: `<lab>/<module>: <việc đã làm>`.
- `git add <file>` từng file; xem `git status` và `git diff --staged` trước khi commit.
- Không commit dữ liệu, file model, kết quả, bí mật. Không force-push nhánh dùng chung.
- Mở Pull Request vào `main` theo template, ghi `Closes #<số issue>`; nhóm trưởng duyệt và merge.
- Trước khi mở PR: chạy `pytest labs/<lab>/tests`, cập nhật `progress_status.md`, ghi việc dùng AI vào `AI_usage.md`.
- Không đổi tên hàm/chữ ký khi chưa báo nhóm trưởng. Với lỗi khó, viết bug report vào `logbook/` của lab.
- Bàn giao A -> B là tuần tự: B chỉ bắt đầu sau khi module của A đã merge vào `main` và checklist bàn giao trong `progress_status.md` đã xong.
