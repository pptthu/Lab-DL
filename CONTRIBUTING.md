# Contributing (Git rules)

- Branch from `main`. Module work: one branch per module, `<lab>/<module>` (`lab1/features`, `lab2/model`, `lab3/train`).
  Work that is not one module: `<type>/<topic>` (`docs/lab2-lab3-setup`, `chore/scaffold-lab1`). Never commit directly to `main`.
- One commit per logical step. Message format (Conventional Commits): `<type>(<lab>/<module>): <what changed>`, see the table below.
  The PR title uses the same format; with "Squash and merge" it becomes the commit on `main`.
- Stage files explicitly: `git add <file>`. Check `git status` and `git diff --staged` before committing.
- Never commit data, model files, outputs, secrets (`.gitignore` covers most of them). Never force-push shared branches.
- Open a Pull Request into `main` using the template; write `Closes #<issue number>`; the team leader reviews and merges.
- Before opening the PR: run `pytest labs/<lab>/tests`, update `progress_status.md`, and add your AI use to `AI_usage.md`.
- Do not rename functions or change signatures without telling the team leader (the interface is a contract).
- Add a bug report (docs/templates/bug_report.md) in the lab's `logbook/` when you fix a non-trivial bug.
- Handoff A -> B is sequential: Member B starts only after Member A's modules are merged into `main` and the handoff checklist in `progress_status.md` is complete.

## Commit messages

Format: `<type>(<scope>): <what changed>`. Scope = `<lab>/<module>` (e.g. `lab1/features`), or a short topic when no module fits
(`scaffold`, `repo`). Description: imperative, lower case, no final period, about 72 characters or fewer.

| Type | Use it when | Example |
|---|---|---|
| `feat` | new behaviour in a module (new function, new pipeline step) | `feat(lab1/model): implement MLP (logits output) and count_parameters` |
| `fix` | a bug fix (behaviour was wrong) | `fix(lab1/main): pass val_ratio instead of val_fraction to make_split` |
| `test` | add or change tests only | `test(lab1/train): add fake-data tests for train_model` |
| `docs` | documentation only: README, `*.md`, docstrings, notebook text or skeleton | `docs(lab1/model): update progress status and AI usage for part A` |
| `refactor` | restructure code, same behaviour and same results | `refactor(lab1/evaluate): reuse one loop for loss and accuracy` |
| `chore` | setup, config, `.gitignore`, requirements, scaffold | `chore(scaffold): setup multi-lab structure with verified smoke tests` |

Old messages on `main` stay as they are (history is not rewritten). How they map to the new format:

| Old message | New format |
|---|---|
| `lab1/model: implement MLP ...` | `feat(lab1/model): implement MLP ...` |
| `lab1/notebook: ...` | `docs(lab1/notebook): ...` (text/skeleton) or `feat(lab1/notebook): ...` (new analysis code) |
| `chore(scaffold): setup multi-lab structure ...` | already correct |
| `feat(lab1): complete EDA ...` | correct; better with the module: `feat(lab1/eda): complete EDA ...` |

---

**Tiếng Việt**

- Tạo nhánh từ `main`. Làm module: mỗi module một nhánh `<lab>/<module>` (`lab1/features`, `lab2/model`...).
  Việc không thuộc một module: `<type>/<chủ đề>` (`docs/lab2-lab3-setup`, `chore/scaffold-lab1`). Không commit trực tiếp vào `main`.
- Mỗi bước logic một commit. Tin nhắn (Conventional Commits): `<type>(<lab>/<module>): <việc đã làm>`, xem bảng dưới.
  Tiêu đề PR dùng cùng định dạng; khi "Squash and merge" tiêu đề này thành commit trên `main`.
- `git add <file>` từng file; xem `git status` và `git diff --staged` trước khi commit.
- Không commit dữ liệu, file model, kết quả, bí mật. Không force-push nhánh dùng chung.
- Mở Pull Request vào `main` theo template, ghi `Closes #<số issue>`; nhóm trưởng duyệt và merge.
- Trước khi mở PR: chạy `pytest labs/<lab>/tests`, cập nhật `progress_status.md`, ghi việc dùng AI vào `AI_usage.md`.
- Không đổi tên hàm/chữ ký khi chưa báo nhóm trưởng. Với lỗi khó, viết bug report vào `logbook/` của lab.
- Bàn giao A -> B là tuần tự: B chỉ bắt đầu sau khi module của A đã merge vào `main` và checklist bàn giao trong `progress_status.md` đã xong.

## Tin nhắn commit

Định dạng: `<type>(<phạm vi>): <việc đã làm>`. Phạm vi = `<lab>/<module>` (vd `lab1/features`), hoặc một chủ đề ngắn nếu không thuộc
module nào (`scaffold`, `repo`). Phần mô tả: viết bằng tiếng Anh, dạng mệnh lệnh, chữ thường, không có dấu chấm cuối, khoảng 72 ký tự trở xuống.

| Type | Dùng khi | Ví dụ |
|---|---|---|
| `feat` | thêm hành vi mới cho module (hàm mới, bước pipeline mới) | `feat(lab1/model): implement MLP (logits output) and count_parameters` |
| `fix` | sửa lỗi (hành vi đang sai) | `fix(lab1/main): pass val_ratio instead of val_fraction to make_split` |
| `test` | chỉ thêm hoặc sửa test | `test(lab1/train): add fake-data tests for train_model` |
| `docs` | chỉ tài liệu: README, `*.md`, docstring, chữ hoặc khung notebook | `docs(lab1/model): update progress status and AI usage for part A` |
| `refactor` | sắp xếp lại code, hành vi và kết quả không đổi | `refactor(lab1/evaluate): reuse one loop for loss and accuracy` |
| `chore` | cài đặt, cấu hình, `.gitignore`, requirements, scaffold | `chore(scaffold): setup multi-lab structure with verified smoke tests` |

Các commit cũ trên `main` giữ nguyên (không viết lại lịch sử). Cách đổi sang định dạng mới:

| Tin nhắn cũ | Định dạng mới |
|---|---|
| `lab1/model: implement MLP ...` | `feat(lab1/model): implement MLP ...` |
| `lab1/notebook: ...` | `docs(lab1/notebook): ...` (chữ/khung) hoặc `feat(lab1/notebook): ...` (thêm code phân tích mới) |
| `chore(scaffold): setup multi-lab structure ...` | đã đúng |
| `feat(lab1): complete EDA ...` | đúng; tốt hơn nếu ghi module: `feat(lab1/eda): complete EDA ...` |
