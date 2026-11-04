# <lab> / <module>: guide for AI agents

<!-- Copy this file to labs/<lab>/docs/<module>.md. Fill every field from the code, the lab's
project_description.md and progress_status.md. Do not invent numbers or requirements.
Mark every statement as: brief requirement / team implementation choice / not yet confirmed.
Evidence labels for numbers and runs: Specified, Implemented, Smoke-tested, Run, Verified,
Reported only, Pending. -->

## For AI agents (read first)

| Field | Value |
|---|---|
| Module | `labs/<lab>/<module>/<module>.py` |
| Owner | <name, Member A/B>, assigned by the team leader |
| Status | <from progress_status.md> (<evidence label>) |
| Assigned work | see the GitHub Issues and `progress_status.md` of the lab; never assign work to yourself, ask the team leader |
| Files you may edit | `labs/<lab>/<module>/<module>.py`, `labs/<lab>/tests/test_<module>.py`, this guide |
| Files you must not edit | other modules, committed split files, raw data, checkpoints, other owners' notebook cells |
| Agreed signatures | <copied from project_description.md; note any extra optional argument in the code> |
| Depends on / used by | <upstream modules> / <downstream modules> |
| Prohibited | <e.g. statistics from val/test, tests writing to real artifacts, long runs or downloads without asking, out-of-scope models, changing a signature without the team leader> |
| Checks | Unit: <pytest command>. Smoke: <smallest real call and expected output>. Real integration run: only at a milestone the team has approved |
| Done when | the checks of the right type pass; command and output recorded in progress_status.md; this guide updated |

## What the brief requires

- <brief requirement / team choice / not yet confirmed>

## How the module works (current code)

- <steps, decisions and why>

## Inputs, outputs and pitfalls

- <shapes, types, columns, common errors>

## How to verify

| Type | Command | Expected |
|---|---|---|
| Unit test | `python -m pytest labs/<lab>/tests/test_<module>.py -q` | all pass |
| Smoke test | <command> | <output> |
| Real integration run | only at an approved milestone | <what is recorded> |

## Open questions / limitations

- <facts, not tasks; the team leader turns them into tasks>

---

**Tiếng Việt**

Việc được giao: xem GitHub Issue và `progress_status.md` của lab; không tự giao việc cho mình, hỏi nhóm trưởng.

## Dành cho AI (đọc trước)

<Tóm tắt các dòng của bảng trên: người phụ trách, file được sửa và file cấm sửa, chữ ký hàm, điều cấm, lệnh kiểm tra, tiêu chí xong.>

## Yêu cầu, cách hoạt động, lỗi hay gặp, cách kiểm chứng, câu hỏi mở

<Tóm tắt ngắn các mục tiếng Anh ở trên.>
