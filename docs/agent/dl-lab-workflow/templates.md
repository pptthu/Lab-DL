# Templates (Mẫu điền)

Copy a block, fill in the `<...>` parts. Keep each file short and accurate.

---

## 1. project_description.md (Define + Design)

```markdown
# <Project name>

## Define
- **Problem:** <1-3 sentences>
- **Target users:** <who uses / grades it>
- **Input / Output:** <data format, shapes> → <what the program produces>
- **Core features:** <3-6 bullets>
- **Technology constraints:** <Python version, PyTorch, CPU/GPU, time limit, libraries allowed>
- **Evaluation criteria:** <metrics, deliverables, what the lecturer checks>

## Design
- **Modules:** <name: purpose> (one script per module)
- **Folder structure:** <tree>
- **Data flow:** <module A → module B → ...>
- **Interfaces:** <function signatures with input/output types and shapes>
- **Milestones:** <M1, M2, ... with a done-check each>
- **Technical risks:** <risk → mitigation>
```

## 2. progress_status.md (update after every step)

```markdown
# Progress status
_Last updated: <date> by <name>_

| Module | Owner | Status (todo / doing / done) | Smoke run | Normal run | Tests | Notes |
|---|---|---|---|---|---|---|
| data | <name> | todo | - | - | - | |

## Audit (what actually runs)
- <command> → <result: pass/fail + key output>

## Log
- <date> <name>: <module> <what changed> → <verified how>

## Open issues
- <issue>: expected / actual / error / env / relevant code / recent changes
```

## 3. Prompt templates

**Plan**
```
Goal: <...>
Input/Output: <...>
Features: <...>
Constraints: <...>
Proposed structure and modules: <...>
Milestones: <...>
Risks: <...>
Task: propose a plan only. Do not write code yet.
```

**Implement**
```
Module: <name> (file <path>)
Goal: <...>
Input/Output: <types and shapes>
Interface: <function signatures>
Rules: change only this module; give a smoke-run command; explain the code in Vietnamese.
```

**Review**
```
Review <file> against the interface in project_description.md.
List: bugs, shape/dtype/device issues, data leakage, missing steps, unclear names.
Do not rewrite; list findings by severity.
```

**Debug**
```
Expected: <...>
Actual: <...>
Error (verbatim): <...>
Env: <python, torch, OS, CPU/GPU>
Relevant code: <snippet>
Recent changes: <...>
Help me find the cause by asking me questions first.
```

## 4. PR / commit description

```markdown
## What
<module and change>
## Why
<goal / task link>
## How verified
<command + real output summary>
## Risks / follow-up
<...>
```
Commit message: `<module>: <what changed>`

## 5. README.md skeleton

```markdown
# <Project name>
## 1. Project overview
## 2. Problem statement
## 3. Main features
## 4. System structure
<folder tree that matches the real repo>
## 5. How to run
<setup, data download, commands>
```

## 6. AI_usage.md skeleton

```markdown
# AI usage
- **Tools used:** <names, versions>
- **Main purpose:** <planning / code draft / debugging / docs>
- **Student contribution:** <what the student designed, wrote, decided>
- **Important AI-generated components:** <module/function list>
- **Prompts purpose:** <one line per important prompt>
- **Student modification:** <what was changed after AI output and why>
- **Validation method:** <how it was checked: smoke run, tests, manual inspection>
```
