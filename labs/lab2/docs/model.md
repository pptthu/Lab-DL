# Lab 2 / model: guide for AI agents

## For AI agents (read first)

| Field | Value |
|---|---|
| Module | `labs/lab2/model/model.py` |
| Owner | Member A (Đại), assigned by the team leader; PR #48 changed VGG unfreezing |
| Status | done in `progress_status.md` (Smoke-tested by unit tests) |
| Assigned work | see the GitHub Issues and `progress_status.md` of the lab; never assign work to yourself, ask the team leader |
| Files you may edit | `labs/lab2/model/model.py`, `labs/lab2/tests/test_model.py`, `labs/lab2/tests/test_model_unfreezing.py`, this guide |
| Files you must not edit | other modules, saved checkpoints, other owners' notebook cells |
| Agreed signatures | `build_model(name: str, num_classes: int = 10, freeze_backbone: bool = True, pretrained: bool = True) -> nn.Module`; `describe_model(model) -> dict`; `unfreeze_last_blocks(model, n_blocks: int) -> None` (`project_description.md`) |
| Depends on / used by | `torchvision.models` / `train`, `experiment`, `main.py`, notebook |
| Prohibited | adding architectures beyond ResNet-18, DenseNet-121, VGG-16 without the team leader; adding softmax to the head (CrossEntropyLoss expects logits); downloading weights in unit tests (use `pretrained=False`) |
| Checks | Unit: `python -m pytest labs/lab2/tests/test_model.py labs/lab2/tests/test_model_unfreezing.py -q` (`test_model.py` loads pretrained ResNet-18/DenseNet-121 weights: download on first run) |
| Done when | tests pass; this guide updated |

## What the brief requires

- Brief requirement: load pre-trained torchvision models, replace the final layer, freeze (transfer learning) or unfreeze the last layers (fine-tuning), print the architecture.
- Team choices: ResNet-18 (baseline), DenseNet-121, VGG-16; ImageNet weights.

## How the module works (current code)

- `build_model`: loads the backbone, sets `requires_grad=False` on all parameters if `freeze_backbone`, then replaces the head with a new `nn.Linear` (trainable): ResNet `fc` (512 -> 10), DenseNet `classifier` (1024 -> 10), VGG `classifier[6]` (4096 -> 10). Unknown names raise an error.
- `describe_model`: total and trainable parameter counts and layer names.
- `unfreeze_last_blocks(model, n)`: ResNet: last `n` of `layer1..layer4` plus `fc`; VGG-16: last `n` convolution blocks (block = convs up to a MaxPool, `n=1` is `features[24:31]`) plus the classifier; DenseNet-121: dense blocks inside `features`.

## Inputs, outputs and pitfalls

- Output logits `[B, 10]`.
- **BatchNorm:** `requires_grad=False` does not freeze BatchNorm running statistics; in `model.train()` they still update during training. The code does not put frozen BatchNorm layers in eval mode. Whether "frozen" should also mean fixed BatchNorm statistics is not decided.

## How to verify

| Type | Command | Expected |
|---|---|---|
| Unit test | see Checks | all pass (`test_model_unfreezing.py`: 9 passed, 2026-11-04) |
| Smoke test | `python -c "from labs.lab2.model.model import build_model, describe_model; print(describe_model(build_model('resnet18', pretrained=False)))"` | trainable parameters = head only (5,130) |

## Open questions / limitations

- BatchNorm behaviour under a frozen backbone (see above).

---

**Tiếng Việt**

- **Việc được giao:** xem GitHub Issue và `progress_status.md` của lab; không tự giao việc cho mình, hỏi nhóm trưởng.
- **Dành cho AI:** người phụ trách Member A (Đại), do nhóm trưởng giao; PR #48 đã sửa phần mở khóa VGG. Được sửa `model.py`, hai file test model, file này. Chữ ký 3 hàm khớp `project_description.md`. Cấm thêm kiến trúc ngoài ResNet-18, DenseNet-121, VGG-16 khi chưa hỏi nhóm trưởng, cấm thêm softmax, cấm tải trọng số trong unit test.
- **Đề:** nạp model pre-trained, thay lớp cuối, đóng băng (transfer learning) hoặc mở khóa các lớp cuối (fine-tuning).
- **Cách hoạt động:** đóng băng toàn bộ rồi thay head mới; `unfreeze_last_blocks` mở `n` khối cuối của ResNet, VGG-16 hoặc DenseNet-121.
- **Lưu ý BatchNorm:** `requires_grad=False` không giữ cố định thống kê của BatchNorm khi model ở chế độ train; nhóm chưa quyết "đóng băng" có bao gồm việc này không.
