# Lab 2: Pre-trained CNNs and Fine-tuning on CIFAR-10 (PyTorch): Progress Status
_Last updated: 2026-10-13 by AI agent (Antigravity)_

## Module Status

| Module | Owner | Status | Smoke run | Notes |
|--------|-------|--------|-----------|-------|
| `eda` | Member A | **VERIFIED** | PASS (synthetic) | run_eda implemented; EDA on real data pending dataset download |
| `preprocessing` | Member A | **VERIFIED** | PASS (36/36 tests) | load_raw_dataset, make_split, save_split, load_split |
| `features` | Member A | **VERIFIED** | PASS (36/36 tests) | build_transforms, make_loaders with _TransformSubset fix |
| `model` | Member A | **VERIFIED** | PASS (36/36 tests) | build_model, describe_model, unfreeze_last_blocks; _FrozenBNModel wrapper |
| `train` | Member B | **VERIFIED** | PASS (36/36 tests) | train_model, save_checkpoint, load_checkpoint |
| `evaluate` | Member B | **VERIFIED** | PASS (36/36 tests) | evaluate, confusion_matrix, top_confusions, full_evaluation |
| `experiment` | Member B | **VERIFIED** | PASS (interface) | run_comparison, save_results; real data run pending |
| `main.py`, README, integration | Team leader | **VERIFIED** | PASS (CLI importable) | All modes implemented; real pipeline run pending dataset |
| `ORAL_EXAM_VI.md` | Agent | **COMPLETED** | N/A | 22 questions with deep explanations |

## Handoff A → B
✅ All Member A modules: VERIFIED
✅ All Member B modules: VERIFIED
✅ All interface contracts: 36/36 tests pass
✅ Full pipeline: integrated in main.py

## Test Results (2026-10-13)

```
Command: pytest labs/lab2/tests/test_smoke.py -v
Result:  36 passed, 0 failed, 0 skipped in 3.57s
```

### Test breakdown:
- `test_module_interface` (7 parametrized) — ALL PASS ✅
- `test_main_importable` — PASS ✅
- `TestModelOutputShape` (3) — ALL PASS ✅
- `TestFrozenParameters` (4) — ALL PASS ✅
- `TestTrainingLoop` (4) — ALL PASS ✅
- `TestEvaluate` (5) — ALL PASS ✅
- `TestPreprocessing` (5) — ALL PASS ✅
- `TestTransforms` (4) — ALL PASS ✅
- `TestDescribeModel` (2) — ALL PASS ✅
- `TestCPUExecution` (1) — ALL PASS ✅

## Experiments

| Experiment | Status | Reason |
|-----------|--------|--------|
| A — Scratch | NOT_RUN | Awaiting CIFAR-10 download (~170MB, slow network, ~60min) |
| B — Feature Extraction | NOT_RUN | Awaiting CIFAR-10 download |
| C — Fine-tuning | NOT_RUN | Awaiting CIFAR-10 download |
| EDA | NOT_RUN | Awaiting CIFAR-10 download |

## Bugs Found and Fixed (2026-10-13)

### Bug 1: `unfreeze_last_blocks` not working
- **Expected:** `n_blocks=1` unfreezes `layer4` of ResNet-18.
- **Actual:** `n_blocks=1` took the last element of `[layer1, layer2, layer3, layer4, fc]`, which was `fc` (already trainable) → no change.
- **Fix:** Removed `fc` from the blocks list in `unfreeze_last_blocks`; backbone-only blocks now = `[layer1, layer2, layer3, layer4]`.
- **Test:** `TestFrozenParameters::test_unfreeze_last_blocks_adds_trainable` → PASS.

### Bug 2: Checkpoint round-trip test failure
- **Expected:** `out_pre ≈ out_post` after save/load checkpoint.
- **Actual:** Model was in train mode when computing `out_pre` (BN uses batch stats), but eval mode for `out_post` (BN uses running stats) → different outputs.
- **Fix:** Set `model.eval()` before computing `out_pre` in the test.
- **Test:** `TestTrainingLoop::test_checkpoint_save_load_roundtrip` → PASS.

## Audit: Commands Actually Run

```bash
# 2026-10-13:
# Environment check:
python3 --version  # → Python 3.14.8
# torch check:
python3 -c "import torch; print(torch.__version__)"  # → 2.9.1+cu128 (CPU)
# Test run 1 (with 2 failures):
pytest labs/lab2/tests/test_smoke.py -v  # → 34/36 pass
# Bug fixes applied.
# Test run 2 (all pass):
pytest labs/lab2/tests/test_smoke.py -v  # → 36/36 pass ✅
# Re-verification of full test suite:
python3 -m pytest labs/lab2/tests -v   # → 36 passed in 8.62s ✅
# ResNet-18 ImageNet pre-trained weights verification:
python3 -c "import torchvision.models as models; m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)" # → Cached successfully (44.7MB) ✅
# Dataset download background task:
# PID task-73 downloading CIFAR-10 into labs/lab2/raw_data/ (progress: ~74MB / 170MB, running)
```

## Open Issues

1. **CIFAR-10 download:** In progress (~47kB/s, ETA ~60 minutes). Run ID: task-73.
   - Once complete: run `python -m labs.lab2.main --mode smoke` to verify.
   - Then: `python -m labs.lab2.main --mode full --experiments scratch,feature_extraction,fine_tuning`.

2. **image_size=64 deviation:** Using 64×64 instead of 224×224 for CPU feasibility.
   - This affects pretrained feature quality. Documented in project_description.md.
   - To run at 224: `python -m labs.lab2.main --mode full --image-size 224` (expect very slow).

3. **TensorBoard:** Requires `tensorboard` package. If not installed, SummaryWriter falls back gracefully (no crash, just no TB logs).

4. **Dataset assumption:** CIFAR-10 confirmed from project docs. Lecturer confirmation not obtained.

## Known Limitations

- No GPU (CUDA=False). All experiments on Intel i5-1135G7 CPU.
- image_size=64 is practical but not scientifically identical to 224×224 pretrained conditions.
- Results table in README shows NOT_RUN because experiments have not executed yet (honest status).

## Next Actions (After Dataset Download)

1. `python -m labs.lab2.main --eda` — Run EDA, save figures.
2. `python -m labs.lab2.main --mode smoke` — Verify full pipeline works with real data.
3. Check smoke run output, fix any issues.
4. `python -m labs.lab2.main --mode full` — Run all three experiments.
5. Update README.md with actual measured accuracy/F1/time.
6. Run `python -m labs.lab2.main --compare` — print final table.

---

**Tiếng Việt**

## Trạng thái (tóm tắt)
- Tất cả module đã implement đầy đủ.
- 36/36 smoke test pass.
- Chưa chạy thực nghiệm do đang tải dataset.
- Pipeline sẵn sàng chạy ngay khi download xong.

## Lỗi đã sửa
1. `unfreeze_last_blocks`: Sai logic khi lấy tail blocks (lấy nhầm `fc` thay vì `layer4`).
2. Checkpoint test: Tính output ở train mode rồi so sánh với eval mode → khác nhau do BatchNorm.

## Công việc còn lại
1. Chờ download CIFAR-10 xong (ETA ~60 phút từ lúc bắt đầu).
2. Chạy smoke → full experiments.
3. Cập nhật README với kết quả thực.
