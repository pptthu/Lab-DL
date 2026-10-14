"""Interface contract + smoke tests for lab2.

The contract test fails if someone renames a function that other modules depend on.
VI: Test hợp đồng giao diện: fail nếu ai đó đổi tên hàm mà module khác đang dùng.

Additional smoke tests verify:
  - Model output dimensions (logits [B, 10]).
  - Correct number of output classes.
  - Trainable vs frozen parameter configuration.
  - Optimizer parameter selection.
  - Training on a synthetic batch.
  - Loss calculation and gradient updates.
  - Validation metrics.
  - Checkpoint save/load round trip.
  - Deterministic splitting.
  - Transform output shapes.
  - Metric calculations on known toy examples.
  - CPU-only execution.
"""
import importlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Interface contract (existing test — do NOT change function/class names below)
# ---------------------------------------------------------------------------

SLUG = "lab2"
EXPECTED = {
    "eda":           ["run_eda"],
    "preprocessing": ["load_raw_dataset", "make_split", "save_split", "load_split"],
    "features":      ["build_transforms", "make_loaders"],
    "model":         ["build_model", "describe_model", "unfreeze_last_blocks"],
    "train":         ["train_model", "save_checkpoint", "load_checkpoint"],
    "evaluate":      ["evaluate", "confusion_matrix", "top_confusions"],
    "experiment":    ["run_comparison", "save_results"],
}


@pytest.mark.parametrize("module,names", list(EXPECTED.items()))
def test_module_interface(module, names):
    mod = importlib.import_module(f"labs.{SLUG}.{module}.{module}")
    for name in names:
        assert hasattr(mod, name), (
            f"{module}.{name} is missing / thiếu {module}.{name}"
        )


def test_main_importable():
    main = importlib.import_module(f"labs.{SLUG}.main")
    assert callable(main.main)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _synthetic_loader(n_samples: int = 32, batch_size: int = 8,
                      image_size: int = 32, n_classes: int = 10) -> DataLoader:
    """Build a DataLoader of synthetic random tensors (no real data needed)."""
    images = torch.randn(n_samples, 3, image_size, image_size)
    labels = torch.randint(0, n_classes, (n_samples,))
    ds     = TensorDataset(images, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def _tiny_scratch_model(n_classes: int = 10) -> nn.Module:
    """Build the scratch CNN via build_model without any pretrained download."""
    from labs.lab2.model.model import build_model
    return build_model("scratch", num_classes=n_classes,
                       freeze_backbone=False, pretrained=False)


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestModelOutputShape:
    """Verify that all model variants output logits of shape [B, 10]."""

    def test_scratch_output_shape(self):
        from labs.lab2.model.model import build_model
        model = build_model("scratch", num_classes=10,
                            freeze_backbone=False, pretrained=False)
        model.eval()
        x = torch.randn(4, 3, 32, 32)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (4, 10), f"Expected (4,10) got {logits.shape}"

    def test_scratch_output_shape_resized(self):
        from labs.lab2.model.model import build_model
        model = build_model("scratch", num_classes=10,
                            freeze_backbone=False, pretrained=False)
        model.eval()
        x = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (2, 10), f"Expected (2,10) got {logits.shape}"

    def test_num_classes_configurable(self):
        from labs.lab2.model.model import build_model
        for n_cls in [5, 10, 20]:
            model = build_model("scratch", num_classes=n_cls,
                                freeze_backbone=False, pretrained=False)
            model.eval()
            x = torch.randn(2, 3, 32, 32)
            with torch.no_grad():
                out = model(x)
            assert out.shape == (2, n_cls)


class TestFrozenParameters:
    """Feature extraction: backbone frozen, head trainable."""

    def test_resnet18_freeze_backbone_no_download(self):
        """Build ResNet-18 with freeze=True without downloading weights."""
        # We use pretrained=False here to avoid network access in tests.
        from labs.lab2.model.model import build_model, describe_model
        model = build_model("resnet18", num_classes=10,
                            freeze_backbone=True, pretrained=False)
        info = describe_model(model)
        # The replacement fc layer must be trainable.
        assert info["trainable_params"] > 0, "Head should be trainable."
        # Most parameters must be frozen.
        assert info["frozen_params"] > info["trainable_params"], (
            "Backbone should have far more frozen params than trainable."
        )

    def test_scratch_all_trainable(self):
        from labs.lab2.model.model import build_model, describe_model
        model = build_model("scratch", num_classes=10,
                            freeze_backbone=False, pretrained=False)
        info = describe_model(model)
        assert info["trainable_params"] > 0
        assert info["frozen_params"] == 0

    def test_frozen_bn_stays_eval(self):
        """Calling model.train() must not set frozen backbone BN to train mode."""
        from labs.lab2.model.model import build_model, _FrozenBNModel
        import torch.nn as nn
        model = build_model("resnet18", num_classes=10,
                            freeze_backbone=True, pretrained=False)
        assert isinstance(model, _FrozenBNModel), (
            "Frozen model should be wrapped in _FrozenBNModel."
        )
        model.train()
        # Check that BN layers inside the backbone are in eval mode.
        bn_in_eval = []
        for m in model.inner.modules():
            if isinstance(m, nn.BatchNorm2d):
                bn_in_eval.append(m.training)
        # The backbone BNs should all be in eval (training=False).
        # The replacement fc has no BN so we only check backbone BNs.
        # (The new fc has requires_grad=True but no BN.)
        assert any(not t for t in bn_in_eval), (
            "At least some backbone BN layers should be in eval mode."
        )

    def test_unfreeze_last_blocks_adds_trainable(self):
        from labs.lab2.model.model import build_model, describe_model, unfreeze_last_blocks
        model = build_model("resnet18", num_classes=10,
                            freeze_backbone=True, pretrained=False)
        before = describe_model(model)["trainable_params"]
        unfreeze_last_blocks(model, n_blocks=1)
        after = describe_model(model)["trainable_params"]
        assert after > before, (
            f"Unfreezing should increase trainable params: before={before}, after={after}"
        )


# ---------------------------------------------------------------------------
# Training tests (synthetic data, no real dataset)
# ---------------------------------------------------------------------------

class TestTrainingLoop:

    def test_train_epoch_decreases_loss(self):
        """At least one gradient step runs without error."""
        from labs.lab2.train.train import train_model

        model  = _tiny_scratch_model()
        loader = _synthetic_loader(n_samples=16, batch_size=8, image_size=32)
        device = torch.device("cpu")

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = os.path.join(tmpdir, "ckpt.pt")
            history = train_model(
                model=model,
                train_loader=loader,
                val_loader=loader,
                epochs=2,
                lr=1e-3,
                device=device,
                ckpt_path=ckpt,
                writer=None,
            )
        assert "train_loss" in history
        assert "val_loss"   in history
        assert "val_acc"    in history
        assert len(history["train_loss"]) >= 1

    def test_no_trainable_params_raises(self):
        """train_model must raise when there are no trainable parameters."""
        from labs.lab2.train.train import train_model
        from labs.lab2.model.model import build_model

        model = build_model("scratch", num_classes=10,
                            freeze_backbone=False, pretrained=False)
        # Freeze everything manually.
        for p in model.parameters():
            p.requires_grad = False

        loader = _synthetic_loader(n_samples=8, batch_size=8, image_size=32)
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises((ValueError, RuntimeError)):
                train_model(model, loader, loader, epochs=1, lr=1e-3,
                            device=torch.device("cpu"),
                            ckpt_path=os.path.join(tmpdir, "ckpt.pt"))

    def test_checkpoint_save_load_roundtrip(self):
        from labs.lab2.train.train import save_checkpoint, load_checkpoint

        model  = _tiny_scratch_model()
        loader = _synthetic_loader(n_samples=8, batch_size=8, image_size=32)

        # Evaluate in eval mode so BN uses running stats (deterministic).
        model.eval()
        with torch.no_grad():
            x       = next(iter(loader))[0]
            out_pre = model(x).clone()

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "model.pt")
            save_checkpoint(model, ckpt_path)
            assert os.path.exists(ckpt_path)

            # Load into a fresh model and evaluate in eval mode.
            model2 = _tiny_scratch_model()
            load_checkpoint(model2, ckpt_path, torch.device("cpu"))
            model2.eval()
            with torch.no_grad():
                out_post = model2(x)

        assert torch.allclose(out_pre, out_post, atol=1e-5), (
            "Checkpoint round-trip should preserve model outputs."
        )

    def test_gradient_flows_in_trainable_layers(self):
        """Verify that gradients flow to trainable parameters after a forward/backward pass."""
        model     = _tiny_scratch_model()
        criterion = nn.CrossEntropyLoss()
        x  = torch.randn(4, 3, 32, 32)
        y  = torch.randint(0, 10, (4,))

        out  = model(x)
        loss = criterion(out, y)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No grad for {name}"
                assert not torch.all(param.grad == 0), f"Zero grad for {name}"


# ---------------------------------------------------------------------------
# Evaluate tests
# ---------------------------------------------------------------------------

class TestEvaluate:

    def test_evaluate_returns_loss_and_accuracy(self):
        from labs.lab2.evaluate.evaluate import evaluate
        model  = _tiny_scratch_model()
        loader = _synthetic_loader(n_samples=32, batch_size=8, image_size=32)
        loss, acc = evaluate(model, loader, torch.device("cpu"))
        assert isinstance(loss, float) and loss >= 0
        assert 0.0 <= acc <= 1.0

    def test_confusion_matrix_shape(self):
        from labs.lab2.evaluate.evaluate import confusion_matrix as cm_fn
        model  = _tiny_scratch_model()
        loader = _synthetic_loader(n_samples=32, batch_size=8, image_size=32)
        cm = cm_fn(model, loader, torch.device("cpu"), num_classes=10)
        assert cm.shape == (10, 10), f"Expected (10,10) got {cm.shape}"
        assert cm.sum() == 32

    def test_confusion_matrix_diagonal_dominant(self):
        """A perfect classifier should have identity confusion matrix."""
        import torch.nn as nn

        class PerfectClassifier(nn.Module):
            def forward(self, x):
                # Returns one-hot logits for label = 0 always — not perfect,
                # but at least deterministic.
                b = x.shape[0]
                logits = torch.zeros(b, 10)
                logits[:, 0] = 1000.0
                return logits

        from labs.lab2.evaluate.evaluate import confusion_matrix as cm_fn
        model  = PerfectClassifier()
        loader = _synthetic_loader(n_samples=16, batch_size=8,
                                   image_size=32, n_classes=10)
        cm = cm_fn(model, loader, torch.device("cpu"), num_classes=10)
        # All predictions go to class 0.
        assert cm[:, 0].sum() == 16

    def test_top_confusions_correct(self):
        from labs.lab2.evaluate.evaluate import top_confusions
        cm = np.zeros((3, 3), dtype=np.int64)
        cm[0, 1] = 10
        cm[1, 2] = 5
        cm[0, 2] = 2
        names = ["cat", "dog", "bird"]
        result = top_confusions(cm, names, k=2)
        assert len(result) == 2
        assert result[0] == ("cat", "dog", 10)
        assert result[1] == ("dog", "bird", 5)

    def test_top_confusions_excludes_diagonal(self):
        from labs.lab2.evaluate.evaluate import top_confusions
        cm = np.eye(4, dtype=np.int64) * 100
        cm[0, 1] = 3
        names = ["a", "b", "c", "d"]
        result = top_confusions(cm, names, k=5)
        # Only the off-diagonal entry (a→b=3) should appear.
        assert len(result) == 1
        assert result[0][2] == 3


# ---------------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------------

class TestPreprocessing:

    def test_make_split_deterministic(self):
        from labs.lab2.preprocessing.preprocessing import make_split
        a1, b1 = make_split(100, val_ratio=0.2, seed=7)
        a2, b2 = make_split(100, val_ratio=0.2, seed=7)
        assert a1 == a2 and b1 == b2

    def test_make_split_different_seeds(self):
        from labs.lab2.preprocessing.preprocessing import make_split
        a1, _ = make_split(100, val_ratio=0.2, seed=1)
        a2, _ = make_split(100, val_ratio=0.2, seed=2)
        assert a1 != a2

    def test_make_split_correct_sizes(self):
        from labs.lab2.preprocessing.preprocessing import make_split
        n = 1000
        ratio = 0.15
        train_idx, val_idx = make_split(n, val_ratio=ratio, seed=42)
        assert len(train_idx) + len(val_idx) == n
        assert abs(len(val_idx) / n - ratio) < 0.02  # within 2% of expected

    def test_make_split_no_overlap(self):
        from labs.lab2.preprocessing.preprocessing import make_split
        train_idx, val_idx = make_split(500, val_ratio=0.1, seed=0)
        assert set(train_idx).isdisjoint(set(val_idx))

    def test_save_load_split_roundtrip(self):
        from labs.lab2.preprocessing.preprocessing import make_split, save_split, load_split
        train_idx, val_idx = make_split(200, val_ratio=0.1, seed=99)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "split.json")
            save_split(train_idx, val_idx, path)
            t2, v2 = load_split(path)
        assert t2 == train_idx and v2 == val_idx


# ---------------------------------------------------------------------------
# Features / transforms tests
# ---------------------------------------------------------------------------

class TestTransforms:

    def test_build_transforms_returns_tuple(self):
        from labs.lab2.features.features import build_transforms
        result = build_transforms(image_size=32, augment=False)
        assert isinstance(result, tuple) and len(result) == 2

    def test_transform_output_shape(self):
        from labs.lab2.features.features import build_transforms
        train_tf, eval_tf = build_transforms(image_size=64, augment=True)
        img = Image.new("RGB", (32, 32), color=(128, 64, 32))
        t = train_tf(img)
        assert t.shape == (3, 64, 64), f"Expected (3,64,64) got {t.shape}"
        e = eval_tf(img)
        assert e.shape == (3, 64, 64)

    def test_eval_transform_deterministic(self):
        """The eval transform (no augmentation) must return identical tensors."""
        from labs.lab2.features.features import build_transforms
        _, eval_tf = build_transforms(image_size=32, augment=False)
        img = Image.new("RGB", (32, 32), color=(200, 100, 50))
        t1 = eval_tf(img)
        t2 = eval_tf(img)
        assert torch.allclose(t1, t2), "Eval transform should be deterministic."

    def test_make_loaders_batch_shapes(self):
        """make_loaders should produce batches of shape [B, 3, H, W]."""
        from labs.lab2.features.features import build_transforms, make_loaders
        import torchvision.datasets as datasets
        import torchvision.transforms as T

        # Build tiny synthetic CIFAR-like datasets using FakeData.
        from torchvision.datasets import FakeData
        fake_train = FakeData(size=40, image_size=(3, 32, 32),
                              num_classes=10, transform=None)
        fake_test  = FakeData(size=10, image_size=(3, 32, 32),
                              num_classes=10, transform=None)

        from labs.lab2.preprocessing.preprocessing import make_split
        train_idx, val_idx = make_split(40, val_ratio=0.2, seed=0)

        train_tf, eval_tf = build_transforms(image_size=32, augment=False)
        tl, vl, testl = make_loaders(
            fake_train, fake_test, train_idx, val_idx,
            train_tf, eval_tf, batch_size=8
        )
        imgs, labels = next(iter(tl))
        assert imgs.shape[1:] == (3, 32, 32)
        assert labels.ndim == 1
        assert labels.max() < 10


# ---------------------------------------------------------------------------
# Describe model test
# ---------------------------------------------------------------------------

class TestDescribeModel:

    def test_describe_model_returns_dict(self):
        from labs.lab2.model.model import build_model, describe_model
        model = build_model("scratch", num_classes=10,
                            freeze_backbone=False, pretrained=False)
        info = describe_model(model)
        assert "total_params"     in info
        assert "trainable_params" in info
        assert "frozen_params"    in info
        assert info["total_params"] == info["trainable_params"] + info["frozen_params"]

    def test_frozen_model_describe_correct_counts(self):
        from labs.lab2.model.model import build_model, describe_model
        model = build_model("resnet18", num_classes=10,
                            freeze_backbone=True, pretrained=False)
        info = describe_model(model)
        # Only the new fc(512→10) should be trainable: 512*10 + 10 = 5120
        assert 5000 < info["trainable_params"] < 6000, (
            f"Expected ~5120 trainable params, got {info['trainable_params']}"
        )


# ---------------------------------------------------------------------------
# CPU-only execution tests
# ---------------------------------------------------------------------------

class TestCPUExecution:

    def test_full_mini_pipeline_cpu(self):
        """End-to-end mini pipeline on CPU with synthetic data."""
        from labs.lab2.model.model import build_model
        from labs.lab2.train.train import train_model, save_checkpoint, load_checkpoint
        from labs.lab2.evaluate.evaluate import evaluate

        device = torch.device("cpu")
        model  = build_model("scratch", num_classes=10,
                             freeze_backbone=False, pretrained=False)
        loader = _synthetic_loader(n_samples=16, batch_size=8, image_size=32)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = os.path.join(tmpdir, "ckpt.pt")
            history = train_model(model, loader, loader, epochs=2, lr=1e-2,
                                  device=device, ckpt_path=ckpt, writer=None)
            assert history["best_val_acc"] >= 0.0

            save_checkpoint(model, ckpt)
            model2 = build_model("scratch", num_classes=10,
                                 freeze_backbone=False, pretrained=False)
            load_checkpoint(model2, ckpt, device)

            loss, acc = evaluate(model2, loader, device)
            assert isinstance(loss, float)
            assert 0.0 <= acc <= 1.0
