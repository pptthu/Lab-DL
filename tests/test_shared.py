import numpy as np
import pytest

from shared.paths import REPO_ROOT, artifact_dir, lab_dir
from shared.seed import set_seed


def test_repo_root_contains_labs():
    assert (REPO_ROOT / "labs").is_dir()


def test_artifact_dir_rejects_unknown_kind():
    with pytest.raises(ValueError):
        artifact_dir("lab1", "unknown")


def test_lab_dir_is_inside_labs():
    assert lab_dir("lab1").parent.name == "labs"


def test_set_seed_is_reproducible():
    set_seed(7)
    a = np.random.rand(3)
    set_seed(7)
    b = np.random.rand(3)
    assert (a == b).all()
