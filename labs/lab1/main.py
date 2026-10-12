"""Entry point of Lab 1: Fashion-MNIST classification with an MLP (PyTorch).

Run from the repo root:  python -m labs.lab1.main
Owner: team leader. Wire the stages here after the modules are implemented
and each one has passed its smoke test.

VI: Điểm chạy chính. Chạy từ gốc repo: python -m labs.lab1.main
Người phụ trách: nhóm trưởng. Nối các bước sau khi từng module đã qua smoke test.
"""
from __future__ import annotations

from shared.seed import set_seed

STAGES = ['eda', 'preprocessing', 'features', 'model', 'train', 'evaluate', 'visualize', 'experiment']


def main(seed: int = 42) -> None:
    """Pipeline order: eda -> preprocessing -> features -> model -> train -> evaluate -> visualize ; experiment wraps model + train

    VI: Thứ tự pipeline ở trên.
    """
    set_seed(seed)
    for name in STAGES:
        print(f"[todo] stage '{name}' is not wired yet / chưa nối bước '{name}'")
    raise NotImplementedError("TODO(leader): wire the stages in main()")


if __name__ == "__main__":
    main()
