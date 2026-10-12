"""Entry point of Lab 3: Hugging Face sentiment analysis and fine-tuning.

Run from the repo root:  python -m labs.lab3.main
Owner: team leader. Wire the stages here after the modules are implemented
and each one has passed its smoke test.

VI: Điểm chạy chính. Chạy từ gốc repo: python -m labs.lab3.main
Người phụ trách: nhóm trưởng. Nối các bước sau khi từng module đã qua smoke test.
"""
from __future__ import annotations

from shared.seed import set_seed

STAGES = ['sentiment_demo', 'eda', 'preprocessing', 'features', 'model', 'train', 'evaluate']


def main(seed: int = 42) -> None:
    """Pipeline order: Ex1: sentiment_demo ; Ex2: eda -> preprocessing -> model (load) -> features (tokenize) -> train (Trainer) -> evaluate

    VI: Thứ tự pipeline ở trên.
    """
    set_seed(seed)
    for name in STAGES:
        print(f"[todo] stage '{name}' is not wired yet / chưa nối bước '{name}'")
    raise NotImplementedError("TODO(leader): wire the stages in main()")


if __name__ == "__main__":
    main()
