"""Repo-relative paths so code works from any working directory.

VI: Đường dẫn tính từ gốc repo, nên chạy ở thư mục nào cũng đúng (không dùng '../').
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_KINDS = ("splits", "models", "figures", "logs", "runs")


def lab_dir(lab_slug: str) -> Path:
    """Return the folder of a lab, e.g. labs/lab1. / Thư mục của một lab."""
    return REPO_ROOT / "labs" / lab_slug


def raw_data_dir(lab_slug: str) -> Path:
    """Folder for raw data (ignored by git). / Thư mục dữ liệu thô (git bỏ qua)."""
    return lab_dir(lab_slug) / "raw_data"


def artifact_dir(lab_slug: str, kind: str) -> Path:
    """Return (and create) an artifact folder: splits, models, figures, logs or runs.

    VI: Trả về (và tạo) thư mục kết quả: splits, models, figures, logs hoặc runs.
    """
    if kind not in ARTIFACT_KINDS:
        raise ValueError(f"kind must be one of {ARTIFACT_KINDS}, got {kind!r}")
    path = lab_dir(lab_slug) / "artifacts" / kind
    path.mkdir(parents=True, exist_ok=True)
    return path
