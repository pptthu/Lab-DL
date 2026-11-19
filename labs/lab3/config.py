"""Lab 3 settings in one place (RULES_v1 §2: change a parameter in one file only).

VI: Mọi tham số của Lab 3 đặt ở đây; module khác đọc từ file này.
Giá trị kèm nguồn ở docs/plans/detail_plan_lab3_md_v1.md §2 (L1–L17).
"""
from __future__ import annotations

LAB = "lab3"
SEED = 42

# Dữ liệu (plan L1, L4): bản gốc Stanford, có URL phim cho từng review.
IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
LABEL_NAMES = {0: "NEGATIVE", 1: "POSITIVE"}  # thư mục neg -> 0, pos -> 1

# Chia val (plan L3): 10 fold theo nhóm (phim + review trùng), lấy 1 fold làm val.
VAL_FOLDS = 10

# Model (plan L5, L6).
CHECKPOINT = "distilbert-base-uncased"
SST2_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
MODEL_MAX_POSITIONS = 512  # DistilBertConfig.max_position_embeddings

# max_length cho lần chạy chính: chốt sau EDA (plan §4 S1).
# None = chưa chốt; features báo lỗi nếu chưa truyền giá trị.
MAX_LENGTH: int | None = 512  # [đề xuất 2026-11-20 sau EDA, chờ Thư chốt] căn cứ:
# docs/experiments/experiment_log.md mục S1 (512 cắt 13% review, 256 cắt 41%; zero-shot cũng cắt ở 512)
MAX_LENGTH_CANDIDATES = (128, 256, 512)  # thí nghiệm max_length (plan §5)

# Smoke run (RULES_v1 bước 7): vài chục review mỗi split.
SMOKE_N_PER_SPLIT = 40
