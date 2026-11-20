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
MAX_LENGTH: int | None = 512  # Thư chốt 2026-11-20 sau EDA; căn cứ:
# docs/experiments/experiment_log.md mục S1 (512 cắt 13% review, 256 cắt 41%; zero-shot cũng cắt ở 512)
MAX_LENGTH_CANDIDATES = (128, 256, 512)  # thí nghiệm max_length (plan §5)

# Smoke run (RULES_v1 bước 7): vài chục review mỗi split.
SMOKE_N_PER_SPLIT = 40

# ---------------------------------------------------------------- Member B (Phúc): fine-tune
# Nguồn từng giá trị: docs/phases/train_v1.md, experiment_v1.md, evaluate_v1.md (plan L8–L13).
LEARNING_RATE = 2e-5          # L9: BERT gốc {5e-5…2e-5}; Mosbach et al. 2021
WARMUP_RATIO = 0.1            # L9: warmup 10% rồi linear decay (Mosbach et al. 2021)
LR_SCHEDULER = "linear"
BATCH_SIZE = 16               # L9: BERT gốc {16, 32}
EVAL_BATCH_SIZE = 64          # chỉ ảnh hưởng tốc độ đánh giá, không ảnh hưởng kết quả
WEIGHT_DECAY = 0.01           # Thư duyệt 2026-11-20 (train_v1.md)
EVALS_PER_EPOCH = 4           # L11: đánh giá val mỗi ~1/4 epoch (Dodge et al. 2020)
SAVE_TOTAL_LIMIT = 2          # train_v1.md: giữ checkpoint tốt nhất + mới nhất
METRIC_FOR_BEST = "accuracy"  # L8: chọn checkpoint theo val accuracy

PILOT_SEED = 42
MAIN_SEEDS = (42, 43, 44)     # Thư duyệt 2026-11-20: cố định trước khi chạy (experiment_v1.md)

# Chốt theo từng bước đo (RULES_v1 §2: số chưa đo thì không chốt trước). None = chưa chốt.
TIMING_MAX_STEPS = 300                # S2 bước 0: kernel ngắn đo thời gian/epoch trên Kaggle GPU
PILOT_MAX_EPOCHS: int | None = 6      # chốt 2026-11-20 sau đo thời gian (8,3 phút/epoch T4): trên mức 2–4 của BERT (experiment_log S2)
MAIN_MAX_EPOCHS: int | None = 6       # chốt sau pilot (AI theo uỷ quyền, Thư duyệt 2026-11-20): = trần pilot để lịch lr giống hệt
PATIENCE: int | None = 8              # chốt sau pilot: 8 lần đánh giá = 2 epoch; pilot cần ≥ 5 để không bỏ lỡ best (experiment_log S2b)
THRESHOLD: float | None = 0.0         # chốt sau pilot: mặc định HF; dao động val (±0,005) > mọi threshold nhỏ
MAXLEN_SEEDS: tuple | None = (42,)    # chốt sau S3: std val acc 3 seed = 0,0013 ≤ 0,003 → 1 seed/giá trị (ngưỡng của plan, experiment_log S3)

# Đánh giá (L13): ngưỡng đạt trên test, trung bình 3 seed.
PASS_ACCURACY = 0.91

# Smoke run của phần B: vài step trên SMOKE_N_PER_SPLIT review.
SMOKE_MAX_STEPS = 6
