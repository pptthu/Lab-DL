# model — nạp DistilBERT pre-trained + đầu phân loại 2 lớp (v1)

- **Created**: 2026-11-20T01:00:14+07:00
- **Last Updated**: 2026-11-20T14:29:08+07:00 (Thư)
- **Người phụ trách**: Thư
- **Status**: Done — audit 2026-11-20

---

## Name
Nạp model và tokenizer (Bài 2.3) — `distilbert-base-uncased` + đầu phân loại 2 lớp, đếm param

## Background
Bài 2.3 của đề: nạp model pre-trained và tokenizer. Plan L5: `distilbert-base-uncased` (DistilBERT paper báo IMDb test acc 92,82). Dùng cho `features` (tokenizer), `train`, `evaluate`.

## Goals / Purpose
- `load_tokenizer()` trả tokenizer của `distilbert-base-uncased`
- `build_model(seed)` trả `AutoModelForSequenceClassification` với `num_labels=2`, `id2label` {0: NEGATIVE, 1: POSITIVE}
- `count_parameters(model)` trả tổng / backbone / đầu phân loại → điền model card
- Không làm:
  - Train — thuộc `train`
  - Pipeline SST-2 của Bài 1 — thuộc `sentiment_demo`

## Input / Output
- **Input:** tên checkpoint (hằng số một chỗ, `main.py`/config đọc từ đây); seed
- **Output:** model, tokenizer (trong bộ nhớ); số param → `docs/models/distilbert_imdb_v1.md`

## How to do it (general plan)
1. `set_seed(seed)` **trước** khi tạo model (đầu phân loại khởi tạo ngẫu nhiên theo seed)
2. `from_pretrained(..., num_labels=2, id2label, label2id)`
3. Đếm param theo nhóm
4. Viết phần Initial model của model card (mermaid + bảng param)
5. Test với config DistilBERT rất nhỏ tạo tại chỗ (không tải trọng số, không cần mạng)
6. Smoke run: nạp thật, forward 1 batch → logits [B, 2]

## Pipeline
```
checkpoint + seed → build_model → DistilBERT (6 lớp, 768 chiều) + pre_classifier + classifier → logits [B, 2]
                  → load_tokenizer → features
```

## Detailed plan
- **Kiến trúc** (`DistilBertConfig` mặc định, tài liệu HF): `n_layers` 6, `dim` 768, `n_heads` 12, `max_position_embeddings` 512, `dropout` 0,1. Đầu phân loại của `DistilBertForSequenceClassification`: `pre_classifier` Linear 768→768 + ReLU + dropout `seq_classif_dropout` 0,2 + `classifier` Linear 768→2 (đã kiểm khi in model, xem mục Triển khai)
- **Fine-tune toàn bộ** (không đóng băng backbone) — cách làm của tutorial HF và BERT gốc
- **Seed:** Dodge et al. 2020 cho thấy khởi tạo đầu phân loại ảnh hưởng kết quả → seed phải đặt trước khi tạo model, mỗi seed của lần chạy chính một khởi tạo khác
- **Gotchas:**
  - Cảnh báo "Some weights … newly initialized: pre_classifier, classifier" là **bình thường**: đầu phân loại mới, chưa train
  - Tokenizer và model phải cùng checkpoint

## Triển khai (2026-11-20)
- **Hàm** (`labs/lab3/model/model.py`): `load_tokenizer(checkpoint=config.CHECKPOINT)` (`:17`); `build_model(seed=42, checkpoint=config.CHECKPOINT, model_config=None)` (`:24`); `count_parameters(model) -> dict` (`:42`); `describe_head(model) -> str` (`:50`); `model_summary(model, revision) -> dict` (`:71`). Chạy: `python -m labs.lab3.model.model` → `outputs/results/model_summary.json`
- **Chỗ code khác plan:** thêm `model_summary` và ghi JSON để notebook, model card đọc số liệu mà không phải nạp model
- **Kiểm kiến trúc khi in model (đã kiểm):** `n_layers` 6, `dim` 768, `n_heads` 12, `hidden_dim` 3072, `max_position_embeddings` 512, `dropout` 0,1, `seq_classif_dropout` 0,2; đầu phân loại `pre_classifier` Linear 768→768, `classifier` Linear 768→2 — khớp tài liệu HF
- **Kết quả:** tổng 66.955.010 param (đều train được): backbone 66.362.880, đầu phân loại 592.130. Commit model `12040accade4e8a0f71eabdb258fecc2e7e948be`. Model card: `docs/models/distilbert_imdb_v1.md` (phần Initial model)
- **Test:** `pytest labs/lab3/tests/test_model.py` 3 passed (DistilBERT nhỏ tạo từ config, không tải trọng số)

## Success criteria
- **Test:** `pytest labs/lab3/tests/test_model.py` đạt: logits [B, 2], `id2label` đúng, cùng seed → cùng trọng số đầu phân loại
- **Smoke run:** nạp thật, forward 1 batch không lỗi
- **Normal run:** số param in ra và ghi vào model card (đếm từ code)

## Kết quả audit (2026-11-20)
- **Test:** `pytest labs/lab3/tests/test_model.py` → 3 passed ✔
- **Smoke run** (`run_member_a(smoke=True)`, 38 giây, máy Thư, 40 review mỗi split): nạp thật `distilbert-base-uncased`, đếm param không lỗi ✔
- **Normal run:** `outputs/results/model_summary.json`: 66.955.010 param, đầu phân loại 592.130, commit `12040acc…`; khớp model card ✔
- Số dòng code ghi ở mục Triển khai khớp code hiện tại ✔

## Links
- Desc lab: [../lab3_description_v1.md](../lab3_description_v1.md) | Plan: [../plans/detail_plan_lab3_md_v1.md](../plans/detail_plan_lab3_md_v1.md)
- Module liên quan: [features_v1.md](features_v1.md), [train_v1.md](train_v1.md) | Khuôn: [../templates/MODEL_CARD_TEMPLATE_v1.md](../templates/MODEL_CARD_TEMPLATE_v1.md)
- Nguồn: Sanh et al. 2019 https://arxiv.org/abs/1910.01108 ; HF DistilBERT https://huggingface.co/docs/transformers/model_doc/distilbert ; Dodge et al. 2020 https://arxiv.org/abs/2002.06305
- Code: `labs/lab3/model/model.py` | Test: `labs/lab3/tests/test_model.py`
