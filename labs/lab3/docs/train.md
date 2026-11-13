# Lab 3 / train: guide for AI agents

## Purpose

- **Fine-tune** the pre-trained DistilBERT on the IMDB train subset with the Hugging Face `Trainer`.
- Keep the **best epoch on validation F1** and **save the fine-tuned model** to disk.
- The test set is never used here.

## For AI agents (read first)

| Field | Value |
|---|---|
| Module | `labs/lab3/train/train.py` |
| Owner | Member B (ngohongphuc2805ag-hash), assigned by the team leader |
| Status | done in `progress_status.md`, merged in PR #39 (Reported only) |
| Assigned work | see the GitHub Issues and `progress_status.md` of the lab; never assign work to yourself, ask the team leader |
| Files you may edit | `labs/lab3/train/train.py`, `labs/lab3/tests/test_train.py`, this guide |
| Files you must not edit | Member A's modules, `main.py` (team leader), saved checkpoints, other owners' notebook cells |
| Agreed signatures | `build_training_args(output_dir: str, epochs: int = 2, lr: float = 2e-5, batch_size: int = 16)`; `train_with_trainer(model, tokenized_train, tokenized_val, args, compute_metrics)` (`project_description.md`). The code adds **optional** arguments only: `build_training_args(..., *, max_steps=-1, seed=42, logging_steps=10)` and `train_with_trainer(..., tokenizer=None)` |
| Depends on / used by | `features` output, `model`, `evaluate.compute_metrics` / `evaluate.evaluate_on_test`, notebook |
| Prohibited | passing the test set to the Trainer; selecting the best epoch on test; starting a full fine-tune without the team leader (cost on CPU); reporting to external services; removing the version detection; saving the model outside `artifact_dir('lab3', 'models')` |
| Checks | Unit: `python -m pytest labs/lab3/tests/test_train.py -q` (fake `transformers`). Smoke: tiny fine-tune with `max_steps` > 0 on a few rows. Real run: only at an approved milestone |
| Done when | unit tests pass; smoke or approved real run recorded in `progress_status.md`; this guide updated |

## What the brief requires

- Brief requirement (Exercise 2): define training arguments, create a `Trainer` and fine-tune.
- Team implementation choice: evaluate and save once per epoch, keep the best epoch on **validation F1**, weight decay 0.01, `save_total_limit=1`, `report_to="none"`.

## How the module works (current code)

- `build_training_args`:
  - checks `epochs`, `lr`, `batch_size` > 0 (`ValueError`).
  - detects `eval_strategy` vs `evaluation_strategy` for the installed `transformers`.
  - sets `load_best_model_at_end=True` and `metric_for_best_model="f1"`.
  - `max_steps > 0` overrides `epochs` (used for smoke runs).
- `train_with_trainer`:
  - refuses an empty train or validation set (`ValueError`).
  - pads per batch with `DataCollatorWithPadding(tokenizer)`; if `tokenizer` is not given it is loaded from `model.name_or_path`.
  - detects `processing_class` vs `tokenizer` for the `Trainer`.
  - trains, then calls `trainer.save_model(args.output_dir)`.
- The test set is never passed here; it is used once in `evaluate.evaluate_on_test`.

## Inputs, outputs and pitfalls

- Input:
  1. Initial model: the `(model, tokenizer)` from `model.load_model_and_tokenizer` (`distilbert-base-uncased`, see `model.md`).
  2. Data: tokenized train and validation subsets from `features`; `compute_metrics` from `evaluate`.
  3. `output_dir`: the folder where the trained model is written.
- Output: the trained `Trainer` holding the best validation epoch, and the saved model files in `output_dir`.
- Pitfalls:
  - `transformers` argument names change between versions: keep the detection.
  - CPU cost: use small subsets, short `max_length`, few epochs; agree the cost before a full run.
  - a forgotten or wrong `output_dir` writes the model to the wrong place (see the next section).

## Where the model is saved

- Implemented (code fact): `train_with_trainer` ends with `trainer.save_model(args.output_dir)`. `train.py` does not choose the path; the caller passes `output_dir` to `build_training_args`.
- Implemented (`main.py`, around line 111): `models_dir = artifact_dir("lab3", "models")` and `output_dir = models_dir / "distilbert_sentiment"`. So the fine-tuned model is saved to `labs/lab3/artifacts/models/distilbert_sentiment/`. At the end `main.py` prints `Model saved to: <path>`.
- Implemented (`shared/paths.py`): `artifact_dir(lab, kind)` creates and returns `labs/<lab>/artifacts/<kind>`, where `kind` is one of `splits`, `models`, `figures`, `logs`, `runs`. Use it instead of typing a path by hand.
- Implemented (Hugging Face behaviour): because evaluation and saving run every epoch, the `Trainer` also writes `checkpoint-<step>` subfolders inside `output_dir`, limited by `save_total_limit=1`.
- Other outputs of `main.py --run-training`: EDA figures in `artifacts/figures/` and the final test metrics in `artifacts/logs/test_metrics.json`. No code seen writes to `artifacts/runs/` or `artifacts/splits/`.
- Not committed (Verified): `git check-ignore -v labs/lab3/artifacts/models/distilbert_sentiment/model.safetensors` reports the rule `labs/*/artifacts/models/*` at line 12 of `.gitignore`, so model files are ignored by git.
- Wrong places to avoid:
  - the repository root or a default `./results` folder.
  - `labs/lab3/raw_data/` (this is only the dataset cache).
  - `labs/lab3/docs/` or any folder that is committed.
  - a hand-typed path that bypasses `artifact_dir`.
- Folder name difference (Implemented in two places, not yet agreed):
  - the notebook (cell with `MODEL_DIR`) saves the model to `Path(artifact_dir('lab3', 'models')) / 'distilbert_imdb'` and its log to `artifact_dir('lab3', 'logs') / 'distilbert_imdb_log.json'`. Its recorded output says it loaded the saved model from `.../labs/lab3/artifacts/models/distilbert_imdb (no training)`. The optional run with another `max_length` saves to `artifacts/models/distilbert_imdb_len<ALT_LEN>`.
  - `main.py` saves to `distilbert_sentiment` and writes `test_metrics.json`.
  - both use the correct root (`artifact_dir`, under `labs/lab3/artifacts/`), but a model trained by one flow is not found by the other.
  - the team leader picks one name (for example change `output_dir` in `main.py` to `distilbert_imdb`, because the recorded notebook results use that folder); until then this guide lists both.
- Loading it back: pass the same folder to `AutoModelForSequenceClassification.from_pretrained(<folder>)` and `AutoTokenizer.from_pretrained(<folder>)` (the tokenizer is saved there too because the `Trainer` receives it).

## How to verify

| Type | Command | Expected |
|---|---|---|
| Unit test | `python -m pytest labs/lab3/tests/test_train.py -q` | all pass, no network |
| Smoke test | tiny fine-tune: `build_training_args(tmp_dir, max_steps=4)` on 64 train reviews (recorded run: 4 steps, 64 train / 32 test reviews) | runs without error; recorded tiny run: accuracy 0.4688, F1 0.3200 (only proves the code runs; Reported only) |
| Entry point smoke run | `python -m labs.lab3.main --run-training --max-steps 2` (downloads the dataset and model the first time) | prints the final test metrics, `Metrics saved to: ...test_metrics.json` and `Model saved to: .../labs/lab3/artifacts/models/distilbert_sentiment` (Pending: no recorded run seen) |
| Check the save location | list `labs/lab3/artifacts/models/distilbert_sentiment/` after the run | contains the model weights, config and tokenizer files |
| Real integration run | only at an approved milestone | recorded full small run (2026-10-31, CPU, MAX_LEN=128, EPOCHS=1, seed 42): fine-tuned test accuracy/F1 0.8220/0.8195 (Reported only) |

## Open questions / limitations

- The recorded full run used 1 epoch while the default is 2; the report must state which setting a result used.
- Tests use fake objects; the real-library run is recorded only in the notebook and `progress_status.md`.
- The notebook saves to `distilbert_imdb` while `main.py` saves to `distilbert_sentiment`: the team leader must agree on one folder name (see "Where the model is saved").
- The defaults of `main.py` (`--max-length 256`, `--epochs 2`) differ from the recorded run (128 and 1); pass `--max-length 128 --epochs 1` for equivalent settings.

---

**Tiếng Việt**

## Mục đích

- **Fine-tune** DistilBERT đã pre-train trên tập train IMDB bằng `Trainer` của Hugging Face.
- Giữ **epoch tốt nhất theo F1 trên validation** và **lưu model đã fine-tune** xuống đĩa.
- Tập test không bao giờ được dùng ở đây.

## Dành cho AI (đọc trước)

- Người phụ trách: Member B (ngohongphuc2805ag-hash), do nhóm trưởng giao. Trạng thái: xong, đã merge PR #39 (chỉ có trong báo cáo).
- Việc được giao: xem GitHub Issue và `progress_status.md`; không tự giao việc cho mình, hỏi nhóm trưởng.
- Được sửa: `train.py`, `tests/test_train.py`, file này.
- Cấm sửa: module của Member A, `main.py` (nhóm trưởng), checkpoint đã lưu, ô notebook của người khác.
- Chữ ký: theo `project_description.md`; code chỉ thêm tham số **tùy chọn** (`max_steps`, `seed`, `logging_steps`, `tokenizer`).
- Cấm: đưa tập test vào Trainer; chọn epoch theo test; tự chạy fine-tune đầy đủ khi chưa hỏi nhóm trưởng (tốn thời gian CPU); báo cáo ra dịch vụ ngoài; bỏ phần tự dò phiên bản; lưu model ngoài `artifact_dir('lab3', 'models')`.
- Kiểm tra: unit test `pytest labs/lab3/tests/test_train.py` (`transformers` giả); smoke test fine-tune rất ngắn với `max_steps` > 0; chạy thật chỉ ở mốc đã duyệt.
- Xong khi: unit test qua, smoke hoặc lần chạy thật đã duyệt được ghi vào `progress_status.md`, đã cập nhật file này.

## Đề yêu cầu gì

- Đề (Bài 2): định nghĩa training arguments, tạo `Trainer` và fine-tune.
- Lựa chọn của nhóm: đánh giá và lưu mỗi epoch, giữ epoch tốt nhất theo **F1 validation**, weight decay 0.01, `save_total_limit=1`, `report_to="none"`.

## Cách hoạt động

- `build_training_args`:
  - kiểm tra `epochs`, `lr`, `batch_size` > 0 (`ValueError`).
  - tự dò `eval_strategy` hay `evaluation_strategy` theo phiên bản `transformers`.
  - đặt `load_best_model_at_end=True` và `metric_for_best_model="f1"`.
  - `max_steps > 0` thay cho `epochs` (dùng cho smoke test).
- `train_with_trainer`:
  - từ chối tập train hoặc validation rỗng (`ValueError`).
  - padding theo batch bằng `DataCollatorWithPadding(tokenizer)`; nếu không truyền `tokenizer` thì tải từ `model.name_or_path`.
  - tự dò `processing_class` hay `tokenizer` cho `Trainer`.
  - train xong thì gọi `trainer.save_model(args.output_dir)`.
- Tập test không được đưa vào đây; chỉ dùng một lần ở `evaluate.evaluate_on_test`.

## Đầu vào, đầu ra và lỗi hay gặp

- Đầu vào:
  1. Model ban đầu: `(model, tokenizer)` từ `model.load_model_and_tokenizer` (`distilbert-base-uncased`, xem `model.md`).
  2. Dữ liệu: tập train và validation đã tokenize từ `features`; `compute_metrics` từ `evaluate`.
  3. `output_dir`: thư mục ghi model sau khi train.
- Đầu ra: `Trainer` đã train, giữ epoch validation tốt nhất, và các file model đã lưu trong `output_dir`.
- Lưu ý:
  - tên tham số của `transformers` đổi theo phiên bản: giữ phần tự dò.
  - tốn CPU: dùng tập con nhỏ, `max_length` ngắn, ít epoch; thống nhất chi phí trước khi chạy đầy đủ.
  - `output_dir` bị quên hoặc sai sẽ ghi model vào sai chỗ (xem mục tiếp theo).

## Model được lưu ở đâu

- Đã có trong code: `train_with_trainer` kết thúc bằng `trainer.save_model(args.output_dir)`. `train.py` không tự chọn đường dẫn; người gọi hàm truyền `output_dir` vào `build_training_args`.
- Đã có (`main.py`, khoảng dòng 111): `models_dir = artifact_dir("lab3", "models")` và `output_dir = models_dir / "distilbert_sentiment"`. Vậy model fine-tune được lưu ở `labs/lab3/artifacts/models/distilbert_sentiment/`. Cuối cùng `main.py` in ra `Model saved to: <đường dẫn>`.
- Đã có (`shared/paths.py`): `artifact_dir(lab, kind)` tạo và trả về `labs/<lab>/artifacts/<kind>`, với `kind` là một trong `splits`, `models`, `figures`, `logs`, `runs`. Nên dùng hàm này thay vì gõ tay đường dẫn.
- Đã có (hành vi của Hugging Face): vì đánh giá và lưu mỗi epoch, `Trainer` còn ghi các thư mục `checkpoint-<step>` trong `output_dir`, bị giới hạn bởi `save_total_limit=1`.
- Các kết quả khác của `main.py --run-training`: biểu đồ EDA ở `artifacts/figures/` và số liệu test cuối cùng ở `artifacts/logs/test_metrics.json`. Không có code nào đã xem ghi vào `artifacts/runs/` hoặc `artifacts/splits/`.
- Không commit (đã kiểm chứng): `git check-ignore -v labs/lab3/artifacts/models/distilbert_sentiment/model.safetensors` báo quy tắc `labs/*/artifacts/models/*` ở dòng 12 của `.gitignore`, nên file model bị git bỏ qua.
- Những chỗ sai cần tránh:
  - thư mục gốc repo hoặc thư mục `./results` mặc định.
  - `labs/lab3/raw_data/` (chỉ là cache dataset).
  - `labs/lab3/docs/` hoặc bất kỳ thư mục nào được commit.
  - đường dẫn gõ tay, không đi qua `artifact_dir`.
- Hai tên thư mục khác nhau (đã có trong code ở hai nơi, chưa thống nhất):
  - notebook (ô có `MODEL_DIR`) lưu model vào `Path(artifact_dir('lab3', 'models')) / 'distilbert_imdb'` và log vào `artifact_dir('lab3', 'logs') / 'distilbert_imdb_log.json'`. Kết quả đã ghi trong notebook cho thấy nó nạp model đã lưu từ `.../labs/lab3/artifacts/models/distilbert_imdb (no training)`. Lần chạy tùy chọn với `max_length` khác lưu vào `artifacts/models/distilbert_imdb_len<ALT_LEN>`.
  - `main.py` lưu vào `distilbert_sentiment` và ghi `test_metrics.json`.
  - cả hai dùng đúng thư mục gốc (`artifact_dir`, nằm trong `labs/lab3/artifacts/`), nhưng model train ở luồng này thì luồng kia không tìm thấy.
  - nhóm trưởng cần chọn một tên (ví dụ đổi `output_dir` trong `main.py` thành `distilbert_imdb`, vì kết quả đã ghi của notebook dùng thư mục đó); trước mắt guide này ghi cả hai.
- Nạp lại model: truyền đúng thư mục đó vào `AutoModelForSequenceClassification.from_pretrained(<thư mục>)` và `AutoTokenizer.from_pretrained(<thư mục>)` (tokenizer cũng được lưu ở đó vì `Trainer` nhận tokenizer).

## Cách kiểm chứng

- Unit test: `python -m pytest labs/lab3/tests/test_train.py -q`, kỳ vọng tất cả qua, không cần mạng.
- Smoke test: fine-tune rất ngắn `build_training_args(tmp_dir, max_steps=4)` trên 64 review train; đã ghi: accuracy 0,4688, F1 0,3200 (chỉ chứng minh code chạy được; chỉ có trong báo cáo).
- Chạy thử điểm vào: `python -m labs.lab3.main --run-training --max-steps 2` (lần đầu tải dataset và model); kỳ vọng in số liệu test cuối, `Metrics saved to: ...test_metrics.json` và `Model saved to: .../labs/lab3/artifacts/models/distilbert_sentiment` (chưa có: chưa thấy lần chạy nào được ghi).
- Kiểm tra chỗ lưu: liệt kê `labs/lab3/artifacts/models/distilbert_sentiment/` sau khi chạy; kỳ vọng thấy file trọng số, cấu hình và tokenizer.
- Chạy tích hợp thật: chỉ ở mốc nhóm đã duyệt; đã ghi (31/10, CPU, MAX_LEN=128, 1 epoch, seed 42): accuracy/F1 trên test 0,8220/0,8195 (chỉ có trong báo cáo).

## Câu hỏi mở

- Lần chạy đầy đủ đã ghi dùng 1 epoch trong khi mặc định là 2; báo cáo phải nêu kết quả dùng thiết lập nào.
- Test dùng object giả; lần chạy với thư viện thật chỉ được ghi trong notebook và `progress_status.md`.
- Notebook lưu vào `distilbert_imdb` còn `main.py` lưu vào `distilbert_sentiment`: nhóm trưởng phải chốt một tên thư mục (xem mục "Model được lưu ở đâu").
- Mặc định của `main.py` (`--max-length 256`, `--epochs 2`) khác lần chạy đã ghi (128 và 1); truyền `--max-length 128 --epochs 1` để có thiết lập tương đương.
