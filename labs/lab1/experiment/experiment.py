"""Module `experiment` (Experiments) of Lab 1: Fashion-MNIST classification with an MLP (PyTorch).

Owner: Member B (see progress_status.md). Do not rename functions or change signatures
without telling the team leader: other modules depend on them.

VI: Module `experiment` (Thí nghiệm). Người phụ trách: Member B. Không đổi tên hàm hoặc chữ ký
khi chưa báo nhóm trưởng vì các module khác phụ thuộc vào chúng.
"""
from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from torch import nn

from shared.paths import artifact_dir
from shared.seed import set_seed
from labs.lab1.model.model import MLP, count_parameters
from labs.lab1.train.train import load_checkpoint, train_model
from labs.lab1.evaluate.evaluate import confusion_matrix, evaluate, per_class_accuracy


def build_model(cfg: Dict[str, Any], num_classes: int = 10) -> nn.Module:
    """Create the MLP of a config.

    Input : cfg; num_classes
    Output: new, untrained model

    VI: Tạo MLP của một config.
    Vào : cfg; num_classes
    Ra  : model mới, chưa huấn luyện
    """
    return MLP(
        hidden_sizes=tuple(cfg.get("hidden_sizes", (256, 128))),
        num_classes=num_classes,
        dropout=float(cfg.get("dropout", 0.0)),
    )


def run_experiments(
    configs: List[Dict[str, Any]],
    loaders: Tuple[Any, Any, Any],
    device: str,
    ckpt_prefix: str = "exp_run",
) -> List[Dict[str, Any]]:
    """Train one model per config (e.g. hidden sizes, lr, dropout, optimizer) and compare on VALIDATION data.

    Input : configs: list of dicts; loaders: (train, val, test); device; ckpt_prefix: checkpoint file name prefix
    Output: list of result dicts: config, best_val_acc, best_val_loss, best_epoch, final_val_loss, n_params, train_seconds, ckpt_path

    VI: Huấn luyện mỗi config một model (hidden sizes, lr, dropout, optimizer...) và so sánh trên VALIDATION.
    Vào : configs: danh sách dict; loaders: (train, val, test); device; ckpt_prefix: tiền tố tên file checkpoint
    Ra  : danh sách dict kết quả: config, best_val_acc, best_val_loss, best_epoch, final_val_loss, n_params, train_seconds, ckpt_path
    """
    train_loader, val_loader, _test_loader = loaders
    results: List[Dict[str, Any]] = []

    models_dir = artifact_dir("lab1", "models")

    for idx, cfg in enumerate(configs):
        seed = cfg.get("seed", 42)
        set_seed(seed)

        lr = float(cfg.get("lr", 1e-3))
        optimizer_name = str(cfg.get("optimizer_name", "adam"))
        weight_decay = float(cfg.get("weight_decay", 0.0))
        epochs = int(cfg.get("epochs", 5))

        model = build_model(cfg)
        n_params = count_parameters(model)

        ckpt_path = str(models_dir / f"{ckpt_prefix}_{idx}.pt")

        t0 = time.perf_counter()
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=lr,
            device=device,
            ckpt_path=ckpt_path,
            optimizer_name=optimizer_name,
            weight_decay=weight_decay,
        )
        train_seconds = float(time.perf_counter() - t0)

        val_accs = history.get("val_acc", [])
        val_losses = history.get("val_loss", [])
        # Report the epoch whose checkpoint was kept (lowest val loss).
        # VI: Báo cáo đúng epoch được lưu checkpoint (val loss thấp nhất).
        best_idx = val_losses.index(min(val_losses)) if val_losses else 0
        best_val_acc = float(val_accs[best_idx]) if val_accs else 0.0
        best_val_loss = float(val_losses[best_idx]) if val_losses else 0.0

        results.append({
            "config": cfg,
            "best_val_acc": best_val_acc,
            "best_val_loss": best_val_loss,
            "best_epoch": best_idx + 1,
            "final_val_loss": float(val_losses[-1]) if val_losses else 0.0,
            "n_params": n_params,
            "train_seconds": train_seconds,
            "ckpt_path": ckpt_path,
        })

    return results


def save_results(results: List[Dict[str, Any]], csv_path: str) -> None:
    """Save the comparison table.

    Input : results, csv_path
    Output: CSV file

    VI: Lưu bảng so sánh.
    Vào : results, csv_path
    Ra  : file CSV
    """
    rows = []
    for r in results:
        row: Dict[str, Any] = {}
        cfg = r.get("config", {})
        for k, v in cfg.items():
            if isinstance(v, (tuple, list)):
                row[k] = "-".join(str(x) for x in v) or "none"
            else:
                row[k] = v
        row["best_val_acc"] = r.get("best_val_acc")
        row["best_val_loss"] = r.get("best_val_loss")
        row["best_epoch"] = r.get("best_epoch")
        row["final_val_loss"] = r.get("final_val_loss")
        row["n_params"] = r.get("n_params")
        row["train_seconds"] = r.get("train_seconds")
        rows.append(row)

    df = pd.DataFrame(rows)
    p = Path(csv_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)


def analyze_seed_runs(
    runs: List[Dict[str, Any]],
    loaders: Tuple[Any, Any, Any],
    class_names: Sequence[str],
    device: str,
    out_dir: str,
    top_k: int = 10,
    prefix: str = "",
    use_test: bool = True,
) -> Dict[str, float]:
    """Measure one fixed configuration across seeds. Use it only for the configuration chosen beforehand, never to choose one by test score.

    Input : runs: results of one configuration (one per seed); loaders: (train, val, test); class_names; device; out_dir; top_k; prefix of the file names; use_test: False skips the test set
    Output: dict: train_acc_mean, val_acc_mean, and test_acc_mean, test_acc_std when use_test; CSV files in out_dir: seed_results.csv, recall_by_class.csv (validation), top_confusions.csv (validation)

    VI: Đo một cấu hình cố định qua nhiều seed. Chỉ dùng cho cấu hình đã chốt trước, không dùng để chọn cấu hình theo điểm test.
    Vào : runs: kết quả của một cấu hình (mỗi seed một dòng); loaders: (train, val, test); class_names; device; out_dir; top_k; prefix của tên file; use_test: False thì bỏ qua tập test
    Ra  : dict: train_acc_mean, val_acc_mean, và test_acc_mean, test_acc_std khi use_test; file CSV trong out_dir: seed_results.csv, recall_by_class.csv (validation), top_confusions.csv (validation)
    """
    train_loader, val_loader, test_loader = loaders
    num_classes = len(class_names)
    seed_rows = []
    recalls = []
    cm_sum = np.zeros((num_classes, num_classes), dtype=np.int64)

    for run in runs:
        cfg = run["config"]
        model = load_checkpoint(build_model(cfg, num_classes), run["ckpt_path"], device=device)

        train_loss, train_acc = evaluate(model, train_loader, device=device)
        val_loss, val_acc = evaluate(model, val_loader, device=device)
        cm = confusion_matrix(model, val_loader, device=device, num_classes=num_classes)
        cm_sum += cm
        recalls.append(list(per_class_accuracy(cm).values()))
        row = {
            "seed": cfg.get("seed"),
            "best_epoch": run.get("best_epoch"),
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        if use_test:
            row["test_loss"], row["test_acc"] = evaluate(model, test_loader, device=device)
        seed_rows.append(row)

    recall_arr = np.asarray(recalls)
    # ddof=1 is undefined for one seed, so report 0 there.
    # VI: ddof=1 không xác định khi chỉ có một seed nên báo 0.
    ddof = 1 if len(runs) > 1 else 0
    lowest = recall_arr.argmin(axis=1)
    recall_df = pd.DataFrame({
        "class": list(class_names),
        "recall_mean": recall_arr.mean(axis=0),
        "recall_std": recall_arr.std(axis=0, ddof=ddof),
        "recall_min": recall_arr.min(axis=0),
        "recall_max": recall_arr.max(axis=0),
        "n_seeds_lowest": [int((lowest == i).sum()) for i in range(num_classes)],
    })

    confusion_rows = [
        {
            "true_class": class_names[i],
            "predicted_class": class_names[j],
            "total": int(cm_sum[i, j]),
            "mean_per_seed": float(cm_sum[i, j]) / len(runs),
        }
        for i in range(num_classes)
        for j in range(num_classes)
        if i != j
    ]
    confusion_df = pd.DataFrame(confusion_rows).sort_values("total", ascending=False).head(top_k)

    seed_df = pd.DataFrame(seed_rows)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seed_df.to_csv(out / f"{prefix}seed_results.csv", index=False)
    recall_df.to_csv(out / f"{prefix}recall_by_class.csv", index=False)
    confusion_df.to_csv(out / f"{prefix}top_confusions.csv", index=False)

    summary = {
        "train_acc_mean": float(seed_df["train_acc"].mean()),
        "val_acc_mean": float(seed_df["val_acc"].mean()),
    }
    if use_test:
        summary["test_acc_mean"] = float(seed_df["test_acc"].mean())
        summary["test_acc_std"] = float(seed_df["test_acc"].std(ddof=ddof))
    return summary


def compare_pairs(values: pd.DataFrame, pairs: Sequence[Tuple[str, str, str]]) -> pd.DataFrame:
    """Paired comparison across shared seeds, with a t-test and Holm correction for several pairs.

    Input : values: table with one row per seed, one column per configuration (points); pairs: (variant, reference, factor)
    Output: DataFrame: factor, variant, reference, mean_diff, std_diff, ci95_low, ci95_high, seeds_higher, rule_clear, p_value, p_holm

    VI: So sánh theo cặp trên cùng các seed, có kiểm định t và hiệu chỉnh Holm khi so nhiều cặp.
    Vào : values: bảng mỗi seed một dòng, mỗi cấu hình một cột (đơn vị điểm); pairs: (biến thể, mốc, yếu tố)
    Ra  : DataFrame: factor, variant, reference, mean_diff, std_diff, ci95_low, ci95_high, seeds_higher, rule_clear, p_value, p_holm
    """
    rows = []
    for variant, reference, factor in pairs:
        diff = values[variant] - values[reference]
        n = len(diff)
        mean, std = float(diff.mean()), float(diff.std(ddof=1))
        half_width = float(stats.t.ppf(0.975, n - 1)) * std / np.sqrt(n)
        same_direction = bool((diff > 0).all() or (diff < 0).all())
        rows.append({
            "factor": factor,
            "variant": variant,
            "reference": reference,
            "mean_diff": mean,
            "std_diff": std,
            "ci95_low": mean - half_width,
            "ci95_high": mean + half_width,
            "seeds_higher": int((diff > 0).sum()),
            # Rule fixed before reading the results.
            # VI: Quy tắc chốt trước khi đọc kết quả.
            "rule_clear": same_direction and abs(mean) >= 2 * std,
            "p_value": float(stats.ttest_rel(values[variant], values[reference]).pvalue),
        })

    df = pd.DataFrame(rows)
    # Holm: multiply the smallest p by m, the next by m-1, and keep the values increasing.
    # VI: Holm: nhân p nhỏ nhất với m, p kế tiếp với m-1, và giữ dãy không giảm.
    order = df["p_value"].to_numpy().argsort()
    adjusted = np.empty(len(df))
    running_max = 0.0
    for rank, i in enumerate(order):
        running_max = max(running_max, float(df["p_value"].iloc[i]) * (len(df) - rank))
        adjusted[i] = min(running_max, 1.0)
    df["p_holm"] = adjusted
    return df
