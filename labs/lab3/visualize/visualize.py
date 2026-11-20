"""Figures and error analysis from the files in outputs/ (no training, no model).

VI: Hình và phân tích lỗi — chỉ đọc file trong outputs/ (history, dự đoán, bảng run).
Desc: labs/lab3/docs/phases/visualize_v1.md
Owner: Phúc (Member B).

Run from the repo root / Chạy từ gốc repo (sau khi tải outputs/ từ Kaggle về):
    python -m labs.lab3.visualize.visualize
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from labs.lab3 import config
from shared.paths import output_dir

TEXT_PREVIEW_CHARS = 300   # visualize_v1.md: cắt hiển thị, giữ uid để tra bản đầy đủ
EXAMPLES_PER_GROUP = 3


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


# ---------------------------------------------------------------- đường huấn luyện
def plot_training_curves(history_csv, out_path: Path | None = None, title: str | None = None) -> Path:
    """Train loss, val loss (left) and val accuracy (right) per evaluation; the best checkpoint is marked.

    VI: Vẽ cả val loss lẫn val acc để thấy chỗ val loss tăng mà acc còn tăng (lý do chọn theo acc, L8).
    Trục trên: epoch.
    """
    plt = _plt()
    history_csv = Path(history_csv)
    h = pd.read_csv(history_csv)
    run = title or history_csv.name.replace("_history.csv", "")
    best = h.loc[h["val_acc"].idxmax()]  # idxmax: lần đầu tiên đạt max (cùng quy tắc với train)
    ratio = (h["epoch"] / h["step"]).median()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    if h["train_loss"].notna().any():
        ax1.plot(h["step"], h["train_loss"], "o-", ms=3, label="train loss")
    ax1.plot(h["step"], h["val_loss"], "o-", ms=3, label="val loss")
    ax1.set_xlabel("step")
    ax1.set_ylabel("loss")
    ax1.legend()
    ax2.plot(h["step"], h["val_acc"], "o-", ms=3, color="tab:green", label="val accuracy")
    ax2.scatter([best["step"]], [best["val_acc"]], s=90, facecolors="none", edgecolors="red", zorder=3,
                label=f"tốt nhất: step {int(best['step'])}, acc {best['val_acc']:.4f}")
    ax2.set_xlabel("step")
    ax2.set_ylabel("val accuracy")
    ax2.legend(loc="lower right")
    for ax in (ax1, ax2):
        ax.axvline(best["step"], color="red", ls=":", lw=1)
        ax.grid(alpha=0.3)
        if np.isfinite(ratio) and ratio > 0:
            top = ax.secondary_xaxis("top", functions=(lambda s, r=ratio: s * r, lambda e, r=ratio: e / r))
            top.set_xlabel("epoch")
    fig.suptitle(run)
    fig.tight_layout()
    out_path = Path(out_path) if out_path is not None else output_dir(config.LAB, "figures") / f"{run}_curves.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------- confusion matrix
def confusion_counts(df: pd.DataFrame) -> np.ndarray:
    """2x2 counts, rows = true label (0 NEGATIVE, 1 POSITIVE), columns = predicted. / Ma trận nhầm lẫn."""
    m = np.zeros((2, 2), dtype=int)
    for t, p in zip(df["label"].astype(int), df["pred"].astype(int)):
        m[t, p] += 1
    return m


def plot_confusion(pred_csvs: dict, out_path: Path | None = None) -> Path:
    """Confusion matrices side by side (counts + share of the true class), e.g. zero-shot vs fine-tuned.

    VI: pred_csvs {tên hiển thị: đường dẫn *_test_predictions.csv}. Tổng mỗi ma trận = số mẫu test.
    """
    plt = _plt()
    names = list(pred_csvs)
    fig, axes = plt.subplots(1, len(names), figsize=(4.6 * len(names), 4))
    axes = np.atleast_1d(axes)
    labels = [config.LABEL_NAMES[0], config.LABEL_NAMES[1]]
    for ax, name in zip(axes, names):
        m = confusion_counts(pd.read_csv(pred_csvs[name]))
        share = m / m.sum(axis=1, keepdims=True)
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{m[i, j]:,}\n({share[i, j]:.1%})", ha="center", va="center",
                        color="white" if share[i, j] > 0.5 else "black")
        ax.set_xticks([0, 1], labels)
        ax.set_yticks([0, 1], labels)
        ax.set_xlabel("dự đoán")
        ax.set_ylabel("nhãn thật")
        ax.set_title(f"{name}\nn = {m.sum():,}")
    fig.tight_layout()
    out_path = (Path(out_path) if out_path is not None
                else output_dir(config.LAB, "figures") / "confusion_zero_shot_vs_finetuned.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------- max_length
def maxlen_table(maxlen_runs, main_runs=None) -> pd.DataFrame:
    """Val accuracy per max_length: mean, std (n-1), n; the main runs give the value of config.MAX_LENGTH.

    VI: Thí nghiệm maxlen bỏ giá trị đã chạy ở main (512) → lấy 512 từ main_runs.csv.
    """
    frames = [pd.read_csv(maxlen_runs) if isinstance(maxlen_runs, (str, Path)) else maxlen_runs]
    if main_runs is not None:
        frames.append(pd.read_csv(main_runs) if isinstance(main_runs, (str, Path)) else main_runs)
    df = pd.concat(frames, ignore_index=True)
    g = df.groupby("max_length")["best_val_acc"]
    out = pd.DataFrame({"mean_val_acc": g.mean(), "std_val_acc": g.std(ddof=1), "n_seeds": g.size()})
    return out.reset_index().sort_values("max_length")


def plot_maxlen(maxlen_runs, main_runs=None, out_path: Path | None = None) -> Path:
    """Bar chart of val accuracy per max_length (error bar = std when there are several seeds)."""
    plt = _plt()
    t = maxlen_table(maxlen_runs, main_runs)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    yerr = t["std_val_acc"].fillna(0).to_numpy()
    x = [str(v) for v in t["max_length"]]
    ax.bar(x, t["mean_val_acc"], yerr=yerr, capsize=6, color="tab:blue", alpha=0.8)
    for xi, (m, n) in enumerate(zip(t["mean_val_acc"], t["n_seeds"])):
        ax.text(xi, m + 0.002, f"{m:.4f}\n(n={n})", ha="center", va="bottom", fontsize=9)
    lo = float(t["mean_val_acc"].min()) - 0.03
    ax.set_ylim(max(0.0, lo), min(1.0, float(t["mean_val_acc"].max()) + 0.02))
    ax.set_xlabel("max_length (token)")
    ax.set_ylabel("val accuracy (checkpoint tốt nhất)")
    ax.set_title("Ảnh hưởng của việc cắt review (so trên val)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = Path(out_path) if out_path is not None else output_dir(config.LAB, "figures") / "maxlen_val_acc.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------- phân tích lỗi
def error_analysis(pred_csv, test_df: pd.DataFrame, token_lengths: pd.DataFrame, max_length: int,
                   results_dir: Path | None = None, demo_dir: Path | None = None,
                   per_group: int = EXAMPLES_PER_GROUP) -> dict:
    """Error rate per group (truncated or not at max_length; original rating) + the most confident errors.

    VI: test_df: uid, text, rating (từ preprocessing); token_lengths: uid, n_tokens (eda_token_lengths.csv,
    đếm cả [CLS]/[SEP]). Bị cắt ⇔ n_tokens > max_length. Lý do sai ghi là giả thuyết, không khẳng định.
    Ghi outputs/results/error_by_group.csv và outputs/demo/error_examples.csv.
    """
    pred = pd.read_csv(pred_csv)
    df = (pred.merge(test_df[["uid", "text", "rating"]], on="uid", how="left")
          .merge(token_lengths[["uid", "n_tokens"]], on="uid", how="left"))
    if df["text"].isna().any() or df["n_tokens"].isna().any():
        raise ValueError("some predictions have no text or token length (uid mismatch)")
    df["error"] = df["label"] != df["pred"]
    df["truncated"] = np.where(df["n_tokens"] > max_length, f"bị cắt (> {max_length} token)",
                               f"không bị cắt (≤ {max_length} token)")
    df["confidence_wrong"] = np.where(df["pred"] == 1, df["prob_pos"], 1 - df["prob_pos"])

    groups = []
    for gtype, col in (("truncation", "truncated"), ("rating", "rating")):
        g = df.groupby(col)["error"]
        t = pd.DataFrame({"n": g.size(), "n_errors": g.sum().astype(int)}).reset_index().rename(columns={col: "group"})
        t.insert(0, "group_type", gtype)
        groups.append(t)
    total = pd.DataFrame([{"group_type": "all", "group": "all", "n": len(df), "n_errors": int(df["error"].sum())}])
    by_group = pd.concat([total, *groups], ignore_index=True)
    by_group["error_rate"] = (by_group["n_errors"] / by_group["n"]).round(4)
    by_group["group"] = by_group["group"].astype(str)

    wrong = df[df["error"]].sort_values("confidence_wrong", ascending=False)
    ex = []
    for gtype, col in (("truncation", "truncated"), ("rating", "rating")):
        for value, part in wrong.groupby(col):
            for r in part.head(per_group).itertuples():
                ex.append({"group_type": gtype, "group": str(value), "uid": r.uid, "label": r.label,
                           "pred": r.pred, "prob_pos": r.prob_pos, "n_tokens": int(r.n_tokens),
                           "rating": r.rating, "text_preview": r.text[:TEXT_PREVIEW_CHARS]})
    examples = pd.DataFrame(ex)

    results_dir = Path(results_dir) if results_dir is not None else output_dir(config.LAB, "results")
    demo_dir = Path(demo_dir) if demo_dir is not None else output_dir(config.LAB, "demo")
    results_dir.mkdir(parents=True, exist_ok=True)
    demo_dir.mkdir(parents=True, exist_ok=True)
    paths = {"by_group": results_dir / "error_by_group.csv", "examples": demo_dir / "error_examples.csv"}
    by_group.to_csv(paths["by_group"], index=False)
    examples.to_csv(paths["examples"], index=False)
    return {"by_group": by_group, "examples": examples, "paths": paths}


# ---------------------------------------------------------------- chạy hết từ file
def run_all(results_dir: Path | None = None, logs_dir: Path | None = None, figures_dir: Path | None = None,
            demo_dir: Path | None = None) -> list[Path]:
    """Make every figure / table whose input file exists; skip the others (print why).

    VI: Vẽ mọi hình có đủ file đầu vào. Confusion matrix và phân tích lỗi dùng seed có val acc cao nhất
    của lần chạy chính (ghi rõ seed trong tiêu đề).
    """
    res = Path(results_dir) if results_dir is not None else output_dir(config.LAB, "results")
    logs = Path(logs_dir) if logs_dir is not None else output_dir(config.LAB, "logs")
    figs = Path(figures_dir) if figures_dir is not None else output_dir(config.LAB, "figures")
    made = []
    for h in sorted(logs.glob("*_history.csv")):
        if "timing" in h.name or pd.read_csv(h).empty:
            continue
        run = h.name.replace("_history.csv", "").split("_", 2)[-1]   # <ts>_<run>: bỏ YYYYMMDD_HHMMSS
        made.append(plot_training_curves(h, figs / f"{run}_curves.png", title=f"{run} ({h.name[:15]})"))
    main_runs, maxlen_runs = res / "main_runs.csv", res / "maxlen_runs.csv"
    if maxlen_runs.exists():
        made.append(plot_maxlen(maxlen_runs, main_runs if main_runs.exists() else None, figs / "maxlen_val_acc.png"))
    zs = res / "zero_shot_test_predictions.csv"
    if main_runs.exists() and zs.exists():
        best = pd.read_csv(main_runs).sort_values("best_val_acc", ascending=False).iloc[0]
        ft = res / f"{best['run']}_test_predictions.csv"
        if ft.exists():
            made.append(plot_confusion({"zero-shot (SST-2)": zs, f"fine-tuned ({best['run']})": ft},
                                       figs / "confusion_zero_shot_vs_finetuned.png"))
            from labs.lab3.preprocessing.preprocessing import clean_text, load_split

            test = load_split("test")
            test["text"] = test["text"].map(clean_text)   # cùng văn bản model đã thấy (độ dài EDA cũng đo trên bản sạch)
            lengths = pd.read_csv(res / "eda_token_lengths.csv")
            out = error_analysis(ft, test, lengths[lengths["split"] == "test"], int(best["max_length"]),
                                 results_dir=res, demo_dir=demo_dir)
            made += list(out["paths"].values())
    else:
        print("[visualize] chưa có main_runs.csv hoặc zero_shot_test_predictions.csv: bỏ confusion / phân tích lỗi")
    return made


def main() -> None:
    from labs.lab3.main import utf8_stdout

    utf8_stdout()
    for p in run_all():
        print(p)


if __name__ == "__main__":
    main()
