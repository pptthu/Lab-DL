"""Load the official IMDB data (aclImdb), clean it and split train/val by movie (Exercise 2.2).

VI: Nạp dữ liệu IMDB gốc (aclImdb), làm sạch, chia train/val theo phim (Bài 2.2).
Desc: labs/lab3/docs/phases/preprocessing_v1.md

Run from the repo root / Chạy từ gốc repo:
    python -m labs.lab3.preprocessing.preprocessing            # tải + tạo split
"""
from __future__ import annotations

import hashlib
import json
import re
import tarfile
import urllib.request
from pathlib import Path

import pandas as pd

from labs.lab3 import config
from shared.paths import REPO_ROOT, data_dir

SPLITS = ("train", "test")
LABEL_DIRS = {"neg": 0, "pos": 1}
REVIEWS_PER_DIR = 12_500  # README aclImdb: 25k train + 25k test, cân bằng nhãn
_MOVIE_RE = re.compile(r"/title/(tt\d+)/")
_BR_RE = re.compile(r"<br\s*/?>", flags=re.IGNORECASE)
_SPACE_RE = re.compile(r"\s+")


# ---------------------------------------------------------------- tải dữ liệu
def raw_root(root: Path | None = None) -> Path:
    """Folder that holds aclImdb/. / Thư mục chứa aclImdb/ (mặc định labs/lab3/data/raw)."""
    return Path(root) if root is not None else data_dir(config.LAB, "raw")


def download_imdb(root: Path | None = None, url: str = config.IMDB_URL) -> Path:
    """Download and extract aclImdb_v1.tar.gz once; return the aclImdb folder.

    VI: Tải và giải nén một lần (đã có thì bỏ qua); kiểm 4 thư mục × 12.500 file.
    """
    root = raw_root(root)
    imdb = root / "aclImdb"
    if not (imdb / "test" / "pos").is_dir():
        archive = root / "aclImdb_v1.tar.gz"
        if not archive.exists():
            print(f"Downloading {url} ...")
            urllib.request.urlretrieve(url, archive)
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(root, filter="data")  # filter="data": chặn đường dẫn lạ trong file nén
    for split in SPLITS:
        for label_dir in LABEL_DIRS:
            n = len(list((imdb / split / label_dir).glob("*.txt")))
            if n != REVIEWS_PER_DIR:
                raise ValueError(f"{split}/{label_dir}: expected {REVIEWS_PER_DIR} files, found {n}")
    return imdb


# ---------------------------------------------------------------- đọc dữ liệu
def read_movie_ids(urls_file: Path) -> list[str]:
    """Movie id (tt...) of each line of urls_[pos|neg].txt; line i = review id i.

    VI: Mã phim của từng dòng. README aclImdb: review id i có URL ở dòng i.
    Dòng không có mã tt... thì báo lỗi (không đoán).
    """
    ids = []
    for i, line in enumerate(Path(urls_file).read_text(encoding="utf-8").splitlines()):
        m = _MOVIE_RE.search(line)
        if m is None:
            raise ValueError(f"{urls_file} line {i}: no movie id in {line!r}")
        ids.append(m.group(1))
    return ids


def load_split(split: str, root: Path | None = None) -> pd.DataFrame:
    """Read one official split (train or test) as a table.

    Columns: uid, split, id, rating, label, movie_id, text (raw, not cleaned).
    VI: Đọc một split chính thức. Tên file <id>_<rating>.txt; nhãn theo thư mục (neg=0, pos=1).
    Số dòng URL phải bằng số review, nếu không thì dừng (plan: không tự đổi cách chia).
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    # Bộ đệm: đọc 25k file nhỏ trên Windows rất chậm (đo 2026-11-20) → lần đầu đọc xong thì lưu
    # một file parquet cạnh aclImdb/ (git bỏ qua); lần sau đọc file này. Xoá file để đọc lại từ gốc.
    cache = raw_root(root) / f"imdb_{split}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    imdb = raw_root(root) / "aclImdb"
    rows = []
    for label_dir, label in LABEL_DIRS.items():
        files = sorted((imdb / split / label_dir).glob("*.txt"))
        movies = read_movie_ids(imdb / split / f"urls_{label_dir}.txt")
        if len(movies) != len(files):
            raise ValueError(f"{split}/{label_dir}: {len(files)} reviews but {len(movies)} URL lines")
        for f in files:
            rid, rating = (int(x) for x in f.stem.split("_"))
            rows.append({
                "uid": f"{split}_{label_dir}_{rid}",
                "split": split,
                "id": rid,
                "rating": rating,
                "label": label,
                "movie_id": movies[rid],
                "text": f.read_text(encoding="utf-8"),
            })
    df = pd.DataFrame(rows).sort_values("uid", kind="stable").reset_index(drop=True)
    df.to_parquet(cache, index=False)
    return df


def clean_text(text: str) -> str:
    """Replace <br /> tags with a space and collapse spaces; keep everything else.

    VI: Chỉ bỏ thẻ <br /> và khoảng trắng thừa. Không bỏ stopword (sẽ mất not/no/nor),
    không hạ chữ thường (tokenizer uncased tự làm). Lý do: phases/preprocessing_v1.md.
    """
    return _SPACE_RE.sub(" ", _BR_RE.sub(" ", text)).strip()


def text_hash(text: str) -> str:
    """Hash of the cleaned text, used to find duplicate reviews. / Băm nội dung đã làm sạch."""
    return hashlib.sha1(clean_text(text).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- chia train / val
def split_groups(df: pd.DataFrame) -> pd.Series:
    """Group id per review: same movie OR same cleaned text -> same group (union-find).

    VI: Nhóm khi chia = phim; review trùng nội dung (dù khác phim) gộp chung nhóm,
    để cả cặp nằm cùng một phía (tránh lộ dữ liệu sang val). Thư duyệt 2026-11-20.
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    hashes = df["text"].map(text_hash)
    first_movie_of_hash: dict[str, str] = {}
    for movie, h in zip(df["movie_id"], hashes):
        find(movie)
        if h in first_movie_of_hash:
            union(movie, first_movie_of_hash[h])
        else:
            first_movie_of_hash[h] = movie
    return df["movie_id"].map(find)


def make_train_val_split(train: pd.DataFrame, seed: int = config.SEED, n_folds: int = config.VAL_FOLDS) -> dict:
    """Pick one of n_folds StratifiedGroupKFold folds as validation (about 1/n_folds of train).

    VI: Cân bằng nhãn (stratified) và không nhóm nào (phim / review trùng) nằm ở cả train và val.
    Trả về dict có danh sách uid train/val và thông tin để ghi file split.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    groups = split_groups(train)
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    _, val_idx = next(sgkf.split(train, train["label"], groups))
    is_val = pd.Series(False, index=train.index)
    is_val.iloc[val_idx] = True
    tr, va = train[~is_val], train[is_val]
    return {
        "method": f"StratifiedGroupKFold(n_splits={n_folds}, shuffle=True, random_state={seed}); fold 0 = val",
        "group": "movie_id, merged with duplicate texts (union-find)",
        "seed": seed,
        "n_train": len(tr), "n_val": len(va),
        "val_pos_share": round(float(va["label"].mean()), 4),
        "n_groups": int(groups.nunique()),
        "groups_in_both": int(len(set(groups[~is_val]) & set(groups[is_val]))),
        "train_uids": tr["uid"].tolist(),
        "val_uids": va["uid"].tolist(),
    }


def default_split_path() -> Path:
    return data_dir(config.LAB, "splits") / "train_val_split.json"


def save_split(split: dict, path: Path | None = None) -> Path:
    """Write the split as JSON (committed, so every run uses the same val). / Ghi split JSON (commit)."""
    path = Path(path) if path is not None else default_split_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def read_split(path: Path | None = None) -> dict:
    path = Path(path) if path is not None else default_split_path()
    return json.loads(path.read_text(encoding="utf-8"))


def load_datasets(root: Path | None = None, split_path: Path | None = None, smoke_n: int | None = None):
    """DatasetDict train / validation / test with cleaned text (datasets library).

    Columns: uid, text (cleaned), label, rating, movie_id.
    VI: Đọc split JSON đã lưu (tạo mới nếu chưa có). smoke_n: lấy smoke_n review mỗi split.
    """
    from datasets import Dataset, DatasetDict

    train_all, test = load_split("train", root), load_split("test", root)
    path = Path(split_path) if split_path is not None else default_split_path()
    split = read_split(path) if path.exists() else read_split(save_split(make_train_val_split(train_all), path))
    by_uid = train_all.set_index("uid")
    parts = {
        "train": by_uid.loc[split["train_uids"]].reset_index(),
        "validation": by_uid.loc[split["val_uids"]].reset_index(),
        "test": test,
    }
    cols = ["uid", "text", "label", "rating", "movie_id"]
    out = {}
    for name, df in parts.items():
        if smoke_n:
            df = sample_per_label(df, smoke_n)
        df = df.assign(text=df["text"].map(clean_text))[cols].reset_index(drop=True)
        out[name] = Dataset.from_pandas(df, preserve_index=False)
    return DatasetDict(out)


def sample_per_label(df: pd.DataFrame, n: int, seed: int = config.SEED) -> pd.DataFrame:
    """n reviews (half per label) for a smoke run. / Lấy n review (mỗi nhãn một nửa) cho smoke run."""
    half = max(1, n // 2)
    parts = [g.sample(n=min(half, len(g)), random_state=seed) for _, g in df.groupby("label")]
    return pd.concat(parts).sort_values("uid", kind="stable").reset_index(drop=True)


def main(argv=None) -> None:
    """Download, build the train/val split, print its checks. / Tải, tạo split, in các kiểm tra."""
    import argparse

    parser = argparse.ArgumentParser(description="Lab 3 preprocessing")
    parser.add_argument("--force", action="store_true", help="tạo lại split dù đã có file")
    args = parser.parse_args(argv)
    download_imdb()
    train = load_split("train")
    path = default_split_path()
    if args.force or not path.exists():
        save_split(make_train_val_split(train), path)
    split = read_split(path)
    print({k: v for k, v in split.items() if not k.endswith("_uids")})
    print("split file:", path.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
