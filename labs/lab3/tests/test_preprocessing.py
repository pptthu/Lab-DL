"""Tests for labs/lab3/preprocessing (offline: fake aclImdb tree)."""
import pytest

from labs.lab3.preprocessing.preprocessing import clean_text, load_split, read_movie_ids, text_hash


def test_load_split_reads_id_rating_label_movie(fake_imdb_root):
    df = load_split("train", fake_imdb_root)
    assert len(df) == 8
    assert set(df.columns) >= {"uid", "split", "id", "rating", "label", "movie_id", "text"}
    row = df.set_index("uid").loc["train_neg_3"]
    assert (row["rating"], row["label"], row["movie_id"]) == (4, 0, "tt0000006")
    assert set(df.loc[df["label"] == 1, "rating"]) <= {7, 8, 9, 10}
    assert set(df.loc[df["label"] == 0, "rating"]) <= {1, 2, 3, 4}


def test_load_split_stops_when_url_lines_do_not_match(fake_imdb_root):
    urls = fake_imdb_root / "aclImdb" / "train" / "urls_pos.txt"
    urls.write_text("\n".join(urls.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(ValueError, match="URL lines"):
        load_split("train", fake_imdb_root)


def test_read_movie_ids_rejects_line_without_movie(tmp_path):
    p = tmp_path / "urls.txt"
    p.write_text("http://www.imdb.com/title/tt0453418/usercomments\nhttp://example.com/x\n")
    with pytest.raises(ValueError, match="no movie id"):
        read_movie_ids(p)


def test_clean_text_removes_br_only():
    assert clean_text("Great.<br /><br />I did NOT like   it<BR/>!") == "Great. I did NOT like it !"
    assert clean_text("no tags, don't change") == "no tags, don't change"


def test_text_hash_ignores_br_and_spaces():
    assert text_hash("a<br />b") == text_hash("a b") == text_hash("a   b")
    assert text_hash("a b") != text_hash("A b")


# ---------------------------------------------------------------- chia train / val
def test_split_groups_merge_same_movie_and_duplicate_text(fake_imdb_root):
    from labs.lab3.preprocessing.preprocessing import split_groups

    df = load_split("train", fake_imdb_root).set_index("uid")
    g = split_groups(df.reset_index()).set_axis(df.index)
    assert g["train_pos_0"] == g["train_pos_1"]          # cùng phim tt0000001
    assert g["train_pos_2"] == g["train_neg_2"]          # khác phim nhưng trùng nội dung
    assert g["train_pos_0"] != g["train_neg_3"]


def test_make_train_val_split_has_no_group_in_both(fake_imdb_root):
    from labs.lab3.preprocessing.preprocessing import make_train_val_split, split_groups

    train = load_split("train", fake_imdb_root)
    s = make_train_val_split(train, seed=0, n_folds=2)
    assert s["groups_in_both"] == 0
    assert s["n_train"] + s["n_val"] == len(train)
    assert not set(s["train_uids"]) & set(s["val_uids"])
    groups = dict(zip(train["uid"], split_groups(train)))
    assert not {groups[u] for u in s["train_uids"]} & {groups[u] for u in s["val_uids"]}


def test_make_train_val_split_is_reproducible(fake_imdb_root):
    from labs.lab3.preprocessing.preprocessing import make_train_val_split

    train = load_split("train", fake_imdb_root)
    assert make_train_val_split(train, seed=3, n_folds=2)["val_uids"] == \
        make_train_val_split(train, seed=3, n_folds=2)["val_uids"]


def test_load_datasets_returns_three_clean_splits(fake_imdb_root, tmp_path):
    from labs.lab3.preprocessing.preprocessing import load_datasets, make_train_val_split, save_split

    path = save_split(make_train_val_split(load_split("train", fake_imdb_root), seed=0, n_folds=2),
                      tmp_path / "split.json")
    ds = load_datasets(root=fake_imdb_root, split_path=path)
    assert set(ds) == {"train", "validation", "test"}
    assert len(ds["train"]) + len(ds["validation"]) == 8 and len(ds["test"]) == 4
    assert set(ds["train"].column_names) == {"uid", "text", "label", "rating", "movie_id"}
    assert not any("<br" in t.lower() for split in ds.values() for t in split["text"])
