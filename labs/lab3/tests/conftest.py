"""Shared test fixtures for Lab 3: a fake aclImdb tree and a tiny offline tokenizer.

VI: Fixture dùng chung: cây aclImdb giả (không tải mạng) và tokenizer WordPiece nhỏ tạo tại chỗ.
"""
from __future__ import annotations

from pathlib import Path

import pytest

WORDS = ["the", "movie", "film", "was", "great", "good", "bad", "boring", "not", "i", "love", "hate",
         "it", "plot", "acting", "but", "and", "very", "a", ".", ",", "!", "<", ">", "/", "br", "##s",
         "story", "ending", "this"]

# (split, label_dir, id, rating, movie_id, text)
FAKE_REVIEWS = [
    ("train", "pos", 0, 9, "tt0000001", "Great movie.<br /><br />I love it!"),
    ("train", "pos", 1, 8, "tt0000001", "Good acting and   good plot."),
    ("train", "pos", 2, 10, "tt0000002", "Very good film."),
    ("train", "pos", 3, 7, "tt0000003", "Not bad at all."),
    ("train", "neg", 0, 1, "tt0000004", "Boring plot.<br />Bad acting."),
    ("train", "neg", 1, 2, "tt0000004", "I hate this movie."),
    ("train", "neg", 2, 3, "tt0000005", "Very good film."),  # trùng nội dung với train/pos/2, phim khác
    ("train", "neg", 3, 4, "tt0000006", "Not good, the ending was bad."),
    ("test", "pos", 0, 9, "tt0000101", "A great story."),
    ("test", "pos", 1, 8, "tt0000102", "Good acting and   good plot."),  # trùng với train/pos/1
    ("test", "neg", 0, 2, "tt0000103", "Bad film, boring ending."),
    ("test", "neg", 1, 1, "tt0000103", "The movie was bad."),
]


def make_fake_imdb(root: Path) -> Path:
    """Write a tiny aclImdb tree with urls_[pos|neg].txt in the real format; return root."""
    by_dir: dict[tuple[str, str], list] = {}
    for split, label_dir, rid, rating, movie, text in FAKE_REVIEWS:
        d = root / "aclImdb" / split / label_dir
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{rid}_{rating}.txt").write_text(text, encoding="utf-8")
        by_dir.setdefault((split, label_dir), []).append((rid, movie))
    for (split, label_dir), items in by_dir.items():
        lines = [f"http://www.imdb.com/title/{m}/usercomments" for _, m in sorted(items)]
        (root / "aclImdb" / split / f"urls_{label_dir}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


@pytest.fixture
def fake_imdb_root(tmp_path) -> Path:
    return make_fake_imdb(tmp_path / "raw")


@pytest.fixture
def tiny_tokenizer(tmp_path):
    """DistilBertTokenizerFast with a ~35-word vocab (no token_type_ids, like the real one); no download."""
    from transformers import DistilBertTokenizerFast

    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + WORDS
    path = tmp_path / "vocab.txt"
    path.write_text("\n".join(vocab) + "\n", encoding="utf-8")
    return DistilBertTokenizerFast(vocab_file=str(path), do_lower_case=True)
