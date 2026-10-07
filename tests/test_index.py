import hashlib
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rag.index import BM25, DENSE_WEIGHT, SUPERSEDED_FACTOR, PolicyIndex, char_ngrams
from rag.ingest import Chunk


class FakeEmbedder:
    """Deterministic bag-of-words embedding: shared words mean higher similarity."""

    model = "fake-embedding"

    def __init__(self) -> None:
        self.calls = 0
        self.texts_embedded = 0

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        self.calls += 1
        self.texts_embedded += len(texts)
        vectors = np.zeros((len(texts), 256), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in text.lower().split():
                bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % 256
                vectors[row, bucket] += 1
        return vectors


def chunk(chunk_id: str, body: str, superseded: str | None = None) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        file="doc.docx",
        doc_code="DOC",
        doc_title="title",
        version="1.0",
        effective="2026",
        status="მოქმედი",
        doc_type="faq" if superseded else "policy",
        article=chunk_id,
        section=None,
        heading=chunk_id,
        kind="text",
        page=None,
        superseded_note=superseded,
        body=body,
    )


CHUNKS = [
    chunk("1", "annual leave notice five working days"),
    chunk("2", "hotel limit london pounds per night"),
    chunk("3", "password length fourteen characters"),
]


def test_build_embeds_once_then_uses_the_cache(tmp_path: Path) -> None:
    first = FakeEmbedder()
    PolicyIndex.build(CHUNKS, first, index_dir=tmp_path)
    assert first.texts_embedded == 3
    assert (tmp_path / "embeddings.npy").exists()
    assert (tmp_path / "embeddings.json").exists()

    second = FakeEmbedder()
    index = PolicyIndex.build(CHUNKS, second, index_dir=tmp_path)
    assert second.calls == 0
    assert index.embedded_count == 0


def test_only_changed_chunks_are_embedded_again(tmp_path: Path) -> None:
    PolicyIndex.build(CHUNKS, FakeEmbedder(), index_dir=tmp_path)
    changed = [*CHUNKS[:2], replace(CHUNKS[2], body="password length sixteen characters")]
    embedder = FakeEmbedder()
    index = PolicyIndex.build(changed, embedder, index_dir=tmp_path)
    assert embedder.texts_embedded == 1
    assert index.embedded_count == 1


def test_a_different_model_does_not_reuse_vectors(tmp_path: Path) -> None:
    PolicyIndex.build(CHUNKS, FakeEmbedder(), index_dir=tmp_path)
    other = FakeEmbedder()
    other.model = "another-model"
    PolicyIndex.build(CHUNKS, other, index_dir=tmp_path)
    assert other.texts_embedded == 3


def test_search_ranks_by_similarity(tmp_path: Path) -> None:
    index = PolicyIndex.build(CHUNKS, FakeEmbedder(), index_dir=tmp_path)
    search = index.search("hotel limit in london", k=2, min_score=0.1)
    assert search.results[0].chunk.chunk_id == "2"
    assert len(search.results) == 2
    assert search.found
    assert search.results[0].score >= search.results[1].score


def test_unrelated_question_is_not_found(tmp_path: Path) -> None:
    index = PolicyIndex.build(CHUNKS, FakeEmbedder(), index_dir=tmp_path)
    search = index.search("parking fines downtown", min_score=0.3)
    assert not search.found


def test_superseded_chunks_rank_below_equal_policy_chunks(tmp_path: Path) -> None:
    same = "carry over unused days"
    chunks = [chunk("faq", same, superseded="outdated"), chunk("policy", same)]
    index = PolicyIndex.build(chunks, FakeEmbedder(), index_dir=tmp_path)
    results = index.search(same, k=2, min_score=0.1).results
    assert [r.chunk.chunk_id for r in results] == ["policy", "faq"]
    policy, faq = results
    assert policy.score == pytest.approx(policy.keyword + DENSE_WEIGHT * policy.dense)
    assert faq.score == pytest.approx((faq.keyword + DENSE_WEIGHT * faq.dense) * SUPERSEDED_FACTOR)


def test_empty_query_returns_nothing(tmp_path: Path) -> None:
    index = PolicyIndex.build(CHUNKS, FakeEmbedder(), index_dir=tmp_path)
    assert index.search("   ").results == []


# --- BM25 over character n-grams ------------------------------------------------


def test_char_ngrams_mark_word_boundaries() -> None:
    assert char_ngrams("Ab cd") == ["_ab", "ab_", "_cd", "cd_"]
    assert char_ngrams("a") == ["_a_"]


def test_inflected_georgian_words_still_match() -> None:
    bm25 = BM25(["შვებულების მოთხოვნა", "პაროლის სიგრძე"])
    scores = bm25.scores("შვებულებას")  # different case ending, no exact word match
    assert scores[0] > 0
    assert scores[1] == 0


def test_bm25_scores_are_normalized() -> None:
    bm25 = BM25(["annual leave notice", "hotel limit london", "password length"])
    scores = bm25.scores("hotel limit london")
    assert 0 < scores.max() <= 1
    assert scores.argmax() == 1


def test_unknown_words_lower_the_score() -> None:
    bm25 = BM25(["hotel limit london", "password length"])
    known = bm25.scores("hotel limit").max()
    with_unknown = bm25.scores("hotel limit xylophone quartz").max()
    assert with_unknown < known
