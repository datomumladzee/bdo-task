"""Hybrid search over the policy chunks: keyword (BM25) plus embeddings.

The eval set (rag/eval_set.json, rag/evaluate.py) showed that embeddings alone
work poorly on Georgian, while BM25 over character 3-grams works well: Georgian
words change their endings (შვებულება, შვებულების, შვებულებას) but keep most
3-letter fragments. Embeddings add a smaller semantic signal for paraphrases.

Ranking score = BM25 (normalized 0..1) + DENSE_WEIGHT * cosine similarity,
multiplied by SUPERSEDED_FACTOR for outdated FAQ/Handbook sections.
"Not found" = the best normalized BM25 score is below RAG_MIN_SCORE.

Each chunk's embedding is cached in .index/ under a hash of (model, chunk text),
so a chunk is embedded once and never again unless its text or the model changes.

    uv run python -m rag.index --build
    uv run python -m rag.index --query "რამდენი დღით ადრე უნდა მოვითხოვო შვებულება?"
"""

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
from dotenv import load_dotenv

from rag.ingest import PROJECT_ROOT, Chunk, load_chunks

INDEX_DIR = PROJECT_ROOT / ".index"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
# Weight of the embedding similarity next to the normalized BM25 score.
DENSE_WEIGHT = 0.3
# Outdated summaries (Handbook, FAQ) are multiplied by this, so the current
# policy wins (Leave Policy Article 1.4, Remote Work Policy Article 1.3).
SUPERSEDED_FACTOR = 0.6
# Below this normalized BM25 score the documents most likely do not answer the
# question. Calibrated on the eval set; override with RAG_MIN_SCORE in .env.
DEFAULT_MIN_SCORE = 0.22
# Outdated passages returned separately, so the answer can point out the
# outdated FAQ/Handbook version without it taking a slot from the current policy.
OUTDATED_K = 2
NGRAM = 3
EMBED_BATCH_SIZE = 64


class Embedder(Protocol):
    model: str

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


class OpenAIEmbedder:
    def __init__(self, model: str | None = None) -> None:
        load_dotenv(PROJECT_ROOT / ".env")
        self.model = model or os.getenv("OPENAI_EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set; add it to .env (see .env.example)")
        from openai import OpenAI

        self._client = OpenAI()

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = list(texts[start : start + EMBED_BATCH_SIZE])
            response = self._client.embeddings.create(model=self.model, input=batch)
            vectors.extend(item.embedding for item in response.data)
        return np.asarray(vectors, dtype=np.float32)


# --- keyword search -----------------------------------------------------------


def char_ngrams(text: str, n: int = NGRAM) -> list[str]:
    """Character n-grams of each word, with word boundaries marked by "_"."""
    grams: list[str] = []
    for word in re.findall(r"[ა-ჰa-z0-9]+", text.lower()):
        padded = f"_{word}_"
        grams.extend(padded[i : i + n] for i in range(max(1, len(padded) - n + 1)))
    return grams


class BM25:
    """Okapi BM25 over character n-grams, with scores normalized to 0..1."""

    def __init__(
        self,
        texts: Sequence[str],
        tokenize: Callable[[str], list[str]] = char_ngrams,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.tokenize = tokenize
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(text)) for text in texts]
        self.lengths = np.array([sum(doc.values()) for doc in self.docs], dtype=np.float64)
        self.avg_length = float(self.lengths.mean()) if len(self.docs) else 0.0
        n = len(self.docs)
        document_frequency = Counter(term for doc in self.docs for term in doc)
        self.idf = {
            term: math.log(1 + (n - df + 0.5) / (df + 0.5))
            for term, df in document_frequency.items()
        }
        # A fragment that appears nowhere counts as maximally rare in the
        # normalization, so unknown words lower the score instead of being ignored.
        self.unseen_idf = math.log(1 + (n + 0.5) / 0.5)

    def scores(self, query: str) -> np.ndarray:
        """BM25 score of every document divided by the query's best possible score."""
        terms = self.tokenize(query)
        result = np.zeros(len(self.docs))
        if not terms or not self.docs:
            return result
        for i, doc in enumerate(self.docs):
            norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg_length)
            for term in terms:
                tf = doc.get(term)
                if tf:
                    result[i] += self.idf[term] * tf * (self.k1 + 1) / (tf + norm)
        best_possible = sum(self.idf.get(t, self.unseen_idf) * (self.k1 + 1) for t in terms)
        return result / best_possible


# --- embedding cache ----------------------------------------------------------


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.where(norms == 0, 1, norms)


def _cache_key(model: str, text: str) -> str:
    return hashlib.sha256(f"{model}\0{text}".encode()).hexdigest()


def _load_cache(index_dir: Path) -> dict[str, np.ndarray]:
    meta_path, vectors_path = index_dir / "embeddings.json", index_dir / "embeddings.npy"
    if not (meta_path.exists() and vectors_path.exists()):
        return {}
    keys = [row["key"] for row in json.loads(meta_path.read_text(encoding="utf-8"))]
    vectors = np.load(vectors_path)
    if len(keys) != len(vectors):
        return {}  # inconsistent files: start over rather than mix up vectors
    return dict(zip(keys, vectors, strict=True))


def _save_cache(
    index_dir: Path, model: str, chunks: list[Chunk], keys: list[str], vectors: np.ndarray
) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {"key": key, "model": model, "chunk_id": chunk.chunk_id, "citation": chunk.citation()}
        for key, chunk in zip(keys, chunks, strict=True)
    ]
    np.save(index_dir / "embeddings.npy", vectors)
    (index_dir / "embeddings.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8"
    )


# --- search -------------------------------------------------------------------


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float  # ranking score: (keyword + DENSE_WEIGHT * dense) x superseded factor
    keyword: float  # normalized BM25, 0..1
    dense: float  # cosine similarity of the embeddings


@dataclass(frozen=True)
class PolicySearch:
    query: str
    results: list[SearchResult]
    keyword_score: float  # best normalized BM25 over all chunks
    min_score: float
    # Best-matching superseded passages not already in results (scored without the
    # penalty), so the answer can say which older statement is outdated.
    outdated: list[SearchResult] = field(default_factory=list)

    @property
    def found(self) -> bool:
        """False means the documents most likely do not answer the question."""
        return self.keyword_score >= self.min_score


@dataclass
class PolicyIndex:
    chunks: list[Chunk]
    vectors: np.ndarray  # one normalized row per chunk
    embedder: Embedder
    bm25: BM25
    embedded_count: int = 0  # how many chunks this build had to embed (0 = all cached)
    _query_cache: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    @classmethod
    def build(
        cls, chunks: list[Chunk], embedder: Embedder, index_dir: Path = INDEX_DIR
    ) -> "PolicyIndex":
        cache = _load_cache(index_dir)
        keys = [_cache_key(embedder.model, chunk.text) for chunk in chunks]
        missing = [i for i, key in enumerate(keys) if key not in cache]
        if missing:
            new_vectors = _normalize(embedder.embed([chunks[i].text for i in missing]))
            for i, vector in zip(missing, new_vectors, strict=True):
                cache[keys[i]] = vector
        vectors = np.stack([cache[key] for key in keys]) if keys else np.zeros((0, 0))
        if missing or len(cache) != len(keys):
            _save_cache(index_dir, embedder.model, chunks, keys, vectors)
        return cls(
            chunks=chunks,
            vectors=vectors,
            embedder=embedder,
            bm25=BM25([chunk.text for chunk in chunks]),
            embedded_count=len(missing),
        )

    def _embed_query(self, query: str) -> np.ndarray:
        if query not in self._query_cache:
            self._query_cache[query] = _normalize(self.embedder.embed([query]))[0]
        return self._query_cache[query]

    def search(self, query: str, k: int = 5, min_score: float | None = None) -> PolicySearch:
        if min_score is None:
            min_score = float(os.getenv("RAG_MIN_SCORE") or DEFAULT_MIN_SCORE)
        if not self.chunks or not query.strip():
            return PolicySearch(query, [], 0.0, min_score)
        keyword = self.bm25.scores(query)
        dense = self.vectors @ self._embed_query(query)
        superseded = np.array([c.superseded_note is not None for c in self.chunks])
        raw = keyword + DENSE_WEIGHT * dense
        ranking = np.where(superseded, raw * SUPERSEDED_FACTOR, raw)
        top = np.argsort(-ranking)[:k]
        results = [
            SearchResult(self.chunks[i], float(ranking[i]), float(keyword[i]), float(dense[i]))
            for i in top
        ]
        shown = set(top.tolist())
        outdated = [
            SearchResult(self.chunks[i], float(raw[i]), float(keyword[i]), float(dense[i]))
            for i in np.argsort(-raw)
            if superseded[i] and i not in shown and keyword[i] >= min_score
        ][:OUTDATED_K]
        return PolicySearch(query, results, float(keyword.max()), min_score, outdated)


_default_index: PolicyIndex | None = None


def load_index() -> PolicyIndex:
    """The index over documents/, embedded with OpenAI and cached in .index/."""
    global _default_index
    if _default_index is None:
        _default_index = PolicyIndex.build(load_chunks(), OpenAIEmbedder())
    return _default_index


def search_policies(query: str, k: int = 5, index: PolicyIndex | None = None) -> PolicySearch:
    return (index or load_index()).search(query, k=k)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build or query the policy index")
    parser.add_argument("--build", action="store_true", help="embed any new or changed chunks")
    parser.add_argument("--query", help="search the documents")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args(argv)

    try:
        index = load_index()
    except RuntimeError as exc:
        parser.exit(1, f"error: {exc}\n")
    if args.build:
        print(
            f"{len(index.chunks)} chunks, {index.embedded_count} newly embedded "
            f"with {index.embedder.model}; cache in {INDEX_DIR}"
        )
    if args.query:
        search = index.search(args.query, k=args.k)
        print(
            f"keyword score {search.keyword_score:.3f} "
            f"(threshold {search.min_score}) found={search.found}"
        )
        for r in search.results:
            note = "  [superseded]" if r.chunk.superseded_note else ""
            print(
                f"  {r.score:.3f} (bm25 {r.keyword:.3f}, dense {r.dense:.3f})  "
                f"{r.chunk.citation()}{note}"
            )


if __name__ == "__main__":
    main()
