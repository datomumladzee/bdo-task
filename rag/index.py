"""Embedding index over the policy chunks, with an on-disk cache.

Each chunk's embedding is cached in .index/ under a hash of (model, chunk text),
so a chunk is embedded once and never again unless its text or the model changes.

    uv run python -m rag.index --build
    uv run python -m rag.index --query "რამდენი დღით ადრე უნდა მოვითხოვო შვებულება?"
"""

import argparse
import hashlib
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
from dotenv import load_dotenv

from rag.ingest import PROJECT_ROOT, Chunk, load_chunks

INDEX_DIR = PROJECT_ROOT / ".index"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
# Below this cosine similarity the best match is treated as "not in the documents".
# Calibrated against real questions; override with RAG_MIN_SCORE in .env.
DEFAULT_MIN_SCORE = 0.30
# Outdated summaries (Handbook, FAQ) rank slightly below the current policies.
SUPERSEDED_PENALTY = 0.05
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


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float  # cosine similarity between the question and the chunk
    rank_score: float  # score after the superseded-document penalty; used for ordering


@dataclass(frozen=True)
class PolicySearch:
    query: str
    results: list[SearchResult]
    min_score: float

    @property
    def best_score(self) -> float:
        return max((r.score for r in self.results), default=0.0)

    @property
    def found(self) -> bool:
        """False means the documents most likely do not answer the question."""
        return self.best_score >= self.min_score


@dataclass
class PolicyIndex:
    chunks: list[Chunk]
    vectors: np.ndarray  # one normalized row per chunk
    embedder: Embedder
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
        return cls(chunks=chunks, vectors=vectors, embedder=embedder, embedded_count=len(missing))

    def _embed_query(self, query: str) -> np.ndarray:
        if query not in self._query_cache:
            self._query_cache[query] = _normalize(self.embedder.embed([query]))[0]
        return self._query_cache[query]

    def search(self, query: str, k: int = 5, min_score: float | None = None) -> PolicySearch:
        if min_score is None:
            min_score = float(os.getenv("RAG_MIN_SCORE") or DEFAULT_MIN_SCORE)
        if not self.chunks or not query.strip():
            return PolicySearch(query, [], min_score)
        scores = self.vectors @ self._embed_query(query)
        penalties = np.array(
            [SUPERSEDED_PENALTY if c.superseded_note else 0.0 for c in self.chunks]
        )
        ranked = scores - penalties
        top = np.argsort(-ranked)[:k]
        results = [SearchResult(self.chunks[i], float(scores[i]), float(ranked[i])) for i in top]
        return PolicySearch(query, results, min_score)


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
            f"best score {search.best_score:.3f} (threshold {search.min_score}) found={search.found}"
        )
        for r in search.results:
            note = "  [superseded]" if r.chunk.superseded_note else ""
            print(f"  {r.score:.3f}  {r.chunk.citation()}{note}")


if __name__ == "__main__":
    main()
