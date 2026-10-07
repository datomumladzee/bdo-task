"""Retrieval evaluation on a fixed set of Georgian questions.

For each answerable question: is one of its expected articles in the top k
(hit@1, hit@k) and at what rank (MRR)? For each unanswerable question: is the
best score below the "not found" threshold? The score ranges of both groups
show where the threshold should sit.

    uv run python -m rag.evaluate
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from rag.index import PolicyIndex, PolicySearch, load_index
from rag.ingest import Chunk

EVAL_SET = Path(__file__).with_name("eval_set.json")


def load_eval_set(path: Path = EVAL_SET) -> dict[str, list[dict]]:
    return json.loads(path.read_text(encoding="utf-8"))


def matches(chunk: Chunk, expected: list[list[str]]) -> bool:
    """An expected article also matches its sub-articles ("8" matches "8.1")."""
    return any(
        chunk.doc_code == doc_code
        and chunk.article is not None
        and (chunk.article == article or chunk.article.startswith(article + "."))
        for doc_code, article in expected
    )


def first_hit_rank(search: PolicySearch, expected: list[list[str]]) -> int | None:
    for rank, result in enumerate(search.results, start=1):
        if matches(result.chunk, expected):
            return rank
    return None


@dataclass
class Report:
    answerable: int
    hit_at_1: int
    hit_at_k: int
    mrr: float
    unanswerable: int
    rejected: int
    lowest_answerable_score: float
    highest_unanswerable_score: float


def evaluate(index: PolicyIndex, k: int = 5, verbose: bool = True) -> Report:
    data = load_eval_set()
    hit1 = hitk = 0
    reciprocal = 0.0
    answerable_scores = []
    for item in data["answerable"]:
        search = index.search(item["question"], k=k)
        rank = first_hit_rank(search, item["expected"])
        hit1 += rank == 1
        hitk += rank is not None
        reciprocal += 1 / rank if rank else 0.0
        answerable_scores.append(search.keyword_score)
        if verbose:
            status = f"rank {rank}" if rank else "MISS  "
            print(f"[{status}] {search.keyword_score:.3f}  {item['topic']}")
            if rank != 1:
                for r in search.results:
                    mark = "*" if matches(r.chunk, item["expected"]) else " "
                    print(f"      {mark} {r.score:.3f} {r.chunk.doc_code} {r.chunk.article}")

    rejected = 0
    unanswerable_scores = []
    for item in data["unanswerable"]:
        search = index.search(item["question"], k=k)
        rejected += not search.found
        unanswerable_scores.append(search.keyword_score)
        if verbose:
            top = search.results[0].chunk if search.results else None
            where = f"{top.doc_code} {top.article}" if top else "-"
            verdict = "not found" if not search.found else "WRONGLY FOUND"
            print(f"[{verdict}] {search.keyword_score:.3f}  {item['topic']}  (top: {where})")

    n = len(data["answerable"])
    report = Report(
        answerable=n,
        hit_at_1=hit1,
        hit_at_k=hitk,
        mrr=reciprocal / n,
        unanswerable=len(data["unanswerable"]),
        rejected=rejected,
        lowest_answerable_score=min(answerable_scores),
        highest_unanswerable_score=max(unanswerable_scores),
    )
    if verbose:
        print(
            f"\nhit@1 {hit1}/{n}   hit@{k} {hitk}/{n}   MRR {report.mrr:.2f}\n"
            f"not found correctly: {rejected}/{report.unanswerable}\n"
            f"answerable best scores >= {report.lowest_answerable_score:.3f}; "
            f"unanswerable best scores <= {report.highest_unanswerable_score:.3f}"
        )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval on the eval set")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args(argv)
    try:
        index = load_index()
    except RuntimeError as exc:
        parser.exit(1, f"error: {exc}\n")
    evaluate(index, k=args.k)


if __name__ == "__main__":
    main()
