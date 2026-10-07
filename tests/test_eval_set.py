"""The eval set must point at articles that really exist (no API key needed)."""

from rag.evaluate import load_eval_set, matches
from rag.ingest import load_chunks


def test_every_expected_article_exists() -> None:
    chunks = load_chunks()
    for item in load_eval_set()["answerable"]:
        for expected in item["expected"]:
            assert any(matches(c, [expected]) for c in chunks), (item["question"], expected)


def test_eval_set_size() -> None:
    data = load_eval_set()
    assert len(data["answerable"]) >= 15
    assert len(data["unanswerable"]) >= 2
    questions = [i["question"] for i in data["answerable"] + data["unanswerable"]]
    assert len(set(questions)) == len(questions)
