import re

import pytest

from rag.ingest import Chunk, load_chunks

DOC_CODES = {
    "HR-POL-02",
    "HR-HB-01",
    "HR-FAQ-01",
    "HR-POL-05",
    "HR-POL-07",
    "SEC-POL-01",
    "FIN-POL-03",
}


@pytest.fixture(scope="module")
def chunks() -> list[Chunk]:
    return load_chunks()


def find(chunks: list[Chunk], code: str, article: str | None, kind: str = "text") -> Chunk:
    matches = [c for c in chunks if c.doc_code == code and c.article == article and c.kind == kind]
    assert matches, (code, article, kind)
    return matches[0]


def test_all_seven_documents_with_metadata(chunks: list[Chunk]) -> None:
    assert {c.doc_code for c in chunks} == DOC_CODES
    policy = find(chunks, "HR-POL-02", "4.4")
    assert (policy.version, policy.status, policy.effective) == (
        "4.0",
        "მოქმედი",
        "2026 წლის 1 იანვრიდან",
    )
    assert policy.doc_title == "შვებულებისა და გაცდენის პოლიტიკა"
    assert (
        find(chunks, "FIN-POL-03", "5.1").doc_title
        == "მივლინებისა და ხარჯების ანაზღაურების პოლიტიკა"
    )


def test_chunk_ids_are_unique(chunks: list[Chunk]) -> None:
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_pdf_page_header_and_footer_are_removed(chunks: list[Chunk]) -> None:
    for chunk in chunks:
        assert not re.search(r"გვერდი \d+", chunk.body), chunk.chunk_id
        assert "შპს „ნორთსტარ სერვისეზი“ |" not in chunk.body, chunk.chunk_id


def test_text_has_no_broken_characters(chunks: list[Chunk]) -> None:
    for chunk in chunks:
        assert "�" not in chunk.body
        assert re.search(r"[ა-ჰ]", chunk.body), chunk.chunk_id  # Georgian text survived


def test_pdf_articles_follow_bold_headings(chunks: list[Chunk]) -> None:
    articles = [c.article for c in chunks if c.doc_code == "FIN-POL-03" and c.kind == "text"]
    for expected in ["1.1", "1.2", "2", "4.4", "6.3", "10", "16"]:
        assert expected in articles
    # A wrapped body line starting with a number is not a heading.
    assert "55" not in articles


def test_pdf_lines_are_joined_and_paragraphs_kept(chunks: list[Chunk]) -> None:
    body = find(chunks, "HR-POL-05", "5.2").body
    assert "მხოლოდ უშუალო ხელმძღვანელის თანხმობა საკმარისი არ არის." in body
    assert len(body.split("\n")) == 2  # two paragraphs


def test_tables_are_separate_chunks(chunks: list[Chunk]) -> None:
    notice = find(chunks, "HR-POL-02", "4.4", kind="table").body
    assert "| 6-დან 15-მდე | 15 სამუშაო დღე |" in notice
    hotels = find(chunks, "FIN-POL-03", "5.1", kind="table").body
    assert "| ლონდონი | 180 გირვანქა სტერლინგი |" in hotels
    # The metadata table is metadata, not a chunk.
    assert not any("დოკუმენტის კოდი" in c.body for c in chunks)


def test_faq_letter_articles_and_intro(chunks: list[Chunk]) -> None:
    assert find(chunks, "HR-FAQ-01", "ა.2").heading.startswith("ა.2 ")
    intro = find(chunks, "HR-FAQ-01", None)
    assert intro.heading == "როგორ გამოვიყენოთ ეს დოკუმენტი"


def test_superseded_notes_follow_document_precedence(chunks: list[Chunk]) -> None:
    assert "HR-POL-02" in (find(chunks, "HR-FAQ-01", "ა.3").superseded_note or "")
    assert "HR-POL-05" in (find(chunks, "HR-FAQ-01", "ბ.1").superseded_note or "")
    assert "HR-POL-02" in (find(chunks, "HR-HB-01", "5.2").superseded_note or "")
    assert "HR-POL-05" in (find(chunks, "HR-HB-01", "6").superseded_note or "")
    assert find(chunks, "HR-FAQ-01", "ე.2").superseded_note is None
    assert find(chunks, "HR-HB-01", "7.1").superseded_note is None
    assert not any(c.superseded_note for c in chunks if c.doc_type == "policy")


def test_citation(chunks: list[Chunk]) -> None:
    assert find(chunks, "HR-POL-02", "4.4").citation() == (
        "შვებულებისა და გაცდენის პოლიტიკა (HR-POL-02, ვერსია 4.0), მუხლი 4.4"
    )
