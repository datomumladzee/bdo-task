"""Read the policy documents and split them into chunks, one per article.

DOCX files are split on their Word heading styles. PDF files are split on bold
numbered headings ("4." or "4.4"), with the repeating page header and footer
removed. Every table becomes its own chunk under the article it appears in.
Each chunk carries the document's code, version, effective date and status so
answers can cite their source.

    uv run python -m rag.ingest            # summary
    uv run python -m rag.ingest --show 12  # print some chunks for review
"""

import argparse
import re
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import pdfplumber
from docx import Document
from docx.table import Table as DocxTable
from docx.text.paragraph import Paragraph as DocxParagraph

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = PROJECT_ROOT / "documents"

# Metadata table labels (first table of every document).
META_KEYS = {
    "დოკუმენტის კოდი": "doc_code",
    "ვერსია": "version",
    "ძალაშია": "effective",
    "ბოლო განახლება": "effective",  # the FAQ has a "last updated" date instead
    "სტატუსი": "status",
}

# PDF page furniture: "შპს „ნორთსტარ სერვისეზი“ | <title>" and "<code> | ... გვერდი N".
PDF_HEADER = re.compile(r"^შპს „ნორთსტარ სერვისეზი“ \| (?P<title>.+)$")
PDF_FOOTER = re.compile(r"გვერდი \d+$")
PDF_HEADING = re.compile(r"^(?P<num>\d+)\.(?P<sub>\d+)?\s+\S")
PDF_HEADING_MIN_SIZE = 11.2  # body text is 11.0, article headings 11.4, sections 14.0
PDF_TITLE_MIN_SIZE = 18.0
PDF_PARAGRAPH_GAP = 1.7  # line distance (in font sizes) that starts a new paragraph

# Article number at the start of a heading: "4.", "4.4", or the FAQ's single
# Georgian letter "ა.", "ა.1". A plain word ("როგორ ...") is not a number.
ARTICLE = re.compile(r"^(?P<article>\d+(?:\.\d+)?|[ა-ჰ]\.\d+|[ა-ჰ](?=\.))\.?\s")

# Precedence, as the documents themselves state it:
# - Leave Policy 1.4: the policy prevails over the Handbook and the FAQ.
# - Handbook 1.2: specific policies prevail; its summaries may be outdated.
# - FAQ introduction: if an answer contradicts a policy, follow the policy.
# - Remote Work Policy 1.3: replaces Handbook article 6 and the FAQ overview.
DOC_TYPES = {"HR-HB-01": "handbook", "HR-FAQ-01": "faq"}  # everything else: "policy"
SUPERSEDED: dict[tuple[str, str], str] = {
    ("HR-HB-01", "5"): (
        "მოძველებული შეჯამება; მოქმედებს შვებულებისა და გაცდენის პოლიტიკა "
        "(HR-POL-02 v4.0, მუხლი 1.4)"
    ),
    ("HR-HB-01", "6"): (
        "მოძველებული შეჯამება; ჩანაცვლებულია დისტანციური და ჰიბრიდული მუშაობის "
        "პოლიტიკით (HR-POL-05 v2.0, მუხლი 1.3)"
    ),
    ("HR-FAQ-01", "ა"): (
        "საცნობარო მასალა, შეიძლება მოძველებული იყოს; მოქმედებს შვებულებისა და "
        "გაცდენის პოლიტიკა (HR-POL-02 v4.0, მუხლი 1.4)"
    ),
    ("HR-FAQ-01", "ბ"): (
        "მოძველებული მიმოხილვა; ჩანაცვლებულია დისტანციური და ჰიბრიდული მუშაობის "
        "პოლიტიკით (HR-POL-05 v2.0, მუხლი 1.3)"
    ),
}


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    file: str
    doc_code: str
    doc_title: str
    version: str
    effective: str
    status: str
    doc_type: str  # "policy", "handbook" or "faq"
    article: str | None  # "4.4", "ა.1"; None for unnumbered headings
    section: str | None  # the level-1 heading
    heading: str  # the article's own heading
    kind: str  # "text" or "table"
    page: int | None  # PDF page where the chunk starts
    superseded_note: str | None
    body: str

    @property
    def text(self) -> str:
        """What gets embedded: where the chunk comes from, then its content."""
        path = [self.doc_title]
        if self.section and self.section != self.heading:
            path.append(self.section)
        path.append(self.heading)
        return " — ".join(path) + "\n" + self.body

    def citation(self) -> str:
        where = f"მუხლი {self.article}" if self.article else self.heading
        return f"{self.doc_title} ({self.doc_code}, ვერსია {self.version}), {where}"


# --- events: a document as a flat stream of headings, paragraphs and tables ----


@dataclass(frozen=True)
class Heading:
    level: int
    text: str
    page: int | None = None


@dataclass(frozen=True)
class Paragraph:
    text: str
    page: int | None = None


@dataclass(frozen=True)
class Table:
    rows: list[list[str]]
    page: int | None = None


Event = Heading | Paragraph | Table


def _clean(text: str | None) -> str:
    return " ".join((text or "").split())


def _is_meta_table(rows: list[list[str]]) -> bool:
    return any(row and row[0] in META_KEYS for row in rows)


def _meta_from_rows(rows: list[list[str]]) -> dict[str, str]:
    return {META_KEYS[row[0]]: row[1] for row in rows if len(row) >= 2 and row[0] in META_KEYS}


def _docx_events(path: Path) -> tuple[str, dict[str, str], list[Event]]:
    doc = Document(str(path))
    title = ""
    meta: dict[str, str] = {}
    events: list[Event] = []
    for element in doc.element.body.iterchildren():
        tag = element.tag.rsplit("}", 1)[-1]
        if tag == "p":
            paragraph = DocxParagraph(element, doc)
            text = _clean(paragraph.text)
            if not text:
                continue
            style = paragraph.style.name if paragraph.style is not None else ""
            if style == "Title":
                title = text
            elif style == "Heading 1":
                events.append(Heading(1, text))
            elif style == "Heading 2":
                events.append(Heading(2, text))
            elif style.startswith("List"):
                events.append(Paragraph("• " + text))
            else:
                events.append(Paragraph(text))
        elif tag == "tbl":
            rows = [
                [_clean(cell.text) for cell in row.cells] for row in DocxTable(element, doc).rows
            ]
            if not meta and _is_meta_table(rows):
                meta = _meta_from_rows(rows)
            else:
                events.append(Table(rows))
    return title, meta, events


@dataclass
class _Line:
    text: str
    top: float
    size: float
    bold: bool
    page: int


def _pdf_lines(
    page: pdfplumber.page.Page, page_no: int, skip: list[tuple[float, float]]
) -> list[_Line]:
    lines = []
    for raw in page.extract_text_lines(return_chars=True):
        top = raw["top"]
        if any(start - 1 <= top <= end + 1 for start, end in skip):
            continue  # inside a table; the table is read separately
        chars = [c for c in raw["chars"] if c["text"].strip()]
        if not chars:
            continue
        bold = sum("Bold" in c["fontname"] for c in chars) > len(chars) / 2
        size = max(c["size"] for c in chars)
        lines.append(_Line(_clean(raw["text"]), top, size, bold, page_no))
    return lines


def _pdf_events(path: Path) -> tuple[str, dict[str, str], list[Event]]:
    title = ""
    meta: dict[str, str] = {}
    events: list[Event] = []
    with pdfplumber.open(str(path)) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            items: list[tuple[float, Event | _Line]] = []
            table_spans = []
            for found in page.find_tables():
                table_spans.append((found.bbox[1], found.bbox[3]))
                rows = [[_clean(cell) for cell in row] for row in found.extract()]
                if not meta and _is_meta_table(rows):
                    meta = _meta_from_rows(rows)
                else:
                    items.append((found.bbox[1], Table(rows, page_no)))
            for line in _pdf_lines(page, page_no, table_spans):
                if match := PDF_HEADER.match(line.text):
                    title = title or match["title"]
                    continue
                if PDF_FOOTER.search(line.text) or line.size >= PDF_TITLE_MIN_SIZE:
                    continue  # footer, or the big title on page 1
                items.append((line.top, line))
            items.sort(key=lambda item: item[0])
            events.extend(_pdf_page_events(item for _, item in items))
    return title, meta, events


def _pdf_page_events(items: Iterator[Event | _Line]) -> list[Event]:
    """Turn sorted lines into headings and paragraphs; wrapped lines are joined."""
    events: list[Event] = []
    paragraph: list[str] = []
    previous: _Line | None = None
    page: int | None = None

    def flush() -> None:
        if paragraph:
            events.append(Paragraph(" ".join(paragraph), page))
            paragraph.clear()

    for item in items:
        if not isinstance(item, _Line):
            flush()
            events.append(item)
            previous = None
            continue
        heading = PDF_HEADING.match(item.text)
        if heading and item.bold and item.size >= PDF_HEADING_MIN_SIZE:
            flush()
            events.append(Heading(2 if heading["sub"] else 1, item.text, item.page))
            previous = None
            continue
        new_paragraph = previous is None or item.top - previous.top > PDF_PARAGRAPH_GAP * item.size
        if new_paragraph:
            flush()
            page = item.page
        paragraph.append(item.text)
        previous = item
    flush()
    return events


# --- chunking -------------------------------------------------------------------


def _article(heading: str) -> str | None:
    match = ARTICLE.match(heading)
    return match["article"] if match else None


def _superseded(doc_code: str, article: str | None) -> str | None:
    if article is None:
        return None
    top = article.split(".")[0]
    return SUPERSEDED.get((doc_code, top))


def _render_table(rows: list[list[str]]) -> str:
    return "\n".join("| " + " | ".join(row) + " |" for row in rows if any(row))


def chunk_document(path: Path) -> list[Chunk]:
    if path.suffix == ".docx":
        title, meta, events = _docx_events(path)
    elif path.suffix == ".pdf":
        title, meta, events = _pdf_events(path)
    else:
        raise ValueError(f"Unsupported document type: {path.name}")
    missing = {"doc_code", "version", "effective", "status"} - set(meta)
    if missing:
        raise ValueError(f"{path.name}: metadata table is missing {sorted(missing)}")

    doc_code = meta["doc_code"]
    chunks: list[Chunk] = []
    section: str | None = None
    heading: str | None = None
    page: int | None = None
    body: list[str] = []
    counter = 0

    def make(kind: str, text: str, at_page: int | None) -> None:
        nonlocal counter
        assert heading is not None
        counter += 1
        article = _article(heading)
        chunks.append(
            Chunk(
                chunk_id=f"{doc_code}:{article or 'x'}:{counter}",
                file=path.name,
                doc_code=doc_code,
                doc_title=title,
                version=meta["version"],
                effective=meta["effective"],
                status=meta["status"],
                doc_type=DOC_TYPES.get(doc_code, "policy"),
                article=article,
                section=section,
                heading=heading,
                kind=kind,
                page=at_page,
                superseded_note=_superseded(doc_code, article),
                body=text,
            )
        )

    def flush() -> None:
        if heading is not None and body:
            make("text", "\n".join(body), page)
        body.clear()

    for event in events:
        if isinstance(event, Heading):
            flush()
            if event.level == 1:
                section = event.text
            heading = event.text
            page = event.page
        elif heading is None:
            continue  # preamble before the first heading (the fictional-company note)
        elif isinstance(event, Paragraph):
            if not body:
                page = event.page or page
            body.append(event.text)
        else:
            make("table", _render_table(event.rows), event.page or page)
    flush()
    return chunks


def load_chunks(documents_dir: Path = DOCUMENTS_DIR) -> list[Chunk]:
    files = sorted(p for p in documents_dir.iterdir() if p.suffix in {".docx", ".pdf"})
    return [chunk for path in files for chunk in chunk_document(path)]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Chunk the policy documents")
    parser.add_argument("--show", type=int, default=0, help="print this many chunks")
    parser.add_argument("--doc", help="only show chunks whose doc_code contains this")
    args = parser.parse_args(argv)

    chunks = load_chunks()
    by_doc: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        by_doc.setdefault(chunk.doc_code, []).append(chunk)
    for code, items in by_doc.items():
        first = items[0]
        tables = sum(c.kind == "table" for c in items)
        print(
            f"{code:<11} v{first.version:<4} {first.doc_type:<8} {len(items):>3} chunks "
            f"({tables} tables)  {first.status} | {first.effective} | {first.doc_title}"
        )
    print(f"total: {len(chunks)} chunks")

    shown = [c for c in chunks if not args.doc or args.doc in c.doc_code][: args.show]
    for chunk in shown:
        print("\n" + "=" * 80)
        fields = {k: v for k, v in asdict(chunk).items() if k != "body"}
        print(fields)
        print("-" * 80)
        print(chunk.text)


if __name__ == "__main__":
    main()
