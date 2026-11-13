from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    start: int
    end: int
    page: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _boundary(text: str, pos: int, window: int = 200) -> int:
    """Move pos back to the nearest paragraph or sentence break within the window."""
    if pos >= len(text):
        return len(text)
    lo = max(0, pos - window)
    segment = text[lo:pos]
    for sep in ("\n\n", "\n", ". ", "; "):
        idx = segment.rfind(sep)
        if idx != -1:
            return lo + idx + len(sep)
    return pos


def chunk_text(doc_id: str, text: str, size: int = 1200, overlap: int = 200, page: int | None = None) -> list[Chunk]:
    if size <= overlap:
        raise ValueError("chunk size must be larger than overlap")
    chunks: list[Chunk] = []
    start = 0
    n = len(text)
    while start < n:
        end = _boundary(text, min(start + size, n))
        if end <= start:
            end = min(start + size, n)
        piece = text[start:end]
        if piece.strip():
            chunks.append(Chunk(f"{doc_id}::{len(chunks)}", doc_id, piece, start, end, page))
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunk_pages(doc_id: str, pages: list[str], size: int = 1200, overlap: int = 200) -> tuple[str, list[Chunk]]:
    """Chunk a paged document, keeping offsets relative to the joined text."""
    full_text = ""
    chunks: list[Chunk] = []
    for page_no, page_text in enumerate(pages, start=1):
        offset = len(full_text)
        for c in chunk_text(doc_id, page_text, size, overlap, page=page_no):
            chunks.append(Chunk(f"{doc_id}::{len(chunks)}", doc_id, c.text, c.start + offset, c.end + offset, page_no))
        full_text += page_text + "\n\n"
    return full_text, chunks
