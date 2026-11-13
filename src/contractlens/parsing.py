from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class DocumentParser(ABC):
    @abstractmethod
    def parse(self, path: str | Path) -> list[str]:
        """Return the text of each page."""


class LocalPdfParser(DocumentParser):
    """PyMuPDF text extraction with optional Tesseract OCR for scanned pages."""

    def __init__(self, ocr: bool = False, min_chars: int = 40):
        self.ocr = ocr
        self.min_chars = min_chars

    def parse(self, path: str | Path) -> list[str]:
        import pymupdf

        pages = []
        with pymupdf.open(str(path)) as doc:
            for page in doc:
                text = page.get_text("text")
                if self.ocr and len(text.strip()) < self.min_chars:
                    text = page.get_textpage_ocr(full=True).extractText()
                pages.append(text)
        return pages


class DocumentAIParser(DocumentParser):
    """Google Cloud Document AI parser. Requires a billing-enabled project and processor."""

    def __init__(self, project_id: str, location: str, processor_id: str):
        self.name = f"projects/{project_id}/locations/{location}/processors/{processor_id}"
        self.location = location

    def parse(self, path: str | Path) -> list[str]:
        from google.api_core.client_options import ClientOptions
        from google.cloud import documentai

        client = documentai.DocumentProcessorServiceClient(
            client_options=ClientOptions(api_endpoint=f"{self.location}-documentai.googleapis.com")
        )
        raw = documentai.RawDocument(content=Path(path).read_bytes(), mime_type="application/pdf")
        document = client.process_document(
            request=documentai.ProcessRequest(name=self.name, raw_document=raw)
        ).document

        def segment_text(layout) -> str:
            return "".join(
                document.text[int(s.start_index): int(s.end_index)] for s in layout.text_anchor.text_segments
            )

        return [segment_text(page.layout) for page in document.pages]


def get_parser(kind: str = "local", **kwargs) -> DocumentParser:
    if kind == "local":
        return LocalPdfParser(**kwargs)
    if kind == "documentai":
        return DocumentAIParser(**kwargs)
    raise ValueError(f"unknown parser: {kind}")
