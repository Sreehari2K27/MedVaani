"""PDF text extraction for package insert."""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional


class PDFExtractionError(Exception):
    pass


class PDFExtractor:
    """Extract text from PDF pages preserving page numbers."""

    def __init__(self) -> None:
        pass

    def extract_pages(self, pdf_path: Path | str) -> List[Dict[str, Any]]:
        """Extract text page by page."""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise PDFExtractionError(f"PDF not found: {pdf_path}")

        pages: List[Dict[str, Any]] = []
        try:
            import pdfplumber
        except ImportError:
            raise PDFExtractionError("pdfplumber not installed")

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""
                    # Clean obvious artifacts
                    text = self._clean_text(text)
                    pages.append({
                        "page_number": i,
                        "text": text,
                        "char_count": len(text),
                    })
        except Exception as e:
            raise PDFExtractionError(f"Failed to extract PDF: {e}")

        return pages

    def _clean_text(self, text: str) -> str:
        if not text:
            return text
        # Remove excessive whitespace/newlines artifacts
        lines = [line.strip() for line in text.splitlines()]
        # Collapse multiple blank lines
        cleaned_lines = []
        for line in lines:
            if line or (cleaned_lines and cleaned_lines[-1]):
                cleaned_lines.append(line)
        return "\n".join(cleaned_lines)


def extract_lantus_pi(pdf_path: Path | str | None = None) -> List[Dict[str, Any]]:
    if pdf_path is None:
        pdf_path = Path("data/reference/Lantus_PI.pdf.pdf")
    extractor = PDFExtractor()
    return extractor.extract_pages(pdf_path)
