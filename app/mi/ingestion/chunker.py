"""Chunker for splitting extracted text into retrieval chunks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional
import hashlib


@dataclass
class Chunk:
    """A chunk of document content with metadata."""

    chunk_id: str
    text: str
    page_number: int
    source_filename: str
    section_heading: Optional[str] = None
    metadata: Dict[str, Any] | None = None


class DocumentChunker:
    """Split pages into meaningful chunks."""

    def __init__(self, max_chunk_size: int = 1000, overlap: int = 50) -> None:
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap

    def chunk_pages(self, pages: List[Dict[str, Any]], source_filename: str = "Lantus_PI.pdf.pdf") -> List[Chunk]:
        chunks: List[Chunk] = []
        for page in pages:
            page_num = page.get("page_number", 0)
            text = page.get("text", "")
            if not text.strip():
                continue  # skip empty pages

            # Simple sentence-aware-ish splitting - split on newlines/paragraphs first
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            if not paragraphs:
                paragraphs = [text.strip()]

            current_chunk_text = []
            for para in paragraphs:
                # If adding this paragraph exceeds max size and we already have content, flush
                test_text = " ".join(current_chunk_text + [para]) if current_chunk_text else para
                if len(test_text) > self.max_chunk_size and current_chunk_text:
                    chunk_text = " ".join(current_chunk_text)
                    chunk = self._create_chunk(chunk_text, page_num, source_filename)
                    chunks.append(chunk)
                    # Start new chunk with overlap from previous
                    if self.overlap > 0 and chunk_text:
                        overlap_text = chunk_text[-self.overlap:] if len(chunk_text) > self.overlap else chunk_text
                        current_chunk_text = [overlap_text, para]
                    else:
                        current_chunk_text = [para]
                else:
                    current_chunk_text.append(para)

            if current_chunk_text:
                chunk_text = " ".join(current_chunk_text)
                chunk = self._create_chunk(chunk_text, page_num, source_filename)
                chunks.append(chunk)

        return chunks

    def _create_chunk(self, text: str, page_num: int, source_filename: str) -> Chunk:
        # Generate deterministic chunk ID
        content_hash = hashlib.sha256(f"{source_filename}:{page_num}:{text}".encode("utf-8")).hexdigest()[:16]
        chunk_id = f"chunk_{page_num}_{content_hash}"
        # Try to extract heading (first line if short)
        lines = text.split("\n")
        section_heading = None
        if lines:
            first = lines[0].strip()
            if len(first) < 80 and first and not first.endswith("."):
                section_heading = first
        return Chunk(
            chunk_id=chunk_id,
            text=text,
            page_number=page_num,
            source_filename=source_filename,
            section_heading=section_heading,
            metadata={
                "source_filename": source_filename,
                "page_number": page_num,
                "chunk_id": chunk_id,
                "section_heading": section_heading,
            },
        )

    def to_dicts(self, chunks: List[Chunk]) -> List[Dict[str, Any]]:
        return [
            {
                "chunk_id": c.chunk_id,
                "text": c.text,
                "page_number": c.page_number,
                "source_filename": c.source_filename,
                "section_heading": c.section_heading,
                "metadata": c.metadata,
            }
            for c in chunks
        ]
