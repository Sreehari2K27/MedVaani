"""Focused unit tests for PDF ingestion pipeline."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import os

import pytest

from app.mi.ingestion.pdf_extractor import PDFExtractor, PDFExtractionError, extract_lantus_pi
from app.mi.ingestion.chunker import DocumentChunker, Chunk


def test_pdf_extraction_finds_document():
    """Test PDF extraction can find and read the Lantus PI."""
    pdf_path = Path("data/reference/Lantus_PI.pdf.pdf")
    assert pdf_path.exists(), f"PDF should exist at {pdf_path}"


def test_pdf_extraction_page_numbers_preserved():
    """Test page numbers are preserved."""
    pages = extract_lantus_pi()
    assert len(pages) > 0
    # Check page numbers are sequential starting from 1
    for i, page in enumerate(pages, start=1):
        assert page["page_number"] == i


def test_pdf_extraction_non_empty_content():
    """Test extracted content is non-empty."""
    pages = extract_lantus_pi()
    assert len(pages) > 0
    total_chars = sum(p.get("char_count", 0) for p in pages)
    assert total_chars > 0
    # First page should have content
    assert len(pages[0]["text"]) > 0


def test_chunk_creation():
    """Test chunks are created from pages."""
    pages = extract_lantus_pi()
    chunker = DocumentChunker()
    chunks = chunker.chunk_pages(pages)
    assert len(chunks) > 0
    assert isinstance(chunks[0], Chunk)


def test_chunk_metadata_present():
    """Test chunk metadata is complete."""
    pages = extract_lantus_pi()
    chunker = DocumentChunker()
    chunks = chunker.chunk_pages(pages)
    chunk = chunks[0]
    assert chunk.chunk_id
    assert chunk.page_number >= 1
    assert chunk.source_filename
    assert chunk.text
    assert chunk.metadata is not None
    assert "source_filename" in chunk.metadata
    assert "page_number" in chunk.metadata
    assert "chunk_id" in chunk.metadata
    assert "section_heading" in chunk.metadata


def test_deterministic_chunking():
    """Test chunking is deterministic."""
    pages = extract_lantus_pi()
    chunker = DocumentChunker()
    chunks1 = chunker.chunk_pages(pages)
    chunks2 = chunker.chunk_pages(pages)
    assert len(chunks1) == len(chunks2)
    # Chunk IDs should be deterministic
    for c1, c2 in zip(chunks1, chunks2):
        assert c1.chunk_id == c2.chunk_id


def test_invalid_pdf_handling():
    """Test handling of invalid/missing PDF cleanly."""
    extractor = PDFExtractor()
    with pytest.raises(PDFExtractionError):
        extractor.extract_pages("/non/existent/file.pdf")
