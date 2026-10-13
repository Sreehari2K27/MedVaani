"""Ingest Lantus package insert and produce chunked output."""

from __future__ import annotations

import json
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.mi.ingestion.pdf_extractor import extract_lantus_pi
from app.mi.ingestion.chunker import DocumentChunker


def main() -> None:
    pdf_path = Path("data/reference/Lantus_PI.pdf.pdf")
    pages = extract_lantus_pi(pdf_path)
    chunker = DocumentChunker()
    chunks = chunker.chunk_pages(pages, source_filename=pdf_path.name)
    
    output_path = Path("data/processed/lantus_chunks.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunker.to_dicts(chunks), f, indent=2, ensure_ascii=False)
    
    print(f"Pages processed: {len(pages)}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
