"""Test retrieval for Lantus."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.mi.retrieval import LantusRetriever


def main() -> None:
    retriever = LantusRetriever()
    query = sys.argv[1] if len(sys.argv) > 1 else "What is Lantus used for?"
    results = retriever.retrieve(query, k=3)
    print(f"Query: {query}")
    print(f"Results: {len(results)}")
    for i, r in enumerate(results, 1):
        print(f"\n{i}. chunk_id={r.chunk_id} page={r.page_number} score={r.relevance_score:.4f}")
        if r.section_heading:
            # Clean encoding issues
            heading = r.section_heading.encode('ascii', errors='ignore').decode('ascii')
            print(f"   heading: {heading}")
        text_preview = r.text[:120].encode('ascii', errors='ignore').decode('ascii')
        print(f"   text_preview: {text_preview}...")


if __name__ == "__main__":
    main()
