"""Ad-hoc probe: run G1 -> classify -> text -> locate/chunk and print chunk inventory."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ssi_extractor.stages.classify import classify_document
from ssi_extractor.stages.locate_chunk import locate_and_chunk
from ssi_extractor.stages.ocr import OcrEngine
from ssi_extractor.stages.page_text import build_document_text

engine = OcrEngine()
for pdf in sorted(Path("input_pdf").glob("*.pdf")):
    print(f"\n================ {pdf.name} ================")
    composition = classify_document(pdf)
    text = build_document_text(composition, engine=engine)
    for page in text.pages:
        print(f"page {page.page}: lines={len(page.lines)} tables={len(page.tables)}")
        for grid in page.tables:
            print(f"   grid cols={grid.column_count} header_rows={len(grid.header_rows)} body_rows={len(grid.rows)}")
            for header in grid.header_rows:
                print(f"     H: {header}")
    located = locate_and_chunk(text)
    print(f"pattern={located.dominant_pattern.value} chunks={len(located.chunks)} amendment_doc={located.is_amendment_document}")
    for note in located.skipped_regions[:6]:
        print("   skipped:", note)
    for chunk in located.chunks[:3]:
        print(f"--- chunk {chunk.index} pages={chunk.page_label} score={chunk.relevance_score:.2f} kinds={[k.value for k in chunk.evidence_kinds]}")
        print(chunk.text[:600])
    if len(located.chunks) > 3:
        print(f"... and {len(located.chunks) - 3} more chunks")
