"""Inspect reconstructed grids and chunk cutting."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ssi_extractor.stages.classify import classify_document
from ssi_extractor.stages.locate_chunk import locate_and_chunk
from ssi_extractor.stages.ocr import OcrEngine
from ssi_extractor.stages.page_text import build_document_text

target = Path(sys.argv[1])
composition = classify_document(target)
text = build_document_text(composition, engine=OcrEngine())
for page in text.pages:
    print(f"page {page.page}: lines={len(page.lines)} tables={len(page.tables)}")
    for grid in page.tables:
        print(f"  cols={grid.column_count} headers={len(grid.header_rows)} body={len(grid.rows)}")
        for row in grid.header_rows:
            print("   H:", [c[:22] for c in row])
        for row in grid.rows[:4]:
            print("   R:", [c.replace(chr(10), '|')[:22] for c in row])
located = locate_and_chunk(text)
print(f"\npattern={located.dominant_pattern.value} chunks={len(located.chunks)}")
for chunk in located.chunks[:2]:
    print(f"--- chunk {chunk.index} pages={chunk.page_label} score={chunk.relevance_score:.2f}")
    print(chunk.text[:700])
print("skipped:", len(located.skipped_regions))
for note in located.skipped_regions[:4]:
    print("   ", note)
