"""Diagnose column/row band detection on a page."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import fitz

from ssi_extractor.stages.page_text import (
    _column_bounds,
    _native_word_lines,
    _row_bands,
)

document = fitz.open(sys.argv[1])
page = document.load_page(0)
fragments = _native_word_lines(page)
print("fragments(groups):", len(fragments), "words:", sum(len(g) for g in fragments))
print("page width:", page.rect.width)
columns = _column_bounds(fragments)
bands = _row_bands(fragments)
print("columns:", len(columns), [f"{a:.0f}-{b:.0f}" for a, b in columns][:20])
print("bands:", len(bands), [f"{a:.0f}-{b:.0f}" for a, b in bands][:20])
widths = sorted((max(w.bbox.x1 for w in g) - min(w.bbox.x0 for w in g)) for g in fragments)
print("fragment width percentiles:", [f"{widths[int(len(widths)*q)]:.0f}" for q in (0.1,0.5,0.9,0.99)])
document.close()
