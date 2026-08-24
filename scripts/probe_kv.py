"""Probe key:value detection on the form-style sample."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ssi_extractor.stages.classify import classify_document
from ssi_extractor.stages.locate_chunk import _key_value_pairs
from ssi_extractor.stages.ocr import OcrEngine
from ssi_extractor.stages.page_text import build_document_text

composition = classify_document(Path("input_pdf/vida_bank.pdf"))
text = build_document_text(composition, engine=OcrEngine())
page = text.pages[0]
print("--- lines and cells ---")
for line in page.lines:
    print(f"  {line.text[:60]!r} cells={line.cells}")
print("--- pairs ---")
for pair in _key_value_pairs(page.lines):
    print(f"  section={pair.section!r} key={pair.key!r} value={pair.value!r}")
