"""Ad-hoc probe: run G1 + Stage 1 over the sample PDFs and print the composition."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ssi_extractor.guardrails.g1_input_gate import run_input_gate
from ssi_extractor.stages.classify import classify_document

for pdf in sorted(Path("input_pdf").glob("*.pdf")):
    gate = run_input_gate(pdf)
    print(f"\n=== {pdf.name} ===")
    print(f"G1: {gate.outcome.value} pages={gate.page_count} id={gate.document_id}")
    if not gate.accepted:
        continue
    composition = classify_document(pdf)
    print("composition:", composition.summary())
    for page in composition.pages:
        print(
            f"  page {page.page}: {page.page_class.value} "
            f"native_chars={page.native_chars} layers={len(page.layers)} "
            f"images={len(page.image_layers)} image_area={page.image_area_ratio}"
        )
        for layer in page.native_layers[:3]:
            print(f"      text[{layer.char_count}] {layer.text[:90]!r}")
