"""Ad-hoc probe: OCR the image-only sample and report confidence and text."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import fitz

from ssi_extractor.stages.classify import classify_document
from ssi_extractor.stages.ocr import OcrEngine, recognise_region

target = Path("input_pdf/vida_bank.pdf")
composition = classify_document(target)
engine = OcrEngine()
print("rapidocr available:", engine.available)

document = fitz.open(target)
started = time.perf_counter()
for page_composition in composition.pages:
    page = document.load_page(page_composition.page - 1)
    for layer in page_composition.image_layers:
        result = recognise_region(
            page, layer.bbox, page_number=page_composition.page, engine=engine
        )
        print(
            f"page {result.page}: words={len(result.words)} "
            f"mean_conf={result.mean_confidence} min_conf={result.min_confidence} "
            f"attempts={result.outcome.attempts} tier={result.outcome.tier_used.value} "
            f"orientation={result.outcome.orientation} cleared={result.outcome.cleared_threshold} "
            f"fallback_needed={result.outcome.needs_vision_fallback}"
        )
        print("ops:", result.outcome.operations)
        print("--- text ---")
        print(result.text)
document.close()
print(f"elapsed {time.perf_counter() - started:.1f}s")
