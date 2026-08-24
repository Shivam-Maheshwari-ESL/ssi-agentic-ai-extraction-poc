"""Probe: harvest candidates, build the deterministic descriptor and its strict schema."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ssi_extractor.schema.harvest import build_deterministic_descriptor, harvest_candidates
from ssi_extractor.schema.model_builder import build_record_model
from ssi_extractor.schema.strict_schema import to_strict_schema
from ssi_extractor.stages.classify import classify_document
from ssi_extractor.stages.locate_chunk import locate_and_chunk
from ssi_extractor.stages.ocr import OcrEngine
from ssi_extractor.stages.page_text import build_document_text

engine = OcrEngine()
for pdf in sorted(Path("input_pdf").glob("*.pdf")):
    print(f"\n================ {pdf.name} ================")
    composition = classify_document(pdf)
    text = build_document_text(composition, engine=engine)
    located = locate_and_chunk(text)
    candidates = harvest_candidates(located)
    descriptor = build_deterministic_descriptor(located, candidates, document_id=pdf.stem)
    print(f"chunks={len(located.chunks)} candidates={len(candidates)} fields={len(descriptor.fields)}")
    print(f"repeating_unit: {descriptor.repeating_unit.layout_pattern.value} anchors={[k.value for k in descriptor.repeating_unit.anchor_kinds]}")
    for field in descriptor.fields:
        print(f"  {field.path:52s} {field.kind.value:16s} conf={field.kind_confidence:<6} card={field.cardinality.value} pages={field.pages}")
    model = build_record_model(descriptor)
    schema = to_strict_schema(model)
    print("strict schema keys:", list(schema.get("properties", {}).keys()))
    instance = model()
    dumped = instance.model_dump(by_alias=True)
    print("empty record sample:", json.dumps(dumped, indent=None)[:300])
