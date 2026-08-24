"""Stage 9 — assembler.

Builds the output JSON from the scored fields and the discovered schema. The shape
follows the descriptor, so it is the document's own structure rather than a fixed
contract; the parts that are always present are the document metadata and the
five-key leaf.

``documentAnalysis`` reports **region-level** composition, not a page-level
native/scanned split: a page with one scanned stamp over a digital table is
reported as mixed, and the count of text-over-image regions is stated explicitly.

Two files are written beside each other: the record JSON, and a schema sidecar
describing the discovered fields. Because the shape varies per document, a
consumer needs the sidecar to interpret an unfamiliar output mechanically.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from ssi_extractor.config.settings import Settings, get_settings
from ssi_extractor.observability.logging import get_logger
from ssi_extractor.schema.descriptor import SchemaDescriptor
from ssi_extractor.schema.leaf import FieldStatus
from ssi_extractor.stages.classify import DocumentComposition
from ssi_extractor.stages.confidence import ScoredField
from ssi_extractor.stages.locate_chunk import InstructionChunk, LocateResult

__all__ = ["AssembledDocument", "assemble_document", "write_outputs"]

_logger = get_logger(__name__)


class AssembledDocument(BaseModel):
    """The assembled output plus the sidecar and a short run summary."""

    model_config = ConfigDict(frozen=True)

    payload: dict[str, Any]
    schema_sidecar: dict[str, Any]
    output_path: Path | None = None
    sidecar_path: Path | None = None
    masked_path: Path | None = None

    @property
    def instruction_count(self) -> int:
        return int(self.payload.get("instructionCount", 0))


def _nest(path: str, leaf_payload: dict[str, Any], target: dict[str, Any]) -> None:
    """Place a leaf at its dotted path, creating groups as needed."""
    parts = path.split(".")
    node = target
    for part in parts[:-1]:
        existing = node.get(part)
        if not isinstance(existing, dict):
            existing = {}
            node[part] = existing
        node = existing
    node[parts[-1]] = leaf_payload


def _row_analysis(
    chunk: InstructionChunk | None,
    fields: list[ScoredField],
    index: int,
    total: int,
) -> str:
    """A one-line, human-readable account of how this instruction was produced."""
    populated = [field for field in fields if field.leaf.value.strip()]
    failed = [field for field in fields if field.leaf.status is FieldStatus.FAILED]
    flagged = [field for field in fields if field.needs_review]

    parts = [f"Instruction {index + 1} of {total}"]
    if chunk is not None:
        parts.append(f"Layout: {chunk.layout_pattern.value}")
        parts.append(f"Page(s): {chunk.page_label}")
        if chunk.stitched_across_pages:
            parts.append("Reassembled across a page break")
        if chunk.is_amendment:
            parts.append("Amendment: unstated fields are NOT_APPLICABLE")
        parts.append(
            "Source: native text" if chunk.ocr_confidence >= 0.999 else f"Source: OCR (mean confidence {chunk.ocr_confidence:.2f})"
        )
    parts.append(f"Fields populated: {len(populated)}")
    if failed:
        parts.append(f"Failed: {len(failed)}")
    if flagged:
        parts.append(f"Flagged for review: {len(flagged)}")
    return " | ".join(parts)


def _document_analysis(
    composition: DocumentComposition,
    located: LocateResult,
    descriptor: SchemaDescriptor,
) -> str:
    parts = [
        composition.summary(),
        f"Instructions: {len(located.chunks)}",
        f"Layout: {located.dominant_pattern.value}",
        f"Discovered fields: {len(descriptor.fields)}",
        f"Schema source: {descriptor.source.value}",
    ]
    if located.is_amendment_document:
        parts.append("Amendment document")
    if located.skipped_regions:
        parts.append(f"Non-SSI regions skipped: {len(located.skipped_regions)}")
    return " | ".join(parts)


def assemble_document(
    *,
    document_name: str,
    composition: DocumentComposition,
    located: LocateResult,
    descriptor: SchemaDescriptor,
    scored_fields: list[ScoredField],
    review_count: int = 0,
) -> AssembledDocument:
    """Build the output payload and its schema sidecar."""
    by_record: dict[int, list[ScoredField]] = {}
    for field in scored_fields:
        by_record.setdefault(field.record_index, []).append(field)

    total = len(located.chunks)
    records: list[dict[str, Any]] = []

    for index in range(total):
        fields = by_record.get(index, [])
        chunk = located.chunks[index] if index < len(located.chunks) else None
        record: dict[str, Any] = {"rowAnalysis": _row_analysis(chunk, fields, index, total)}
        for field in sorted(fields, key=lambda item: item.path):
            _nest(
                field.path,
                {
                    "value": field.leaf.value,
                    "status": field.leaf.status.value,
                    "confidence": field.leaf.confidence,
                    "evidence": field.leaf.evidence,
                    "page": list(field.leaf.page),
                },
                record,
            )
        records.append(record)

    statuses = {status.value: 0 for status in FieldStatus}
    for field in scored_fields:
        statuses[field.leaf.status.value] += 1

    payload: dict[str, Any] = {
        "documentName": document_name,
        "status": "COMPLETED" if records else "NO_INSTRUCTIONS_FOUND",
        "documentAnalysis": _document_analysis(composition, located, descriptor),
        "pageCount": composition.page_count,
        "nativeTextPages": composition.native_text_pages,
        "scannedPages": composition.scanned_pages,
        "mixedPages": composition.mixed_pages,
        "instructionCount": len(records),
        "settlementInstructionRecords": records,
        "extractionSummary": {
            "fieldsValidated": statuses[FieldStatus.VALIDATED.value],
            "fieldsFailed": statuses[FieldStatus.FAILED.value],
            "fieldsNotApplicable": statuses[FieldStatus.NOT_APPLICABLE.value],
            "fieldsQueuedForReview": review_count,
            "schemaDescriptorHash": descriptor.descriptor_hash,
        },
    }

    sidecar = {
        "documentName": document_name,
        "descriptorHash": descriptor.descriptor_hash,
        "source": descriptor.source.value,
        "repeatingUnit": descriptor.repeating_unit.model_dump(mode="json"),
        "notes": list(descriptor.notes),
        "fields": [
            {
                "path": field.path,
                "label": field.label,
                "group": list(field.group_path),
                "kind": field.kind.value,
                "kindConfidence": field.kind_confidence,
                "cardinality": field.cardinality.value,
                "isPii": field.is_pii,
                "hints": field.hints.model_dump(mode="json"),
                "discoveredBy": field.source_pattern,
                "pages": list(field.pages),
            }
            for field in descriptor.fields
        ],
        "skippedRegions": list(located.skipped_regions),
    }

    return AssembledDocument(payload=payload, schema_sidecar=sidecar)


def write_outputs(
    assembled: AssembledDocument,
    *,
    input_stem: str,
    masked_payload: dict[str, Any] | None = None,
    settings: Settings | None = None,
) -> AssembledDocument:
    """Write the record JSON, the schema sidecar and (optionally) the masked export."""
    settings = settings or get_settings()
    directory = settings.paths.output_dir
    directory.mkdir(parents=True, exist_ok=True)

    output_path = directory / f"{input_stem}.json"
    sidecar_path = directory / f"{input_stem}.schema.json"
    output_path.write_text(
        json.dumps(assembled.payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    sidecar_path.write_text(
        json.dumps(assembled.schema_sidecar, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    masked_path: Path | None = None
    if masked_payload is not None:
        masked_path = directory / f"{input_stem}.masked.json"
        masked_path.write_text(
            json.dumps(masked_payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    _logger.info(
        "Stage 9 wrote %s (%s instruction(s))%s",
        output_path.name,
        assembled.instruction_count,
        f" and {masked_path.name}" if masked_path else "",
    )
    return assembled.model_copy(
        update={
            "output_path": output_path,
            "sidecar_path": sidecar_path,
            "masked_path": masked_path,
        }
    )
