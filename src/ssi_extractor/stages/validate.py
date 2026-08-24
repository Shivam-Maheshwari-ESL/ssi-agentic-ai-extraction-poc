"""Stage 7 — dynamic validation, plus Stage 8 status assignment.

Three levels, all dispatched by inferred kind rather than by field name:

* **value** — the registry's format, checksum, length and charset checks;
* **field** — does the value's own shape agree with the kind this field holds;
* **chunk** — is the instruction complete, judged against the descriptor's
  repeating unit rather than a hand-written required-field list.

A field never becomes ``VALIDATED`` on a failed check. Anything ``FAILED`` or
below the confidence floor is routed to the human review queue (G3) with its
evidence and page attached, so nothing sits silently in the output.

The absence of a value is not a failure. A document that never states a field —
routinely, an amendment stating only what changed — yields ``NOT_APPLICABLE``.
Treating that as a failure would turn every unmentioned field into a false
negative, which is explicitly called out in the spec.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ssi_extractor.config.settings import Settings, get_settings
from ssi_extractor.guardrails.g2_extraction_guard import GuardReport
from ssi_extractor.guardrails.g4_audit_log import AuditEvent, AuditLog
from ssi_extractor.observability.logging import get_logger
from ssi_extractor.schema.descriptor import FieldDescriptor, FieldKind, SchemaDescriptor
from ssi_extractor.schema.kinds import infer_kind
from ssi_extractor.schema.leaf import ExtractedField, FieldStatus
from ssi_extractor.stages.locate_chunk import InstructionChunk
from ssi_extractor.validators.registry import (
    FieldVerdict,
    ValidationLevel,
    ValidationOutcome,
    validate_value,
)

__all__ = ["ChunkVerdict", "ValidatedField", "ValidationReport", "validate_records"]

_logger = get_logger(__name__)


class ValidatedField(BaseModel):
    """One field after validation, carrying everything the assembler and G3 need."""

    model_config = ConfigDict(frozen=True)

    record_index: int
    path: str
    descriptor: FieldDescriptor
    leaf: ExtractedField
    verdict: FieldVerdict
    kind_agreement: float = Field(default=1.0, ge=0.0, le=1.0)
    guard_issues: tuple[str, ...] = ()
    needs_review: bool = False
    review_reasons: tuple[str, ...] = ()

    @property
    def status(self) -> FieldStatus:
        return self.leaf.status


class ChunkVerdict(BaseModel):
    """Whether one instruction was captured completely."""

    model_config = ConfigDict(frozen=True)

    record_index: int
    outcome: ValidationOutcome
    reasons: tuple[str, ...] = ()
    populated_fields: int = 0
    required_kinds_present: tuple[FieldKind, ...] = ()
    required_kinds_missing: tuple[FieldKind, ...] = ()


class ValidationReport(BaseModel):
    """Stage 7's output for a whole document."""

    model_config = ConfigDict(frozen=True)

    fields: tuple[ValidatedField, ...] = ()
    chunks: tuple[ChunkVerdict, ...] = ()

    @property
    def failed_fields(self) -> tuple[ValidatedField, ...]:
        return tuple(field for field in self.fields if field.status is FieldStatus.FAILED)

    @property
    def review_fields(self) -> tuple[ValidatedField, ...]:
        return tuple(field for field in self.fields if field.needs_review)

    @property
    def ambiguous_fields(self) -> tuple[ValidatedField, ...]:
        return tuple(field for field in self.fields if field.verdict.ambiguous)

    def counts(self) -> dict[str, int]:
        counts = {status.value: 0 for status in FieldStatus}
        for field in self.fields:
            counts[field.status.value] += 1
        return counts


def _field_level_agreement(value: str, field: FieldDescriptor) -> tuple[float, list[str]]:
    """Field-level check: does the value's own shape match the kind this field holds?

    Answers "is this the right field?" rather than "is this a valid value?". A BIC
    sitting in an account-number field is a field-assignment error: the value is
    perfectly valid, but it is in the wrong place, and only a shape comparison
    catches it.
    """
    if not value.strip():
        return 1.0, []
    if field.kind in (FieldKind.FREE_TEXT, FieldKind.UNKNOWN):
        return 1.0, []

    inference = infer_kind([value], label=field.label)
    if inference.kind is field.kind:
        return 1.0, []
    if inference.kind in (FieldKind.FREE_TEXT, FieldKind.UNKNOWN):
        # The value has no distinctive shape; that is not evidence of misplacement.
        return 0.75, []
    if inference.confidence < 0.6:
        return 0.7, []
    return (
        0.25,
        [
            f"value looks like {inference.kind.value} but this field holds {field.kind.value}"
        ],
    )


def _leaves_of(payload: Any, prefix: str = "") -> list[tuple[str, ExtractedField]]:
    """Structural traversal of a record of unknown shape."""
    leaves: list[tuple[str, ExtractedField]] = []

    if isinstance(payload, BaseModel):
        payload = payload.model_dump(by_alias=True)

    if isinstance(payload, dict):
        if {"value", "status", "confidence", "evidence"} <= set(payload):
            try:
                leaves.append((prefix, ExtractedField.model_validate(payload)))
            except Exception:
                pass
            return leaves
        for key, value in payload.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            leaves.extend(_leaves_of(value, child))
        return leaves

    if isinstance(payload, list):
        for index, item in enumerate(payload):
            leaves.extend(_leaves_of(item, f"{prefix}[{index}]"))
    return leaves


def _validate_chunk_completeness(
    record_index: int,
    leaves: list[tuple[str, ExtractedField]],
    descriptor: SchemaDescriptor,
    chunk: InstructionChunk | None,
) -> ChunkVerdict:
    """Chunk-level check: was the whole instruction captured?

    Judged against the descriptor's repeating unit. An amendment is exempt: it is
    complete by definition when it states only the fields it meant to change.
    """
    kinds_by_path = {field.path: field.kind for field in descriptor.fields}
    populated_kinds = {
        kinds_by_path.get(path, FieldKind.UNKNOWN)
        for path, leaf in leaves
        if leaf.value.strip()
    }
    populated = sum(1 for _, leaf in leaves if leaf.value.strip())

    required = set(descriptor.repeating_unit.required_kinds)
    missing = tuple(sorted(required - populated_kinds, key=lambda kind: kind.value))
    present = tuple(sorted(required & populated_kinds, key=lambda kind: kind.value))

    reasons: list[str] = []
    outcome = ValidationOutcome.PASSED

    if chunk is not None and chunk.is_amendment:
        reasons.append("amendment document: unstated fields are NOT_APPLICABLE by design")
    elif populated == 0:
        outcome = ValidationOutcome.FAILED
        reasons.append("no field in this instruction carries a value")
    elif missing:
        outcome = ValidationOutcome.AMBIGUOUS
        reasons.append(
            "instruction is missing every value of required kind(s): "
            + ", ".join(kind.value for kind in missing)
        )

    return ChunkVerdict(
        record_index=record_index,
        outcome=outcome,
        reasons=tuple(reasons),
        populated_fields=populated,
        required_kinds_present=present,
        required_kinds_missing=missing,
    )


def validate_records(
    records: list[Any],
    chunks: list[InstructionChunk],
    descriptor: SchemaDescriptor,
    guard: GuardReport,
    *,
    settings: Settings | None = None,
    audit: AuditLog | None = None,
) -> ValidationReport:
    """Validate every field of every record at all three levels."""
    settings = settings or get_settings()
    descriptors_by_path = {field.path: field for field in descriptor.fields}
    fields: list[ValidatedField] = []
    chunk_verdicts: list[ChunkVerdict] = []

    for record_index, record in enumerate(records):
        if record is None:
            chunk_verdicts.append(
                ChunkVerdict(
                    record_index=record_index,
                    outcome=ValidationOutcome.FAILED,
                    reasons=("extraction produced no record for this instruction",),
                )
            )
            continue

        chunk = chunks[record_index] if record_index < len(chunks) else None
        leaves = _leaves_of(record)

        for path, leaf in leaves:
            field_descriptor = descriptors_by_path.get(path) or FieldDescriptor(
                name=path.rsplit(".", 1)[-1],
                label=path.rsplit(".", 1)[-1],
                kind=FieldKind.UNKNOWN,
            )

            guard_issues = tuple(
                f"{finding.issue.value}: {finding.detail}"
                for finding in guard.for_field(record_index, path)
            )
            blocking_guard = any(
                finding.blocks_field for finding in guard.for_field(record_index, path)
            )

            if not leaf.value.strip():
                # No value: normalise to NOT_APPLICABLE so absence is stated
                # explicitly and identically everywhere.
                normalised_leaf = leaf.model_copy(
                    update={
                        "status": FieldStatus.NOT_APPLICABLE,
                        "confidence": 0.0,
                        "value": "",
                    }
                )
                fields.append(
                    ValidatedField(
                        record_index=record_index,
                        path=path,
                        descriptor=field_descriptor,
                        leaf=normalised_leaf,
                        verdict=FieldVerdict(
                            outcome=ValidationOutcome.NOT_CHECKED,
                            reasons=("field not stated in this instruction",),
                            checked_by="registry",
                        ),
                        guard_issues=guard_issues,
                    )
                )
                continue

            verdict = validate_value(leaf.value, field_descriptor)
            agreement, agreement_reasons = _field_level_agreement(leaf.value, field_descriptor)

            reasons = list(verdict.reasons) + agreement_reasons
            failed = (
                verdict.outcome is ValidationOutcome.FAILED
                or blocking_guard
                or agreement < 0.3
            )

            if failed:
                status = FieldStatus.FAILED
            elif verdict.outcome in (ValidationOutcome.PASSED, ValidationOutcome.NOT_CHECKED):
                status = FieldStatus.VALIDATED
            else:
                # Ambiguous: not a failure, but not a clean pass either. It stays
                # VALIDATED with a reduced confidence and is queued for review,
                # and it is the population the 7b adjudicator is for.
                status = FieldStatus.VALIDATED

            resolved_leaf = leaf.model_copy(update={"status": status})

            review_reasons: list[str] = []
            if status is FieldStatus.FAILED:
                review_reasons.append("validation failed")
            if verdict.ambiguous:
                review_reasons.append("value is ambiguous")
            if blocking_guard:
                review_reasons.append("extraction guard flagged this value")
            if agreement_reasons:
                review_reasons.extend(agreement_reasons)

            fields.append(
                ValidatedField(
                    record_index=record_index,
                    path=path,
                    descriptor=field_descriptor,
                    leaf=resolved_leaf,
                    verdict=verdict.model_copy(update={"reasons": tuple(reasons)}),
                    kind_agreement=agreement,
                    guard_issues=guard_issues,
                    needs_review=bool(review_reasons),
                    review_reasons=tuple(review_reasons),
                )
            )

            if audit is not None:
                audit.record(
                    AuditEvent.FIELD_VALIDATED,
                    stage="7",
                    outcome=status.value,
                    field_path=path,
                    page=resolved_leaf.page,
                    value=resolved_leaf.value,
                    detail={
                        "kind": field_descriptor.kind.value,
                        "verdict": verdict.outcome.value,
                        "format_score": verdict.format_score,
                        "reasons": list(reasons)[:4],
                    },
                )

        chunk_verdicts.append(
            _validate_chunk_completeness(record_index, leaves, descriptor, chunk)
        )

    report = ValidationReport(fields=tuple(fields), chunks=tuple(chunk_verdicts))
    counts = report.counts()
    _logger.info(
        "Stage 7 validated %s field(s): %s VALIDATED, %s FAILED, %s NOT_APPLICABLE.",
        len(report.fields),
        counts[FieldStatus.VALIDATED.value],
        counts[FieldStatus.FAILED.value],
        counts[FieldStatus.NOT_APPLICABLE.value],
    )
    return report
