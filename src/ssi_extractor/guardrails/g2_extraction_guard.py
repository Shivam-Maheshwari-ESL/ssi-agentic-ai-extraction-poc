"""G2 — extraction guard.

Runs after the extraction agent and before anything downstream trusts its output.
Four independent checks, each with a distinct failure mode:

* **Schema** — enforced by validating against the runtime-built model. Handled in
  the agent's ``parse`` (reject and retry, never coerce); the verdict is recorded
  here so the audit trail shows it happened.
* **Hallucination** — every populated value must appear in its own source chunk.
  A value with no match in the text it supposedly came from is the single most
  dangerous failure mode in this pipeline, because it looks exactly like a good
  extraction.
* **Duplicate / conflict** — two instructions that identify the same market but
  state different values cannot both be right; a human decides.
* **Reference drift** — when a large share of a document's identifiers fail their
  format or registry checks, the likely cause is the wrong document type or
  systemic OCR corruption, not a series of unlucky fields. That is a
  document-level signal and is reported as one.
"""

from __future__ import annotations

from collections import defaultdict
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ssi_extractor.config.settings import Settings, get_settings
from ssi_extractor.guardrails.g4_audit_log import AuditEvent, AuditLog
from ssi_extractor.observability.logging import get_logger
from ssi_extractor.schema.descriptor import FieldKind, SchemaDescriptor
from ssi_extractor.schema.leaf import ExtractedField, FieldStatus
from ssi_extractor.stages.locate_chunk import InstructionChunk
from ssi_extractor.utils.text import substring_present
from ssi_extractor.validators.formats import check_bic, check_iban, check_isin, check_lei

__all__ = ["GuardFinding", "GuardIssue", "GuardReport", "run_extraction_guard"]

_logger = get_logger(__name__)

# Above this share of failing identifiers, the problem is the document, not the field.
_DRIFT_THRESHOLD = 0.5

# A document needs at least this many identifier fields before a drift ratio means
# anything; two failures out of three is noise, not a trend.
_DRIFT_MIN_SAMPLE = 6

_IDENTIFIER_CHECKS = {
    FieldKind.BIC: check_bic,
    FieldKind.IBAN: check_iban,
    FieldKind.ISIN: check_isin,
    FieldKind.LEI: check_lei,
}


class GuardIssue(StrEnum):
    """What G2 found."""

    SCHEMA_REJECTED = "SCHEMA_REJECTED"
    HALLUCINATED_VALUE = "HALLUCINATED_VALUE"
    EVIDENCE_NOT_IN_CHUNK = "EVIDENCE_NOT_IN_CHUNK"
    DUPLICATE_INSTRUCTION = "DUPLICATE_INSTRUCTION"
    CONFLICTING_INSTRUCTION = "CONFLICTING_INSTRUCTION"
    REFERENCE_DRIFT = "REFERENCE_DRIFT"
    MISSING_PAGE_CITATION = "MISSING_PAGE_CITATION"


class GuardFinding(BaseModel):
    """One issue, addressed to a specific field or instruction."""

    model_config = ConfigDict(frozen=True)

    issue: GuardIssue
    detail: str
    record_index: int | None = None
    field_path: str | None = None
    value_present: bool = False

    @property
    def blocks_field(self) -> bool:
        """Whether this finding must stop the field being reported as VALIDATED."""
        return self.issue in (
            GuardIssue.HALLUCINATED_VALUE,
            GuardIssue.SCHEMA_REJECTED,
        )


class GuardReport(BaseModel):
    """Everything G2 concluded about one document."""

    model_config = ConfigDict(frozen=True)

    findings: tuple[GuardFinding, ...] = ()
    identifiers_checked: int = 0
    identifiers_failing: int = 0
    drift_verdict: str = "INDETERMINATE"

    @property
    def ok(self) -> bool:
        return not any(finding.blocks_field for finding in self.findings)

    def for_field(self, record_index: int, field_path: str) -> tuple[GuardFinding, ...]:
        return tuple(
            finding
            for finding in self.findings
            if finding.record_index == record_index and finding.field_path == field_path
        )

    def summary(self) -> str:
        if not self.findings:
            return "G2: no issues"
        counts: dict[str, int] = defaultdict(int)
        for finding in self.findings:
            counts[finding.issue.value] += 1
        return "G2: " + ", ".join(f"{issue} x{count}" for issue, count in sorted(counts.items()))


def _iter_leaves(payload: Any, prefix: str = "") -> list[tuple[str, ExtractedField]]:
    """Walk a record payload and yield every leaf with its dotted path.

    The shape is discovered per document, so traversal is structural: anything
    carrying the five leaf keys is a leaf, anything else with children is a group.
    """
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
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            leaves.extend(_iter_leaves(value, child_prefix))
        return leaves

    if isinstance(payload, list):
        for index, item in enumerate(payload):
            leaves.extend(_iter_leaves(item, f"{prefix}[{index}]"))
    return leaves


def _identity_of(
    leaves: list[tuple[str, ExtractedField]], descriptor: SchemaDescriptor
) -> tuple[str, ...]:
    """Build an instruction's identity from its anchor-kind values.

    Identity comes from the descriptor's anchor kinds (typically the market or
    currency), not from a named column, so duplicate detection works on a document
    whose market column is called something else entirely.
    """
    anchors = descriptor.repeating_unit.anchor_kinds or (FieldKind.COUNTRY, FieldKind.CURRENCY)
    anchor_paths = {field.path for field in descriptor.fields if field.kind in anchors}
    values = [
        field.value.strip().upper()
        for path, field in leaves
        if path in anchor_paths and field.value.strip()
    ]
    return tuple(sorted(values))


def run_extraction_guard(
    records: list[Any],
    chunks: list[InstructionChunk],
    descriptor: SchemaDescriptor,
    *,
    settings: Settings | None = None,
    audit: AuditLog | None = None,
    schema_failures: dict[int, str] | None = None,
) -> GuardReport:
    """Check extracted records against their source chunks and against each other."""
    settings = settings or get_settings()
    findings: list[GuardFinding] = []
    identifiers_checked = 0
    identifiers_failing = 0

    kinds_by_path = {field.path: field.kind for field in descriptor.fields}
    by_identity: dict[tuple[str, ...], list[tuple[int, dict[str, str]]]] = defaultdict(list)

    for record_index, record in enumerate(records):
        if record is None:
            message = (schema_failures or {}).get(record_index, "extraction produced no record")
            findings.append(
                GuardFinding(
                    issue=GuardIssue.SCHEMA_REJECTED,
                    detail=message,
                    record_index=record_index,
                )
            )
            continue

        chunk = chunks[record_index] if record_index < len(chunks) else None
        source_text = chunk.text if chunk is not None else ""
        leaves = _iter_leaves(record)
        values_for_identity: dict[str, str] = {}

        for path, leaf in leaves:
            if not leaf.value.strip():
                continue
            values_for_identity[path] = leaf.value.strip()

            # Hallucination check: the value must be present in the text it claims
            # to come from. Tolerant of OCR noise and layout separators, but not of
            # a value the chunk never contained.
            if source_text and not substring_present(leaf.value, source_text):
                findings.append(
                    GuardFinding(
                        issue=GuardIssue.HALLUCINATED_VALUE,
                        detail=(
                            f"value has no match in its source chunk (page {chunk.page_label if chunk else '?'})"
                        ),
                        record_index=record_index,
                        field_path=path,
                        value_present=True,
                    )
                )

            if leaf.evidence.strip() and source_text and not substring_present(
                leaf.evidence, source_text
            ):
                findings.append(
                    GuardFinding(
                        issue=GuardIssue.EVIDENCE_NOT_IN_CHUNK,
                        detail="quoted evidence does not appear in the source chunk",
                        record_index=record_index,
                        field_path=path,
                    )
                )

            if not leaf.page and leaf.status is FieldStatus.VALIDATED:
                findings.append(
                    GuardFinding(
                        issue=GuardIssue.MISSING_PAGE_CITATION,
                        detail="validated value carries no page citation",
                        record_index=record_index,
                        field_path=path,
                    )
                )

            check = _IDENTIFIER_CHECKS.get(kinds_by_path.get(path, FieldKind.UNKNOWN))
            if check is not None:
                identifiers_checked += 1
                if not check(leaf.value):
                    identifiers_failing += 1

        by_identity[_identity_of(leaves, descriptor)].append((record_index, values_for_identity))

    # Duplicate and conflict detection across instructions sharing an identity.
    for identity, entries in by_identity.items():
        if not identity or len(entries) < 2:
            continue
        first_index, first_values = entries[0]
        for other_index, other_values in entries[1:]:
            shared = set(first_values) & set(other_values)
            differing = [path for path in shared if first_values[path] != other_values[path]]
            issue = (
                GuardIssue.CONFLICTING_INSTRUCTION if differing else GuardIssue.DUPLICATE_INSTRUCTION
            )
            detail = (
                f"instruction {other_index + 1} repeats identity {'/'.join(identity)} "
                f"from instruction {first_index + 1}"
            )
            if differing:
                detail += f"; differing field(s): {', '.join(sorted(differing)[:5])}"
            findings.append(
                GuardFinding(issue=issue, detail=detail, record_index=other_index)
            )

    drift_verdict = "INDETERMINATE"
    if identifiers_checked >= _DRIFT_MIN_SAMPLE:
        ratio = identifiers_failing / identifiers_checked
        if ratio >= _DRIFT_THRESHOLD:
            drift_verdict = "DRIFT"
            findings.append(
                GuardFinding(
                    issue=GuardIssue.REFERENCE_DRIFT,
                    detail=(
                        f"{identifiers_failing}/{identifiers_checked} identifiers failed their "
                        "format checks; the document type or the scan quality is the likely cause, "
                        "not individual fields"
                    ),
                )
            )
        else:
            drift_verdict = "OK"

    report = GuardReport(
        findings=tuple(findings),
        identifiers_checked=identifiers_checked,
        identifiers_failing=identifiers_failing,
        drift_verdict=drift_verdict,
    )

    if audit is not None:
        audit.record(
            AuditEvent.EXTRACTION_GUARD_VERDICT,
            stage="G2",
            outcome="PASS" if report.ok else "ISSUES",
            detail={
                "findings": len(report.findings),
                "identifiers_checked": identifiers_checked,
                "identifiers_failing": identifiers_failing,
                "drift": drift_verdict,
            },
        )

    _logger.info("%s (drift=%s)", report.summary(), drift_verdict)
    return report
