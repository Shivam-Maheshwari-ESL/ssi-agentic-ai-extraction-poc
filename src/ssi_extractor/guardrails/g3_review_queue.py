"""G3 — human review queue.

Anything ``FAILED`` or below the confidence floor lands here **with its evidence
and page citation attached**. The requirement it satisfies is specific: a
low-confidence field must not merely be recorded in the output JSON, where nobody
would notice it; it must arrive somewhere a person is going to look.

The queue is append-only. A revised assessment supersedes an earlier entry by
reference rather than overwriting it, so the history of what was flagged and why
survives — the same reasoning as the audit log.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from ssi_extractor.config.settings import Settings, get_settings
from ssi_extractor.guardrails.g4_audit_log import AuditEvent, AuditLog
from ssi_extractor.observability.logging import get_logger
from ssi_extractor.schema.descriptor import SchemaDescriptor
from ssi_extractor.stages.confidence import ScoredField
from ssi_extractor.stages.locate_chunk import InstructionChunk

__all__ = ["ReviewEntry", "ReviewQueue", "enqueue_for_review"]

_logger = get_logger(__name__)


class ReviewEntry(BaseModel):
    """One field awaiting human judgement."""

    model_config = ConfigDict(frozen=True)

    entry_id: str
    document_id: str
    document_name: str
    created_at: str
    record_index: int
    field_path: str
    field_label: str
    field_kind: str
    value: str
    status: str
    confidence: float
    confidence_breakdown: str
    evidence: str
    page: tuple[int, ...]
    layout_pattern: str = ""
    reasons: tuple[str, ...] = ()
    supersedes_entry_id: str | None = None


class ReviewQueue(BaseModel):
    """The queue written for one document."""

    model_config = ConfigDict(frozen=True)

    path: Path
    entries: tuple[ReviewEntry, ...] = ()

    @property
    def count(self) -> int:
        return len(self.entries)


def enqueue_for_review(
    scored_fields: list[ScoredField],
    chunks: list[InstructionChunk],
    descriptor: SchemaDescriptor,
    *,
    document_id: str,
    document_name: str,
    settings: Settings | None = None,
    audit: AuditLog | None = None,
) -> ReviewQueue:
    """Write every flagged field to the document's review queue."""
    settings = settings or get_settings()
    directory = settings.paths.review_queue_dir
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{document_id}.review.jsonl"

    descriptors_by_path = {field.path: field for field in descriptor.fields}
    timestamp = datetime.now(UTC).isoformat()
    entries: list[ReviewEntry] = []

    for scored in scored_fields:
        if not scored.needs_review:
            continue

        field_descriptor = descriptors_by_path.get(scored.path)
        chunk = chunks[scored.record_index] if scored.record_index < len(chunks) else None
        entries.append(
            ReviewEntry(
                entry_id=f"{document_id}:{scored.record_index}:{scored.path}",
                document_id=document_id,
                document_name=document_name,
                created_at=timestamp,
                record_index=scored.record_index,
                field_path=scored.path,
                field_label=field_descriptor.label if field_descriptor else scored.path,
                field_kind=field_descriptor.kind.value if field_descriptor else "UNKNOWN",
                value=scored.leaf.value,
                status=scored.leaf.status.value,
                confidence=scored.leaf.confidence,
                confidence_breakdown=scored.breakdown.explain(),
                evidence=scored.leaf.evidence,
                page=scored.leaf.page,
                layout_pattern=chunk.layout_pattern.value if chunk else "",
                reasons=scored.review_reasons,
            )
        )

    if entries:
        with path.open("a", encoding="utf-8") as handle:
            for entry in entries:
                handle.write(entry.model_dump_json() + "\n")

        if audit is not None:
            for entry in entries:
                audit.record(
                    AuditEvent.REVIEW_ENQUEUED,
                    stage="G3",
                    outcome="ENQUEUED",
                    field_path=entry.field_path,
                    page=entry.page,
                    value=entry.value,
                    confidence=entry.confidence,
                    detail={"reasons": list(entry.reasons)[:4]},
                )

    _logger.info(
        "G3 queued %s field(s) for human review%s",
        len(entries),
        f" -> {path.name}" if entries else "",
    )
    return ReviewQueue(path=path, entries=tuple(entries))
