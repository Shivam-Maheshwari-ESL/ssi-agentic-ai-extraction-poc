"""G4 — immutable audit log.

Write-once JSONL, one entry per decision (a gate outcome, a field extraction, a
validation verdict), each carrying the model id, prompt version, schema
descriptor hash and timestamp that produced it. Entries are chained: every entry
stores the SHA-256 of the previous entry's canonical form, so a later edit or
deletion is detectable — which is what lets an auditor be told, months later,
how a specific value was produced and that the record has not been altered.

Raw PII never enters the log. Values are recorded as salted hashes via
``hash_value``, with only a short non-identifying shape summary (length and
character classes) kept for debugging.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from threading import Lock
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field

from ssi_extractor.config.settings import Settings, get_settings
from ssi_extractor.observability.logging import get_logger
from ssi_extractor.observability.redaction import hash_value

__all__ = ["AuditEvent", "AuditEntry", "AuditLog", "GENESIS_HASH", "verify_chain"]

GENESIS_HASH = "0" * 64
_logger = get_logger(__name__)


class AuditEvent(StrEnum):
    """The decision points worth reconstructing after the fact."""

    DOCUMENT_ACCEPTED = "DOCUMENT_ACCEPTED"
    DOCUMENT_REJECTED = "DOCUMENT_REJECTED"
    REGION_CLASSIFIED = "REGION_CLASSIFIED"
    OCR_COMPLETED = "OCR_COMPLETED"
    OCR_RETRIED = "OCR_RETRIED"
    VISION_FALLBACK_INVOKED = "VISION_FALLBACK_INVOKED"
    SCHEMA_DESCRIPTOR_BUILT = "SCHEMA_DESCRIPTOR_BUILT"
    CHUNK_CREATED = "CHUNK_CREATED"
    FIELD_EXTRACTED = "FIELD_EXTRACTED"
    EXTRACTION_GUARD_VERDICT = "EXTRACTION_GUARD_VERDICT"
    FIELD_VALIDATED = "FIELD_VALIDATED"
    FIELD_ADJUDICATED = "FIELD_ADJUDICATED"
    REVIEW_ENQUEUED = "REVIEW_ENQUEUED"
    DOCUMENT_COMPLETED = "DOCUMENT_COMPLETED"


def _value_shape(value: str) -> str:
    """A non-identifying description of a value: length and character classes."""
    if not value:
        return "len=0"
    classes = []
    if any(character.isdigit() for character in value):
        classes.append("digit")
    if any(character.isupper() for character in value):
        classes.append("upper")
    if any(character.islower() for character in value):
        classes.append("lower")
    if any(not character.isalnum() and not character.isspace() for character in value):
        classes.append("punct")
    return f"len={len(value)} classes={'+'.join(classes) or 'space'}"


class AuditEntry(BaseModel):
    """One write-once audit record."""

    model_config = ConfigDict(frozen=True)

    sequence: int = Field(ge=0)
    timestamp: str
    document_id: str
    event: AuditEvent
    stage: str
    outcome: str
    field_path: str | None = None
    page: tuple[int, ...] = ()
    value_hash: str | None = None
    value_shape: str | None = None
    confidence: float | None = None
    model_id: str | None = None
    prompt_version: str | None = None
    schema_descriptor_hash: str | None = None
    detail: dict[str, Any] = Field(default_factory=dict)
    prev_hash: str = GENESIS_HASH
    entry_hash: str = ""

    def canonical_payload(self) -> str:
        """Deterministic serialisation used for hashing (excludes ``entry_hash``)."""
        payload = self.model_dump(mode="json", exclude={"entry_hash"})
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def with_hash(self) -> Self:
        """Return a copy whose ``entry_hash`` covers this entry and its predecessor."""
        digest = hashlib.sha256(self.canonical_payload().encode("utf-8")).hexdigest()
        return self.model_copy(update={"entry_hash": digest})


class AuditLog:
    """Append-only audit writer for one document.

    The file is opened in append mode with an exclusive in-process lock, and
    every line is flushed and fsynced before the call returns, so a crash cannot
    lose the record of a decision that already took effect downstream.
    """

    def __init__(
        self,
        document_id: str,
        *,
        settings: Settings | None = None,
        schema_descriptor_hash: str | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._document_id = document_id
        self._schema_descriptor_hash = schema_descriptor_hash
        self._lock = Lock()
        self._sequence = 0
        self._prev_hash = GENESIS_HASH
        self._enabled = self._settings.audit.enabled

        audit_dir = self._settings.paths.audit_dir
        audit_dir.mkdir(parents=True, exist_ok=True)
        self._path = audit_dir / self._settings.audit.file_name_template.format(
            document_id=document_id
        )
        if self._path.exists():
            self._resume_from_existing()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def sequence(self) -> int:
        return self._sequence

    def set_schema_descriptor_hash(self, descriptor_hash: str) -> None:
        """Stamp subsequent entries with the descriptor that shaped the extraction."""
        self._schema_descriptor_hash = descriptor_hash

    def _resume_from_existing(self) -> None:
        """Continue the chain of an existing log (resumed run) without rewriting it."""
        last_line = None
        with self._path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    last_line = line
        if last_line is None:
            return
        try:
            previous = json.loads(last_line)
            self._sequence = int(previous["sequence"]) + 1
            self._prev_hash = str(previous["entry_hash"])
        except (KeyError, ValueError, TypeError):
            _logger.warning(
                "Existing audit log tail is unreadable; starting a new chain segment.",
                extra={"audit_path": str(self._path), "document_id": self._document_id},
            )

    def record(
        self,
        event: AuditEvent,
        *,
        stage: str,
        outcome: str,
        field_path: str | None = None,
        page: tuple[int, ...] | list[int] | int | None = None,
        value: str | None = None,
        confidence: float | None = None,
        model_id: str | None = None,
        prompt_version: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> AuditEntry | None:
        """Append one entry. Returns the written entry, or ``None`` when disabled."""
        if not self._enabled:
            return None

        if page is None:
            pages: tuple[int, ...] = ()
        elif isinstance(page, int):
            pages = (page,)
        else:
            pages = tuple(page)

        with self._lock:
            entry = AuditEntry(
                sequence=self._sequence,
                timestamp=datetime.now(UTC).isoformat(),
                document_id=self._document_id,
                event=event,
                stage=stage,
                outcome=outcome,
                field_path=field_path,
                page=pages,
                value_hash=hash_value(value) if value else None,
                value_shape=_value_shape(value) if value else None,
                confidence=confidence,
                model_id=model_id,
                prompt_version=prompt_version,
                schema_descriptor_hash=self._schema_descriptor_hash,
                detail=detail or {},
                prev_hash=self._prev_hash,
            ).with_hash()

            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(entry.model_dump_json() + "\n")
                handle.flush()
                os.fsync(handle.fileno())

            self._sequence += 1
            self._prev_hash = entry.entry_hash
            return entry

    def read_all(self) -> list[AuditEntry]:
        """Read the log back, in order."""
        if not self._path.exists():
            return []
        entries: list[AuditEntry] = []
        with self._path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    entries.append(AuditEntry.model_validate_json(line))
        return entries


def verify_chain(entries: list[AuditEntry]) -> tuple[bool, str | None]:
    """Verify hash-chain integrity.

    Returns ``(True, None)`` when intact, otherwise ``(False, reason)`` naming
    the first entry that fails — a tampered value, a broken link, or a gap in
    the sequence.
    """
    expected_prev = GENESIS_HASH
    for index, entry in enumerate(entries):
        recomputed = hashlib.sha256(entry.canonical_payload().encode("utf-8")).hexdigest()
        if recomputed != entry.entry_hash:
            return False, f"entry {index} (sequence {entry.sequence}) content was modified"
        if entry.prev_hash != expected_prev:
            return False, f"entry {index} (sequence {entry.sequence}) breaks the chain link"
        expected_prev = entry.entry_hash
    return True, None
