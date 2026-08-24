"""G4: the audit log must be append-only, hash-chained, and free of raw PII."""

from __future__ import annotations

import json

from ssi_extractor.config.settings import Settings
from ssi_extractor.guardrails.g4_audit_log import (
    GENESIS_HASH,
    AuditEvent,
    AuditLog,
    verify_chain,
)
from ssi_extractor.observability.redaction import hash_value

ACCOUNT = "0654716099"


def test_entries_are_chained_and_verifiable(settings: Settings) -> None:
    log = AuditLog("doc-1", settings=settings)
    log.set_schema_descriptor_hash("descriptor-abc")

    first = log.record(
        AuditEvent.DOCUMENT_ACCEPTED, stage="G1", outcome="ACCEPTED", detail={"pages": 3}
    )
    second = log.record(
        AuditEvent.FIELD_EXTRACTED,
        stage="5",
        outcome="EXTRACTED",
        field_path="parties.party_1.account_identification",
        page=(2, 3),
        value=ACCOUNT,
        confidence=0.93,
        model_id="gpt-5.4-mini",
        prompt_version="extraction-v1",
    )

    assert first is not None and second is not None
    assert first.prev_hash == GENESIS_HASH
    assert second.prev_hash == first.entry_hash
    assert second.schema_descriptor_hash == "descriptor-abc"
    assert second.page == (2, 3)

    intact, reason = verify_chain(log.read_all())
    assert intact, reason


def test_values_are_hashed_not_stored(settings: Settings) -> None:
    log = AuditLog("doc-2", settings=settings)
    log.record(
        AuditEvent.FIELD_VALIDATED,
        stage="7",
        outcome="VALIDATED",
        field_path="party_1.account",
        value=ACCOUNT,
    )

    contents = log.path.read_text(encoding="utf-8")
    assert ACCOUNT not in contents

    entry = json.loads(contents.splitlines()[0])
    assert entry["value_hash"] == hash_value(ACCOUNT)
    assert entry["value_shape"] == "len=10 classes=digit"


def test_tampering_is_detected(settings: Settings) -> None:
    log = AuditLog("doc-3", settings=settings)
    log.record(AuditEvent.DOCUMENT_ACCEPTED, stage="G1", outcome="ACCEPTED")
    log.record(AuditEvent.DOCUMENT_COMPLETED, stage="9", outcome="COMPLETED", confidence=0.88)

    entries = log.read_all()
    intact, _ = verify_chain(entries)
    assert intact

    tampered = list(entries)
    tampered[0] = tampered[0].model_copy(update={"outcome": "REJECTED"})
    intact, reason = verify_chain(tampered)
    assert not intact
    assert reason is not None and "modified" in reason


def test_resumed_log_continues_the_sequence(settings: Settings) -> None:
    first_run = AuditLog("doc-4", settings=settings)
    first_run.record(AuditEvent.DOCUMENT_ACCEPTED, stage="G1", outcome="ACCEPTED")
    first_run.record(AuditEvent.CHUNK_CREATED, stage="4", outcome="CREATED", page=1)

    resumed = AuditLog("doc-4", settings=settings)
    entry = resumed.record(AuditEvent.DOCUMENT_COMPLETED, stage="9", outcome="COMPLETED")

    assert entry is not None
    assert entry.sequence == 2
    intact, reason = verify_chain(resumed.read_all())
    assert intact, reason


def test_disabled_audit_writes_nothing(settings: Settings) -> None:
    disabled = settings.model_copy(
        update={"audit": settings.audit.model_copy(update={"enabled": False})}
    )
    log = AuditLog("doc-5", settings=disabled)

    assert log.record(AuditEvent.DOCUMENT_ACCEPTED, stage="G1", outcome="ACCEPTED") is None
    assert not log.path.exists()
