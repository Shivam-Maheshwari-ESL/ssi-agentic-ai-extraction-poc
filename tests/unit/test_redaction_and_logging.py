"""Raw PII must not be able to reach a log, however the log call was written."""

from __future__ import annotations

import json
from pathlib import Path

from ssi_extractor.config.settings import Settings
from ssi_extractor.observability.logging import configure_logging, get_logger
from ssi_extractor.observability.redaction import (
    hash_value,
    redact,
    redact_structure,
    register_literal_secret,
)

ACCOUNT = "4773002145"
IBAN = "GB29NWBK60161331926819"
BIC = "INVLESMMXXX"
EMAIL = "settlements@examplebank.com"


def test_redact_replaces_pii_shapes_with_stable_tokens() -> None:
    text = f"account {ACCOUNT} iban {IBAN} bic {BIC} contact {EMAIL}"
    redacted = redact(text)

    for secret in (ACCOUNT, IBAN, BIC, EMAIL):
        assert secret not in redacted

    assert f"<ACCOUNT:{hash_value(ACCOUNT)}>" in redacted
    assert f"<IBAN:{hash_value(IBAN)}>" in redacted
    assert f"<BIC:{hash_value(BIC)}>" in redacted
    assert f"<EMAIL:{hash_value(EMAIL)}>" in redacted
    assert redact(text) == redacted, "tokens must be stable so logs stay correlatable"


def test_redact_is_not_reversible_but_is_distinguishing() -> None:
    assert hash_value(ACCOUNT) != hash_value(ACCOUNT[:-1] + "6")
    assert ACCOUNT not in hash_value(ACCOUNT)


def test_registered_literal_secret_is_redacted() -> None:
    secret = "sk-abcdef1234567890"
    register_literal_secret(secret)
    assert secret not in redact(f"authorization: {secret}")


def test_redact_structure_walks_nested_payloads() -> None:
    payload = {"party": {"accounts": [ACCOUNT, IBAN]}, "page": 3, "confidence": 0.91}
    result = redact_structure(payload)

    serialised = json.dumps(result)
    assert ACCOUNT not in serialised
    assert IBAN not in serialised
    assert result["page"] == 3
    assert result["confidence"] == 0.91


def test_logging_file_handler_never_writes_raw_pii(settings: Settings) -> None:
    configure_logging(settings)
    logger = get_logger("test.redaction")

    logger.info("extracted account %s for iban %s", ACCOUNT, IBAN)
    logger.info("structured", extra={"evidence": f"A/C {ACCOUNT}", "page": 1})

    log_path: Path = settings.paths.logs_dir / settings.logging.file_name
    contents = log_path.read_text(encoding="utf-8")

    assert ACCOUNT not in contents
    assert IBAN not in contents
    assert f"<ACCOUNT:{hash_value(ACCOUNT, length=settings.privacy.log_hash_length)}>" in contents

    lines = [json.loads(line) for line in contents.splitlines() if line.strip()]
    assert len(lines) == 2
    assert lines[1]["page"] == 1
    assert ACCOUNT not in lines[1]["evidence"]
