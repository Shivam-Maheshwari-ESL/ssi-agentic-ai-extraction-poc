"""PII redaction for logs, traces and audit entries — applied by construction.

The pipeline's primary JSON keeps raw values (it is the auditable source of
truth), but nothing that reaches a log, a trace or an audit entry may contain a
raw account number, name, BIC or key. Rather than trusting every call site to
remember that, redaction is installed as a logging filter over the *rendered*
message, so a new ``logger.info(f"... {account}")`` written months from now is
still covered.

Redaction is one-way: a detected value is replaced by ``<KIND:hash>`` where the
hash is a truncated SHA-256 of the value. The same value always yields the same
token, so a log remains correlatable without being reversible.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping
from typing import Final

__all__ = [
    "REDACTION_PATTERNS",
    "hash_value",
    "redact",
    "redact_structure",
    "register_literal_secret",
]

_HASH_SALT: Final = b"ssi-extractor-log-redaction-v1"
_DEFAULT_HASH_LENGTH: Final = 12

# Ordered longest/most-specific first: IBAN before BIC before bare digit runs, so
# an IBAN is not partially consumed by the digit-run rule.
REDACTION_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]{2,}\b")),
    # ISO 9362: 4-letter institution + 2-letter country + 2-char location, plus an
    # optional 3-char branch — 8 or 11 characters, never 9 or 10.
    ("BIC", re.compile(r"\b[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b")),
    ("LEI", re.compile(r"\b[A-Z0-9]{18}\d{2}\b")),
    ("PHONE", re.compile(r"(?<![\w.])\+\d[\d\s().-]{7,}\d(?![\w.])")),
    ("ACCOUNT", re.compile(r"(?<![\w.])\d[\d\s-]{5,}\d(?![\w.])")),
)

_literal_secrets: set[str] = set()


def register_literal_secret(secret: str | None) -> None:
    """Register an exact string (an API key, a password) for unconditional redaction.

    Short strings are ignored: redacting a 3-character "secret" would blank out
    unrelated log text.
    """
    if secret and len(secret) >= 8:
        _literal_secrets.add(secret)


def hash_value(value: str, *, length: int = _DEFAULT_HASH_LENGTH) -> str:
    """Return a stable, salted, truncated SHA-256 digest of ``value``."""
    digest = hashlib.sha256(_HASH_SALT + value.encode("utf-8")).hexdigest()
    return digest[:length]


def _token(kind: str, value: str, length: int) -> str:
    return f"<{kind}:{hash_value(value, length=length)}>"


def redact(text: str, *, hash_length: int = _DEFAULT_HASH_LENGTH) -> str:
    """Replace every detected secret or PII-shaped substring in ``text`` with a token."""
    if not text:
        return text

    result = text
    for secret in sorted(_literal_secrets, key=len, reverse=True):
        if secret in result:
            result = result.replace(secret, _token("SECRET", secret, hash_length))

    for kind, pattern in REDACTION_PATTERNS:
        result = pattern.sub(
            lambda match, _kind=kind: _token(_kind, match.group(0), hash_length),
            result,
        )
    return result


def redact_structure(
    value: object,
    *,
    hash_length: int = _DEFAULT_HASH_LENGTH,
    _depth: int = 0,
) -> object:
    """Recursively redact strings inside mappings, sequences and scalars.

    Used for structured log payloads and trace attributes. Depth is bounded so a
    self-referential structure cannot hang the logger.
    """
    if _depth > 12:
        return "<REDACTION_DEPTH_EXCEEDED>"
    if isinstance(value, str):
        return redact(value, hash_length=hash_length)
    if isinstance(value, Mapping):
        return {
            key: redact_structure(item, hash_length=hash_length, _depth=_depth + 1)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)) or (
        isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray))
    ):
        return [
            redact_structure(item, hash_length=hash_length, _depth=_depth + 1) for item in value
        ]
    return value
