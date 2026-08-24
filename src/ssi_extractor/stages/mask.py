"""Stage 6 — PII masking.

Runs after extraction and before anything is exported, logged or traced. The order
matters: the extraction agent must see raw values, because a tokenised account
number cannot be copied character-exactly, so masking cannot come earlier.

The primary JSON stays unmasked — it is the auditable source of truth. A separate
masked copy is produced for ``--masked-export``, for logs and for traces.

Masking is **format-preserving and reversible within a run**: a 10-digit account
number becomes a different 10-digit string, so a masked export still exercises
downstream length and charset checks. The token↔value map lives only in memory and
is purged when the document finishes, which is why the export is safe to share
while the mapping never reaches disk.

Recognisers are registered **per field kind** from the discovered schema, so
masking follows whatever fields the document turned out to have rather than a
fixed list of column names. Presidio is used for the kinds that need language
models (names, addresses); the deterministic kinds are masked by their own rules,
which is both faster and exact.
"""

from __future__ import annotations

import hashlib
import string
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ssi_extractor.config.settings import Settings, get_settings
from ssi_extractor.observability.logging import get_logger
from ssi_extractor.schema.descriptor import FieldKind, SchemaDescriptor

__all__ = ["MaskingResult", "PiiVault", "mask_payload"]

_logger = get_logger(__name__)

# Kinds masked deterministically, preserving each character's class so the masked
# value still passes the same length and charset checks as the original.
_FORMAT_PRESERVED_KINDS = frozenset(
    {
        FieldKind.BIC,
        FieldKind.IBAN,
        FieldKind.ACCOUNT_NUMBER,
        FieldKind.SORT_CODE,
        FieldKind.PARTICIPANT_ID,
        FieldKind.LEI,
        FieldKind.PHONE,
    }
)

# Kinds replaced with a labelled token: their length carries no meaning, and a
# character-substituted name would read as a real (wrong) name.
_TOKENISED_KINDS = frozenset(
    {
        FieldKind.PERSON_NAME,
        FieldKind.ORG_NAME,
        FieldKind.ADDRESS,
        FieldKind.EMAIL,
    }
)

_SALT = b"ssi-extractor-masking-v1"


class PiiVault:
    """In-memory, per-document token↔value map.

    Deliberately not persisted. Reversibility is needed *within* a run — to
    reconcile a masked export against the source of truth, or to re-expand a value
    for a reviewer — and persisting the map would recreate the exposure the masking
    exists to prevent.
    """

    def __init__(self) -> None:
        self._to_token: dict[str, str] = {}
        self._to_value: dict[str, str] = {}

    def __len__(self) -> int:
        return len(self._to_token)

    def token_for(self, value: str, generator: Any) -> str:
        """Return this value's stable token, creating it on first sight."""
        existing = self._to_token.get(value)
        if existing is not None:
            return existing
        token = generator(value)
        # Two different values must never share a token, or a masked export would
        # merge two accounts into one.
        attempt = 0
        while token in self._to_value and self._to_value[token] != value:
            attempt += 1
            token = generator(f"{value}#{attempt}")
        self._to_token[value] = token
        self._to_value[token] = value
        return token

    def original(self, token: str) -> str | None:
        return self._to_value.get(token)

    def purge(self) -> None:
        """Erase the mapping. Called when a document finishes processing."""
        self._to_token.clear()
        self._to_value.clear()


class MaskingResult(BaseModel):
    """The masked payload and what was masked in it."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    payload: dict[str, Any]
    masked_field_count: int = 0
    masked_kinds: tuple[str, ...] = ()
    presidio_available: bool = False
    vault_size: int = 0
    notes: tuple[str, ...] = Field(default_factory=tuple)


def _digits_from(seed: str, count: int) -> str:
    """Deterministic pseudo-random digits derived from the value."""
    digest = hashlib.sha256(_SALT + seed.encode("utf-8")).hexdigest()
    return "".join(str(int(character, 16) % 10) for character in digest)[:count] or "0"


def _letters_from(seed: str, count: int) -> str:
    digest = hashlib.sha256(_SALT + b"alpha" + seed.encode("utf-8")).digest()
    alphabet = string.ascii_uppercase
    return "".join(alphabet[byte % 26] for byte in digest)[:count] or "X"


def _format_preserving(value: str) -> str:
    """Replace characters class-for-class, keeping separators and length.

    Digits stay digits and letters stay letters, so ``GB12BARC20199003809072``
    masks to another well-formed 22-character string. The point is that a masked
    export can still be validated for length and charset without revealing
    anything.
    """
    digits = _digits_from(value, sum(1 for character in value if character.isdigit()) or 1)
    letters = _letters_from(value, sum(1 for character in value if character.isalpha()) or 1)
    digit_index = letter_index = 0
    output: list[str] = []

    for character in value:
        if character.isdigit():
            output.append(digits[digit_index % len(digits)])
            digit_index += 1
        elif character.isalpha():
            replacement = letters[letter_index % len(letters)]
            output.append(replacement if character.isupper() else replacement.lower())
            letter_index += 1
        else:
            output.append(character)
    return "".join(output)


def _tokenised(kind: FieldKind, value: str) -> str:
    return f"<{kind.value}_{hashlib.sha256(_SALT + value.encode('utf-8')).hexdigest()[:10]}>"


def _presidio_anonymiser() -> Any | None:
    """Build a Presidio engine pair, or ``None`` when unavailable.

    Absence is handled explicitly rather than silently: without the spaCy model,
    the kind-driven rules still mask every deterministic identifier, and the
    shortfall (free-text names and addresses) is reported in the result notes and
    the log instead of passing PII through unnoticed.
    """
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine
    except ImportError:
        return None

    # Presidio downloads a missing spaCy model on construction, which on a
    # restricted network blocks for 15 seconds per attempt and can stall the
    # pipeline for minutes. Availability is therefore checked locally first: no
    # model present means no Presidio, decided instantly and offline.
    try:
        import spacy.util

        if not any(
            spacy.util.is_package(name)
            for name in ("en_core_web_lg", "en_core_web_md", "en_core_web_sm")
        ):
            _logger.warning(
                "No spaCy model is installed; using kind-driven masking only. "
                "Install one with: python -m spacy download en_core_web_lg"
            )
            return None
    except ImportError:
        return None

    try:
        return AnalyzerEngine(), AnonymizerEngine()
    except Exception as exc:  # a missing spaCy model surfaces here
        _logger.warning(
            "Presidio could not initialise (%s); falling back to kind-driven masking only.",
            type(exc).__name__,
        )
        return None


def _mask_free_text(text: str, engines: Any, vault: PiiVault) -> str:
    """Mask names, addresses and similar spans inside free text via Presidio."""
    analyzer, anonymizer = engines
    try:
        results = analyzer.analyze(text=text, language="en")
        if not results:
            return text
        anonymised = anonymizer.anonymize(text=text, analyzer_results=results)
        masked = anonymised.text
    except Exception as exc:
        _logger.warning("Presidio failed on a value (%s); tokenising it instead.", type(exc).__name__)
        return vault.token_for(text, lambda value: _tokenised(FieldKind.FREE_TEXT, value))
    return masked


def mask_payload(
    payload: dict[str, Any],
    descriptor: SchemaDescriptor,
    *,
    vault: PiiVault | None = None,
    settings: Settings | None = None,
) -> MaskingResult:
    """Produce a masked copy of an assembled payload.

    The input is not modified: the caller keeps the unmasked payload as the source
    of truth and writes the returned copy wherever the data is shared.
    """
    settings = settings or get_settings()
    vault = vault or PiiVault()
    kinds_by_path = {field.path: field.kind for field in descriptor.fields}
    engines = _presidio_anonymiser()
    notes: list[str] = []
    if engines is None:
        notes.append(
            "Presidio unavailable: identifiers were masked by kind, but free-text "
            "names and addresses were tokenised wholesale rather than span-masked."
        )

    masked_count = 0
    masked_kinds: set[str] = set()

    def mask_leaf(path: str, leaf: dict[str, Any]) -> dict[str, Any]:
        nonlocal masked_count
        value = str(leaf.get("value", ""))
        if not value.strip():
            return leaf

        kind = kinds_by_path.get(path, FieldKind.UNKNOWN)
        if not kind.is_pii:
            return leaf

        if kind in _FORMAT_PRESERVED_KINDS:
            masked_value = vault.token_for(value, _format_preserving)
        elif kind in _TOKENISED_KINDS:
            masked_value = vault.token_for(value, lambda item: _tokenised(kind, item))
        elif engines is not None:
            masked_value = _mask_free_text(value, engines, vault)
        else:
            masked_value = vault.token_for(value, lambda item: _tokenised(kind, item))

        masked_count += 1
        masked_kinds.add(kind.value)

        # Evidence quotes the source text, so it carries the same PII as the value
        # and must be masked with it.
        evidence = str(leaf.get("evidence", ""))
        masked_evidence = evidence.replace(value, masked_value) if evidence else evidence
        return {**leaf, "value": masked_value, "evidence": masked_evidence}

    def walk(node: Any, prefix: str = "") -> Any:
        if isinstance(node, dict):
            if {"value", "status", "confidence", "evidence"} <= set(node):
                return mask_leaf(prefix, node)
            return {
                key: walk(value, f"{prefix}.{key}" if prefix else str(key))
                for key, value in node.items()
            }
        if isinstance(node, list):
            return [walk(item, prefix) for item in node]
        return node

    masked_records = [
        walk(record) for record in payload.get("settlementInstructionRecords", [])
    ]
    masked_payload = {
        **payload,
        "settlementInstructionRecords": masked_records,
        "maskingPolicy": {
            "primaryJsonIsUnmasked": True,
            "maskedFieldCount": masked_count,
            "maskedKinds": sorted(masked_kinds),
            "formatPreservingKinds": sorted(kind.value for kind in _FORMAT_PRESERVED_KINDS),
            "tokenMapPersisted": False,
            "presidioAvailable": engines is not None,
        },
    }

    _logger.info(
        "Stage 6 masked %s value(s) across kind(s): %s.",
        masked_count,
        ", ".join(sorted(masked_kinds)) or "none",
    )
    return MaskingResult(
        payload=masked_payload,
        masked_field_count=masked_count,
        masked_kinds=tuple(sorted(masked_kinds)),
        presidio_available=engines is not None,
        vault_size=len(vault),
        notes=tuple(notes),
    )
