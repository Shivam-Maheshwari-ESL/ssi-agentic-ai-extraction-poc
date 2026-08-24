"""Value-shape kind inference — how the pipeline stays generic.

Field *kind* is inferred from the observed values, not from the label. That is
what lets the pipeline work on a document whose labels are missing, abbreviated,
or written in a language it has never seen: a value that passes the mod-97 IBAN
checksum is an IBAN whatever the column heading says.

Labels are used only as a weak tie-breaker for kinds that have no distinguishing
shape (a person's name and an organisation's name look alike), never to override
a checksum-confirmed kind.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from ssi_extractor.schema.descriptor import Cardinality, FieldKind, ValidatorHints
from ssi_extractor.validators.formats import (
    charset_profile,
    check_bic,
    check_cfi,
    check_country,
    check_country_name,
    check_currency,
    check_date,
    check_iban,
    check_isin,
    check_lei,
    check_percentage,
    check_sort_code,
)

__all__ = ["KindInference", "infer_kind"]

# Minimum share of non-empty samples that must satisfy a detector before its kind
# is accepted. Below this, a single stray value cannot rename the field.
_ACCEPTANCE_RATIO = 0.6

# Detectors in specific-to-general order. The first kind clearing the acceptance
# ratio wins, so an IBAN is never demoted to a bare digit string.
_DETECTORS: tuple[tuple[FieldKind, object], ...] = (
    # Written country names come first: they collide with the CFI (six letters) and
    # BIC (eleven characters) formats by length, so checking codes first would type
    # "FRANCE" as a securities classification and "NETHERLANDS" as a bank code.
    (FieldKind.COUNTRY, check_country_name),
    (FieldKind.IBAN, check_iban),
    (FieldKind.ISIN, check_isin),
    (FieldKind.LEI, check_lei),
    (FieldKind.BIC, check_bic),
    (FieldKind.CFI, check_cfi),
    (FieldKind.SORT_CODE, check_sort_code),
    (FieldKind.PERCENTAGE, check_percentage),
    (FieldKind.CURRENCY, check_currency),
    (FieldKind.COUNTRY, check_country),
    (FieldKind.DATE, check_date),
)

_EMAIL_RE = re.compile(r"^[\w.+-]+@[\w-]+\.[\w.-]{2,}$")
_PHONE_RE = re.compile(r"^\+?[\d\s().-]{7,}\d$")
_ACCOUNT_RE = re.compile(r"^[A-Z]{0,4}[\d][\d\s./-]{2,}[\dA-Z]$")
_DIGIT_RUN_RE = re.compile(r"^\d[\d\s.-]{3,}\d$")
_PARTICIPANT_ID_RE = re.compile(r"^[A-Z0-9]{2,6}[-/]?[A-Z0-9]{0,6}$")

_ORG_TOKENS = frozenset(
    {
        "ag", "bank", "banca", "banco", "banque", "bv", "co", "company", "corp",
        "custody", "depository", "gmbh", "group", "holdings", "inc", "limited",
        "llc", "ltd", "nv", "plc", "sa", "sarl", "securities", "services", "spa",
        "trust", "trustee",
    }
)

# Label tokens used only where shape cannot discriminate. Deliberately small:
# this is a tie-breaker, not a field dictionary, and the pipeline must not start
# depending on English labels.
_LABEL_HINTS: tuple[tuple[FieldKind, frozenset[str]], ...] = (
    (FieldKind.ORG_NAME, frozenset({"bank", "custodian", "agent", "institution", "counterparty"})),
    (FieldKind.PERSON_NAME, frozenset({"contact", "signatory", "signatory_name", "officer"})),
    (FieldKind.ADDRESS, frozenset({"address", "street", "city", "postcode", "zip", "domicile"})),
    (FieldKind.ACCOUNT_NUMBER, frozenset({"account", "acct", "ac", "cuenta", "konto", "compte"})),
    (FieldKind.PARTICIPANT_ID, frozenset({"participant", "member", "clearing", "psafe", "pset"})),
    (FieldKind.PHONE, frozenset({"phone", "tel", "telephone", "fax"})),
    (FieldKind.EMAIL, frozenset({"email", "mail"})),
)


class KindInference(BaseModel):
    """The inferred kind for a field, with the evidence that produced it."""

    model_config = ConfigDict(frozen=True)

    kind: FieldKind = FieldKind.UNKNOWN
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    cardinality: Cardinality = Cardinality.SINGLE
    hints: ValidatorHints = Field(default_factory=ValidatorHints)
    reasons: tuple[str, ...] = ()
    checked_values: int = 0


def _sample_confidence(matched: int, total: int) -> float:
    """Scale the pass ratio by how much evidence there was.

    One matching value is weak evidence; five are strong. Without this, a column
    with a single populated cell would claim the same confidence as a full one.
    """
    if total == 0:
        return 0.0
    ratio = matched / total
    evidence_factor = min(1.0, 0.55 + 0.15 * total)
    return round(min(1.0, ratio * evidence_factor), 4)


def _looks_like_org(value: str) -> bool:
    tokens = {token.strip(".,").lower() for token in value.split()}
    return bool(tokens & _ORG_TOKENS)


def _label_hint_kind(label: str | None) -> FieldKind | None:
    if not label:
        return None
    tokens = {token for token in re.split(r"[^a-z]+", label.lower()) if token}
    for kind, keywords in _LABEL_HINTS:
        if tokens & keywords:
            return kind
    return None


def _derive_hints(values: Sequence[str], kind: FieldKind) -> ValidatorHints:
    """Build deterministic constraints from what the document actually contains."""
    cleaned = [value.strip() for value in values if value.strip()]
    if not cleaned:
        return ValidatorHints()

    lengths = {len(value) for value in cleaned}
    charsets = {charset_profile(value) for value in cleaned}
    unique_values = sorted(set(cleaned))

    enum_values: tuple[str, ...] = ()
    if kind is FieldKind.ENUM or (
        kind in (FieldKind.FREE_TEXT, FieldKind.UNKNOWN)
        and len(unique_values) <= 5
        and len(cleaned) >= 3
    ):
        enum_values = tuple(unique_values)

    return ValidatorHints(
        exact_length=lengths.pop() if len(lengths) == 1 else None,
        min_length=min(len(value) for value in cleaned),
        max_length=max(len(value) for value in cleaned),
        enum_values=enum_values,
        charset=charsets.pop() if len(charsets) == 1 else "any",
    )


def _multi_valued(values: Sequence[str]) -> bool:
    """Detect a cell that holds several values (a market with three sub-accounts).

    Deliberately conservative: only newline or an explicit separator followed by
    another populated segment counts, so a name containing a slash is not split.
    """
    for value in values:
        if "\n" in value.strip():
            return True
        segments = [segment.strip() for segment in re.split(r"\s{2,}[/;,]\s*|\s*;\s*", value)]
        populated = [segment for segment in segments if len(segment) >= 4]
        if len(populated) > 1:
            return True
    return False


def infer_kind(values: Sequence[str], *, label: str | None = None) -> KindInference:
    """Infer a field's kind from its observed values.

    Args:
        values: sample values seen for this field across the document.
        label: the verbatim label, used only as a tie-breaker for shape-identical
            kinds. It never overrides a checksum-confirmed kind.
    """
    samples = [value.strip() for value in values if value and value.strip()]
    cardinality = Cardinality.MULTI if _multi_valued(samples) else Cardinality.SINGLE

    if not samples:
        hinted = _label_hint_kind(label)
        return KindInference(
            kind=hinted or FieldKind.UNKNOWN,
            confidence=0.2 if hinted else 0.0,
            cardinality=cardinality,
            reasons=("no sample values; label hint only",) if hinted else ("no sample values",),
        )

    # A multi-valued cell is judged on its segments, so "acct A / acct B" is still
    # recognised as an account field rather than as free text.
    judged: list[str] = []
    for value in samples:
        parts = [part.strip() for part in re.split(r"[\n;]|\s{2,}", value) if part.strip()]
        judged.extend(parts or [value])

    for kind, detector in _DETECTORS:
        matched = sum(1 for value in judged if detector(value))  # type: ignore[operator]
        if matched and matched / len(judged) >= _ACCEPTANCE_RATIO:
            confidence = _sample_confidence(matched, len(judged))
            return KindInference(
                kind=kind,
                confidence=confidence,
                cardinality=cardinality,
                hints=_derive_hints(judged, kind),
                reasons=(f"{matched}/{len(judged)} values satisfy the {kind.value} format check",),
                checked_values=len(judged),
            )

    matched = sum(1 for value in judged if _EMAIL_RE.match(value))
    if matched / len(judged) >= _ACCEPTANCE_RATIO:
        return KindInference(
            kind=FieldKind.EMAIL,
            confidence=_sample_confidence(matched, len(judged)),
            cardinality=cardinality,
            hints=_derive_hints(judged, FieldKind.EMAIL),
            reasons=(f"{matched}/{len(judged)} values match the EMAIL pattern",),
            checked_values=len(judged),
        )

    # A bare digit run is an account number far more often than a phone number in
    # settlement data, so PHONE requires an explicit international prefix or a
    # phone-shaped label. Without this, every 10-digit account reads as a phone.
    phone_matched = sum(1 for value in judged if _PHONE_RE.match(value))
    if phone_matched / len(judged) >= _ACCEPTANCE_RATIO and (
        all(value.strip().startswith("+") for value in judged)
        or _label_hint_kind(label) is FieldKind.PHONE
    ):
        return KindInference(
            kind=FieldKind.PHONE,
            confidence=_sample_confidence(phone_matched, len(judged)),
            cardinality=cardinality,
            hints=_derive_hints(judged, FieldKind.PHONE),
            reasons=(f"{phone_matched}/{len(judged)} values match the PHONE pattern",),
            checked_values=len(judged),
        )

    digit_like = sum(1 for value in judged if _DIGIT_RUN_RE.match(value) or _ACCOUNT_RE.match(value))
    if digit_like / len(judged) >= _ACCEPTANCE_RATIO:
        hinted = _label_hint_kind(label)
        kind = (
            hinted
            if hinted in (FieldKind.PARTICIPANT_ID, FieldKind.ACCOUNT_NUMBER)
            else FieldKind.ACCOUNT_NUMBER
        )
        return KindInference(
            kind=kind,
            confidence=_sample_confidence(digit_like, len(judged)),
            cardinality=cardinality,
            hints=_derive_hints(judged, kind),
            reasons=(f"{digit_like}/{len(judged)} values are identifier-shaped digit strings",),
            checked_values=len(judged),
        )

    unique = sorted({value for value in judged})
    if len(judged) >= 3 and len(unique) <= max(2, len(judged) // 3):
        return KindInference(
            kind=FieldKind.ENUM,
            confidence=_sample_confidence(len(judged), len(judged)) * 0.8,
            cardinality=cardinality,
            hints=_derive_hints(judged, FieldKind.ENUM),
            reasons=(f"only {len(unique)} distinct values across {len(judged)} samples",),
            checked_values=len(judged),
        )

    org_like = sum(1 for value in judged if _looks_like_org(value))
    if org_like / len(judged) >= _ACCEPTANCE_RATIO:
        return KindInference(
            kind=FieldKind.ORG_NAME,
            confidence=_sample_confidence(org_like, len(judged)) * 0.9,
            cardinality=cardinality,
            hints=_derive_hints(judged, FieldKind.ORG_NAME),
            reasons=(f"{org_like}/{len(judged)} values contain an organisation token",),
            checked_values=len(judged),
        )

    if short_id := sum(
        1 for value in judged if _PARTICIPANT_ID_RE.match(value) and 2 <= len(value) <= 12
    ):
        if short_id / len(judged) >= _ACCEPTANCE_RATIO and _label_hint_kind(
            label
        ) is FieldKind.PARTICIPANT_ID:
            return KindInference(
                kind=FieldKind.PARTICIPANT_ID,
                confidence=_sample_confidence(short_id, len(judged)) * 0.8,
                cardinality=cardinality,
                hints=_derive_hints(judged, FieldKind.PARTICIPANT_ID),
                reasons=("short alphanumeric codes with a participant-style label",),
                checked_values=len(judged),
            )

    hinted = _label_hint_kind(label)
    # A label hint may not contradict the values. "Account Name" hints at an
    # account, but a column of institution names is not an account number, and
    # accepting the hint would put a length-and-digits validator on free text.
    if hinted in (FieldKind.ACCOUNT_NUMBER, FieldKind.SORT_CODE, FieldKind.PARTICIPANT_ID):
        if not any(character.isdigit() for value in judged for character in value):
            hinted = FieldKind.ORG_NAME if org_like else None

    kind = hinted or FieldKind.FREE_TEXT
    return KindInference(
        kind=kind,
        confidence=0.35 if hinted else 0.25,
        cardinality=cardinality,
        hints=_derive_hints(judged, kind),
        reasons=(
            "no format check matched; label hint applied" if hinted else "no format check matched",
        ),
        checked_values=len(judged),
    )
