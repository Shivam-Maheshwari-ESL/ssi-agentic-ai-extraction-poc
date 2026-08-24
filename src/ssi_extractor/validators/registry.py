"""Validator registry — dispatch by inferred field *kind*, never by field name.

This is what makes validation work on a schema nobody wrote in advance. A field
called ``psetPartyIdentifierBic``, ``PSET``, ``Depositario`` or ``column_2`` is
validated identically if its values are BICs, because the rule set is chosen from
the kind that was inferred from the values themselves.

Registered rules come in three levels, matching the spec:

* **value** — type, length, charset, checksum, ISO membership;
* **field** — does the value's shape agree with the kind the field is supposed to
  hold (i.e. is this the right field for this value);
* **chunk** — is the instruction complete, judged against the descriptor's
  repeating unit.

Digit integrity is enforced here in code. A model is never asked whether a
10-digit account number kept all ten digits.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from ssi_extractor.schema.descriptor import FieldDescriptor, FieldKind, ValidatorHints
from ssi_extractor.validators.formats import (
    CheckResult,
    charset_profile,
    check_bic,
    check_cfi,
    check_country,
    check_currency,
    check_date,
    check_iban,
    check_isin,
    check_lei,
    check_percentage,
    check_sort_code,
    normalise_identifier,
)

__all__ = [
    "FieldVerdict",
    "ValidationLevel",
    "ValidationOutcome",
    "validate_value",
]


class ValidationLevel(StrEnum):
    VALUE = "VALUE"
    FIELD = "FIELD"
    CHUNK = "CHUNK"


class ValidationOutcome(StrEnum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    NOT_CHECKED = "NOT_CHECKED"
    AMBIGUOUS = "AMBIGUOUS"


class FieldVerdict(BaseModel):
    """The result of validating one value, with reasons a human can act on."""

    model_config = ConfigDict(frozen=True)

    outcome: ValidationOutcome
    level: ValidationLevel = ValidationLevel.VALUE
    reasons: tuple[str, ...] = ()
    normalised_value: str | None = None
    format_score: float = Field(default=0.0, ge=0.0, le=1.0)
    checked_by: str = ""

    @property
    def passed(self) -> bool:
        return self.outcome is ValidationOutcome.PASSED

    @property
    def ambiguous(self) -> bool:
        return self.outcome is ValidationOutcome.AMBIGUOUS


# Kind -> checker. Checkers are the same functions kind inference used, so a value
# cannot be classified as an IBAN during discovery and then judged by a different
# rule during validation.
_CHECKERS: dict[FieldKind, Callable[[str], CheckResult]] = {
    FieldKind.BIC: lambda value: check_bic(value, use_registry=True),
    FieldKind.IBAN: check_iban,
    FieldKind.ISIN: check_isin,
    FieldKind.CFI: check_cfi,
    FieldKind.LEI: check_lei,
    FieldKind.SORT_CODE: check_sort_code,
    FieldKind.COUNTRY: check_country,
    FieldKind.CURRENCY: check_currency,
    FieldKind.DATE: check_date,
    FieldKind.PERCENTAGE: check_percentage,
}

# Kinds whose length and charset must be preserved exactly. For these, a hint
# mismatch is a hard failure rather than a warning: it is the guarantee that no
# digit or character was dropped between the page and the JSON.
_EXACT_KINDS = frozenset(
    {
        FieldKind.ACCOUNT_NUMBER,
        FieldKind.SORT_CODE,
        FieldKind.PARTICIPANT_ID,
        FieldKind.BIC,
        FieldKind.IBAN,
        FieldKind.ISIN,
        FieldKind.LEI,
        FieldKind.CFI,
    }
)


def _check_hints(value: str, kind: FieldKind, hints: ValidatorHints) -> list[str]:
    """Apply the constraints observed in the document itself."""
    problems: list[str] = []
    comparable = normalise_identifier(value) if kind.is_identifier else value.strip()

    if hints.exact_length is not None and len(comparable) != hints.exact_length:
        problems.append(
            f"length {len(comparable)} does not match the {hints.exact_length} "
            "characters this field uses elsewhere in the document"
        )
    if hints.min_length is not None and len(comparable) < hints.min_length:
        problems.append(f"length {len(comparable)} is below the observed minimum {hints.min_length}")
    if hints.max_length is not None and len(comparable) > hints.max_length:
        problems.append(f"length {len(comparable)} is above the observed maximum {hints.max_length}")
    if hints.enum_values and value.strip() not in hints.enum_values:
        allowed = ", ".join(hints.enum_values[:6])
        problems.append(f"value is not one of the values used in this document ({allowed})")
    if hints.charset and hints.charset not in ("any", "empty"):
        actual = charset_profile(value)
        if actual != hints.charset:
            problems.append(
                f"character composition '{actual}' differs from '{hints.charset}' used elsewhere"
            )
    return problems


def _account_number_checks(value: str) -> list[str]:
    """Digit-integrity checks for account-like identifiers.

    Deliberately narrow: an account number's *format* varies by market, so the only
    universal rules are that it must contain digits and must not have been
    truncated with an ellipsis or a "..." style abbreviation.
    """
    problems: list[str] = []
    candidate = normalise_identifier(value)
    if not any(character.isdigit() for character in candidate):
        problems.append("account identifier contains no digits")
    if "..." in value or "…" in value:
        problems.append("value appears truncated (contains an ellipsis)")
    if len(candidate) < 3:
        problems.append(f"account identifier is implausibly short ({len(candidate)} characters)")
    return problems


def validate_value(value: str, field: FieldDescriptor) -> FieldVerdict:
    """Validate one value against its field's kind and observed constraints."""
    text = value.strip()
    if not text:
        return FieldVerdict(
            outcome=ValidationOutcome.NOT_CHECKED,
            reasons=("no value to validate",),
            checked_by="registry",
        )

    reasons: list[str] = []
    normalised: str | None = None
    format_score = 0.5
    ambiguous = False

    checker = _CHECKERS.get(field.kind)
    if checker is not None:
        result = checker(text)
        normalised = result.normalised
        if not result.ok:
            reasons.append(result.reason or f"failed the {field.kind.value} format check")
            format_score = 0.0
        else:
            format_score = 1.0
            if result.reason:
                # A pass with a caveat: a structurally valid BIC missing from the
                # registry, or a date whose day/month order is genuinely ambiguous.
                reasons.append(result.reason)
                if "ambiguous" in result.reason:
                    ambiguous = True
                    format_score = 0.7
                else:
                    format_score = 0.85

    if field.kind in (FieldKind.ACCOUNT_NUMBER, FieldKind.PARTICIPANT_ID):
        problems = _account_number_checks(text)
        reasons.extend(problems)
        format_score = 0.0 if problems else max(format_score, 0.8)

    hint_problems = _check_hints(text, field.kind, field.hints)
    if hint_problems:
        reasons.extend(hint_problems)
        # For exact kinds a hint mismatch is a failure; for free text it is a note,
        # because prose legitimately varies in length between instructions.
        if field.kind in _EXACT_KINDS:
            format_score = 0.0

    if format_score == 0.0:
        outcome = ValidationOutcome.FAILED
    elif ambiguous:
        outcome = ValidationOutcome.AMBIGUOUS
    elif checker is None and field.kind not in (
        FieldKind.ACCOUNT_NUMBER,
        FieldKind.PARTICIPANT_ID,
    ):
        # Nothing deterministic to check (free text, names, addresses): report that
        # honestly rather than claiming a pass the code did not earn.
        outcome = ValidationOutcome.NOT_CHECKED
        format_score = 0.6
        reasons.append(f"no deterministic check exists for kind {field.kind.value}")
    else:
        outcome = ValidationOutcome.PASSED

    return FieldVerdict(
        outcome=outcome,
        level=ValidationLevel.VALUE,
        reasons=tuple(reasons),
        normalised_value=normalised,
        format_score=round(format_score, 4),
        checked_by=f"kind:{field.kind.value}",
    )
