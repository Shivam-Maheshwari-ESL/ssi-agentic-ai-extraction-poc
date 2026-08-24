"""Deterministic format checks — the single implementation of "is this a valid X".

Both kind inference (Stage 4c) and validation (Stage 7) call these, so a value
is judged by exactly the same code in both places. Every check is pure, offline
and side-effect free.

Checksums are implemented directly rather than delegated, for two reasons: the
pipeline must never depend on a library being importable to decide whether a
digit was dropped, and a local implementation gives a specific failure reason
instead of a bare boolean. ``schwifty`` is used as an *additional* registry check
where it is available, never as the only one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache

__all__ = [
    "CheckResult",
    "charset_profile",
    "check_bic",
    "check_cfi",
    "check_country",
    "check_country_alpha2",
    "check_country_name",
    "check_currency",
    "check_date",
    "check_iban",
    "check_isin",
    "check_lei",
    "check_percentage",
    "check_sort_code",
    "normalise_identifier",
]


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Outcome of one format check."""

    ok: bool
    reason: str | None = None
    normalised: str | None = None

    def __bool__(self) -> bool:
        return self.ok


def _fail(reason: str) -> CheckResult:
    return CheckResult(ok=False, reason=reason)


def _pass(normalised: str) -> CheckResult:
    return CheckResult(ok=True, normalised=normalised)


def normalise_identifier(value: str) -> str:
    """Strip the whitespace and separators OCR and layout introduce into identifiers.

    Only characters that are never semantically part of an identifier are
    removed. Digits and letters are never altered — dropping or "correcting" a
    character here would violate the no-digit-lost constraint.
    """
    return re.sub(r"[\s ._/-]", "", value).strip().upper()


def _reject_prose(value: str, code_name: str) -> CheckResult | None:
    """Guard the all-uppercase code checks against ordinary words.

    Registered codes (BIC, CFI, ISIN, LEI) are written in upper case wherever
    they appear. Without this guard, uppercasing during normalisation would let
    an English word of the right length ("Create" -> "CREATE") satisfy a code
    pattern and silently mis-type the field.
    """
    if any(character.islower() for character in value):
        return _fail(f"{code_name} codes are upper case; '{value}' contains lower-case letters")
    return None


# ---------------------------------------------------------------------------
# Charset profiling
# ---------------------------------------------------------------------------

def charset_profile(value: str) -> str:
    """Classify a value's character composition, used as a validator hint."""
    stripped = re.sub(r"\s", "", value)
    if not stripped:
        return "empty"
    if stripped.isdigit():
        return "digits"
    if stripped.isalpha():
        return "upper_alpha" if stripped.isupper() else "alpha"
    if stripped.isalnum():
        return "upper_alnum" if stripped.upper() == stripped else "alnum"
    return "any"


# ---------------------------------------------------------------------------
# BIC / SWIFT
# ---------------------------------------------------------------------------

_BIC_RE = re.compile(r"^[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?$")


def check_bic(value: str, *, use_registry: bool = False) -> CheckResult:
    """ISO 9362: 4-letter institution, 2-letter ISO country, 2-char location, optional branch.

    Args:
        use_registry: also consult the local ``schwifty`` BIC registry. Off by
            default because kind inference runs this over every sampled value and
            loading the registry there would cost far more than it is worth;
            Stage 7 validation opts in, where an unknown-but-well-formed BIC is a
            meaningful confidence signal.
    """
    if (prose := _reject_prose(value, "BIC")) is not None:
        return prose
    candidate = normalise_identifier(value)
    if not candidate:
        return _fail("empty")
    if len(candidate) not in (8, 11):
        return _fail(f"BIC must be 8 or 11 characters, got {len(candidate)}")
    if not _BIC_RE.match(candidate):
        return _fail("BIC does not match the ISO 9362 character pattern")
    if not check_country_alpha2(candidate[4:6]):
        return _fail(f"BIC country code '{candidate[4:6]}' is not a valid ISO 3166-1 alpha-2 code")

    if use_registry and _schwifty_bic_known(candidate) is False:
        return CheckResult(
            ok=True,
            reason="structurally valid but not present in the local BIC registry",
            normalised=candidate,
        )
    return _pass(candidate)


@lru_cache(maxsize=4096)
def _schwifty_bic_known(candidate: str) -> bool | None:
    """Registry lookup via schwifty when importable; ``None`` when unavailable."""
    try:
        from schwifty import BIC
    except ImportError:
        return None
    try:
        return BIC(candidate).exists
    except Exception:  # schwifty raises library-specific validation errors
        return False


# ---------------------------------------------------------------------------
# IBAN
# ---------------------------------------------------------------------------

_IBAN_RE = re.compile(r"^[A-Z]{2}\d{2}[A-Z0-9]{10,30}$")


def check_iban(value: str) -> CheckResult:
    """ISO 13616 structure plus the mod-97-10 checksum, which catches dropped digits."""
    candidate = normalise_identifier(value)
    if not candidate:
        return _fail("empty")
    if not _IBAN_RE.match(candidate):
        return _fail("IBAN does not match the ISO 13616 character pattern")
    if not check_country_alpha2(candidate[:2]):
        return _fail(f"IBAN country code '{candidate[:2]}' is not a valid ISO 3166-1 alpha-2 code")

    rearranged = candidate[4:] + candidate[:4]
    digits = "".join(
        str(ord(character) - 55) if character.isalpha() else character for character in rearranged
    )
    if int(digits) % 97 != 1:
        return _fail("IBAN checksum failed (mod-97), which usually means a character was lost")
    return _pass(candidate)


# ---------------------------------------------------------------------------
# ISIN / CFI
# ---------------------------------------------------------------------------

_ISIN_RE = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}\d$")


def check_isin(value: str) -> CheckResult:
    """ISO 6166: 12 characters with a Luhn check digit over letter-expanded digits."""
    if (prose := _reject_prose(value, "ISIN")) is not None:
        return prose
    candidate = normalise_identifier(value)
    if len(candidate) != 12:
        return _fail(f"ISIN must be 12 characters, got {len(candidate)}")
    if not _ISIN_RE.match(candidate):
        return _fail("ISIN does not match the ISO 6166 character pattern")

    expanded = "".join(
        str(ord(character) - 55) if character.isalpha() else character for character in candidate[:11]
    )
    total = 0
    for index, character in enumerate(reversed(expanded)):
        digit = int(character)
        if index % 2 == 0:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    if (10 - total % 10) % 10 != int(candidate[11]):
        return _fail("ISIN check digit failed")
    return _pass(candidate)


_CFI_RE = re.compile(r"^[A-Z]{6}$")


def check_cfi(value: str) -> CheckResult:
    """ISO 10962 classification code: six uppercase letters."""
    if (prose := _reject_prose(value, "CFI")) is not None:
        return prose
    candidate = normalise_identifier(value)
    if not _CFI_RE.match(candidate):
        return _fail("CFI must be exactly six uppercase letters")
    return _pass(candidate)


# ---------------------------------------------------------------------------
# LEI
# ---------------------------------------------------------------------------

_LEI_RE = re.compile(r"^[A-Z0-9]{18}\d{2}$")


def check_lei(value: str) -> CheckResult:
    """ISO 17442: 20 characters with an ISO 7064 mod-97-10 checksum."""
    if (prose := _reject_prose(value, "LEI")) is not None:
        return prose
    candidate = normalise_identifier(value)
    if len(candidate) != 20:
        return _fail(f"LEI must be 20 characters, got {len(candidate)}")
    if not _LEI_RE.match(candidate):
        return _fail("LEI does not match the ISO 17442 character pattern")

    digits = "".join(
        str(ord(character) - 55) if character.isalpha() else character for character in candidate
    )
    if int(digits) % 97 != 1:
        return _fail("LEI checksum failed (mod-97-10)")
    return _pass(candidate)


# ---------------------------------------------------------------------------
# ISO country / currency
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _iso_country_codes() -> frozenset[str]:
    import pycountry

    codes = {country.alpha_2 for country in pycountry.countries}
    # Reserved and commonly used exceptional codes that pycountry omits but that
    # appear in settlement data (e.g. XS for international securities).
    codes.update({"XS", "EU"})
    return frozenset(codes)


@lru_cache(maxsize=1)
def _iso_currency_codes() -> frozenset[str]:
    import pycountry

    codes = {currency.alpha_3 for currency in pycountry.currencies}
    codes.update({"XAU", "XAG", "XPT", "XPD", "XDR"})
    return frozenset(codes)


def check_country_alpha2(value: str) -> CheckResult:
    candidate = value.strip().upper()
    if len(candidate) != 2:
        return _fail(f"country code must be 2 characters, got {len(candidate)}")
    if candidate not in _iso_country_codes():
        return _fail(f"'{candidate}' is not a valid ISO 3166-1 alpha-2 code")
    return _pass(candidate)


@lru_cache(maxsize=1)
def _country_name_index() -> dict[str, str]:
    """Map lower-cased country names, official names and alpha-3 codes to alpha-2.

    SSI documents identify markets by name at least as often as by code
    ("AUSTRIA", "United Kingdom"), so country detection cannot be alpha-2 only.
    """
    import pycountry

    index: dict[str, str] = {}
    for country in pycountry.countries:
        index[country.alpha_3.lower()] = country.alpha_2
        for attribute in ("name", "official_name", "common_name"):
            name = getattr(country, attribute, None)
            if name:
                index[name.lower()] = country.alpha_2
    return index


def check_country_name(value: str) -> CheckResult:
    """Match a written country *name* only, never a code.

    Needed because names collide with code formats by length: ``CANADA`` and
    ``FRANCE`` are six upper-case letters, which is exactly the CFI pattern, and
    ``NETHERLANDS`` is eleven, which is exactly a BIC with a branch code. Checking
    names first — and only names, so a genuine ``CEDELULL`` is untouched — keeps a
    document that spells its markets out from having them typed as securities
    codes.
    """
    candidate = value.strip()
    if len(candidate) < 4:
        return _fail("too short to be a country name")
    resolved = _country_name_index().get(candidate.lower())
    if resolved is None:
        return _fail(f"'{candidate}' is not a recognised ISO 3166-1 country name")
    return _pass(resolved)


def check_country(value: str) -> CheckResult:
    """Accept an alpha-2 code, an alpha-3 code, or a country name; normalise to alpha-2."""
    candidate = value.strip()
    if not candidate:
        return _fail("empty")
    if len(candidate) == 2:
        return check_country_alpha2(candidate)

    resolved = _country_name_index().get(candidate.lower())
    if resolved is None:
        return _fail(f"'{candidate}' is not a recognised ISO 3166-1 country name or code")
    return _pass(resolved)


def check_currency(value: str) -> CheckResult:
    candidate = value.strip().upper()
    if len(candidate) != 3:
        return _fail(f"currency code must be 3 characters, got {len(candidate)}")
    if candidate not in _iso_currency_codes():
        return _fail(f"'{candidate}' is not a valid ISO 4217 code")
    return _pass(candidate)


# ---------------------------------------------------------------------------
# Dates, percentages, sort codes
# ---------------------------------------------------------------------------

# Day-first before month-first: SSI documents from European counterparties
# dominate, and an ambiguous value is reported as ambiguous rather than guessed.
_DATE_FORMATS: tuple[str, ...] = (
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%d.%m.%Y",
    "%d %b %Y",
    "%d %B %Y",
    "%b %d, %Y",
    "%B %d, %Y",
    "%Y/%m/%d",
    "%m/%d/%Y",
    "%d/%m/%y",
    "%Y%m%d",
)


def check_date(value: str) -> CheckResult:
    """Parse a date and normalise it to ISO 8601, flagging genuine ambiguity.

    A value like ``05/04/2025`` parses under both day-first and month-first
    conventions. Rather than silently choosing, the result is reported as valid
    but ambiguous, which is what routes it to the adjudicator instead of being
    quietly wrong.
    """
    candidate = value.strip()
    if not candidate:
        return _fail("empty")

    parsed: date | None = None
    matched_format: str | None = None
    for date_format in _DATE_FORMATS:
        try:
            parsed = datetime.strptime(candidate, date_format).date()
            matched_format = date_format
            break
        except ValueError:
            continue

    if parsed is None or matched_format is None:
        return _fail("value does not parse as a date in any supported format")

    ambiguous = False
    if matched_format in ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d/%m/%y"):
        parts = re.split(r"[/.\-]", candidate)
        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
            first, second = int(parts[0]), int(parts[1])
            ambiguous = first <= 12 and second <= 12 and first != second

    return CheckResult(
        ok=True,
        reason="ambiguous day/month order" if ambiguous else None,
        normalised=parsed.isoformat(),
    )


_PERCENTAGE_RE = re.compile(r"^(\d{1,3}(?:[.,]\d{1,4})?)\s*%$")


def check_percentage(value: str) -> CheckResult:
    match = _PERCENTAGE_RE.match(value.strip())
    if match is None:
        return _fail("value is not a percentage")
    numeric = float(match.group(1).replace(",", "."))
    if not 0.0 <= numeric <= 100.0:
        return _fail(f"percentage {numeric} is outside 0-100")
    return _pass(f"{numeric:g}%")


_SORT_CODE_RE = re.compile(r"^\d{6}$")


def check_sort_code(value: str) -> CheckResult:
    """UK sort code: six digits, conventionally written ``12-34-56``."""
    candidate = normalise_identifier(value)
    if not _SORT_CODE_RE.match(candidate):
        return _fail("sort code must be exactly six digits")
    return _pass(candidate)
