"""Reference data ("golden set") loaders, keyed by field kind.

Ships **empty by default and marked provisional**. No values derived from the
sample documents are baked in: seeding a BIC directory from two samples would make
the pipeline look accurate on those files and wrong everywhere else.

The important behaviour is how absence is handled. When a reference set is not
loaded, a lookup reports ``available=False`` and the confidence blend uses a
neutral score, so an unknown BIC neither raises nor lowers confidence, and the
document-level drift check reports ``INDETERMINATE`` instead of firing on every
document. ISO country and currency membership is always available through
``pycountry``, so those kinds are checked regardless.

Populate ``config/reference/`` to switch the signal on:

* ``bic_directory.csv`` — ``bic,institution,country`` (header row required)
* ``country_pset_map.csv`` — ``country,pset_bic``
* ``account_patterns.yaml`` — ``{market: [regex, ...]}``
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from ssi_extractor.config.settings import Settings, get_settings
from ssi_extractor.observability.logging import get_logger
from ssi_extractor.schema.descriptor import FieldKind
from ssi_extractor.validators.formats import check_country, check_currency, normalise_identifier

__all__ = ["LookupResult", "ReferenceData"]

_logger = get_logger(__name__)


class LookupResult(BaseModel):
    """Outcome of a reference lookup."""

    model_config = ConfigDict(frozen=True)

    available: bool = False
    found: bool = False
    source: str = ""
    detail: str = ""


class ReferenceData(BaseModel):
    """Kind-keyed reference sets, loaded from ``config/reference/``."""

    model_config = ConfigDict(frozen=True)

    bic_directory: frozenset[str] = frozenset()
    country_pset_map: dict[str, str] = Field(default_factory=dict)
    account_patterns: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    provisional: bool = True
    neutral_score: float = 0.5
    loaded_from: tuple[str, ...] = ()

    @classmethod
    def load(cls, *, settings: Settings | None = None) -> "ReferenceData":
        """Load whatever reference files exist; missing files are not an error."""
        settings = settings or get_settings()
        directory = settings.paths.reference_dir
        loaded: list[str] = []

        bics: set[str] = set()
        bic_path = directory / "bic_directory.csv"
        if bic_path.is_file():
            with bic_path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    value = (row.get("bic") or "").strip().upper()
                    if value:
                        bics.add(value)
            loaded.append(bic_path.name)

        pset: dict[str, str] = {}
        pset_path = directory / "country_pset_map.csv"
        if pset_path.is_file():
            with pset_path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    country = (row.get("country") or "").strip().upper()
                    depository = (row.get("pset_bic") or "").strip().upper()
                    if country and depository:
                        pset[country] = depository
            loaded.append(pset_path.name)

        patterns: dict[str, tuple[str, ...]] = {}
        patterns_path = directory / "account_patterns.yaml"
        if patterns_path.is_file():
            import yaml

            raw = yaml.safe_load(patterns_path.read_text(encoding="utf-8")) or {}
            for market, expressions in raw.items():
                if isinstance(expressions, str):
                    expressions = [expressions]
                patterns[str(market).upper()] = tuple(str(item) for item in expressions or ())
            loaded.append(patterns_path.name)

        if loaded:
            _logger.info("Reference data loaded from: %s", ", ".join(loaded))
        else:
            _logger.info(
                "No reference data present in %s; lookups will report INDETERMINATE "
                "and contribute neutrally to confidence.",
                directory,
            )

        return cls(
            bic_directory=frozenset(bics),
            country_pset_map=pset,
            account_patterns=patterns,
            loaded_from=tuple(loaded),
        )

    @property
    def has_bic_directory(self) -> bool:
        return bool(self.bic_directory)

    def lookup(self, kind: FieldKind, value: str) -> LookupResult:
        """Look a value up in whatever reference set covers its kind."""
        text = value.strip()
        if not text:
            return LookupResult(available=False, detail="no value")

        if kind is FieldKind.BIC:
            if not self.bic_directory:
                return LookupResult(available=False, source="bic_directory", detail="not loaded")
            candidate = normalise_identifier(text)
            # An 11-character BIC is present in a directory of 8-character entries
            # via its institution prefix, so both forms are tried.
            found = candidate in self.bic_directory or candidate[:8] in self.bic_directory
            return LookupResult(available=True, found=found, source="bic_directory")

        if kind is FieldKind.COUNTRY:
            result = check_country(text)
            return LookupResult(available=True, found=bool(result), source="iso_3166")

        if kind is FieldKind.CURRENCY:
            result = check_currency(text)
            return LookupResult(available=True, found=bool(result), source="iso_4217")

        if kind is FieldKind.ACCOUNT_NUMBER and self.account_patterns:
            candidate = normalise_identifier(text)
            for market, expressions in self.account_patterns.items():
                for expression in expressions:
                    if re.fullmatch(expression, candidate):
                        return LookupResult(
                            available=True,
                            found=True,
                            source="account_patterns",
                            detail=f"matched {market}",
                        )
            return LookupResult(available=True, found=False, source="account_patterns")

        return LookupResult(available=False, detail=f"no reference set for kind {kind.value}")

    def expected_pset(self, country: str) -> str | None:
        """The depository a country's instructions are expected to settle at."""
        resolved = check_country(country)
        key = (resolved.normalised or country).strip().upper()
        return self.country_pset_map.get(key)
