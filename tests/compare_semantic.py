"""Semantic comparison against a reference JSON of a *different* shape.

The reference files were produced by an earlier system whose key names, grouping
and nesting differ from this pipeline's discovered schema, and they carry no page
citations. A key-by-key diff would therefore report hundreds of differences that
say nothing about extraction quality.

So comparison is semantic: both sides are flattened to a bag of
``(inferred kind, normalised value)`` facts and matched on that. What it measures
is the question that actually matters — **did we recover the same values, with the
same characters, from the same document** — independent of what anything is called
or where it sits in the tree.

It also reports the two failure directions separately, because they mean different
things:

* **missing** — a value the reference has and we do not: a recall failure.
* **extra** — a value we have and the reference does not: either a recall failure
  *in the reference*, or a hallucination in ours. The pipeline's own G2 check
  distinguishes those, so extras are reported, not scored as errors.

Values are compared after character folding, but **length is never folded away**:
a dropped digit changes the folded value and is reported as a mismatch, which is
the whole point of the no-character-lost constraint.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field

from ssi_extractor.schema.descriptor import FieldKind
from ssi_extractor.schema.kinds import infer_kind
from ssi_extractor.utils.text import fold_for_comparison, similarity
from ssi_extractor.validators.formats import check_country, check_currency, check_date

__all__ = ["ComparisonReport", "Fact", "compare_files", "compare_payloads", "extract_facts"]

# Placeholders that mean "absent" in either document; comparing them would inflate
# both sides with meaningless agreement.
_NULL_VALUES = frozenset(
    {"", "-", "--", "n/a", "na", "none", "nil", "null", "n.a.", "not applicable"}
)

# Kinds worth scoring. Free text and prose vary in wording between systems without
# either being wrong, so they are counted separately rather than driving the score.
_SCORED_KINDS = frozenset(
    {
        FieldKind.BIC,
        FieldKind.IBAN,
        FieldKind.ACCOUNT_NUMBER,
        FieldKind.SORT_CODE,
        FieldKind.PARTICIPANT_ID,
        FieldKind.LEI,
        FieldKind.ISIN,
        FieldKind.CFI,
        FieldKind.COUNTRY,
        FieldKind.CURRENCY,
        FieldKind.DATE,
        FieldKind.PERCENTAGE,
    }
)

_LEAF_KEYS = {"value", "status", "confidence", "evidence"}


class Fact(BaseModel):
    """One comparable assertion: a value of an inferred kind, found at a location."""

    model_config = ConfigDict(frozen=True)

    kind: FieldKind
    value: str
    folded: str
    path: str = ""
    record_index: int | None = None

    @property
    def key(self) -> tuple[str, str]:
        return (self.kind.value, self.folded)


class ComparisonReport(BaseModel):
    """The outcome of comparing one extraction against one reference."""

    model_config = ConfigDict(frozen=True)

    document: str = ""
    reference_records: int = 0
    actual_records: int = 0
    matched: tuple[Fact, ...] = ()
    missing: tuple[Fact, ...] = ()
    extra: tuple[Fact, ...] = ()
    near_misses: tuple[tuple[Fact, Fact, float], ...] = ()
    unscored_reference: int = 0
    unscored_actual: int = 0
    by_kind: dict[str, dict[str, int]] = Field(default_factory=dict)

    @property
    def recall(self) -> float:
        """Share of the reference's scored values that we also produced."""
        total = len(self.matched) + len(self.missing)
        return round(len(self.matched) / total, 4) if total else 1.0

    @property
    def precision(self) -> float:
        """Share of our scored values that the reference also has."""
        total = len(self.matched) + len(self.extra)
        return round(len(self.matched) / total, 4) if total else 1.0

    @property
    def f1(self) -> float:
        precision, recall = self.precision, self.recall
        if precision + recall == 0:
            return 0.0
        return round(2 * precision * recall / (precision + recall), 4)

    def render(self) -> str:
        lines = [
            f"=== {self.document} ===",
            f"records: reference {self.reference_records} vs extracted {self.actual_records}",
            f"scored facts: matched {len(self.matched)}, missing {len(self.missing)}, extra {len(self.extra)}",
            f"recall {self.recall:.3f}  precision {self.precision:.3f}  F1 {self.f1:.3f}",
            f"unscored (free-text) facts: reference {self.unscored_reference}, extracted {self.unscored_actual}",
        ]
        if self.by_kind:
            lines.append("per kind (matched/missing/extra):")
            for kind, counts in sorted(self.by_kind.items()):
                lines.append(
                    f"  {kind:16s} {counts.get('matched', 0):3d} / "
                    f"{counts.get('missing', 0):3d} / {counts.get('extra', 0):3d}"
                )
        if self.near_misses:
            lines.append("near misses (likely a dropped or altered character):")
            for reference_fact, actual_fact, ratio in self.near_misses[:15]:
                lines.append(
                    f"  {reference_fact.kind.value}: reference {reference_fact.value!r} "
                    f"vs extracted {actual_fact.value!r} (similarity {ratio:.2f})"
                )
        if self.missing:
            lines.append("missing (in reference, not extracted):")
            for fact in self.missing[:20]:
                lines.append(f"  {fact.kind.value}: {fact.value!r}")
        if self.extra:
            lines.append("extra (extracted, not in reference):")
            for fact in self.extra[:20]:
                lines.append(f"  {fact.kind.value}: {fact.value!r} at {fact.path}")
        return "\n".join(lines)


def _is_null(value: str) -> bool:
    return value.strip().lower() in _NULL_VALUES


_IDENTIFIER_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9./-]{3,}")
_SPACED_IBAN_RE = re.compile(r"\b([A-Z]{2}\d{2}(?:\s?[A-Z0-9]{2,6}){2,8})\b")


def _atomic_values(value: str) -> list[str]:
    """Break a value into every identifier it contains.

    Two systems package the same facts differently: one emits an account number in
    its own field, another writes ``IBAN: GB12 BARC 2019 9003 8090 72 A/C:
    03809072`` into a single combined field, sometimes with the IBAN grouped in
    fours. Comparing whole strings would score those as disagreement when the
    underlying values are identical, so both sides are decomposed the same way —
    separator-split parts, plus every identifier-shaped token inside them, plus any
    space-grouped IBAN rejoined.
    """
    if _is_null(value):
        return []

    atoms: list[str] = []

    for part in re.split(r"[\n;,]|\s{2,}|\s/\s", value):
        text = part.strip(" .:/-")
        if text and not _is_null(text):
            atoms.append(text)

    # A space-grouped IBAN is one value written with cosmetic spacing. Its groups
    # are recorded as consumed so the token scan below does not also emit "2019"
    # and "9003" as separate account numbers, which would invent facts neither
    # document contains.
    consumed: set[str] = set()
    for match in _SPACED_IBAN_RE.finditer(value):
        rejoined = re.sub(r"\s+", "", match.group(1))
        if len(rejoined) >= 15:
            atoms.append(rejoined)
            consumed.update(match.group(1).split())

    # Identifier-shaped tokens inside a longer string, so a combined field still
    # yields its constituent facts.
    if len(value.split()) > 1:
        for match in _IDENTIFIER_TOKEN_RE.finditer(value):
            token = match.group(0).strip(" .:/-")
            # Only tokens carrying a digit are harvested from inside a longer
            # string. Without this, an ordinary word in a name ("IICs AUSTRIA")
            # would be promoted to a country fact that neither side intended.
            if (
                token
                and not _is_null(token)
                and any(ch.isdigit() for ch in token)
                and token not in consumed
            ):
                atoms.append(token)

    seen: set[str] = set()
    unique: list[str] = []
    for atom in atoms:
        folded = fold_for_comparison(atom)
        if folded and folded not in seen:
            seen.add(folded)
            unique.append(atom)
    return unique or ([value.strip()] if not _is_null(value) else [])


def _walk_leaves(node: Any, path: str = "", record_index: int | None = None) -> Iterable[tuple[str, str, int | None]]:
    """Yield ``(path, value, record_index)`` for every leaf, in either shape.

    Handles both this pipeline's output and the reference's: any dict carrying the
    leaf keys is a leaf; a list under ``settlementInstructionRecords`` sets the
    record index so per-record counts stay meaningful.
    """
    if isinstance(node, dict):
        if _LEAF_KEYS <= set(node):
            value = node.get("value")
            if isinstance(value, str):
                yield path, value, record_index
            return
        for key, child in node.items():
            child_path = f"{path}.{key}" if path else str(key)
            yield from _walk_leaves(child, child_path, record_index)
        return

    if isinstance(node, list):
        for index, item in enumerate(node):
            next_index = index if path.endswith("settlementInstructionRecords") else record_index
            yield from _walk_leaves(item, f"{path}[{index}]", next_index)
        return

    if isinstance(node, str) and path:
        yield path, node, record_index


def extract_facts(payload: dict[str, Any]) -> tuple[list[Fact], int]:
    """Flatten a payload into scored facts, returning also the unscored count."""
    facts: list[Fact] = []
    unscored = 0
    unscored_identifiers: list[str] = []

    for path, raw_value, record_index in _walk_leaves(payload):
        # Metadata is not an extracted fact.
        if path.split(".")[0] in {
            "documentName",
            "status",
            "documentAnalysis",
            "pageCount",
            "nativeTextPages",
            "scannedPages",
            "mixedPages",
            "instructionCount",
            "extractionSummary",
            "maskingPolicy",
        }:
            continue
        if path.endswith("rowAnalysis"):
            continue

        for atom in _atomic_values(raw_value):
            inference = infer_kind([atom])
            if inference.kind in _SCORED_KINDS:
                facts.append(
                    Fact(
                        kind=inference.kind,
                        value=atom,
                        folded=_canonical_form(inference.kind, atom),
                        path=path,
                        record_index=record_index,
                    )
                )
            else:
                if _looks_like_identifier(atom):
                    unscored_identifiers.append(atom)
                unscored += 1
    return facts, unscored


def _canonical_form(kind: FieldKind, value: str) -> str:
    """Fold a value for comparison, canonicalising where a standard form exists.

    Two systems can state the same country as "AUSTRIA", "Austria" or "AT" and
    the same date as "25/04/2025" or "2025-04-25". Comparing the printed form
    would report those as disagreements, so kinds with an ISO representation are
    normalised to it first. Identifiers are never canonicalised beyond folding:
    for those, a difference in characters is exactly what must be reported.
    """
    if kind is FieldKind.COUNTRY:
        result = check_country(value)
        if result.normalised:
            return result.normalised.lower()
    elif kind is FieldKind.CURRENCY:
        result = check_currency(value)
        if result.normalised:
            return result.normalised.lower()
    elif kind is FieldKind.DATE:
        result = check_date(value)
        if result.normalised:
            return result.normalised
    return fold_for_comparison(value)


def _looks_like_identifier(value: str) -> bool:
    """Whether an unscored value still looks like a code worth diffing.

    The reference files contain identifiers that fail their own format checks
    (a BIC with a character dropped). Those never become scored facts, so without
    this they would vanish from the comparison and our correct value would appear
    as an unexplained extra.
    """
    candidate = value.strip()
    if not 6 <= len(candidate) <= 24 or " " in candidate:
        return False
    return candidate.isalnum() and candidate.upper() == candidate


def compare_payloads(
    reference: dict[str, Any],
    actual: dict[str, Any],
    *,
    document: str = "",
    near_miss_threshold: float = 0.8,
) -> ComparisonReport:
    """Compare two payloads semantically."""
    reference_facts, reference_unscored = extract_facts(reference)
    actual_facts, actual_unscored = extract_facts(actual)

    actual_pool: dict[tuple[str, str], list[Fact]] = defaultdict(list)
    for fact in actual_facts:
        actual_pool[fact.key].append(fact)

    matched: list[Fact] = []
    missing: list[Fact] = []

    for fact in reference_facts:
        pool = actual_pool.get(fact.key)
        if pool:
            pool.pop()
            matched.append(fact)
        else:
            missing.append(fact)

    extra = [fact for pool in actual_pool.values() for fact in pool]

    # Pair each missing value with its closest surviving extra of the same kind.
    # A high similarity is the signature of a single altered or dropped character —
    # exactly the class of error the pipeline exists to prevent, and the class the
    # reference files themselves contain.
    near_misses: list[tuple[Fact, Fact, float]] = []
    available = list(extra)
    for fact in missing:
        candidates = [other for other in available if other.kind is fact.kind]
        if not candidates:
            continue
        best = max(candidates, key=lambda other: similarity(fact.value, other.value))
        ratio = similarity(fact.value, best.value)
        if ratio >= near_miss_threshold:
            near_misses.append((fact, best, round(ratio, 4)))
            available.remove(best)

    by_kind: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for label, facts in (("matched", matched), ("missing", missing), ("extra", extra)):
        for fact in facts:
            by_kind[fact.kind.value][label] += 1

    return ComparisonReport(
        document=document,
        reference_records=len(reference.get("settlementInstructionRecords", []) or []),
        actual_records=len(actual.get("settlementInstructionRecords", []) or []),
        matched=tuple(matched),
        missing=tuple(missing),
        extra=tuple(extra),
        near_misses=tuple(sorted(near_misses, key=lambda item: -item[2])),
        unscored_reference=reference_unscored,
        unscored_actual=actual_unscored,
        by_kind={kind: dict(counts) for kind, counts in by_kind.items()},
    )


def _load(path: Path) -> dict[str, Any]:
    """Load a JSON payload, tolerating the reference files' ``.txt`` extension."""
    return json.loads(path.read_text(encoding="utf-8-sig"))


def compare_files(reference_path: Path | str, actual_path: Path | str) -> ComparisonReport:
    reference_path, actual_path = Path(reference_path), Path(actual_path)
    return compare_payloads(
        _load(reference_path),
        _load(actual_path),
        document=actual_path.name,
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Semantically compare an extraction against a reference.")
    parser.add_argument("reference", type=Path)
    parser.add_argument("actual", type=Path)
    arguments = parser.parse_args(argv)

    report = compare_files(arguments.reference, arguments.actual)
    print(report.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
