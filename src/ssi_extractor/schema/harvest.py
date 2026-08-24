"""Deterministic field harvesting — the first half of schema discovery.

Reads candidate fields out of located chunks without an LLM: table header bands
become group paths, key:value pairs become labelled fields, SWIFT tags resolve
through their own map, and prose falls back to label/value adjacency. Each
candidate carries the sample values actually observed, which is what lets kind
inference decide what the field *is* independently of what it is called.

This stage alone is sufficient to build a working descriptor. The synthesis agent
refines it; it is never required, which is why an unreachable model degrades the
output rather than failing the document.
"""

from __future__ import annotations

import re
from collections import defaultdict

from pydantic import BaseModel, ConfigDict, Field

from ssi_extractor.observability.logging import get_logger
from ssi_extractor.schema.descriptor import (
    DescriptorSource,
    FieldDescriptor,
    FieldKind,
    LayoutPattern,
    RepeatingUnit,
    SchemaDescriptor,
    slugify,
)
from ssi_extractor.schema.kinds import infer_kind
from ssi_extractor.stages.locate_chunk import InstructionChunk, LocateResult
from ssi_extractor.utils.text import collapse_whitespace

__all__ = ["CandidateField", "harvest_candidates", "build_deterministic_descriptor"]

_logger = get_logger(__name__)

# Values that mean "this row has nothing here" rather than being data. Treated as
# absence so a column of dashes does not become a FREE_TEXT field full of "-".
_NULL_TOKENS = frozenset({"-", "--", "---", "n/a", "na", "none", "nil", "n.a.", "", "–", "—"})

# SWIFT field tags carry their meaning in the standard, not in the document, so a
# small map turns embedded MT messages into properly labelled fields.
_SWIFT_TAG_LABELS: dict[str, tuple[str, FieldKind]] = {
    "95P": ("Party BIC", FieldKind.BIC),
    "95Q": ("Party Name And Address", FieldKind.ORG_NAME),
    "95R": ("Party Proprietary Identification", FieldKind.PARTICIPANT_ID),
    "97A": ("Safekeeping Account", FieldKind.ACCOUNT_NUMBER),
    "97B": ("Safekeeping Account Proprietary", FieldKind.ACCOUNT_NUMBER),
    "22F": ("Indicator", FieldKind.ENUM),
    "35B": ("Identification Of Financial Instrument", FieldKind.ISIN),
    "36B": ("Quantity Of Financial Instrument", FieldKind.FREE_TEXT),
    "98A": ("Date", FieldKind.DATE),
    "19A": ("Amount", FieldKind.FREE_TEXT),
    "16R": ("Start Of Block", FieldKind.ENUM),
    "16S": ("End Of Block", FieldKind.ENUM),
    "20C": ("Reference", FieldKind.FREE_TEXT),
    "94": ("Place", FieldKind.FREE_TEXT),
}

_SWIFT_LINE_RE = re.compile(r"^:(?P<tag>\d{2}[A-Z]?)::?(?P<qualifier>[A-Z]{4})?/?/?(?P<value>.*)$")


class CandidateField(BaseModel):
    """One harvested field candidate, before kind inference and synthesis."""

    model_config = ConfigDict(frozen=True)

    label: str
    group_path: tuple[str, ...] = ()
    sample_values: tuple[str, ...] = ()
    pages: tuple[int, ...] = ()
    source_pattern: str = ""
    kind_hint: FieldKind | None = None
    occurrences: int = 0

    @property
    def name(self) -> str:
        return slugify(self.label)

    @property
    def path(self) -> str:
        return ".".join((*(slugify(part) for part in self.group_path), self.name))


def _is_null_token(value: str) -> bool:
    return collapse_whitespace(value).lower() in _NULL_TOKENS


def _split_group_and_label(header: str) -> tuple[tuple[str, ...], str]:
    """Turn a composed header path ("BENEFICIARY / Bic Code") into groups and a label."""
    parts = [collapse_whitespace(part) for part in header.split(" / ") if collapse_whitespace(part)]
    if not parts:
        return (), "field"
    if len(parts) == 1:
        return (), parts[0]
    return tuple(parts[:-1]), parts[-1]


def _harvest_table(chunks: list[InstructionChunk]) -> list[CandidateField]:
    """One candidate per column, with the column's values as samples."""
    by_header: dict[str, list[str]] = defaultdict(list)
    pages: dict[str, set[int]] = defaultdict(set)

    for chunk in chunks:
        for header, value in chunk.cells:
            by_header[header].append(value)
            pages[header].update(chunk.pages)

    candidates: list[CandidateField] = []
    for header, values in by_header.items():
        group_path, label = _split_group_and_label(header)
        populated = [value for value in values if not _is_null_token(value)]
        candidates.append(
            CandidateField(
                label=label,
                group_path=group_path,
                sample_values=tuple(populated[:12]),
                pages=tuple(sorted(pages[header])),
                source_pattern="table_header",
                occurrences=len(populated),
            )
        )
    return candidates


def _harvest_key_values(chunks: list[InstructionChunk]) -> list[CandidateField]:
    """One candidate per distinct label, grouped by the section it appeared under."""
    grouped: dict[tuple[tuple[str, ...], str], list[str]] = defaultdict(list)
    pages: dict[tuple[tuple[str, ...], str], set[int]] = defaultdict(set)

    for chunk in chunks:
        for pair in chunk.key_values:
            group = (collapse_whitespace(pair.section),) if pair.section else ()
            key = (group, collapse_whitespace(pair.key))
            grouped[key].append(pair.value)
            pages[key].add(pair.page)

    return [
        CandidateField(
            label=label,
            group_path=group,
            sample_values=tuple(
                value for value in values if not _is_null_token(value)
            )[:12],
            pages=tuple(sorted(pages[(group, label)])),
            source_pattern="key_value",
            occurrences=len(values),
        )
        for (group, label), values in grouped.items()
    ]


def _harvest_swift(chunks: list[InstructionChunk]) -> list[CandidateField]:
    """Candidates from embedded MT-style message lines."""
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    pages: dict[tuple[str, str], set[int]] = defaultdict(set)
    hints: dict[tuple[str, str], FieldKind] = {}

    for chunk in chunks:
        for line in chunk.text.splitlines():
            match = _SWIFT_LINE_RE.match(line.strip())
            if not match:
                continue
            tag = match.group("tag")
            qualifier = match.group("qualifier") or ""
            value = collapse_whitespace(match.group("value"))
            if not value:
                continue
            label, kind = _SWIFT_TAG_LABELS.get(tag, (f"SWIFT Tag {tag}", FieldKind.UNKNOWN))
            key = (qualifier or tag, f"{label} ({qualifier})" if qualifier else label)
            grouped[key].append(value)
            pages[key].update(chunk.pages)
            hints[key] = kind

    return [
        CandidateField(
            label=label,
            group_path=("SWIFT Message",),
            sample_values=tuple(values[:12]),
            pages=tuple(sorted(pages[(qualifier, label)])),
            source_pattern="swift_tag",
            kind_hint=hints[(qualifier, label)],
            occurrences=len(values),
        )
        for (qualifier, label), values in grouped.items()
    ]


def _harvest_narrative(chunks: list[InstructionChunk]) -> list[CandidateField]:
    """Label/value adjacency in prose.

    Prose SSI letters state values inline ("held at Euroclear under account
    21625"), so candidates come from identifier-shaped tokens plus the words that
    precede them. The label is weak by nature; kind inference carries the weight.
    """
    grouped: dict[str, list[str]] = defaultdict(list)
    pages: dict[str, set[int]] = defaultdict(set)

    # Three characters is enough for a currency or country code, which a document
    # often prints on a line of its own as a section band. Noise is filtered by
    # kind inference below rather than by length, so only tokens that actually
    # resolve to a recognised kind become candidates.
    token_re = re.compile(r"[A-Z0-9][A-Z0-9./-]{2,}")
    for chunk in chunks:
        for line in chunk.text.splitlines():
            for match in token_re.finditer(line):
                token = match.group(0).strip(".,;")
                inference = infer_kind([token])
                if inference.kind in (FieldKind.FREE_TEXT, FieldKind.UNKNOWN):
                    continue
                preceding = collapse_whitespace(line[: match.start()])
                words = preceding.split()[-4:]
                label = " ".join(words).strip(":,. ") or inference.kind.value.title().replace("_", " ")
                grouped[label].append(token)
                pages[label].update(chunk.pages)

    return [
        CandidateField(
            label=label,
            sample_values=tuple(values[:12]),
            pages=tuple(sorted(pages[label])),
            source_pattern="narrative_adjacency",
            occurrences=len(values),
        )
        for label, values in grouped.items()
    ]


def _deduplicate_same_concept(candidates: list[CandidateField]) -> list[CandidateField]:
    """Collapse candidates that are the same field found by two harvesters.

    A form's line can be picked up as a key:value pair under one section heading,
    again under another, and once more by adjacency scanning — producing
    ``cash.sort_code``, ``gbp.sort_code`` and ``sort_code`` for one printed value.
    Emitting all three would triple the field and report the same value as three
    facts.

    The discriminator is the *values*, not the label: two candidates sharing a name
    **and** an identical set of observed values are one field seen twice, while a
    genuinely repeated label (a "Swift Address" column for the global custodian and
    another for the local one) holds different values and is left alone. The most
    general grouping wins, so the surviving path is the shortest.
    """
    grouped: dict[tuple[str, tuple[str, ...]], list[CandidateField]] = defaultdict(list)
    for candidate in candidates:
        signature = tuple(sorted({value.strip() for value in candidate.sample_values if value.strip()}))
        grouped[(candidate.name, signature)].append(candidate)

    survivors: list[CandidateField] = []
    for (_, signature), group in grouped.items():
        if len(group) == 1 or not signature:
            survivors.extend(group)
            continue

        winner = min(group, key=lambda item: (len(item.group_path), item.path))
        pages = sorted({page for item in group for page in item.pages})
        survivors.append(
            winner.model_copy(
                update={
                    "pages": tuple(pages),
                    "occurrences": max(item.occurrences for item in group),
                    "kind_hint": next(
                        (item.kind_hint for item in group if item.kind_hint is not None), None
                    ),
                }
            )
        )
    return survivors


def harvest_candidates(located: LocateResult) -> list[CandidateField]:
    """Harvest field candidates from located chunks, by their structural pattern."""
    chunks = list(located.chunks)
    if not chunks:
        return []

    tabular = [chunk for chunk in chunks if chunk.cells]
    key_valued = [chunk for chunk in chunks if chunk.key_values]
    swift = [chunk for chunk in chunks if chunk.layout_pattern is LayoutPattern.SWIFT_MESSAGE]
    narrative = [
        chunk
        for chunk in chunks
        if not chunk.cells and not chunk.key_values and chunk not in swift
    ]

    candidates: list[CandidateField] = []
    candidates.extend(_harvest_table(tabular))
    candidates.extend(_harvest_key_values(key_valued))
    candidates.extend(_harvest_swift(swift))
    candidates.extend(_harvest_narrative(narrative))

    # Adjacency harvesting also runs over form chunks. A form states plenty of
    # values on lines that are not key:value pairs — an effective date under a
    # letterhead, a currency band over a block — and structured harvesting alone
    # would leave those fields out of the schema entirely.
    if key_valued:
        candidates.extend(_harvest_narrative(key_valued))

    candidates = _deduplicate_same_concept(candidates)

    # Merge candidates that resolved to the same path from different harvesters.
    merged: dict[str, CandidateField] = {}
    for candidate in candidates:
        existing = merged.get(candidate.path)
        if existing is None:
            merged[candidate.path] = candidate
            continue
        merged[candidate.path] = existing.model_copy(
            update={
                "sample_values": tuple(
                    dict.fromkeys((*existing.sample_values, *candidate.sample_values))
                )[:12],
                "pages": tuple(sorted(set(existing.pages) | set(candidate.pages))),
                "occurrences": existing.occurrences + candidate.occurrences,
                "kind_hint": existing.kind_hint or candidate.kind_hint,
            }
        )

    result = sorted(merged.values(), key=lambda item: (item.group_path, item.label))
    _logger.info("Harvested %s field candidate(s) from %s chunk(s).", len(result), len(chunks))
    return result


def build_deterministic_descriptor(
    located: LocateResult,
    candidates: list[CandidateField],
    *,
    document_id: str = "",
) -> SchemaDescriptor:
    """Build a descriptor from harvested candidates and inferred kinds, with no LLM.

    This is both the input to the synthesis agent and the fallback when synthesis
    is unavailable or rejected, so a document is never blocked on a model call.
    """
    fields: list[FieldDescriptor] = []
    for candidate in candidates:
        inference = infer_kind(list(candidate.sample_values), label=candidate.label)
        kind = candidate.kind_hint or inference.kind
        confidence = inference.confidence if candidate.kind_hint is None else max(
            inference.confidence, 0.6
        )
        fields.append(
            FieldDescriptor(
                name=candidate.name,
                label=candidate.label,
                group_path=candidate.group_path,
                kind=kind,
                kind_confidence=confidence,
                cardinality=inference.cardinality,
                hints=inference.hints,
                pages=candidate.pages,
                source_pattern=candidate.source_pattern,
            )
        )

    anchor_kinds = tuple(
        kind
        for kind in (FieldKind.COUNTRY, FieldKind.CURRENCY)
        if any(field.kind is kind for field in fields)
    )
    required_kinds = tuple(
        kind
        for kind in (FieldKind.BIC, FieldKind.ACCOUNT_NUMBER, FieldKind.IBAN)
        if any(field.kind is kind for field in fields)
    )

    descriptor = SchemaDescriptor(
        document_id=document_id,
        source=DescriptorSource.DETERMINISTIC,
        fields=tuple(fields),
        repeating_unit=RepeatingUnit(
            description=(
                f"One settlement instruction per {located.dominant_pattern.value.lower()} unit"
            ),
            layout_pattern=located.dominant_pattern,
            anchor_kinds=anchor_kinds,
            required_kinds=required_kinds,
        ),
        notes=(
            f"{len(located.chunks)} instruction chunk(s) located",
            f"layout pattern: {located.dominant_pattern.value}",
        ),
    )
    _logger.info(
        "Deterministic descriptor: %s field(s), hash %s.",
        len(descriptor.fields),
        descriptor.descriptor_hash[:12],
    )
    return descriptor
