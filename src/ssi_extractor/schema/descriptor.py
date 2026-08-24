"""The runtime-discovered schema description.

A ``SchemaDescriptor`` is what makes the pipeline generic: it is built per
document from what was actually found on the page, and it is the single source of
truth for the extraction model, the extraction prompt, the validator selection
and the assembled output shape. Nothing downstream may reference a literal field
name — it asks the descriptor instead.

Fields are stored flat with an explicit ``group_path``, and nesting is derived on
demand. That keeps the union-merge across chunks simple and total: merging two
flat sets cannot produce a contradictory tree.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from enum import StrEnum
from typing import Any, Iterator, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "Cardinality",
    "DescriptorSource",
    "FieldDescriptor",
    "FieldKind",
    "LayoutPattern",
    "RepeatingUnit",
    "SchemaDescriptor",
    "slugify",
]


class FieldKind(StrEnum):
    """What a value *is*, inferred from its shape rather than its label.

    Kind drives validator selection, masking, and confidence — which is why it
    must be inferable without understanding the label. A document with
    non-English or missing labels still yields usable kinds.
    """

    BIC = "BIC"
    IBAN = "IBAN"
    ACCOUNT_NUMBER = "ACCOUNT_NUMBER"
    SORT_CODE = "SORT_CODE"
    PARTICIPANT_ID = "PARTICIPANT_ID"
    LEI = "LEI"
    ISIN = "ISIN"
    CFI = "CFI"
    COUNTRY = "COUNTRY"
    CURRENCY = "CURRENCY"
    DATE = "DATE"
    PERCENTAGE = "PERCENTAGE"
    ENUM = "ENUM"
    PERSON_NAME = "PERSON_NAME"
    ORG_NAME = "ORG_NAME"
    ADDRESS = "ADDRESS"
    PHONE = "PHONE"
    EMAIL = "EMAIL"
    FREE_TEXT = "FREE_TEXT"
    UNKNOWN = "UNKNOWN"

    @property
    def is_pii(self) -> bool:
        """Whether Stage 6 must mask values of this kind.

        ``FREE_TEXT`` and ``UNKNOWN`` count as PII: an unclassified value may
        hold an account number or a name, and defaulting to "not PII" would leak.
        """
        return self in _PII_KINDS

    @property
    def is_identifier(self) -> bool:
        """Identifier kinds are digit/charset exact — no character may be dropped."""
        return self in _IDENTIFIER_KINDS


_PII_KINDS = frozenset(
    {
        FieldKind.BIC,
        FieldKind.IBAN,
        FieldKind.ACCOUNT_NUMBER,
        FieldKind.SORT_CODE,
        FieldKind.PARTICIPANT_ID,
        FieldKind.LEI,
        FieldKind.PERSON_NAME,
        FieldKind.ORG_NAME,
        FieldKind.ADDRESS,
        FieldKind.PHONE,
        FieldKind.EMAIL,
        FieldKind.FREE_TEXT,
        FieldKind.UNKNOWN,
    }
)

_IDENTIFIER_KINDS = frozenset(
    {
        FieldKind.BIC,
        FieldKind.IBAN,
        FieldKind.ACCOUNT_NUMBER,
        FieldKind.SORT_CODE,
        FieldKind.PARTICIPANT_ID,
        FieldKind.LEI,
        FieldKind.ISIN,
        FieldKind.CFI,
    }
)


class Cardinality(StrEnum):
    """Whether one instruction holds one value or several for this field."""

    SINGLE = "SINGLE"
    MULTI = "MULTI"


class LayoutPattern(StrEnum):
    """The structural pattern a document or section follows.

    Determines which cutter Stage 4 applies. ``UNRECOGNISED`` is the only path
    that escalates to the adjudicator, and even then every guardrail still runs.
    """

    TABLE_ROW = "TABLE_ROW"
    SECTION_BLOCK = "SECTION_BLOCK"
    NARRATIVE = "NARRATIVE"
    FORM_KEY_VALUE = "FORM_KEY_VALUE"
    SWIFT_MESSAGE = "SWIFT_MESSAGE"
    AMENDMENT = "AMENDMENT"
    MULTI_COLUMN = "MULTI_COLUMN"
    UNRECOGNISED = "UNRECOGNISED"


class DescriptorSource(StrEnum):
    """How a descriptor came to be — audited, because it shapes everything after."""

    DETERMINISTIC = "DETERMINISTIC"
    SYNTHESISED = "SYNTHESISED"
    HINT_BIASED = "HINT_BIASED"
    MERGED = "MERGED"


def slugify(label: str) -> str:
    """Turn a document label into a stable snake_case field name.

    Accents are folded and non-alphanumerics collapse to underscores, so
    ``"Local Sub A/C"`` and ``"Local Sub A/C "`` produce the same name while
    remaining recognisable to a human reading the output.
    """
    normalised = unicodedata.normalize("NFKD", label)
    ascii_only = normalised.encode("ascii", "ignore").decode("ascii")
    collapsed = re.sub(r"[^0-9a-zA-Z]+", "_", ascii_only).strip("_").lower()
    collapsed = re.sub(r"_{2,}", "_", collapsed)
    if not collapsed:
        # A label of only non-ASCII characters (e.g. CJK) still needs a stable
        # name; hash the original so it is deterministic and collision-resistant.
        return f"field_{hashlib.sha256(label.encode('utf-8')).hexdigest()[:10]}"
    if collapsed[0].isdigit():
        collapsed = f"f_{collapsed}"
    return collapsed


class ValidatorHints(BaseModel):
    """Deterministic constraints for this field, gathered from the document itself."""

    model_config = ConfigDict(frozen=True)

    exact_length: int | None = Field(default=None, ge=1)
    min_length: int | None = Field(default=None, ge=0)
    max_length: int | None = Field(default=None, ge=1)
    regex: str | None = None
    enum_values: tuple[str, ...] = ()
    charset: str | None = Field(
        default=None, description="e.g. 'digits', 'upper_alnum', 'alnum', 'any'"
    )

    def merged_with(self, other: "ValidatorHints") -> "ValidatorHints":
        """Union two hint sets, keeping the *weaker* constraint on conflict.

        Two chunks may legitimately show different lengths for the same concept
        (a 10-digit account in one market, 12 in another). Keeping the stricter
        value would manufacture validation failures, so conflicts widen instead.
        """
        exact = self.exact_length if self.exact_length == other.exact_length else None
        lengths = [value for value in (self.min_length, other.min_length) if value is not None]
        maxes = [value for value in (self.max_length, other.max_length) if value is not None]
        return ValidatorHints(
            exact_length=exact,
            min_length=min(lengths) if lengths else None,
            max_length=max(maxes) if maxes else None,
            regex=self.regex if self.regex == other.regex else None,
            enum_values=tuple(sorted(set(self.enum_values) | set(other.enum_values))),
            charset=self.charset if self.charset == other.charset else "any",
        )


class FieldDescriptor(BaseModel):
    """One discovered leaf field."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Stable snake_case identity used in the output JSON.")
    label: str = Field(description="Verbatim label as it appeared in the document.")
    group_path: tuple[str, ...] = Field(
        default=(), description="Display group names, outermost first."
    )
    kind: FieldKind = FieldKind.UNKNOWN
    kind_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    cardinality: Cardinality = Cardinality.SINGLE
    hints: ValidatorHints = Field(default_factory=ValidatorHints)
    aliases: tuple[str, ...] = Field(
        default=(), description="Other labels seen for this same concept."
    )
    canonical_concept: str | None = Field(
        default=None,
        description="Ontology concept this maps to, when recognised. A hint only — "
        "an unmatched field is still emitted.",
    )
    pages: tuple[int, ...] = ()
    source_pattern: str | None = Field(
        default=None, description="Which harvester found it (table_header, key_value, swift_tag...)."
    )

    @property
    def path(self) -> str:
        """Dotted path used for audit entries and review-queue references."""
        return ".".join((*(slugify(part) for part in self.group_path), self.name))

    @property
    def is_pii(self) -> bool:
        return self.kind.is_pii

    def merged_with(self, other: "FieldDescriptor") -> "FieldDescriptor":
        """Union two descriptions of the same field seen in different chunks."""
        keep_kind = self.kind if self.kind_confidence >= other.kind_confidence else other.kind
        keep_confidence = max(self.kind_confidence, other.kind_confidence)
        if self.kind is not other.kind and FieldKind.UNKNOWN in (self.kind, other.kind):
            keep_kind = self.kind if other.kind is FieldKind.UNKNOWN else other.kind

        cardinality = (
            Cardinality.MULTI
            if Cardinality.MULTI in (self.cardinality, other.cardinality)
            else Cardinality.SINGLE
        )
        aliases = set(self.aliases) | set(other.aliases)
        if other.label != self.label:
            aliases.add(other.label)

        return self.model_copy(
            update={
                "kind": keep_kind,
                "kind_confidence": keep_confidence,
                "cardinality": cardinality,
                "hints": self.hints.merged_with(other.hints),
                "aliases": tuple(sorted(aliases)),
                "canonical_concept": self.canonical_concept or other.canonical_concept,
                "pages": tuple(sorted(set(self.pages) | set(other.pages))),
            }
        )


class RepeatingUnit(BaseModel):
    """What constitutes one settlement instruction in this document.

    Stage 7's chunk-level completeness check needs to know which kinds must be
    present for a unit to count as captured, and the chunker needs to know what
    it is cutting on. Both come from here rather than from a hard-coded rule.
    """

    model_config = ConfigDict(frozen=True)

    description: str = ""
    layout_pattern: LayoutPattern = LayoutPattern.UNRECOGNISED
    anchor_kinds: tuple[FieldKind, ...] = Field(
        default=(),
        description="Kinds whose presence marks a new instruction (e.g. COUNTRY, BIC).",
    )
    required_kinds: tuple[FieldKind, ...] = Field(
        default=(),
        description="Kinds that must appear for the unit to count as fully captured.",
    )


class SchemaDescriptor(BaseModel):
    """The discovered output shape for one document."""

    model_config = ConfigDict(frozen=True)

    document_id: str = ""
    descriptor_version: str = "1"
    source: DescriptorSource = DescriptorSource.DETERMINISTIC
    fields: tuple[FieldDescriptor, ...] = ()
    repeating_unit: RepeatingUnit = Field(default_factory=RepeatingUnit)
    notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _reject_duplicate_paths(self) -> Self:
        paths = [field.path for field in self.fields]
        duplicates = sorted({path for path in paths if paths.count(path) > 1})
        if duplicates:
            raise ValueError(f"duplicate field paths in descriptor: {', '.join(duplicates)}")
        return self

    def __iter__(self) -> Iterator[FieldDescriptor]:  # type: ignore[override]
        return iter(self.fields)

    def __len__(self) -> int:
        return len(self.fields)

    @property
    def is_empty(self) -> bool:
        return not self.fields

    def field_by_path(self, path: str) -> FieldDescriptor | None:
        return next((field for field in self.fields if field.path == path), None)

    def fields_of_kind(self, *kinds: FieldKind) -> tuple[FieldDescriptor, ...]:
        wanted = set(kinds)
        return tuple(field for field in self.fields if field.kind in wanted)

    def pii_fields(self) -> tuple[FieldDescriptor, ...]:
        return tuple(field for field in self.fields if field.is_pii)

    def group_tree(self) -> dict[str, Any]:
        """Nested view derived from ``group_path``, for assembly and prompting."""
        tree: dict[str, Any] = {}
        for field in self.fields:
            node = tree
            for part in field.group_path:
                node = node.setdefault(part, {})
            node[field.name] = field
        return tree

    @property
    def descriptor_hash(self) -> str:
        """Stable hash of the shape, recorded in G4 so an output can be traced to it."""
        payload = {
            "descriptor_version": self.descriptor_version,
            "repeating_unit": self.repeating_unit.model_dump(mode="json"),
            "fields": [
                {
                    "path": field.path,
                    "kind": field.kind.value,
                    "cardinality": field.cardinality.value,
                    "hints": field.hints.model_dump(mode="json"),
                }
                for field in sorted(self.fields, key=lambda item: item.path)
            ],
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def with_fields(self, fields: tuple[FieldDescriptor, ...]) -> "SchemaDescriptor":
        return self.model_copy(update={"fields": fields})

    def merged_with(self, other: "SchemaDescriptor") -> "SchemaDescriptor":
        """Union-merge two descriptors so the record array stays homogeneous.

        A field present in only one chunk survives the merge and is emitted
        ``NOT_APPLICABLE`` for instructions that do not state it — which is what
        keeps an amendment or a sparse row from looking like a failed extraction.
        """
        merged: dict[str, FieldDescriptor] = {field.path: field for field in self.fields}
        for field in other.fields:
            existing = merged.get(field.path)
            merged[field.path] = existing.merged_with(field) if existing else field

        unit = self.repeating_unit
        if unit.layout_pattern is LayoutPattern.UNRECOGNISED:
            unit = other.repeating_unit
        elif other.repeating_unit.layout_pattern not in (
            LayoutPattern.UNRECOGNISED,
            unit.layout_pattern,
        ):
            unit = unit.model_copy(
                update={
                    "anchor_kinds": tuple(
                        sorted(
                            set(unit.anchor_kinds) | set(other.repeating_unit.anchor_kinds),
                            key=lambda kind: kind.value,
                        )
                    ),
                    "required_kinds": tuple(
                        sorted(
                            set(unit.required_kinds) & set(other.repeating_unit.required_kinds),
                            key=lambda kind: kind.value,
                        )
                    ),
                }
            )

        return SchemaDescriptor(
            document_id=self.document_id or other.document_id,
            descriptor_version=self.descriptor_version,
            source=DescriptorSource.MERGED,
            fields=tuple(sorted(merged.values(), key=lambda field: field.path)),
            repeating_unit=unit,
            notes=tuple(dict.fromkeys((*self.notes, *other.notes))),
        )
