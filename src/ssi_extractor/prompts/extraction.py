"""Extraction prompt, rendered from the runtime schema descriptor.

The field catalogue is generated from the descriptor rather than written by hand,
which is what makes pointing the pipeline at a different document shape a
no-code change. The instructions themselves carry no institution-specific field
names, only rules about *how* to copy values.
"""

from __future__ import annotations

from ssi_extractor.schema.descriptor import FieldDescriptor, FieldKind, SchemaDescriptor

__all__ = ["render_field_catalogue", "render_system_prompt", "render_user_prompt"]

# Per-kind copying rules. These are format rules, not field semantics, so they
# stay valid for any institution's labels.
_KIND_RULES: dict[FieldKind, str] = {
    FieldKind.BIC: "8 or 11 characters, upper case. Copy every character exactly.",
    FieldKind.IBAN: "Copy every character; keep the full length. Do not reformat or space it.",
    FieldKind.ACCOUNT_NUMBER: (
        "Copy every digit exactly, including leading zeros. Never abbreviate, "
        "truncate, round, or summarise an account number."
    ),
    FieldKind.SORT_CODE: "Six digits; keep the document's separators if present.",
    FieldKind.PARTICIPANT_ID: "Short code; copy exactly as printed.",
    FieldKind.LEI: "20 characters; copy exactly.",
    FieldKind.ISIN: "12 characters; copy exactly.",
    FieldKind.CFI: "Six letters; copy exactly.",
    FieldKind.COUNTRY: "Copy the country as printed (name or code); do not translate it.",
    FieldKind.CURRENCY: "Three-letter currency code as printed.",
    FieldKind.DATE: (
        "Copy the date as an ISO 8601 date (YYYY-MM-DD) when the printed form is "
        "unambiguous; otherwise copy it verbatim."
    ),
    FieldKind.PERCENTAGE: "Include the percent sign.",
    FieldKind.ENUM: "Copy one of the values used in the document.",
    FieldKind.PERSON_NAME: "Copy the name as printed.",
    FieldKind.ORG_NAME: "Copy the full legal name as printed, including suffixes.",
    FieldKind.ADDRESS: "Copy the address as printed, on one line.",
    FieldKind.PHONE: "Copy the number as printed.",
    FieldKind.EMAIL: "Copy the address exactly.",
    FieldKind.FREE_TEXT: "Copy the text as printed.",
    FieldKind.UNKNOWN: "Copy the value as printed.",
}

_SYSTEM_PROMPT = """You extract settlement instruction data from one chunk of a banking document into a fixed JSON structure.

{untrusted_preamble}

How to fill each field:
1. Copy values verbatim from the chunk. Never invent, complete, correct, normalise or infer a value that is not present in the text.
2. If the chunk does not state a field, return it with an empty value, status NOT_APPLICABLE, confidence 0.0 and empty evidence. A missing value is a normal outcome, not a failure — do not guess to fill a gap.
3. Never drop a character. Identifiers such as account numbers, IBANs and BIC codes must retain every character, including leading zeros and trailing branch codes.
4. For every field you populate, put in "evidence" the exact substring of the chunk you took the value from, and in "page" the page number or numbers that substring came from. Evidence must be text that appears in the chunk.
5. Set "status" to VALIDATED only when you copied a value present in the chunk. Set FAILED when the chunk clearly intends a value but it is unreadable or contradictory. Otherwise use NOT_APPLICABLE.
6. Set "confidence" to how certain you are that the value belongs to that field, between 0.0 and 1.0. Deterministic checks run after you and will adjust it; do not inflate it.
7. Placeholders such as "-", "N/A" or "none" mean the field is not applicable. Return NOT_APPLICABLE with an empty value, not the placeholder text.
8. A value that plainly belongs to a different field than the one whose label carries it must still be reported under the label the document used. Do not reorganise the document.

Return only the JSON object required by the schema."""

_USER_PROMPT = """Document: {document_name}
Chunk {chunk_index} of {chunk_total} — layout: {layout_pattern} — page(s): {pages}
{amendment_note}
Fields to fill (path — kind — rule):
{field_catalogue}

{chunk_block}"""

_AMENDMENT_NOTE = (
    "This document is an amendment: it states only the fields that changed. "
    "Every field it does not mention must be NOT_APPLICABLE, never FAILED.\n"
)


def _describe(field: FieldDescriptor) -> str:
    rule = _KIND_RULES.get(field.kind, _KIND_RULES[FieldKind.UNKNOWN])
    hint_parts: list[str] = []
    if field.hints.exact_length:
        hint_parts.append(f"expected length {field.hints.exact_length}")
    if field.hints.enum_values:
        allowed = ", ".join(field.hints.enum_values[:8])
        hint_parts.append(f"values seen in this document: {allowed}")
    if field.aliases:
        hint_parts.append(f"also labelled: {', '.join(field.aliases[:3])}")
    hints = f" ({'; '.join(hint_parts)})" if hint_parts else ""
    label = " / ".join((*field.group_path, field.label))
    return f'- {field.path} — "{label}" — {field.kind.value} — {rule}{hints}'


def render_field_catalogue(descriptor: SchemaDescriptor) -> str:
    """Render the descriptor's fields as prompt lines."""
    return "\n".join(_describe(field) for field in descriptor.fields)


def render_system_prompt(untrusted_preamble: str) -> str:
    return _SYSTEM_PROMPT.format(untrusted_preamble=untrusted_preamble)


def render_user_prompt(
    *,
    descriptor: SchemaDescriptor,
    document_name: str,
    chunk_index: int,
    chunk_total: int,
    layout_pattern: str,
    pages: str,
    chunk_block: str,
    is_amendment: bool,
) -> str:
    return _USER_PROMPT.format(
        document_name=document_name,
        chunk_index=chunk_index,
        chunk_total=chunk_total,
        layout_pattern=layout_pattern,
        pages=pages,
        amendment_note=_AMENDMENT_NOTE if is_amendment else "",
        field_catalogue=render_field_catalogue(descriptor),
        chunk_block=chunk_block,
    )
