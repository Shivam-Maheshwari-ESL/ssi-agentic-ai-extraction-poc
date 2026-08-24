"""Stage 4 — SSI locator, layout classifier and chunker.

Three jobs, in order:

1. **Classify the structure** of each page or section: table rows, section blocks,
   a narrative letter, a key:value form, raw SWIFT text, or an amendment. The
   spec is explicit that "find the table" is not enough, because a document may
   be prose, a form, or a delta.
2. **Locate SSI content** and skip everything else — cover pages, email threads,
   disclaimers, headers, footers, logos — however much of the document they take
   up. Relevance is scored on *structural and value-kind* signals rather than
   English keywords, so a Spanish or German document locates just as well.
3. **Cut one chunk per instruction**, stitching across layers, regions and page
   breaks, and tagging every chunk with its contributing pages and boxes.

Only located chunks ever reach a model. The whole document never does.
"""

from __future__ import annotations

import re
from typing import Iterable

from pydantic import BaseModel, ConfigDict, Field

from ssi_extractor.observability.logging import get_logger
from ssi_extractor.schema.descriptor import FieldKind, LayoutPattern
from ssi_extractor.schema.kinds import infer_kind
from ssi_extractor.stages.page_text import DocumentText, PageText, TableGrid, TextLine
from ssi_extractor.utils.geometry import BBox, union_all
from ssi_extractor.utils.text import collapse_whitespace, fold_for_comparison

__all__ = ["InstructionChunk", "KeyValuePair", "LocateResult", "locate_and_chunk"]

_logger = get_logger(__name__)

# Kinds whose presence is evidence that a region carries settlement data. Chosen
# because they are identifiers no cover page or disclaimer would contain.
_SSI_EVIDENCE_KINDS = frozenset(
    {
        FieldKind.BIC,
        FieldKind.IBAN,
        FieldKind.ACCOUNT_NUMBER,
        FieldKind.SORT_CODE,
        FieldKind.PARTICIPANT_ID,
        FieldKind.ISIN,
        FieldKind.LEI,
    }
)

# Weak evidence: common in SSI content but also in ordinary prose.
_SUPPORTING_KINDS = frozenset({FieldKind.COUNTRY, FieldKind.CURRENCY, FieldKind.DATE})

_KEY_VALUE_RE = re.compile(r"^\s*(?P<key>[^:：]{2,60})[:：]\s*(?P<value>.+?)\s*$")
_SWIFT_TAG_RE = re.compile(r":\d{2}[A-Z]?:")

# Amendment vocabulary, deliberately multilingual and configurable. An amendment
# states only what changed, so its unstated fields must end up NOT_APPLICABLE
# rather than FAILED — mistaking one for a failed full extraction would be a
# false negative on every field the document never intended to restate.
_AMENDMENT_TOKENS = frozenset(
    {
        "amendment", "amend", "amended", "addendum", "supersedes", "superseding",
        "revision", "revised", "change", "changes", "changed", "update", "updated",
        "modification", "modificacion", "enmienda", "anderung", "aenderung",
        "modifica", "aggiornamento", "alteracao",
    }
)

# Front matter that is never settlement content. Structural checks do the real
# work; these tokens only add confidence, and an unmatched language simply relies
# on the structural signal instead.
_FRONT_MATTER_TOKENS = frozenset(
    {
        "disclaimer", "confidential", "confidentiality", "unsubscribe", "sent from",
        "original message", "forwarded message", "kind regards", "best regards",
        "privacy notice", "table of contents", "page", "aviso legal",
    }
)

_EMAIL_HEADER_RE = re.compile(r"^\s*(from|to|cc|bcc|sent|subject|date)\s*:", re.IGNORECASE)


class KeyValuePair(BaseModel):
    """One label/value pair located in a form or narrative chunk."""

    model_config = ConfigDict(frozen=True)

    key: str
    value: str
    page: int
    bbox: BBox
    section: str = ""


class InstructionChunk(BaseModel):
    """One settlement instruction's worth of text, ready for extraction.

    A chunk is the unit that fixes ``instructionCount``: the chunker decides how
    many instructions a document holds, not the model, which is what keeps a
    13-row table from yielding 11 records.
    """

    model_config = ConfigDict(frozen=True)

    index: int
    layout_pattern: LayoutPattern
    text: str
    pages: tuple[int, ...]
    bboxes: tuple[BBox, ...] = ()
    header_context: tuple[str, ...] = Field(
        default=(), description="Column headers or section titles that give the values meaning."
    )
    cells: tuple[tuple[str, str], ...] = Field(
        default=(), description="(header, value) pairs for tabular chunks."
    )
    key_values: tuple[KeyValuePair, ...] = ()
    ocr_confidence: float = 1.0
    is_amendment: bool = False
    stitched_across_pages: bool = False
    relevance_score: float = 0.0
    evidence_kinds: tuple[FieldKind, ...] = ()

    @property
    def page_label(self) -> str:
        return ", ".join(str(page) for page in self.pages)


class LocateResult(BaseModel):
    """Everything Stage 4 produced, including what it deliberately skipped."""

    model_config = ConfigDict(frozen=True)

    chunks: tuple[InstructionChunk, ...] = ()
    dominant_pattern: LayoutPattern = LayoutPattern.UNRECOGNISED
    skipped_regions: tuple[str, ...] = ()
    unrecognised_pages: tuple[int, ...] = ()
    is_amendment_document: bool = False


def _kinds_in(values: Iterable[str]) -> set[FieldKind]:
    """Infer the kinds present in a set of values, one value at a time.

    Judged individually rather than as a column, because a chunk's values are
    heterogeneous and a single BIC is meaningful evidence on its own.
    """
    found: set[FieldKind] = set()
    for value in values:
        text = collapse_whitespace(value)
        if not text:
            continue
        for token in re.split(r"[\s,;|]+", text):
            if len(token) < 2:
                continue
            inference = infer_kind([token])
            if inference.kind in _SSI_EVIDENCE_KINDS or inference.kind in _SUPPORTING_KINDS:
                found.add(inference.kind)
    return found


def _relevance(text: str, kinds: set[FieldKind]) -> float:
    """Score how likely a region carries settlement data.

    Strong-kind identifiers dominate; front-matter shape subtracts. The score is
    kept explicit so a skipped region can be explained rather than silently lost.
    """
    strong = len(kinds & _SSI_EVIDENCE_KINDS)
    supporting = len(kinds & _SUPPORTING_KINDS)
    score = min(1.0, 0.4 * strong + 0.12 * supporting)

    folded = fold_for_comparison(text)
    penalties = sum(1 for token in _FRONT_MATTER_TOKENS if fold_for_comparison(token) in folded)
    if _EMAIL_HEADER_RE.search(text):
        penalties += 2
    return max(0.0, score - 0.12 * penalties)


def _is_amendment(text: str) -> bool:
    folded = {fold_for_comparison(token) for token in re.split(r"[^\w]+", text) if token}
    return bool(folded & {fold_for_comparison(token) for token in _AMENDMENT_TOKENS})


def _header_paths(grid: TableGrid) -> list[tuple[str, ...]]:
    """Turn possibly-merged header rows into one path per column.

    A two-band header ("BENEFICIARY" over "Account Name") becomes
    ``("BENEFICIARY", "Account Name")``, which is what gives the discovered schema
    its grouping without anyone naming the groups in advance. Blank cells inherit
    the last non-blank value to the left, which is how merged cells appear once
    extracted.
    """
    if not grid.header_rows:
        return [() for _ in range(grid.column_count)]

    filled_rows: list[list[str]] = []
    for row in grid.header_rows:
        filled: list[str] = []
        last = ""
        for index in range(grid.column_count):
            cell = row[index] if index < len(row) else ""
            if cell:
                last = cell
            filled.append(cell or last)
        filled_rows.append(filled)

    paths: list[tuple[str, ...]] = []
    for column in range(grid.column_count):
        parts: list[str] = []
        for row in filled_rows:
            value = row[column]
            if value and (not parts or parts[-1] != value):
                parts.append(value)
        paths.append(tuple(parts))
    return paths


def _unique_headers(headers: list[str]) -> list[str]:
    """Make repeated column headers distinct.

    Wide institutional tables legitimately repeat a label under different group
    bands — a "Swift Address" column for the global custodian and another for the
    local custodian. Left identical, the two columns would collapse into one
    discovered field and an entire column of values would be lost, so repeats are
    numbered while the first occurrence keeps the document's own wording.
    """
    seen: dict[str, int] = {}
    result: list[str] = []
    for header in headers:
        count = seen.get(header, 0) + 1
        seen[header] = count
        result.append(header if count == 1 else f"{header} ({count})")
    return result


def _table_chunks(
    grid: TableGrid,
    page: PageText,
    start_index: int,
) -> list[InstructionChunk]:
    """One chunk per body row, carrying its header context."""
    paths = _header_paths(grid)
    headers = _unique_headers(
        [" / ".join(path) if path else f"column_{index + 1}" for index, path in enumerate(paths)]
    )
    chunks: list[InstructionChunk] = []

    for offset, row in enumerate(grid.rows):
        cells = [
            (headers[index] if index < len(headers) else f"column_{index + 1}", value)
            for index, value in enumerate(row)
        ]
        populated = [(header, value) for header, value in cells if value.strip()]
        if len(populated) < 2:
            # A single-cell row is a section banner inside the table
            # ("Securities non Eligible in T2S"), not an instruction.
            continue

        text_lines = [f"{header}: {value}" for header, value in populated]
        row_text = "\n".join(text_lines)
        kinds = _kinds_in(value for _, value in populated)
        score = _relevance(row_text, kinds)

        chunks.append(
            InstructionChunk(
                index=start_index + len(chunks),
                layout_pattern=LayoutPattern.TABLE_ROW,
                text=row_text,
                pages=(grid.page,),
                bboxes=(grid.bbox,),
                header_context=tuple(headers),
                cells=tuple(populated),
                ocr_confidence=page.mean_ocr_confidence if page.has_scanned_content else 1.0,
                is_amendment=_is_amendment(row_text),
                relevance_score=score,
                evidence_kinds=tuple(sorted(kinds, key=lambda kind: kind.value)),
            )
        )
    return chunks


def _section_of_lines(lines: Iterable[TextLine]) -> list[tuple[str, TextLine]]:
    """Tag each line with the section heading in force above it.

    Used to keep a chunk's text verbatim while still knowing which block each line
    belongs to. A heading is a short, digit-free, upper-case line with no value
    beside it — structural, so it holds for any language.
    """
    tagged: list[tuple[str, TextLine]] = []
    section = ""
    for line in lines:
        text_value = line.text.strip()
        if not text_value:
            continue
        if _is_section_heading(text_value):
            section = text_value
            # The heading itself stays in the chunk text. A heading is frequently
            # the value of a field rather than decoration — a "GBP" band over a
            # cash block states the settlement currency — so dropping it would
            # delete data the document plainly provides.
            tagged.append((section, line))
            continue
        tagged.append((section, line))
    return tagged


def _is_section_heading(text_value: str) -> bool:
    if len(text_value) > 30 or ":" in text_value or len(text_value.split()) > 3:
        return False
    if any(character.isdigit() for character in text_value):
        return False
    return text_value.upper() == text_value and any(character.isalpha() for character in text_value)


def _key_value_pairs(lines: Iterable[TextLine]) -> list[KeyValuePair]:
    """Extract label/value pairs from both punctuated and column-aligned lines."""
    pairs: list[KeyValuePair] = []
    section = ""

    for line in lines:
        text = line.text.strip()
        if not text:
            continue

        # An all-caps short line with no value is a section heading (CASH,
        # CUSTODY), which becomes the group for the pairs that follow it.
        if _is_section_heading(text):
            section = text
            continue

        match = _KEY_VALUE_RE.match(text)
        if match:
            key, value = match.group("key").strip(), match.group("value").strip()
            if key and value:
                pairs.append(
                    KeyValuePair(key=key, value=value, page=line.page, bbox=line.bbox, section=section)
                )
                continue

        # Column-aligned form: "Account Name    Vida Bank Limited".
        if len(line.cells) == 2:
            key, value = line.cells[0].strip(), line.cells[1].strip()
            if key and value and len(key) <= 40:
                pairs.append(
                    KeyValuePair(key=key, value=value, page=line.page, bbox=line.bbox, section=section)
                )
    return pairs


def _form_chunks(page: PageText, start_index: int) -> list[InstructionChunk]:
    """A key:value form or narrative page becomes one chunk per coherent block.

    Sections are honoured when present (a CASH block and a CUSTODY block are
    different instruction contexts); otherwise the page is one instruction.
    """
    pairs = _key_value_pairs(page.lines)
    if not pairs:
        return []

    sections: dict[str, list[KeyValuePair]] = {}
    for pair in pairs:
        sections.setdefault(pair.section, []).append(pair)

    # Sections that hold no identifier of their own are context for the document,
    # not separate instructions, so they merge into a single chunk.
    strong_sections = {
        name: items
        for name, items in sections.items()
        if _kinds_in(item.value for item in items) & _SSI_EVIDENCE_KINDS
    }
    grouped = strong_sections or {"": pairs}

    # Several sections describe *one* instruction unless they repeat the same
    # labels. A cash block and a custody block state different things about the
    # same account and belong together; two blocks both stating "Account Name" and
    # "BIC" are two instructions. Deciding on label repetition rather than on
    # section count keeps this true for any document that groups by topic.
    if len(grouped) > 1:
        label_sets = [
            {collapse_whitespace(item.key).lower() for item in items} for items in grouped.values()
        ]
        shared = set.intersection(*label_sets) if label_sets else set()
        smallest = min(len(labels) for labels in label_sets)
        if smallest and len(shared) < max(1, smallest // 2):
            grouped = {"": pairs}

    chunks: list[InstructionChunk] = []
    shared = [pair for pair in pairs if pair.section not in grouped]
    tagged_lines = _section_of_lines(page.lines)

    for name, items in grouped.items():
        members = items + [pair for pair in shared if pair not in items]
        # The chunk carries the region's *verbatim* lines, not the parsed pairs.
        # Rebuilding the text from pairs silently discarded every line the parser
        # could not split — and OCR routinely returns a label and its value as one
        # box — so fields that were plainly on the page never reached the model.
        # Parsed pairs stay attached separately for schema harvesting.
        sections_included = {name} | {pair.section for pair in members}
        line_texts = [
            line.text
            for section, line in tagged_lines
            if not name or section in sections_included or not section
        ]
        text = "\n".join(line_texts) if line_texts else "\n".join(
            f"{pair.key}: {pair.value}" for pair in members
        )
        kinds = _kinds_in(pair.value for pair in members)
        boxes = union_all([pair.bbox for pair in members])
        chunks.append(
            InstructionChunk(
                index=start_index + len(chunks),
                layout_pattern=LayoutPattern.FORM_KEY_VALUE,
                text=text,
                pages=(page.page,),
                bboxes=(boxes,) if boxes else (),
                header_context=(name,) if name else (),
                key_values=tuple(members),
                ocr_confidence=page.mean_ocr_confidence if page.has_scanned_content else 1.0,
                is_amendment=_is_amendment(page.text),
                relevance_score=_relevance(text, kinds),
                evidence_kinds=tuple(sorted(kinds, key=lambda kind: kind.value)),
            )
        )
    return chunks


def _narrative_chunks(page: PageText, start_index: int) -> list[InstructionChunk]:
    """Prose: cut on blank-line paragraph boundaries, keep only SSI-bearing blocks."""
    paragraphs: list[list[TextLine]] = [[]]
    previous_bottom: float | None = None

    for line in page.lines:
        if previous_bottom is not None and (line.bbox.y0 - previous_bottom) > line.bbox.height * 1.4:
            paragraphs.append([])
        paragraphs[-1].append(line)
        previous_bottom = line.bbox.y1

    chunks: list[InstructionChunk] = []
    for block in paragraphs:
        if not block:
            continue
        text = "\n".join(line.text for line in block)
        kinds = _kinds_in([text])
        if not (kinds & _SSI_EVIDENCE_KINDS):
            continue
        boxes = union_all([line.bbox for line in block])
        pattern = (
            LayoutPattern.SWIFT_MESSAGE if _SWIFT_TAG_RE.search(text) else LayoutPattern.NARRATIVE
        )
        chunks.append(
            InstructionChunk(
                index=start_index + len(chunks),
                layout_pattern=pattern,
                text=text,
                pages=(page.page,),
                bboxes=(boxes,) if boxes else (),
                ocr_confidence=page.mean_ocr_confidence if page.has_scanned_content else 1.0,
                is_amendment=_is_amendment(text),
                relevance_score=_relevance(text, kinds),
                evidence_kinds=tuple(sorted(kinds, key=lambda kind: kind.value)),
            )
        )
    return chunks


def _classify_page_pattern(page: PageText) -> LayoutPattern:
    """Decide which cutter a page needs."""
    if any(grid.has_body and grid.column_count >= 3 for grid in page.tables):
        return LayoutPattern.TABLE_ROW
    if _SWIFT_TAG_RE.search(page.text):
        return LayoutPattern.SWIFT_MESSAGE

    pairs = _key_value_pairs(page.lines)
    populated_lines = [line for line in page.lines if line.text.strip()]
    if pairs and len(pairs) >= max(3, len(populated_lines) // 4):
        return LayoutPattern.FORM_KEY_VALUE

    if any(grid.has_body for grid in page.tables):
        return LayoutPattern.SECTION_BLOCK
    if populated_lines:
        return LayoutPattern.NARRATIVE
    return LayoutPattern.UNRECOGNISED


def _stitch_page_breaks(chunks: list[InstructionChunk]) -> list[InstructionChunk]:
    """Merge a table row split by a page break into one instruction.

    The signature is a last row on page N whose leading columns are populated and
    a first row on page N+1 that continues it with the leading columns empty. Both
    contributing page numbers are kept, because the citation must show both.
    """
    if len(chunks) < 2:
        return chunks

    merged: list[InstructionChunk] = []
    index = 0
    while index < len(chunks):
        current = chunks[index]
        if index + 1 < len(chunks):
            following = chunks[index + 1]
            same_shape = current.header_context == following.header_context
            crosses_page = bool(set(following.pages) - set(current.pages))
            continuation = same_shape and crosses_page and _is_continuation(current, following)
            if continuation:
                combined_cells = tuple(list(current.cells) + list(following.cells))
                current = current.model_copy(
                    update={
                        "text": f"{current.text}\n{following.text}",
                        "pages": tuple(sorted(set(current.pages) | set(following.pages))),
                        "bboxes": tuple(list(current.bboxes) + list(following.bboxes)),
                        "cells": combined_cells,
                        "stitched_across_pages": True,
                        "evidence_kinds": tuple(
                            sorted(
                                set(current.evidence_kinds) | set(following.evidence_kinds),
                                key=lambda kind: kind.value,
                            )
                        ),
                        "relevance_score": max(current.relevance_score, following.relevance_score),
                        "ocr_confidence": min(current.ocr_confidence, following.ocr_confidence),
                    }
                )
                index += 2
                merged.append(current)
                continue
        merged.append(current)
        index += 1

    return [chunk.model_copy(update={"index": position}) for position, chunk in enumerate(merged)]


def _is_continuation(first: InstructionChunk, second: InstructionChunk) -> bool:
    """Whether ``second`` continues ``first`` rather than starting a new instruction.

    A continuation row lacks the identifying leading values (the market/country
    column) that every new instruction carries, and adds values for headers the
    first row left empty.
    """
    first_headers = {header for header, value in first.cells if value.strip()}
    second_headers = {header for header, value in second.cells if value.strip()}
    if not second_headers or second_headers & first_headers:
        return False

    leading = first.header_context[0] if first.header_context else None
    if leading and leading in first_headers and leading not in second_headers:
        return True
    return len(second_headers) < len(first_headers) / 2


def locate_and_chunk(
    document_text: DocumentText,
    *,
    relevance_floor: float = 0.2,
) -> LocateResult:
    """Locate SSI regions and cut them into one chunk per instruction."""
    chunks: list[InstructionChunk] = []
    skipped: list[str] = []
    unrecognised: list[int] = []
    patterns: list[LayoutPattern] = []

    for page in document_text.pages:
        pattern = _classify_page_pattern(page)
        patterns.append(pattern)
        before = len(chunks)

        if pattern is LayoutPattern.TABLE_ROW:
            for grid in page.tables:
                if grid.has_body and grid.column_count >= 3:
                    chunks.extend(_table_chunks(grid, page, len(chunks)))
        elif pattern is LayoutPattern.FORM_KEY_VALUE:
            chunks.extend(_form_chunks(page, len(chunks)))
        elif pattern in (LayoutPattern.NARRATIVE, LayoutPattern.SWIFT_MESSAGE):
            chunks.extend(_narrative_chunks(page, len(chunks)))
        elif pattern is LayoutPattern.SECTION_BLOCK:
            for grid in page.tables:
                if grid.has_body:
                    chunks.extend(_table_chunks(grid, page, len(chunks)))
            if len(chunks) == before:
                chunks.extend(_form_chunks(page, len(chunks)))
        else:
            unrecognised.append(page.page)

        if len(chunks) == before and page.text.strip():
            skipped.append(f"page {page.page}: no SSI-bearing region located ({pattern.value})")

    relevant: list[InstructionChunk] = []
    for chunk in chunks:
        if chunk.relevance_score < relevance_floor:
            skipped.append(
                f"page {chunk.page_label}: region skipped, relevance {chunk.relevance_score:.2f} "
                f"below floor {relevance_floor:.2f}"
            )
            continue
        relevant.append(chunk)

    stitched = _stitch_page_breaks(relevant)
    dominant = max(set(patterns), key=patterns.count) if patterns else LayoutPattern.UNRECOGNISED
    is_amendment_document = bool(stitched) and all(chunk.is_amendment for chunk in stitched)

    _logger.info(
        "Stage 4 located %s instruction chunk(s); pattern=%s; skipped %s region(s).",
        len(stitched),
        dominant.value,
        len(skipped),
    )
    return LocateResult(
        chunks=tuple(stitched),
        dominant_pattern=dominant,
        skipped_regions=tuple(skipped),
        unrecognised_pages=tuple(unrecognised),
        is_amendment_document=is_amendment_document,
    )
