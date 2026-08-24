"""Unified page text, word geometry and table reconstruction.

Everything after this point — layout classification, chunking, harvesting,
evidence resolution — works against one representation regardless of whether the
characters came from the PDF's own text layer or from OCR. That is what lets a
single logical instruction be stitched out of a native region and a scanned
region on the same page without special-casing either.

Table reconstruction is geometric, not delimiter- or ruling-based. Real
institutional SSI tables are frequently whitespace-aligned with no drawn borders
(the sample Inversis document has none, and PyMuPDF's ruling-based finder returns
nothing for it), and OCR output never has borders at all. Columns are therefore
found as vertical whitespace corridors that persist across rows, which works
identically for native and recognised text.

Native lines carry a confidence of 1.0 by construction: their characters are
exact. Penalising them for sharing a page with a scan would drag a mixed row's
confidence down for no reason.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from pydantic import BaseModel, ConfigDict, Field

from ssi_extractor.observability.logging import get_logger
from ssi_extractor.stages.classify import DocumentComposition, LayerKind
from ssi_extractor.stages.ocr import OcrEngine, OcrResult, recognise_region
from ssi_extractor.utils.geometry import BBox, union_all
from ssi_extractor.utils.text import collapse_whitespace

__all__ = [
    "DocumentText",
    "PageText",
    "TableGrid",
    "TextLine",
    "TextSource",
    "Word",
    "build_document_text",
]

_logger = get_logger(__name__)

# A gap wider than this multiple of the page's typical word gap separates columns.
_COLUMN_GAP_RATIO = 2.2

# Minimum width, in points, of a whitespace corridor that counts as a column
# boundary. Narrower than this and ordinary inter-word spacing would split cells.
_MIN_CORRIDOR_WIDTH = 5.0

# Minimum height of a horizontal corridor that separates two table rows. Smaller
# than inter-line leading, so wrapped cell lines split and are re-merged later
# by continuation detection rather than being fused into their neighbours.
_MIN_ROW_CORRIDOR = 2.0

# A page needs at least this many text fragments before table reconstruction is
# attempted; below it, the page is prose or a short form, not a table.
_MIN_TABLE_FRAGMENTS = 12


class Word(BaseModel):
    """One token with its box and its own confidence."""

    model_config = ConfigDict(frozen=True)

    text: str
    bbox: BBox
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class TextSource(BaseModel):
    """Where a line's characters came from — needed for confidence and citation."""

    model_config = ConfigDict(frozen=True)

    layer: LayerKind
    ocr_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    ocr_attempts: int = 0
    enhancement_operations: tuple[str, ...] = ()

    @property
    def is_native(self) -> bool:
        return self.layer is LayerKind.NATIVE_TEXT


class TextLine(BaseModel):
    """One line of text with its page, box, words and provenance."""

    model_config = ConfigDict(frozen=True)

    page: int
    text: str
    bbox: BBox
    source: TextSource
    words: tuple[Word, ...] = ()
    cells: tuple[str, ...] = Field(
        default=(),
        description="Column-split segments, when wide gaps indicate tabular structure.",
    )


class TableGrid(BaseModel):
    """A reconstructed table: header rows plus body rows, each row keeping its box."""

    model_config = ConfigDict(frozen=True)

    page: int
    bbox: BBox
    header_rows: tuple[tuple[str, ...], ...] = ()
    rows: tuple[tuple[str, ...], ...] = ()
    row_boxes: tuple[BBox, ...] = ()
    column_count: int = 0
    column_bounds: tuple[tuple[float, float], ...] = ()

    @property
    def has_body(self) -> bool:
        return bool(self.rows)


class PageText(BaseModel):
    """All text available for one page, from every layer."""

    model_config = ConfigDict(frozen=True)

    page: int
    lines: tuple[TextLine, ...] = ()
    tables: tuple[TableGrid, ...] = ()
    ocr_results: tuple[OcrResult, ...] = ()
    width: float = 0.0
    height: float = 0.0

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    @property
    def mean_ocr_confidence(self) -> float:
        scored = [line.source.ocr_confidence for line in self.lines]
        return round(sum(scored) / len(scored), 4) if scored else 0.0

    @property
    def has_scanned_content(self) -> bool:
        return any(not line.source.is_native for line in self.lines)


class DocumentText(BaseModel):
    """Text for the whole document, page by page."""

    model_config = ConfigDict(frozen=True)

    pages: tuple[PageText, ...] = ()

    def page(self, number: int) -> PageText | None:
        return next((page for page in self.pages if page.page == number), None)

    @property
    def text(self) -> str:
        return "\n".join(page.text for page in self.pages)


# ---------------------------------------------------------------------------
# Word collection
# ---------------------------------------------------------------------------

def _native_word_lines(page: Any) -> list[list[Word]]:
    """Words grouped into lines from the PDF text layer."""
    grouped: list[list[Word]] = []
    data = page.get_text("words")
    if not data:
        return grouped

    # get_text("words") yields (x0, y0, x1, y1, word, block, line, word_no); the
    # block/line indices give exact line membership without geometric guessing.
    by_line: dict[tuple[int, int], list[Word]] = {}
    for entry in data:
        x0, y0, x1, y1, text, block_no, line_no = entry[0], entry[1], entry[2], entry[3], entry[4], entry[5], entry[6]
        if not str(text).strip():
            continue
        by_line.setdefault((int(block_no), int(line_no)), []).append(
            Word(text=collapse_whitespace(str(text)), bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y1))
        )

    for key in sorted(by_line, key=lambda item: (by_line[item][0].bbox.y0, item)):
        words = sorted(by_line[key], key=lambda word: word.bbox.x0)
        grouped.append(words)
    return grouped


def _ocr_word_lines(result: OcrResult) -> list[list[Word]]:
    """Words grouped into lines from an OCR result, by vertical overlap."""
    if result.is_empty:
        return []

    lines: list[list[Word]] = []
    for recognised in sorted(result.words, key=lambda word: (word.bbox.y0, word.bbox.x0)):
        word = Word(text=recognised.text, bbox=recognised.bbox, confidence=recognised.confidence)
        for line in lines:
            if line[0].bbox.vertical_overlap(word.bbox) >= 0.45:
                line.append(word)
                break
        else:
            lines.append([word])

    for line in lines:
        line.sort(key=lambda word: word.bbox.x0)
    return lines


def _cluster_rows(word_lines: Sequence[Sequence[Word]]) -> list[list[Word]]:
    """Group word fragments into visual rows by vertical overlap.

    Line membership cannot be taken from the PDF's own block/line indices: many
    generators emit every table cell as its own text block, so a 13-row table
    arrives as 200 single-cell "lines". OCR has no line structure at all. Both
    become rows the same way here — by vertical overlap — which is also what makes
    a native region and a scanned region on the same page line up into one row.
    """
    fragments = [list(line) for line in word_lines if line]
    if not fragments:
        return []

    fragments.sort(key=lambda words: min(word.bbox.y0 for word in words))
    rows: list[list[Word]] = []
    row_boxes: list[BBox] = []

    for words in fragments:
        box = union_all([word.bbox for word in words])
        if box is None:
            continue
        for index, existing in enumerate(row_boxes):
            if existing.vertical_overlap(box) >= 0.45:
                rows[index].extend(words)
                row_boxes[index] = existing.union(box)
                break
        else:
            rows.append(list(words))
            row_boxes.append(box)

    for row in rows:
        row.sort(key=lambda word: word.bbox.x0)
    order = sorted(range(len(rows)), key=lambda index: row_boxes[index].y0)
    return [rows[index] for index in order]


# ---------------------------------------------------------------------------
# Column reconstruction
# ---------------------------------------------------------------------------

def _typical_word_gap(word_lines: Sequence[Sequence[Word]]) -> float:
    """The page's ordinary inter-word gap, used as the unit for column detection."""
    gaps: list[float] = []
    for line in word_lines:
        for left, right in zip(line, line[1:]):
            gap = right.bbox.x0 - left.bbox.x1
            if 0 < gap < 40:
                gaps.append(gap)
    if not gaps:
        return 4.0
    gaps.sort()
    return max(1.5, gaps[len(gaps) // 4])


def _ink_bands(
    intervals: Sequence[tuple[float, float]],
    *,
    min_corridor: float,
    step: float = 1.0,
) -> list[tuple[float, float]]:
    """Find ink bands along one axis, separated by corridors of empty space.

    Used for both axes: vertical corridors give columns, horizontal corridors give
    row bands. Aggregating every fragment's extent means a corridor only survives
    if it is empty across the whole table, which is what distinguishes a real
    column or row boundary from a coincidental gap in a single line.
    """
    if not intervals:
        return []

    low = min(start for start, _ in intervals)
    high = max(end for _, end in intervals)
    if high - low <= 0:
        return []

    slots = int((high - low) / step) + 2
    occupied = bytearray(slots)
    for start, end in intervals:
        first = max(0, int((start - low) / step))
        last = min(slots, int((end - low) / step) + 1)
        for index in range(first, last):
            occupied[index] = 1

    corridor_slots = max(1, int(min_corridor / step))
    bands: list[tuple[float, float]] = []
    band_start: int | None = None
    empty_run = 0

    for index, is_occupied in enumerate(occupied):
        if is_occupied:
            if band_start is None:
                band_start = index
            empty_run = 0
        else:
            empty_run += 1
            if band_start is not None and empty_run >= corridor_slots:
                bands.append((low + band_start * step, low + (index - empty_run + 1) * step))
                band_start = None

    if band_start is not None:
        bands.append((low + band_start * step, high))
    return bands


def _column_bounds(
    word_groups: Sequence[Sequence[Word]],
    bands: Sequence[tuple[float, float]] | None = None,
) -> list[tuple[float, float]]:
    """Column extents, from a projection profile counted in *rows*, not fragments.

    Requiring a corridor to be completely empty does not survive real documents: a
    single full-width banner, letterhead or title line covers every corridor and
    the table becomes invisible. Counting how many distinct rows have ink at each
    x position fixes that — a column boundary is where only a handful of rows have
    anything, so one banner cannot erase it while a column populated across the
    table always survives.
    """
    fragments = [group for group in word_groups if group]
    if not fragments:
        return []

    if bands is None:
        bands = _row_bands(fragments)
    if not bands:
        return []

    words = [word for group in fragments for word in group]
    low = min(word.bbox.x0 for word in words)
    high = max(word.bbox.x1 for word in words)
    if high - low <= 0:
        return []

    step = 1.0
    slots = int((high - low) / step) + 2
    counts = [0] * slots

    for band_start, band_end in bands:
        band_slots = bytearray(slots)
        for word in words:
            if not (band_start <= word.bbox.centre_y <= band_end):
                continue
            first = max(0, int((word.bbox.x0 - low) / step))
            last = min(slots, int((word.bbox.x1 - low) / step) + 1)
            for index in range(first, last):
                band_slots[index] = 1
        for index, occupied in enumerate(band_slots):
            if occupied:
                counts[index] += 1

    # A column must carry content in at least this many rows to be a column rather
    # than an artefact of one banner or footnote.
    threshold = max(2, int(round(0.15 * len(bands))))
    corridor_slots = max(1, int(_MIN_CORRIDOR_WIDTH / step))

    bounds: list[tuple[float, float]] = []
    band_start_index: int | None = None
    empty_run = 0
    for index, count in enumerate(counts):
        if count >= threshold:
            if band_start_index is None:
                band_start_index = index
            empty_run = 0
        else:
            empty_run += 1
            if band_start_index is not None and empty_run >= corridor_slots:
                bounds.append(
                    (low + band_start_index * step, low + (index - empty_run + 1) * step)
                )
                band_start_index = None
    if band_start_index is not None:
        bounds.append((low + band_start_index * step, high))
    return bounds


def _row_bands(word_groups: Sequence[Sequence[Word]]) -> list[tuple[float, float]]:
    """Row extents, from horizontal whitespace corridors between text fragments."""
    intervals = [(word.bbox.y0, word.bbox.y1) for group in word_groups for word in group]
    return _ink_bands(intervals, min_corridor=_MIN_ROW_CORRIDOR)


def _assign_columns(
    line: Sequence[Word], bounds: Sequence[tuple[float, float]]
) -> tuple[str, ...]:
    """Place each word in the column its horizontal centre falls in.

    Centre rather than left edge, so a value that starts slightly left of its
    column (common with right-aligned numbers) still lands in the right cell.
    """
    if not bounds:
        return tuple(word.text for word in line)

    buckets: list[list[str]] = [[] for _ in bounds]
    for word in line:
        centre = (word.bbox.x0 + word.bbox.x1) / 2
        best_index = min(
            range(len(bounds)),
            key=lambda index: 0.0
            if bounds[index][0] <= centre <= bounds[index][1]
            else min(abs(centre - bounds[index][0]), abs(centre - bounds[index][1])),
        )
        buckets[best_index].append(word.text)
    return tuple(" ".join(bucket).strip() for bucket in buckets)


def _split_cells(line: Sequence[Word], typical_gap: float) -> tuple[str, ...]:
    """Split one line on gaps that are wide relative to the page's word spacing."""
    if len(line) < 2:
        return tuple(word.text for word in line)

    threshold = max(_MIN_CORRIDOR_WIDTH, typical_gap * _COLUMN_GAP_RATIO)
    cells: list[str] = []
    current = [line[0].text]
    for left, right in zip(line, line[1:]):
        if (right.bbox.x0 - left.bbox.x1) >= threshold:
            cells.append(" ".join(current))
            current = [right.text]
        else:
            current.append(right.text)
    cells.append(" ".join(current))
    return tuple(cell.strip() for cell in cells if cell.strip())


def _is_label_row(row: Sequence[str]) -> bool:
    """Whether a row reads as column labels rather than data or letterhead.

    Structural only: labels are short, carry no identifiers or digits, and several
    of them sit side by side. No known column name is matched, because the next
    institution will use different ones.
    """
    populated = [cell.strip() for cell in row if cell.strip()]
    if len(populated) < 3:
        return False
    if any(len(cell) > 40 for cell in populated):
        return False
    if any(character.isdigit() for cell in populated for character in cell):
        return False
    if any("@" in cell for cell in populated):
        return False
    return True


def _find_header_span(rows: Sequence[tuple[str, ...]]) -> tuple[int, int]:
    """Locate the header band as ``(first_header_row, first_data_row)``.

    Works downwards to the first row that carries data — a row with identifiers or
    digits in it — then upwards over every contiguous label-like row above it. That
    captures a multi-band header ("BENEFICIARY" spanning four columns above
    "Account Name", "Bic Code", ...) without assuming how many bands there are,
    and leaves anything above the header as page front matter to be dropped.
    """
    def carries_data(row: Sequence[str]) -> bool:
        populated = [cell.strip() for cell in row if cell.strip()]
        if len(populated) < 3:
            return False
        return any(any(character.isdigit() for character in cell) for cell in populated)

    def is_header_like(row: Sequence[str]) -> bool:
        populated = [cell.strip() for cell in row if cell.strip()]
        if len(populated) < 2:
            return False
        if any(len(cell) > 40 for cell in populated):
            return False
        if any("@" in cell for cell in populated):
            return False
        return not any(any(character.isdigit() for character in cell) for cell in populated)

    data_start = next(
        (index for index, row in enumerate(rows) if carries_data(row)),
        None,
    )
    if data_start is None:
        # No row carries data: treat the first row as labels so the caller still
        # gets a usable grid rather than nothing.
        return 0, min(1, len(rows))

    header_start = data_start
    while header_start > 0 and is_header_like(rows[header_start - 1]):
        header_start -= 1

    return header_start, data_start


def _merge_wrapped_rows(
    rows: list[tuple[str, ...]], boxes: list[BBox]
) -> tuple[list[tuple[str, ...]], list[BBox]]:
    """Join lines that are continuations of the row above.

    Multi-line cells are the norm in SSI tables — a market whose local sub-account
    reads ``Documented Acc: CBL 26724 / Non-Documented Acc: CBL 26882`` occupies
    three physical lines but one instruction. A line is a continuation when its
    first column is empty while the row above has one, i.e. it carries no new
    market identity.
    """
    if not rows:
        return rows, boxes

    merged_rows: list[list[str]] = [list(rows[0])]
    merged_boxes: list[BBox] = [boxes[0]]

    for row, box in zip(rows[1:], boxes[1:]):
        first_cell = row[0].strip() if row else ""
        populated = sum(1 for cell in row if cell.strip())
        previous_first = merged_rows[-1][0].strip() if merged_rows[-1] else ""

        is_continuation = not first_cell and previous_first and populated
        if is_continuation:
            for index, cell in enumerate(row):
                if not cell.strip():
                    continue
                while len(merged_rows[-1]) <= index:
                    merged_rows[-1].append("")
                existing = merged_rows[-1][index]
                merged_rows[-1][index] = f"{existing}\n{cell}".strip() if existing else cell
            merged_boxes[-1] = merged_boxes[-1].union(box)
        else:
            merged_rows.append(list(row))
            merged_boxes.append(box)

    return [tuple(row) for row in merged_rows], merged_boxes


def _reconstruct_grid(
    word_groups: Sequence[Sequence[Word]], page_number: int
) -> TableGrid | None:
    """Reconstruct a borderless table from fragment geometry alone.

    Fragments (not lines) are the input: a generator that emits each cell as its
    own block, and OCR that emits each detected box, both produce fragments, and
    both are handled identically. Columns and rows are found as ink bands on their
    respective axes, then each fragment is placed in the (row, column) whose bands
    contain its centre.
    """
    fragments = [group for group in word_groups if group]
    if sum(len(group) for group in fragments) < _MIN_TABLE_FRAGMENTS:
        return None

    bands = _row_bands(fragments)
    if len(bands) < 2:
        return None

    columns = _column_bounds(fragments, bands)
    if len(columns) < 3:
        return None

    # A table needs several rows that actually span multiple columns; otherwise
    # these are stacked paragraphs that merely happen to leave vertical gaps.
    cells_by_band: list[list[list[str]]] = [[[] for _ in columns] for _ in bands]
    boxes_by_band: list[BBox | None] = [None for _ in bands]
    straddles_by_band = [0 for _ in bands]

    for group in fragments:
        for word in group:
            band_index = _index_of(word.bbox.centre_y, bands)
            column_index = _index_of((word.bbox.x0 + word.bbox.x1) / 2, columns)
            cells_by_band[band_index][column_index].append(word.text)
            if _straddles_columns(word.bbox, columns):
                straddles_by_band[band_index] += 1
            existing = boxes_by_band[band_index]
            boxes_by_band[band_index] = word.bbox if existing is None else existing.union(word.bbox)

    rows: list[tuple[str, ...]] = []
    boxes: list[BBox] = []
    straddles: list[int] = []
    for band_index, band_cells in enumerate(cells_by_band):
        box = boxes_by_band[band_index]
        if box is None:
            continue
        row = tuple(" ".join(cell).strip() for cell in band_cells)
        if not any(row):
            continue
        rows.append(row)
        boxes.append(box)
        straddles.append(straddles_by_band[band_index])

    span_start, span_end = _longest_tabular_span(rows, straddles)
    span_rows = rows[span_start:span_end]
    span_boxes = boxes[span_start:span_end]
    multi_column_rows = sum(1 for row in span_rows if sum(1 for cell in row if cell) >= 3)
    if multi_column_rows < 3:
        return None

    header_offset, body_offset = _find_header_span(span_rows)
    header_rows = list(span_rows[header_offset:body_offset])

    # Group bands sit above the tabular span because they straddle columns by
    # design; recover them so the discovered schema keeps its grouping.
    cursor = span_start + header_offset
    while cursor > 0 and _is_group_band(rows[cursor - 1], straddles[cursor - 1]):
        header_rows.insert(0, rows[cursor - 1])
        cursor -= 1

    # Wrapped-row merging runs on the body only. Applied across the header
    # boundary it would fuse letterhead into the column labels, because a banner
    # row also leaves its leading column empty.
    body_rows, body_boxes = _merge_wrapped_rows(
        list(span_rows[body_offset:]), list(span_boxes[body_offset:])
    )
    if len(body_rows) < 1:
        return None

    table_box = union_all(span_boxes)
    assert table_box is not None
    return TableGrid(
        page=page_number,
        bbox=table_box,
        header_rows=tuple(header_rows),
        rows=tuple(body_rows),
        row_boxes=tuple(body_boxes),
        column_count=len(columns),
        column_bounds=tuple(columns),
    )


def _longest_tabular_span(
    rows,
    straddles,
):
    """Find the row range that behaves like a table body, as ``(start, end)``.

    The discriminator is geometric rather than lexical: a table cell sits inside
    one column, while letterhead, a contact block or a title runs across column
    boundaries. Counting fragments that straddle a boundary therefore separates
    the table from the page furniture around it without reading a single word — so
    it holds for any language, and for scanned pages just as well as native ones.
    """

    def is_tabular(index: int) -> bool:
        populated = sum(1 for cell in rows[index] if cell.strip())
        return populated >= 3 and straddles[index] == 0

    best_start = best_end = 0
    current_start: int | None = None
    for index in range(len(rows)):
        if is_tabular(index):
            if current_start is None:
                current_start = index
            if index + 1 - current_start > best_end - best_start:
                best_start, best_end = current_start, index + 1
        else:
            # One sparse or straddling row inside the table is a wrapped cell or a
            # section banner, not the end of the table, so the span tolerates it.
            if current_start is not None and index + 1 < len(rows) and is_tabular(index + 1):
                continue
            current_start = None

    if best_end == best_start:
        return 0, len(rows)
    return best_start, best_end


def _is_group_band(row, straddle_count: int) -> bool:
    """Whether a row is a merged header band above the column labels.

    A group band ("BENEFICIARY" over four columns) is label-like *and* straddles
    column boundaries — that combination is what distinguishes it from both a data
    row and page letterhead, which straddles but carries digits, addresses or
    over-long text.
    """
    populated = [cell.strip() for cell in row if cell.strip()]
    if len(populated) < 2 or straddle_count < 1:
        return False
    if any(len(cell) > 40 for cell in populated):
        return False
    if any("@" in cell for cell in populated):
        return False
    return not any(any(character.isdigit() for character in cell) for cell in populated)


def _straddles_columns(bbox, columns, margin: float = 2.0) -> bool:
    """Whether a fragment crosses a column boundary rather than sitting inside one."""
    touched = 0
    for column_start, column_end in columns:
        if bbox.x1 > column_start + margin and bbox.x0 < column_end - margin:
            touched += 1
            if touched > 1:
                return True
    return False


def _index_of(value: float, bands: Sequence[tuple[float, float]]) -> int:
    """Index of the band containing ``value``, or the nearest one."""
    for index, (start, end) in enumerate(bands):
        if start <= value <= end:
            return index
    return min(
        range(len(bands)),
        key=lambda index: min(abs(value - bands[index][0]), abs(value - bands[index][1])),
    )


def _ruled_tables(page: Any, page_number: int) -> list[TableGrid]:
    """Ruling-line tables, when the PDF actually draws them."""
    grids: list[TableGrid] = []
    try:
        finder = page.find_tables()
    except Exception as exc:
        _logger.debug("Ruled-table detection unavailable on page %s: %s", page_number, exc)
        return grids

    for table in getattr(finder, "tables", []):
        try:
            extracted = table.extract()
        except Exception:
            continue
        rows = [
            tuple(collapse_whitespace(cell or "") for cell in row)
            for row in extracted
            if any((cell or "").strip() for cell in row)
        ]
        if len(rows) < 2:
            continue
        header_start, body_start = _find_header_span(rows)
        if len(rows) <= body_start:
            continue
        grids.append(
            TableGrid(
                page=page_number,
                bbox=BBox.from_tuple(table.bbox),
                header_rows=tuple(rows[header_start:body_start]),
                rows=tuple(rows[body_start:]),
                column_count=max(len(row) for row in rows),
            )
        )
    return grids


def _build_lines(
    word_lines: Iterable[Sequence[Word]],
    *,
    page_number: int,
    source: TextSource,
    typical_gap: float,
) -> list[TextLine]:
    lines: list[TextLine] = []
    for words in word_lines:
        if not words:
            continue
        box = union_all([word.bbox for word in words])
        assert box is not None
        confidences = [word.confidence for word in words]
        line_source = source.model_copy(
            update={"ocr_confidence": round(sum(confidences) / len(confidences), 4)}
        )
        lines.append(
            TextLine(
                page=page_number,
                text=collapse_whitespace(" ".join(word.text for word in words)),
                bbox=box,
                source=line_source,
                words=tuple(words),
                cells=_split_cells(words, typical_gap),
            )
        )
    return lines


def build_document_text(
    composition: DocumentComposition,
    *,
    engine: OcrEngine | None = None,
    ocr_enabled: bool = True,
) -> DocumentText:
    """Assemble all available text for a document, running OCR only where needed.

    Native layers are read directly; image layers go through enhancement and OCR.
    A page with both contributes both, which is how a combo region survives.
    """
    import fitz

    engine = engine or OcrEngine()
    document = fitz.open(composition.path)
    pages: list[PageText] = []

    try:
        for page_composition in composition.pages:
            page = document.load_page(page_composition.page - 1)
            page_number = page_composition.page

            native_word_lines = _native_word_lines(page)
            ocr_results: list[OcrResult] = []
            ocr_word_lines: list[list[Word]] = []

            if ocr_enabled:
                for layer in page_composition.image_layers:
                    # Skip decorative images: a logo smaller than this cannot hold
                    # an instruction, and recognising it only adds noise.
                    if layer.bbox.area < 4000:
                        continue
                    result = recognise_region(
                        page, layer.bbox, page_number=page_number, engine=engine
                    )
                    if not result.is_empty:
                        ocr_results.append(result)
                        ocr_word_lines.extend(_ocr_word_lines(result))

            native_rows = _cluster_rows(native_word_lines)
            all_rows = _cluster_rows(native_word_lines + ocr_word_lines)
            typical_gap = _typical_word_gap(native_word_lines + ocr_word_lines)

            lines = _build_lines(
                native_rows,
                page_number=page_number,
                source=TextSource(layer=LayerKind.NATIVE_TEXT, ocr_confidence=1.0),
                typical_gap=typical_gap,
            )
            for result in ocr_results:
                lines.extend(
                    _build_lines(
                        _cluster_rows(_ocr_word_lines(result)),
                        page_number=page_number,
                        source=TextSource(
                            layer=LayerKind.IMAGE,
                            ocr_attempts=result.outcome.attempts,
                            enhancement_operations=result.outcome.operations,
                        ),
                        typical_gap=typical_gap,
                    )
                )
            lines.sort(key=lambda line: (line.bbox.y0, line.bbox.x0))

            tables = _ruled_tables(page, page_number)
            if not tables:
                reconstructed = _reconstruct_grid(
                    native_word_lines + ocr_word_lines, page_number
                )
                if reconstructed is not None:
                    tables = [reconstructed]

            pages.append(
                PageText(
                    page=page_number,
                    lines=tuple(lines),
                    tables=tuple(tables),
                    ocr_results=tuple(ocr_results),
                    width=page_composition.width,
                    height=page_composition.height,
                )
            )
    finally:
        document.close()

    _logger.info(
        "Prepared text for %s page(s): %s line(s), %s table(s).",
        len(pages),
        sum(len(page.lines) for page in pages),
        sum(len(page.tables) for page in pages),
    )
    return DocumentText(pages=tuple(pages))
