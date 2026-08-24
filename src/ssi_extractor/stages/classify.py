"""Stage 1 — page/region/layer classifier.

Classification is per *layer within a region*, not per page. A page carrying a
digital table plus one scanned stamp is ``MIXED``: its native layers skip OCR and
only the image layer is enhanced and recognised. Reporting that page as fully
"scanned" would send perfectly good text through OCR and lose characters; calling
it fully "native" would silently drop whatever the stamp says.

The document-level counters (``nativeTextPages``, ``scannedPages``,
``mixedPages``) are derived from this layer composition, which is what the output
contract means by region-level composition.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from ssi_extractor.observability.logging import get_logger
from ssi_extractor.utils.geometry import BBox, union_all
from ssi_extractor.utils.text import collapse_whitespace

__all__ = [
    "DocumentComposition",
    "LayerKind",
    "PageClass",
    "PageComposition",
    "RegionLayer",
    "classify_document",
]

_logger = get_logger(__name__)

# A text block must overlap an image by at least this fraction of its own area to
# count as sharing the region. Small values would make every block near a
# letterhead "mixed"; large values would miss text wrapped around a logo.
_OVERLAP_THRESHOLD = 0.15

# Below this, a page's "native" text is not real content: scanners and exporters
# often leave a stray character or a footer on an otherwise imaged page.
_MIN_NATIVE_CHARS = 24


class LayerKind(StrEnum):
    """What one layer of a region is made of."""

    NATIVE_TEXT = "NATIVE_TEXT"
    IMAGE = "IMAGE"
    VECTOR = "VECTOR"


class PageClass(StrEnum):
    """Page-level summary, derived from its layers."""

    NATIVE = "NATIVE"
    SCANNED = "SCANNED"
    MIXED = "MIXED"
    EMPTY = "EMPTY"


class RegionLayer(BaseModel):
    """One layer: a rectangle on a page with either text or pixels behind it."""

    model_config = ConfigDict(frozen=True)

    page: int
    kind: LayerKind
    bbox: BBox
    text: str = ""
    char_count: int = 0
    overlaps_image: bool = Field(
        default=False,
        description="A native-text layer sitting on top of an image — the combo-region case.",
    )
    image_index: int | None = None

    @property
    def needs_ocr(self) -> bool:
        return self.kind is not LayerKind.NATIVE_TEXT


class PageComposition(BaseModel):
    """Layer inventory for one page."""

    model_config = ConfigDict(frozen=True)

    page: int
    page_class: PageClass
    width: float
    height: float
    rotation: int = 0
    layers: tuple[RegionLayer, ...] = ()
    native_chars: int = 0
    image_area_ratio: float = 0.0

    @property
    def native_layers(self) -> tuple[RegionLayer, ...]:
        return tuple(layer for layer in self.layers if layer.kind is LayerKind.NATIVE_TEXT)

    @property
    def image_layers(self) -> tuple[RegionLayer, ...]:
        return tuple(layer for layer in self.layers if layer.kind is LayerKind.IMAGE)

    @property
    def native_text(self) -> str:
        return "\n".join(layer.text for layer in self.native_layers if layer.text)


class DocumentComposition(BaseModel):
    """Document-level composition, feeding the output metadata directly."""

    model_config = ConfigDict(frozen=True)

    path: Path
    page_count: int
    pages: tuple[PageComposition, ...] = ()

    @property
    def native_text_pages(self) -> int:
        return sum(1 for page in self.pages if page.page_class is PageClass.NATIVE)

    @property
    def scanned_pages(self) -> int:
        return sum(1 for page in self.pages if page.page_class is PageClass.SCANNED)

    @property
    def mixed_pages(self) -> int:
        return sum(1 for page in self.pages if page.page_class is PageClass.MIXED)

    @property
    def empty_pages(self) -> int:
        return sum(1 for page in self.pages if page.page_class is PageClass.EMPTY)

    def page(self, number: int) -> PageComposition | None:
        return next((page for page in self.pages if page.page == number), None)

    def summary(self) -> str:
        """Human-readable composition line for ``documentAnalysis``."""
        parts = [
            f"{self.page_count} page(s)",
            f"{self.native_text_pages} native",
            f"{self.scanned_pages} scanned",
            f"{self.mixed_pages} mixed",
        ]
        if self.empty_pages:
            parts.append(f"{self.empty_pages} empty")
        combo_layers = sum(
            1 for page in self.pages for layer in page.layers if layer.overlaps_image
        )
        if combo_layers:
            parts.append(f"{combo_layers} text-over-image region(s)")
        return " | ".join(parts)


def _image_bboxes(page: object) -> list[tuple[int, BBox]]:
    """Image placements on a page, as (index, bbox).

    ``get_image_info`` is preferred over ``get_images`` because it reports where
    each image is drawn, which is what layer overlap needs.
    """
    boxes: list[tuple[int, BBox]] = []
    try:
        for index, info in enumerate(page.get_image_info(xrefs=True)):  # type: ignore[attr-defined]
            bbox = info.get("bbox")
            if bbox:
                boxes.append((index, BBox.from_tuple(bbox)))
    except Exception:  # older bindings, or a page whose resources cannot be walked
        try:
            for index, image in enumerate(page.get_images(full=True)):  # type: ignore[attr-defined]
                rects = page.get_image_rects(image[0])  # type: ignore[attr-defined]
                for rect in rects:
                    boxes.append((index, BBox.from_tuple(rect)))
        except Exception:
            _logger.warning("Could not determine image placement on page %s.", page.number + 1)  # type: ignore[attr-defined]
    return boxes


def _classify_page(page: object, page_number: int) -> PageComposition:
    import fitz

    rect = page.rect  # type: ignore[attr-defined]
    page_box = BBox.from_tuple((rect.x0, rect.y0, rect.x1, rect.y1))
    image_boxes = _image_bboxes(page)
    layers: list[RegionLayer] = []
    native_chars = 0

    text_page = page.get_text("dict")  # type: ignore[attr-defined]
    for block in text_page.get("blocks", []):
        if block.get("type") != 0:
            continue
        lines = []
        for line in block.get("lines", []):
            spans = [span.get("text", "") for span in line.get("spans", [])]
            joined = collapse_whitespace("".join(spans))
            if joined:
                lines.append(joined)
        text = "\n".join(lines)
        if not text:
            continue

        bbox = BBox.from_tuple(block["bbox"])
        overlaps = any(
            bbox.overlap_ratio(image_box) >= _OVERLAP_THRESHOLD for _, image_box in image_boxes
        )
        native_chars += len(text)
        layers.append(
            RegionLayer(
                page=page_number,
                kind=LayerKind.NATIVE_TEXT,
                bbox=bbox,
                text=text,
                char_count=len(text),
                overlaps_image=overlaps,
            )
        )

    for index, image_box in image_boxes:
        layers.append(
            RegionLayer(
                page=page_number,
                kind=LayerKind.IMAGE,
                bbox=image_box,
                image_index=index,
            )
        )

    image_union = union_all([box for _, box in image_boxes])
    image_area_ratio = (image_union.area / page_box.area) if image_union and page_box.area else 0.0

    has_native = native_chars >= _MIN_NATIVE_CHARS
    has_image = bool(image_boxes)
    combo = any(layer.overlaps_image for layer in layers)

    if not has_native and not has_image:
        page_class = PageClass.EMPTY
    elif has_native and (combo or (has_image and image_area_ratio >= 0.05)):
        page_class = PageClass.MIXED
    elif has_native:
        page_class = PageClass.NATIVE
    else:
        page_class = PageClass.SCANNED

    return PageComposition(
        page=page_number,
        page_class=page_class,
        width=page_box.width,
        height=page_box.height,
        rotation=int(getattr(page, "rotation", 0) or 0),
        layers=tuple(layers),
        native_chars=native_chars,
        image_area_ratio=round(image_area_ratio, 4),
    )


def classify_document(path: Path | str) -> DocumentComposition:
    """Classify every page of a PDF into regions and layers."""
    import fitz

    path = Path(path)
    document = fitz.open(path)
    try:
        pages = tuple(
            _classify_page(document.load_page(index), index + 1)
            for index in range(document.page_count)
        )
        composition = DocumentComposition(
            path=path, page_count=document.page_count, pages=pages
        )
    finally:
        document.close()

    _logger.info(
        "Stage 1 classified %s: %s",
        path.name,
        composition.summary(),
        extra={"document": path.name},
    )
    return composition
