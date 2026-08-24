"""Stage 3 — local OCR with per-word confidence, bounding boxes, and one retry.

RapidOCR runs the PP-OCR detection and recognition models through onnxruntime.
That choice is forced and deliberate: ``paddlepaddle`` has no Python 3.14
distribution, so PaddleOCR itself cannot run here, while the same model family
via ONNX can — no second interpreter, no accuracy trade.

The retry is the point of this stage. A region below the tier-1 confidence
threshold is re-enhanced at the aggressive tier and recognised once more, locally.
Only if *that* still misses the threshold does the region become a candidate for
the external vision fallback. Without this, the expensive, privacy-sensitive path
would carry work that a second local attempt resolves.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ssi_extractor.config.settings import Settings, get_settings
from ssi_extractor.observability.logging import get_logger
from ssi_extractor.stages.enhance import (
    EnhancementTier,
    assess_quality,
    enhance_image,
    render_region,
)
from ssi_extractor.utils.geometry import BBox

__all__ = ["OcrEngine", "OcrOutcome", "OcrResult", "OcrWord", "recognise_region"]

_logger = get_logger(__name__)


class OcrWord(BaseModel):
    """One recognised text unit with its own confidence and box.

    Confidence is per unit, not per page: a row whose account number was read
    cleanly must not inherit the low confidence of a smudged neighbouring cell.
    """

    model_config = ConfigDict(frozen=True)

    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BBox
    page: int


class OcrOutcome(BaseModel):
    """What happened to one region, including whether the retry was needed."""

    model_config = ConfigDict(frozen=True)

    attempts: int = 0
    tier_used: EnhancementTier = EnhancementTier.STANDARD
    orientation: int = 0
    cleared_threshold: bool = False
    needs_vision_fallback: bool = False
    operations: tuple[str, ...] = ()


class OcrResult(BaseModel):
    """Recognised text for one region."""

    model_config = ConfigDict(frozen=True)

    page: int
    bbox: BBox
    text: str = ""
    words: tuple[OcrWord, ...] = ()
    mean_confidence: float = 0.0
    min_confidence: float = 0.0
    outcome: OcrOutcome = Field(default_factory=OcrOutcome)

    @property
    def is_empty(self) -> bool:
        return not self.words


class OcrEngine:
    """Thin, lazily-initialised wrapper over RapidOCR.

    The engine loads ONNX models on first use (seconds), so it is constructed
    once per run and shared. Kept behind this class so the OCR backend can be
    swapped without touching pipeline code.
    """

    def __init__(self) -> None:
        self._engine: Any | None = None

    @property
    def available(self) -> bool:
        try:
            import rapidocr  # noqa: F401
        except ImportError:
            return False
        return True

    def _ensure_engine(self) -> Any:
        if self._engine is None:
            from rapidocr import RapidOCR

            _logger.info("Loading RapidOCR (PP-OCR ONNX) models.")
            self._engine = RapidOCR()
        return self._engine

    def run(self, image: Any, *, page: int, offset: BBox | None = None, scale: float = 1.0) -> list[OcrWord]:
        """Recognise one raster, mapping boxes back to page coordinates."""
        engine = self._ensure_engine()
        raw, _elapsed = engine(image)
        if not raw:
            return []

        words: list[OcrWord] = []
        for entry in raw:
            box, text, score = entry[0], entry[1], entry[2]
            if not text or not str(text).strip():
                continue
            xs = [float(point[0]) for point in box]
            ys = [float(point[1]) for point in box]
            bbox = BBox(x0=min(xs), y0=min(ys), x1=max(xs), y1=max(ys))
            if scale != 1.0:
                bbox = BBox(
                    x0=bbox.x0 / scale, y0=bbox.y0 / scale, x1=bbox.x1 / scale, y1=bbox.y1 / scale
                )
            if offset is not None:
                bbox = BBox(
                    x0=bbox.x0 + offset.x0,
                    y0=bbox.y0 + offset.y0,
                    x1=bbox.x1 + offset.x0,
                    y1=bbox.y1 + offset.y0,
                )
            words.append(
                OcrWord(
                    text=str(text).strip(),
                    confidence=max(0.0, min(1.0, float(score))),
                    bbox=bbox,
                    page=page,
                )
            )
        return words


def _assemble_text(words: list[OcrWord]) -> str:
    """Reconstruct reading order: group words into lines, then order left to right.

    Layout matters downstream — the harvester reads table headers and key:value
    pairs out of this text — so words are grouped by vertical overlap rather than
    concatenated in detection order.
    """
    if not words:
        return ""

    remaining = sorted(words, key=lambda word: (word.bbox.y0, word.bbox.x0))
    lines: list[list[OcrWord]] = []
    for word in remaining:
        placed = False
        for line in lines:
            if line[0].bbox.vertical_overlap(word.bbox) >= 0.45:
                line.append(word)
                placed = True
                break
        if not placed:
            lines.append([word])

    rendered = []
    for line in lines:
        line.sort(key=lambda word: word.bbox.x0)
        rendered.append("  ".join(word.text for word in line))
    return "\n".join(rendered)


def _confidences(words: list[OcrWord]) -> tuple[float, float]:
    if not words:
        return 0.0, 0.0
    scores = [word.confidence for word in words]
    return round(sum(scores) / len(scores), 4), round(min(scores), 4)


def recognise_region(
    page: Any,
    bbox: BBox | None,
    *,
    page_number: int,
    engine: OcrEngine,
    settings: Settings | None = None,
) -> OcrResult:
    """Recognise one image region, retrying once at the aggressive tier if needed."""
    settings = settings or get_settings()
    thresholds = settings.ocr
    region_box = bbox or BBox.from_tuple((page.rect.x0, page.rect.y0, page.rect.x1, page.rect.y1))

    if not engine.available:
        _logger.error(
            "RapidOCR is not installed; region on page %s cannot be recognised locally.",
            page_number,
        )
        return OcrResult(
            page=page_number,
            bbox=region_box,
            outcome=OcrOutcome(needs_vision_fallback=True),
        )

    best: tuple[float, list[OcrWord], EnhancementTier, int, tuple[str, ...]] | None = None
    attempts = 0

    for tier in (EnhancementTier.STANDARD, EnhancementTier.AGGRESSIVE):
        raster = render_region(
            page,
            region_box,
            dpi=int(72 * (thresholds.standard_upscale if tier is EnhancementTier.STANDARD else thresholds.aggressive_upscale) * 2),
        )
        metrics = assess_quality(raster)
        enhanced = enhance_image(raster, tier=tier, metrics=metrics)
        rendered_scale = (enhanced.image.shape[1] / max(1.0, region_box.width))

        # Orientation trial: a rotated or upside-down scan reads as noise at 0
        # degrees, so each orientation is attempted and the most confident wins.
        for orientation in thresholds.orientations:
            attempts += 1
            image = enhanced.image
            if orientation:
                import cv2

                rotations = {
                    90: cv2.ROTATE_90_CLOCKWISE,
                    180: cv2.ROTATE_180,
                    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
                }
                image = cv2.rotate(image, rotations[orientation])

            words = engine.run(
                image,
                page=page_number,
                offset=None if orientation else region_box,
                scale=rendered_scale if not orientation else 1.0,
            )
            mean_confidence, _ = _confidences(words)
            if best is None or mean_confidence > best[0]:
                best = (mean_confidence, words, tier, orientation, enhanced.operations)

            # A confident non-rotated read is the common case; stop trying
            # orientations rather than paying for four passes on every region.
            if mean_confidence >= thresholds.confidence_tier1:
                break

        if best is not None and best[0] >= thresholds.confidence_tier1:
            break
        if tier is EnhancementTier.STANDARD:
            _logger.info(
                "OCR confidence %.3f below tier-1 %.2f on page %s; retrying with aggressive enhancement.",
                best[0] if best else 0.0,
                thresholds.confidence_tier1,
                page_number,
            )

    assert best is not None
    mean_confidence, words, tier_used, orientation, operations = best
    mean_value, min_value = _confidences(words)
    cleared = mean_value >= thresholds.confidence_tier1

    result = OcrResult(
        page=page_number,
        bbox=region_box,
        text=_assemble_text(words),
        words=tuple(words),
        mean_confidence=mean_value,
        min_confidence=min_value,
        outcome=OcrOutcome(
            attempts=attempts,
            tier_used=tier_used,
            orientation=orientation,
            cleared_threshold=cleared,
            needs_vision_fallback=not cleared and mean_value < thresholds.confidence_floor,
            operations=operations,
        ),
    )
    _logger.info(
        "Stage 3 OCR page %s: %s words, mean confidence %.3f, tier %s, orientation %s%s",
        page_number,
        len(words),
        mean_value,
        tier_used.value,
        orientation,
        "" if cleared else " (below tier-1 threshold)",
        extra={"page": page_number},
    )
    return result
