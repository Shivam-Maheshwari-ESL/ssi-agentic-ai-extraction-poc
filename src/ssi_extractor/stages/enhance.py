"""Stage 2 — image enhancement for scanned and photographed regions.

This is the cheapest lever on both accuracy and cost: every character recovered
here is one the OCR does not lose and the vision fallback is not called to guess.
Two tiers exist so the retry means something — ``STANDARD`` on the first pass and
``AGGRESSIVE`` only after OCR came back below the tier-1 threshold, which keeps
the external escalation genuinely rare rather than routine.

Enhancement never runs on native-text layers. Their characters are already exact
and rasterising them could only lose information.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from ssi_extractor.observability.logging import get_logger
from ssi_extractor.utils.geometry import BBox

__all__ = [
    "EnhancementResult",
    "EnhancementTier",
    "QualityMetrics",
    "assess_quality",
    "enhance_image",
    "render_region",
]

_logger = get_logger(__name__)


class EnhancementTier(StrEnum):
    """How hard to work on a region."""

    STANDARD = "STANDARD"
    AGGRESSIVE = "AGGRESSIVE"


class QualityMetrics(BaseModel):
    """Measured condition of a region, used to choose operations and to explain them."""

    model_config = ConfigDict(frozen=True)

    blur_variance: float
    skew_degrees: float
    noise_estimate: float
    contrast: float
    mean_intensity: float
    is_inverted: bool
    estimated_dpi: float

    @property
    def is_blurred(self) -> bool:
        return self.blur_variance < 120.0

    @property
    def is_skewed(self) -> bool:
        return abs(self.skew_degrees) >= 0.4

    @property
    def is_noisy(self) -> bool:
        return self.noise_estimate > 6.0

    @property
    def is_low_contrast(self) -> bool:
        return self.contrast < 45.0


class EnhancementResult(BaseModel):
    """The enhanced raster plus what was done to it."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    image: Any
    tier: EnhancementTier
    metrics: QualityMetrics
    operations: tuple[str, ...] = ()
    rotation_applied: float = 0.0
    scale: float = 1.0


def render_region(
    page: Any,
    bbox: BBox | None = None,
    *,
    dpi: int = 300,
) -> np.ndarray:
    """Rasterise a page or a region of it as a greyscale array.

    A clip is used rather than rendering the whole page and cropping, so a small
    stamp on a large page costs a small raster — the token-and-time argument for
    cropping before any model sees anything.
    """
    import fitz

    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    clip = fitz.Rect(*bbox.as_tuple()) if bbox is not None else None
    pixmap = page.get_pixmap(matrix=matrix, clip=clip, colorspace=fitz.csGRAY, alpha=False)
    array = np.frombuffer(pixmap.samples, dtype=np.uint8)
    return array.reshape(pixmap.height, pixmap.width).copy()


def _estimate_skew(image: np.ndarray) -> float:
    """Estimate text skew in degrees.

    Uses the dominant near-horizontal Hough line angle, which tracks text
    baselines, and falls back to the minimum-area rectangle of ink pixels when no
    lines are found (sparse regions such as a stamp).
    """
    import cv2

    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 360, threshold=100, minLineLength=max(40, image.shape[1] // 4), maxLineGap=12
    )
    if lines is not None and len(lines):
        angles = []
        # OpenCV has returned both (N, 1, 4) and (N, 4) across versions; reshaping
        # makes this independent of which one this build produces.
        for x0, y0, x1, y1 in np.asarray(lines).reshape(-1, 4):
            if x1 == x0:
                continue
            angle = np.degrees(np.arctan2(float(y1 - y0), float(x1 - x0)))
            if abs(angle) <= 20.0:
                angles.append(angle)
        if angles:
            return float(np.median(angles))

    inverted = cv2.bitwise_not(image)
    _, binary = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    coordinates = cv2.findNonZero(binary)
    if coordinates is None:
        return 0.0
    angle = cv2.minAreaRect(coordinates)[-1]
    if angle < -45:
        angle += 90
    elif angle > 45:
        angle -= 90
    return float(angle)


def assess_quality(image: np.ndarray, *, dpi: int = 300) -> QualityMetrics:
    """Measure a region before deciding what to do to it."""
    import cv2

    blur = float(cv2.Laplacian(image, cv2.CV_64F).var())
    mean = float(image.mean())
    contrast = float(image.std())
    denoised = cv2.medianBlur(image, 3)
    noise = float(np.abs(image.astype(np.int16) - denoised.astype(np.int16)).mean())

    return QualityMetrics(
        blur_variance=round(blur, 3),
        skew_degrees=round(_estimate_skew(image), 3),
        noise_estimate=round(noise, 3),
        contrast=round(contrast, 3),
        mean_intensity=round(mean, 3),
        is_inverted=mean < 110.0,
        estimated_dpi=float(dpi),
    )


def _deskew(image: np.ndarray, angle: float) -> np.ndarray:
    import cv2

    height, width = image.shape[:2]
    centre = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(centre, angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _normalise_illumination(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """Divide out a slowly varying background.

    Handles the two conditions that break global thresholding: uneven lighting
    and shadow in phone captures, and watermarks or letterhead tints behind text.
    """
    import cv2

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    background = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    background = cv2.GaussianBlur(background, (0, 0), kernel_size / 6.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalised = np.where(background > 0, image.astype(np.float32) / background.astype(np.float32), 1.0)
    return np.clip(normalised * 255.0, 0, 255).astype(np.uint8)


def enhance_image(
    image: np.ndarray,
    *,
    tier: EnhancementTier = EnhancementTier.STANDARD,
    metrics: QualityMetrics | None = None,
) -> EnhancementResult:
    """Clean a region for OCR, doing only what its measured condition calls for.

    Operations are conditional rather than unconditional: binarising a clean
    300-DPI render, or sharpening an already sharp one, destroys detail and costs
    accuracy. The tier widens both the operation set and its strength.
    """
    import cv2

    metrics = metrics or assess_quality(image)
    aggressive = tier is EnhancementTier.AGGRESSIVE
    operations: list[str] = []
    working = image
    rotation = 0.0

    if metrics.is_inverted:
        working = cv2.bitwise_not(working)
        operations.append("invert")

    if metrics.is_skewed:
        rotation = -metrics.skew_degrees
        working = _deskew(working, rotation)
        operations.append(f"deskew({rotation:+.2f}deg)")

    scale = 2.0 if aggressive else 1.5
    if aggressive or metrics.estimated_dpi < 300 or min(working.shape[:2]) < 900:
        working = cv2.resize(working, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        operations.append(f"upscale(x{scale})")
    else:
        scale = 1.0

    if metrics.is_noisy or aggressive:
        working = (
            cv2.fastNlMeansDenoising(working, None, 10, 7, 21)
            if aggressive
            else cv2.bilateralFilter(working, 5, 55, 55)
        )
        operations.append("denoise")

    if metrics.is_low_contrast or aggressive:
        clahe = cv2.createCLAHE(clipLimit=3.0 if aggressive else 2.0, tileGridSize=(8, 8))
        working = clahe.apply(working)
        operations.append("clahe")

    if aggressive:
        working = _normalise_illumination(working, kernel_size=31)
        operations.append("illumination_normalise")

    if metrics.is_blurred or aggressive:
        blurred = cv2.GaussianBlur(working, (0, 0), 1.2)
        working = cv2.addWeighted(working, 1.6, blurred, -0.6, 0)
        operations.append("unsharp")

    if aggressive:
        # Only the aggressive tier binarises. Modern recognisers read greyscale
        # better than a thresholded image, so this is a last resort for pages the
        # first pass could not read.
        working = cv2.adaptiveThreshold(
            working, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
        )
        operations.append("adaptive_threshold")

    _logger.debug(
        "Stage 2 enhanced a region (%s): %s",
        tier.value,
        ", ".join(operations) or "no-op",
    )
    return EnhancementResult(
        image=working,
        tier=tier,
        metrics=metrics,
        operations=tuple(operations),
        rotation_applied=rotation,
        scale=scale,
    )
