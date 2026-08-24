"""Bounding-box geometry.

Region classification, layer overlap detection and chunk stitching all reason
about rectangles, so the arithmetic lives in one place. Coordinates follow the
PDF convention used by PyMuPDF: origin top-left, y increasing downwards.
"""

from __future__ import annotations

from typing import Iterable

from pydantic import BaseModel, ConfigDict, model_validator

__all__ = ["BBox", "union_all"]


class BBox(BaseModel):
    """An axis-aligned rectangle on one page."""

    model_config = ConfigDict(frozen=True)

    x0: float
    y0: float
    x1: float
    y1: float

    @model_validator(mode="after")
    def _order_coordinates(self) -> "BBox":
        if self.x1 < self.x0 or self.y1 < self.y0:
            object.__setattr__(self, "x0", min(self.x0, self.x1))
            object.__setattr__(self, "x1", max(self.x0, self.x1))
            object.__setattr__(self, "y0", min(self.y0, self.y1))
            object.__setattr__(self, "y1", max(self.y0, self.y1))
        return self

    @classmethod
    def from_tuple(cls, values: Iterable[float]) -> "BBox":
        x0, y0, x1, y1 = tuple(float(value) for value in values)
        return cls(x0=x0, y0=y0, x1=x1, y1=y1)

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def centre_y(self) -> float:
        return (self.y0 + self.y1) / 2

    def intersection(self, other: "BBox") -> "BBox | None":
        x0, y0 = max(self.x0, other.x0), max(self.y0, other.y0)
        x1, y1 = min(self.x1, other.x1), min(self.y1, other.y1)
        if x1 <= x0 or y1 <= y0:
            return None
        return BBox(x0=x0, y0=y0, x1=x1, y1=y1)

    def overlap_ratio(self, other: "BBox") -> float:
        """Intersection area as a fraction of *this* box's area.

        Asymmetric on purpose: "how much of this text block sits on top of that
        image" is the question that decides whether a region is mixed content.
        """
        if self.area <= 0:
            return 0.0
        overlap = self.intersection(other)
        return 0.0 if overlap is None else overlap.area / self.area

    def union(self, other: "BBox") -> "BBox":
        return BBox(
            x0=min(self.x0, other.x0),
            y0=min(self.y0, other.y0),
            x1=max(self.x1, other.x1),
            y1=max(self.y1, other.y1),
        )

    def vertical_overlap(self, other: "BBox") -> float:
        """Shared vertical extent as a fraction of the shorter box's height.

        This is how table rows are grouped: cells belonging to one row overlap
        vertically even when their x-ranges are disjoint.
        """
        shorter = min(self.height, other.height)
        if shorter <= 0:
            return 0.0
        overlap = min(self.y1, other.y1) - max(self.y0, other.y0)
        return max(0.0, overlap) / shorter

    def expanded(self, margin: float) -> "BBox":
        return BBox(
            x0=self.x0 - margin,
            y0=self.y0 - margin,
            x1=self.x1 + margin,
            y1=self.y1 + margin,
        )


def union_all(boxes: Iterable[BBox]) -> BBox | None:
    """Smallest box containing all inputs, or ``None`` for an empty iterable."""
    result: BBox | None = None
    for box in boxes:
        result = box if result is None else result.union(box)
    return result
