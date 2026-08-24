"""The one part of the output contract that is fixed: the leaf object.

Field names, grouping and nesting are discovered per document, but every leaf is
always this five-key object, so a consumer can walk an unknown shape and still
know what it is looking at.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = ["ExtractedField", "FieldStatus"]


class FieldStatus(StrEnum):
    """Per the output contract: exactly three states.

    ``NOT_APPLICABLE`` means the document legitimately does not state the field —
    an amendment listing only changed fields, or a layout that omits a concept
    entirely. It is not a failure and must never be reported as one.
    """

    VALIDATED = "VALIDATED"
    FAILED = "FAILED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class ExtractedField(BaseModel):
    """One extracted value with its validation status, confidence and citation."""

    model_config = ConfigDict(extra="forbid")

    value: str = ""
    status: FieldStatus = FieldStatus.NOT_APPLICABLE
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: str = ""
    page: tuple[int, ...] = ()

    @field_validator("page", mode="before")
    @classmethod
    def _normalise_page(cls, value: object) -> tuple[int, ...]:
        """Accept a single page, a list, or a tuple; always store a sorted tuple.

        A value stitched from regions spanning several pages must cite all of
        them, so the citation is always a collection even when it holds one page.
        """
        if value is None:
            return ()
        if isinstance(value, int):
            return (value,)
        if isinstance(value, (list, tuple, set)):
            return tuple(sorted({int(item) for item in value}))
        raise TypeError(f"page must be an int or a sequence of ints, got {type(value).__name__}")

    @classmethod
    def not_applicable(cls, *, evidence: str = "") -> "ExtractedField":
        """A field the document does not state — absent, not failed."""
        return cls(status=FieldStatus.NOT_APPLICABLE, evidence=evidence)

    @property
    def is_present(self) -> bool:
        return bool(self.value) and self.status is not FieldStatus.NOT_APPLICABLE
