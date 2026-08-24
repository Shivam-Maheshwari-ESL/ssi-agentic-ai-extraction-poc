"""Stage 8 — blended confidence and page citation.

Confidence must reflect real signals, not a model's self-report. Four are blended
with configurable weights: OCR word confidence for the region the value came from,
the model's own confidence, whether the deterministic format check passed, and
whether reference data recognised the value.

Two rules matter for correctness:

* **Native-text layers are pinned at OCR confidence 1.0.** Their characters are
  exact. Without this, a mixed row would be penalised across every field because
  one sub-field came from a scanned stamp.
* **Empty reference data contributes neutrally.** No BIC directory has been loaded
  yet, so an unknown BIC must neither raise nor lower confidence; scoring it as a
  miss would make every document look wrong.

Page citation is resolved here too: a value stitched from regions spanning several
pages cites every contributing page, which is what the output contract requires.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from ssi_extractor.config.settings import Settings, get_settings
from ssi_extractor.observability.logging import get_logger
from ssi_extractor.schema.leaf import ExtractedField, FieldStatus
from ssi_extractor.stages.locate_chunk import InstructionChunk
from ssi_extractor.stages.page_text import DocumentText
from ssi_extractor.stages.validate import ValidatedField
from ssi_extractor.utils.text import fold_for_comparison, substring_present
from ssi_extractor.validators.reference_data import ReferenceData

__all__ = ["ConfidenceBreakdown", "ScoredField", "score_fields"]

_logger = get_logger(__name__)


class ConfidenceBreakdown(BaseModel):
    """The individual signals behind a field's score, kept for audit and review."""

    model_config = ConfigDict(frozen=True)

    ocr: float = 1.0
    model: float = 0.0
    format: float = 0.0
    reference: float = 0.5
    blended: float = 0.0
    native_layer: bool = True
    reference_available: bool = False

    def explain(self) -> str:
        return (
            f"ocr={self.ocr:.2f} model={self.model:.2f} format={self.format:.2f} "
            f"reference={self.reference:.2f} -> {self.blended:.3f}"
        )


class ScoredField(BaseModel):
    """A validated field with its final confidence and page citation."""

    model_config = ConfigDict(frozen=True)

    record_index: int
    path: str
    leaf: ExtractedField
    breakdown: ConfidenceBreakdown
    needs_review: bool = False
    review_reasons: tuple[str, ...] = ()


def _resolve_pages(
    leaf: ExtractedField,
    chunk: InstructionChunk | None,
    document_text: DocumentText | None,
) -> tuple[int, ...]:
    """Determine which page or pages a value actually came from.

    Three tiers, most precise first: the pages whose text contains the value, the
    model's own citation when it is consistent with the chunk, and finally the
    chunk's own pages. The first tier is what makes a stitched row cite both of its
    pages rather than whichever one the model happened to name.
    """
    chunk_pages = tuple(chunk.pages) if chunk else ()

    if document_text is not None and leaf.value.strip():
        matched = tuple(
            page.page
            for page in document_text.pages
            if (not chunk_pages or page.page in chunk_pages)
            and substring_present(leaf.value, page.text)
        )
        if matched:
            return matched

    if leaf.page:
        # Trust the model's citation only where it agrees with the chunk it was
        # given; a page it never saw is not evidence.
        consistent = tuple(page for page in leaf.page if not chunk_pages or page in chunk_pages)
        if consistent:
            return consistent

    return chunk_pages


def _reference_score(
    field: ValidatedField, reference: ReferenceData
) -> tuple[float, bool]:
    """Score a value against loaded reference data.

    Returns the score and whether any reference set was actually available for this
    kind, so the blend can treat "not checked" differently from "not found".
    """
    lookup = reference.lookup(field.descriptor.kind, field.leaf.value)
    if not lookup.available:
        return reference.neutral_score, False
    return (1.0 if lookup.found else 0.25), True


def score_fields(
    fields: list[ValidatedField],
    chunks: list[InstructionChunk],
    *,
    document_text: DocumentText | None = None,
    reference: ReferenceData | None = None,
    settings: Settings | None = None,
) -> list[ScoredField]:
    """Blend signals into a final per-field confidence and attach page citations."""
    settings = settings or get_settings()
    weights = settings.confidence
    reference = reference or ReferenceData.load(settings=settings)

    scored: list[ScoredField] = []
    for field in fields:
        chunk = chunks[field.record_index] if field.record_index < len(chunks) else None
        pages = _resolve_pages(field.leaf, chunk, document_text)

        if field.leaf.status is FieldStatus.NOT_APPLICABLE or not field.leaf.value.strip():
            # An unstated field has nothing to be confident about. Reporting 0.0
            # keeps "absent" distinguishable from "present but uncertain".
            leaf = field.leaf.model_copy(
                update={"confidence": 0.0, "status": FieldStatus.NOT_APPLICABLE, "page": pages}
            )
            scored.append(
                ScoredField(
                    record_index=field.record_index,
                    path=field.path,
                    leaf=leaf,
                    breakdown=ConfidenceBreakdown(ocr=1.0, blended=0.0),
                    needs_review=False,
                )
            )
            continue

        native = chunk.ocr_confidence >= 0.999 if chunk else True
        ocr_score = 1.0 if native else float(chunk.ocr_confidence if chunk else 1.0)
        model_score = max(0.0, min(1.0, field.leaf.confidence))
        format_score = field.verdict.format_score
        reference_score, reference_available = _reference_score(field, reference)

        blended = (
            weights.weight_ocr * ocr_score
            + weights.weight_llm * model_score
            + weights.weight_format * format_score
            + weights.weight_reference * reference_score
        )

        # A field-level shape disagreement means the value may be in the wrong
        # place; that uncertainty belongs in the score, not only in a note.
        blended *= field.kind_agreement

        # Evidence that cannot be traced back to the source text is not evidence.
        if field.leaf.evidence.strip() and chunk is not None:
            if fold_for_comparison(field.leaf.evidence) not in fold_for_comparison(chunk.text):
                blended *= 0.85

        if field.leaf.status is FieldStatus.FAILED:
            blended = min(blended, 0.35)

        blended = round(max(0.0, min(1.0, blended)), 4)
        breakdown = ConfidenceBreakdown(
            ocr=round(ocr_score, 4),
            model=round(model_score, 4),
            format=round(format_score, 4),
            reference=round(reference_score, 4),
            blended=blended,
            native_layer=native,
            reference_available=reference_available,
        )

        below_floor = blended < weights.review_floor
        review_reasons = list(field.review_reasons)
        if below_floor:
            review_reasons.append(
                f"confidence {blended:.2f} is below the review floor {weights.review_floor:.2f}"
            )

        scored.append(
            ScoredField(
                record_index=field.record_index,
                path=field.path,
                leaf=field.leaf.model_copy(update={"confidence": blended, "page": pages}),
                breakdown=breakdown,
                needs_review=field.needs_review or below_floor,
                review_reasons=tuple(review_reasons),
            )
        )

    flagged = sum(1 for field in scored if field.needs_review)
    _logger.info(
        "Stage 8 scored %s field(s); %s flagged for human review.", len(scored), flagged
    )
    return scored
