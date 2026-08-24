"""Typed LangGraph state for one document.

One Pydantic model carries the whole run. Keeping it typed (rather than a loose
dict) is what makes the graph debuggable: a checkpoint can be loaded and read, and
every node's contribution is a named field with a known shape.

The state is also the resume unit. Because it is serialisable and the document id
is content-addressed, re-running the same file resumes its checkpoint while an
edited file starts a fresh one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ssi_extractor.guardrails.g1_input_gate import GateResult
from ssi_extractor.guardrails.g2_extraction_guard import GuardReport
from ssi_extractor.llm.port import LlmUsage
from ssi_extractor.schema.descriptor import SchemaDescriptor
from ssi_extractor.stages.classify import DocumentComposition
from ssi_extractor.stages.confidence import ScoredField
from ssi_extractor.stages.locate_chunk import LocateResult
from ssi_extractor.stages.page_text import DocumentText
from ssi_extractor.stages.validate import ValidationReport

__all__ = ["PipelineState", "StageStatus"]


class StageStatus(BaseModel):
    """Per-stage outcome, so a partial run explains itself."""

    model_config = ConfigDict(frozen=True)

    name: str
    ok: bool = True
    detail: str = ""
    duration_seconds: float = 0.0


class PipelineState(BaseModel):
    """Everything known about one document as it moves through the graph."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # --- inputs -----------------------------------------------------------
    pdf_path: Path
    document_name: str = ""
    document_id: str = ""
    masked_export: bool = False

    # --- guardrails and stages -------------------------------------------
    gate: GateResult | None = None
    composition: DocumentComposition | None = None
    document_text: DocumentText | None = None
    located: LocateResult | None = None
    descriptor: SchemaDescriptor | None = None
    records: list[Any] = Field(default_factory=list)
    schema_failures: dict[int, str] = Field(default_factory=dict)
    guard: GuardReport | None = None
    validation: ValidationReport | None = None
    scored_fields: list[ScoredField] = Field(default_factory=list)
    review_count: int = 0

    # --- outputs and bookkeeping -----------------------------------------
    output_path: Path | None = None
    sidecar_path: Path | None = None
    masked_path: Path | None = None
    payload: dict[str, Any] | None = None
    stages: list[StageStatus] = Field(default_factory=list)
    usage: LlmUsage = Field(default_factory=LlmUsage)
    llm_calls: int = 0
    errors: list[str] = Field(default_factory=list)
    rejected: bool = False
    rejection_reason: str = ""

    @property
    def accepted(self) -> bool:
        return self.gate is not None and self.gate.accepted and not self.rejected

    def record_stage(self, name: str, *, ok: bool = True, detail: str = "", seconds: float = 0.0) -> None:
        self.stages.append(
            StageStatus(name=name, ok=ok, detail=detail, duration_seconds=round(seconds, 3))
        )

    def summary(self) -> str:
        parts = [f"{self.document_name}"]
        if self.rejected:
            return f"{self.document_name}: REJECTED — {self.rejection_reason}"
        if self.located is not None:
            parts.append(f"{len(self.located.chunks)} instruction(s)")
        if self.descriptor is not None:
            parts.append(f"{len(self.descriptor.fields)} discovered field(s)")
        if self.scored_fields:
            parts.append(f"{len(self.scored_fields)} scored field(s)")
        if self.review_count:
            parts.append(f"{self.review_count} queued for review")
        if self.usage.total_tokens:
            parts.append(f"{self.usage.total_tokens} tokens in {self.llm_calls} call(s)")
        return " | ".join(parts)
