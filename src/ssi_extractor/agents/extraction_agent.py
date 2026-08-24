"""Stage 5 — the extraction agent.

One chunk in, one populated record out, against a model built at runtime from the
document's own discovered schema. The agent has exactly one capability: return
JSON matching that model. It has no tools, no retrieval and no ability to act,
which is the structural half of prompt-injection resistance; the delimiting in
``BaseAgent`` is the other half.

It sees unmasked values on purpose. Masking before extraction would hand the model
tokenised text and make character-exact copying impossible, so masking is Stage 6
— after extraction, before anything is exported or logged.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from ssi_extractor.agents.base import UNTRUSTED_PREAMBLE, BaseAgent
from ssi_extractor.config.settings import Settings
from ssi_extractor.llm.port import LlmPort
from ssi_extractor.prompts.extraction import render_system_prompt, render_user_prompt
from ssi_extractor.schema.descriptor import SchemaDescriptor
from ssi_extractor.schema.model_builder import build_record_model
from ssi_extractor.schema.strict_schema import to_strict_schema
from ssi_extractor.stages.locate_chunk import InstructionChunk

__all__ = ["ExtractionAgent"]


class ExtractionAgent(BaseAgent[BaseModel]):
    """Maps one located chunk onto the runtime-built record model."""

    prompt_version_setting = "extraction_prompt_version"

    def __init__(
        self,
        llm: LlmPort,
        descriptor: SchemaDescriptor,
        *,
        document_name: str,
        settings: Settings | None = None,
    ) -> None:
        super().__init__(llm, settings=settings)
        self._descriptor = descriptor
        self._document_name = document_name
        self._model = build_record_model(descriptor)
        self._schema = to_strict_schema(self._model)

    @property
    def name(self) -> str:
        return "extraction_agent"

    @property
    def record_model(self) -> type[BaseModel]:
        return self._model

    def system_prompt(self) -> str:
        return render_system_prompt(UNTRUSTED_PREAMBLE)

    def json_schema(self) -> dict[str, Any]:
        return self._schema

    def parse(self, payload: dict[str, Any]) -> BaseModel:
        """Validate against the runtime model — reject, never coerce.

        Validation failure raises, which sends the agent back for another attempt
        with the specific error quoted. That is G2's schema check operating at the
        earliest possible point.
        """
        return self._model.model_validate(payload)

    def build_user_prompt(self, chunk: InstructionChunk, *, chunk_total: int) -> str:
        """Render the per-chunk prompt, with the chunk text delimited as data."""
        return render_user_prompt(
            descriptor=self._descriptor,
            document_name=self._document_name,
            chunk_index=chunk.index + 1,
            chunk_total=chunk_total,
            layout_pattern=chunk.layout_pattern.value,
            pages=chunk.page_label,
            chunk_block=self.wrap_untrusted(chunk.text),
            is_amendment=chunk.is_amendment,
        )
