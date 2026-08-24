"""Provider-agnostic LLM port.

Every agent talks to this interface, never to a vendor SDK, so switching
provider is a config change rather than a code change. The port is deliberately
narrow: a single structured-output call. Agents get no free-form completion and
no tool access, which is what keeps prompt-injected document text from being
able to make an agent do anything except return JSON.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["LlmError", "LlmPort", "LlmResponse", "LlmUsage"]


class LlmUsage(BaseModel):
    """Token accounting, aggregated per document for cost control."""

    model_config = ConfigDict(frozen=True)

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: "LlmUsage") -> "LlmUsage":
        return LlmUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )


class LlmResponse(BaseModel):
    """One structured response, with the provenance G4 needs to audit it."""

    model_config = ConfigDict(frozen=True)

    payload: dict[str, Any] = Field(default_factory=dict)
    raw_text: str = ""
    model_id: str = ""
    usage: LlmUsage = Field(default_factory=LlmUsage)
    truncated: bool = False


class LlmError(RuntimeError):
    """Raised when a provider call fails after its retries are exhausted."""


class LlmPort(ABC):
    """The only surface the agents may use."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Pinned model identity, recorded on every audit entry."""

    @abstractmethod
    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, Any],
        schema_name: str,
        max_output_tokens: int | None = None,
        images: list[bytes] | None = None,
    ) -> LlmResponse:
        """Return a JSON object conforming to ``json_schema``.

        Implementations must not repair or coerce a non-conforming response;
        surfacing it lets G2 reject and retry with the failure recorded.
        """
