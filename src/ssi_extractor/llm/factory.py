"""Provider selection. The pipeline asks for a port; configuration picks the adapter."""

from __future__ import annotations

from ssi_extractor.config.settings import Settings, get_settings
from ssi_extractor.llm.port import LlmPort

__all__ = ["build_llm"]


def build_llm(settings: Settings | None = None) -> LlmPort:
    """Construct the configured provider adapter.

    Imports are local so a missing optional SDK for the unused provider cannot
    break the pipeline at import time.
    """
    settings = settings or get_settings()
    provider = settings.llm.provider

    if provider == "azure_openai":
        from ssi_extractor.llm.azure_openai import AzureOpenAIAdapter

        return AzureOpenAIAdapter(settings=settings)

    if provider == "anthropic":
        from ssi_extractor.llm.anthropic import AnthropicAdapter

        return AnthropicAdapter(settings=settings)

    raise ValueError(f"Unknown LLM provider: {provider!r}")
