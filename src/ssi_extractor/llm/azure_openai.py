"""Azure OpenAI adapter — the default provider.

Uses Chat Completions with a JSON-schema response format so the model is
constrained by the runtime-built schema rather than asked politely to comply.
Credentials come from ``AzureOpeapiKeys.txt`` through the credential loader, and
the resolved key is registered with the log redactor before the client exists.
"""

from __future__ import annotations

import base64
import json
import time
from typing import Any

from ssi_extractor.config.credentials import AzureOpenAICredentials, load_azure_credentials
from ssi_extractor.config.settings import Settings, get_settings
from ssi_extractor.llm.port import LlmError, LlmPort, LlmResponse, LlmUsage
from ssi_extractor.observability.logging import get_logger
from ssi_extractor.observability.redaction import register_literal_secret

__all__ = ["AzureOpenAIAdapter"]

_logger = get_logger(__name__)

# Retried because they are transient. Anything else fails fast, so a prompt or
# schema mistake is not hidden behind a retry loop.
_RETRYABLE_MARKERS = ("429", "500", "502", "503", "504", "timeout", "timed out", "connection")


class AzureOpenAIAdapter(LlmPort):
    """Structured-output calls against one pinned Azure deployment."""

    def __init__(
        self,
        *,
        credentials: AzureOpenAICredentials | None = None,
        settings: Settings | None = None,
        max_attempts: int = 3,
    ) -> None:
        self._settings = settings or get_settings()
        self._credentials = credentials or load_azure_credentials()
        self._max_attempts = max_attempts
        register_literal_secret(self._credentials.api_key.get_secret_value())

        from openai import AzureOpenAI

        self._client = AzureOpenAI(
            api_key=self._credentials.api_key.get_secret_value(),
            azure_endpoint=self._credentials.endpoint,
            api_version=self._credentials.api_version,
            timeout=self._settings.llm.request_timeout_seconds,
            max_retries=0,
        )

    @property
    def model_id(self) -> str:
        return f"azure/{self._credentials.deployment_name}@{self._credentials.api_version}"

    def _messages(
        self, system_prompt: str, user_prompt: str, images: list[bytes] | None
    ) -> list[dict[str, Any]]:
        if not images:
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

        content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        for image in images:
            encoded = base64.b64encode(image).decode("ascii")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded}", "detail": "high"},
                }
            )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

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
        request: dict[str, Any] = {
            "model": self._credentials.deployment_name,
            "messages": self._messages(system_prompt, user_prompt, images),
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": json_schema, "strict": True},
            },
            "max_completion_tokens": max_output_tokens or self._settings.llm.max_output_tokens,
        }
        # Reasoning-capable deployments reject a non-default temperature. The
        # pipeline wants determinism, so it is sent only while it is accepted and
        # dropped on the first complaint rather than failing the document.
        if self._settings.llm.temperature != 1.0:
            request["temperature"] = self._settings.llm.temperature

        last_error: Exception | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                completion = self._client.chat.completions.create(**request)
            except Exception as exc:  # the provider SDK raises a wide range of errors
                message = str(exc).lower()
                last_error = exc
                if "temperature" in message and "temperature" in request:
                    request.pop("temperature")
                    _logger.info("Deployment rejected temperature; retrying without it.")
                    continue
                if attempt < self._max_attempts and any(
                    marker in message for marker in _RETRYABLE_MARKERS
                ):
                    delay = 2 ** (attempt - 1)
                    _logger.warning(
                        "LLM call failed (attempt %s/%s); retrying in %ss.",
                        attempt,
                        self._max_attempts,
                        delay,
                        extra={"error_type": type(exc).__name__},
                    )
                    time.sleep(delay)
                    continue
                raise LlmError(f"Azure OpenAI call failed: {exc}") from exc

            choice = completion.choices[0]
            text = choice.message.content or ""
            usage = LlmUsage(
                prompt_tokens=getattr(completion.usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(completion.usage, "completion_tokens", 0) or 0,
            )
            truncated = choice.finish_reason == "length"

            try:
                payload = json.loads(text) if text else {}
            except json.JSONDecodeError as exc:
                raise LlmError(
                    "Model returned text that is not valid JSON "
                    f"(finish_reason={choice.finish_reason})"
                ) from exc

            return LlmResponse(
                payload=payload,
                raw_text=text,
                model_id=self.model_id,
                usage=usage,
                truncated=truncated,
            )

        raise LlmError(
            f"Azure OpenAI call failed after {self._max_attempts} attempts: {last_error}"
        )
