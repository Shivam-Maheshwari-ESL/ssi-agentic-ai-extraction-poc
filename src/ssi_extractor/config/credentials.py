"""Azure OpenAI credential loading from the project's ``AzureOpeapiKeys.txt``.

The file is a Spring-style YAML fragment in which every value is written as a
``${ENV_VAR:default}`` placeholder::

    azure:
      openai:
        api-key: ${AZURE_OPENAI_API_KEY:literal-key}
        endpoint: ${AZURE_OPENAI_ENDPOINT:https://example.openai.azure.com/}
        deployment-name: ${AZURE_OPENAI_DEPLOYMENT_NAME:gpt-5.4-mini}
        api-version: ${AZURE_OPENAI_API_VERSION:2024-10-21}

Placeholders resolve with Spring semantics: the named environment variable when
it is set and non-empty, otherwise the inline default. That keeps the file
usable as-is while letting any deployment override a single value from the
environment without editing it.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, SecretStr

__all__ = [
    "AzureOpenAICredentials",
    "CredentialError",
    "default_credentials_path",
    "load_azure_credentials",
]

_PLACEHOLDER_RE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)(?::(.*))?\}$", re.DOTALL)

# Matched against the raw file so a missing/reordered parent key cannot silently
# yield an empty configuration. The keys are unique enough in this fragment that
# a line-oriented scan is more robust than depending on exact YAML indentation.
_FIELD_KEYS = {
    "api-key": "api_key",
    "endpoint": "endpoint",
    "deployment-name": "deployment_name",
    "api-version": "api_version",
}

_ENV_FILE_OVERRIDE = "SSI_AZURE_CREDENTIALS_FILE"
_CREDENTIALS_FILENAME = "AzureOpeapiKeys.txt"


class CredentialError(RuntimeError):
    """Raised when credentials cannot be located, parsed, or fully resolved."""


class AzureOpenAICredentials(BaseModel):
    """Resolved Azure OpenAI chat/completions credentials."""

    model_config = ConfigDict(frozen=True)

    api_key: SecretStr
    endpoint: str
    deployment_name: str
    api_version: str
    source_path: Path = Field(description="File the credentials were loaded from.")

    @property
    def masked_summary(self) -> str:
        """A log-safe description that never contains the key."""
        return (
            f"endpoint={self.endpoint} deployment={self.deployment_name} "
            f"api_version={self.api_version} source={self.source_path.name}"
        )


def default_credentials_path() -> Path:
    """Locate ``AzureOpeapiKeys.txt``.

    ``SSI_AZURE_CREDENTIALS_FILE`` wins when set, which is how a Key Vault or
    secret-mount path is injected without a code change. Otherwise the file is
    searched for in the current working directory and then upwards through its
    parents, so the CLI works from anywhere inside the project.
    """
    override = os.environ.get(_ENV_FILE_OVERRIDE, "").strip()
    if override:
        return Path(override).expanduser()

    package_root = Path(__file__).resolve().parents[3]
    search_roots = [Path.cwd(), *Path.cwd().parents, package_root, *package_root.parents]
    seen: set[Path] = set()
    for root in search_roots:
        if root in seen:
            continue
        seen.add(root)
        candidate = root / _CREDENTIALS_FILENAME
        if candidate.is_file():
            return candidate

    return Path.cwd() / _CREDENTIALS_FILENAME


def _resolve_placeholder(raw_value: str, *, key: str) -> str:
    """Resolve one ``${VAR:default}`` expression, or pass a literal through."""
    value = raw_value.strip().strip('"').strip("'")
    match = _PLACEHOLDER_RE.match(value)
    if match is None:
        return value

    var_name, inline_default = match.group(1), match.group(2)
    from_env = os.environ.get(var_name, "").strip()
    if from_env:
        return from_env
    if inline_default is None:
        raise CredentialError(
            f"'{key}' resolves to ${{{var_name}}} with no default and "
            f"{var_name} is not set in the environment."
        )
    return inline_default.strip()


def _parse_fragment(text: str) -> dict[str, str]:
    """Extract the four Azure OpenAI values from the YAML fragment."""
    found: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, separator, remainder = stripped.partition(":")
        if not separator:
            continue
        field = _FIELD_KEYS.get(key.strip().lower())
        if field is None or field in found:
            continue
        comment_free = remainder.split(" #", 1)[0]
        found[field] = comment_free.strip()
    return found


def load_azure_credentials(path: Path | str | None = None) -> AzureOpenAICredentials:
    """Load and resolve Azure OpenAI credentials.

    Raises:
        CredentialError: if the file is missing, unparsable, or any required
            value resolves to an empty string.
    """
    credentials_path = Path(path).expanduser() if path is not None else default_credentials_path()
    if not credentials_path.is_file():
        raise CredentialError(
            f"Azure credentials file not found at {credentials_path}. "
            f"Place {_CREDENTIALS_FILENAME} in the project root or set {_ENV_FILE_OVERRIDE}."
        )

    try:
        text = credentials_path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise CredentialError(f"Cannot read {credentials_path}: {exc}") from exc

    raw_fields = _parse_fragment(text)
    missing = sorted(set(_FIELD_KEYS.values()) - raw_fields.keys())
    if missing:
        raise CredentialError(
            f"{credentials_path.name} is missing required key(s): {', '.join(missing)}."
        )

    resolved = {
        field: _resolve_placeholder(raw_value, key=field)
        for field, raw_value in raw_fields.items()
    }
    empty = sorted(field for field, value in resolved.items() if not value)
    if empty:
        raise CredentialError(
            f"Credential value(s) resolved to empty: {', '.join(empty)}. "
            "Set the corresponding environment variable or fix the inline default."
        )

    return AzureOpenAICredentials(
        api_key=SecretStr(resolved["api_key"]),
        endpoint=resolved["endpoint"].rstrip("/") + "/",
        deployment_name=resolved["deployment_name"],
        api_version=resolved["api_version"],
        source_path=credentials_path,
    )
