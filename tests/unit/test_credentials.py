"""Credential loading from the Spring-style AzureOpeapiKeys.txt fragment."""

from __future__ import annotations

from pathlib import Path

import pytest

from ssi_extractor.config.credentials import (
    CredentialError,
    default_credentials_path,
    load_azure_credentials,
)

FRAGMENT = """# Azure OpenAI Configuration (for Chat/Completions only)
azure:
   openai:
     api-key: ${AZURE_OPENAI_API_KEY:inline-default-key}
     endpoint: ${AZURE_OPENAI_ENDPOINT:https://example-resource.openai.azure.com/}
     deployment-name: ${AZURE_OPENAI_DEPLOYMENT_NAME:gpt-5.4-mini}
     api-version: ${AZURE_OPENAI_API_VERSION:2024-10-21}
"""


@pytest.fixture
def fragment_file(tmp_path: Path) -> Path:
    path = tmp_path / "AzureOpeapiKeys.txt"
    path.write_text(FRAGMENT, encoding="utf-8")
    return path


def test_inline_defaults_are_used_when_environment_is_unset(
    fragment_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for variable in (
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_DEPLOYMENT_NAME",
        "AZURE_OPENAI_API_VERSION",
    ):
        monkeypatch.delenv(variable, raising=False)

    credentials = load_azure_credentials(fragment_file)

    assert credentials.api_key.get_secret_value() == "inline-default-key"
    assert credentials.endpoint == "https://example-resource.openai.azure.com/"
    assert credentials.deployment_name == "gpt-5.4-mini"
    assert credentials.api_version == "2024-10-21"
    assert credentials.source_path == fragment_file


def test_environment_overrides_the_inline_default(
    fragment_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "from-environment")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-5.4")

    credentials = load_azure_credentials(fragment_file)

    assert credentials.api_key.get_secret_value() == "from-environment"
    assert credentials.deployment_name == "gpt-5.4"


def test_secret_is_not_exposed_by_repr_or_summary(
    fragment_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "super-secret-value")
    credentials = load_azure_credentials(fragment_file)

    assert "super-secret-value" not in repr(credentials)
    assert "super-secret-value" not in credentials.masked_summary
    assert "gpt-5.4-mini" in credentials.masked_summary


def test_bom_and_utf8_are_tolerated(tmp_path: Path) -> None:
    path = tmp_path / "AzureOpeapiKeys.txt"
    path.write_text(FRAGMENT, encoding="utf-8-sig")
    assert load_azure_credentials(path).deployment_name == "gpt-5.4-mini"


def test_missing_file_raises_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(CredentialError, match="not found"):
        load_azure_credentials(tmp_path / "absent.txt")


def test_missing_key_is_reported_by_name(tmp_path: Path) -> None:
    path = tmp_path / "AzureOpeapiKeys.txt"
    path.write_text("azure:\n  openai:\n    api-key: ${A:k}\n", encoding="utf-8")

    with pytest.raises(CredentialError, match="api_version"):
        load_azure_credentials(path)


def test_placeholder_without_default_and_without_environment_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SSI_TEST_UNSET_KEY", raising=False)
    path = tmp_path / "AzureOpeapiKeys.txt"
    path.write_text(
        "azure:\n  openai:\n"
        "    api-key: ${SSI_TEST_UNSET_KEY}\n"
        "    endpoint: ${E:https://x.openai.azure.com/}\n"
        "    deployment-name: ${D:gpt-5.4-mini}\n"
        "    api-version: ${V:2024-10-21}\n",
        encoding="utf-8",
    )

    with pytest.raises(CredentialError, match="no default"):
        load_azure_credentials(path)


def test_override_environment_variable_selects_the_file(
    fragment_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SSI_AZURE_CREDENTIALS_FILE", str(fragment_file))
    assert default_credentials_path() == fragment_file


def test_real_project_credentials_file_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    """The credentials file shipped in the project root must actually parse."""
    monkeypatch.delenv("SSI_AZURE_CREDENTIALS_FILE", raising=False)
    project_file = Path(__file__).resolve().parents[2] / "AzureOpeapiKeys.txt"
    if not project_file.is_file():
        pytest.skip("AzureOpeapiKeys.txt is not present in this checkout")

    credentials = load_azure_credentials(project_file)

    assert credentials.endpoint.startswith("https://")
    assert credentials.endpoint.endswith("/")
    assert credentials.deployment_name
    assert credentials.api_version
    assert credentials.api_key.get_secret_value()
