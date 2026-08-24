"""Pipeline configuration.

Every threshold, cap, weight and toggle the pipeline depends on lives here, so
no stage carries a tuned magic number in its source. Values are overridable from
the environment with the ``SSI_`` prefix and nested delimiter ``__``, e.g.
``SSI_OCR__CONFIDENCE_TIER1=0.75`` or ``SSI_PRIVACY__EXTERNAL_FALLBACK_ENABLED=true``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field, PositiveFloat, PositiveInt, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = [
    "AuditSettings",
    "ConfidenceSettings",
    "InputGateSettings",
    "LlmSettings",
    "LoggingSettings",
    "OcrSettings",
    "PathSettings",
    "PrivacySettings",
    "Settings",
    "get_settings",
]


def _project_root() -> Path:
    """The project directory (the parent of ``src/``)."""
    return Path(__file__).resolve().parents[3]


class PathSettings(BaseModel):
    """Filesystem layout. Relative paths resolve against the project root."""

    project_root: Path = Field(default_factory=_project_root)
    input_dir: Path = Path("input_pdf")
    output_dir: Path = Path("output_json")
    logs_dir: Path = Path("logs")
    audit_dir: Path = Path("audit")
    review_queue_dir: Path = Path("review_queue")
    checkpoint_dir: Path = Path(".checkpoints")
    reference_dir: Path = Path("config/reference")
    rules_dir: Path = Path("config/rules")
    ontology_file: Path = Path("config/ontology/ssi_kinds.yaml")

    @model_validator(mode="after")
    def _absolutise(self) -> "PathSettings":
        root = self.project_root.resolve()
        object.__setattr__(self, "project_root", root)
        for name in type(self).model_fields:
            if name == "project_root":
                continue
            value = getattr(self, name)
            if not value.is_absolute():
                object.__setattr__(self, name, root / value)
        return self

    def ensure_writable_dirs(self) -> None:
        """Create the directories the pipeline writes to."""
        for name in (
            "input_dir",
            "output_dir",
            "logs_dir",
            "audit_dir",
            "review_queue_dir",
            "checkpoint_dir",
        ):
            getattr(self, name).mkdir(parents=True, exist_ok=True)


class InputGateSettings(BaseModel):
    """G1 input safety gate limits."""

    max_file_bytes: PositiveInt = 200 * 1024 * 1024
    max_pages: PositiveInt = 1000
    per_page_timeout_seconds: PositiveFloat = 30.0
    max_parse_memory_mb: PositiveInt = 2048
    allowed_magic_prefixes: tuple[bytes, ...] = (b"%PDF-",)


class OcrSettings(BaseModel):
    """Stage 3 local OCR thresholds and the single-retry policy."""

    confidence_tier1: float = Field(default=0.80, ge=0.0, le=1.0)
    confidence_floor: float = Field(default=0.60, ge=0.0, le=1.0)
    max_retries: int = Field(default=1, ge=0, le=1)
    orientations: tuple[int, ...] = (0, 90, 180, 270)
    native_layer_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    standard_upscale: PositiveFloat = 2.0
    aggressive_upscale: PositiveFloat = 3.0

    @model_validator(mode="after")
    def _floor_below_tier1(self) -> "OcrSettings":
        if self.confidence_floor > self.confidence_tier1:
            raise ValueError("ocr.confidence_floor must not exceed ocr.confidence_tier1")
        return self


class PrivacySettings(BaseModel):
    """Privacy-by-default policy for anything leaving the process."""

    external_fallback_enabled: bool = False
    mask_before_external_call: bool = True
    local_vision_model_skips_masking: bool = False
    purge_unmasked_artifacts: bool = True
    log_hash_length: int = Field(default=12, ge=8, le=64)


class LlmSettings(BaseModel):
    """Model governance and cost control."""

    provider: str = Field(default="azure_openai", pattern="^(azure_openai|anthropic)$")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_output_tokens: PositiveInt = 16000
    max_calls_per_chunk: PositiveInt = 3
    max_calls_per_document: PositiveInt = 500
    request_timeout_seconds: PositiveFloat = 120.0
    extraction_prompt_version: str = "extraction-v1"
    adjudicator_prompt_version: str = "adjudicator-v1"
    synthesis_prompt_version: str = "schema-synthesis-v1"


class ConfidenceSettings(BaseModel):
    """Stage 8 blending weights and the human-review floor."""

    weight_ocr: float = Field(default=0.30, ge=0.0, le=1.0)
    weight_llm: float = Field(default=0.30, ge=0.0, le=1.0)
    weight_format: float = Field(default=0.25, ge=0.0, le=1.0)
    weight_reference: float = Field(default=0.15, ge=0.0, le=1.0)
    review_floor: float = Field(default=0.70, ge=0.0, le=1.0)
    neutral_reference_score: float = Field(default=0.5, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> "ConfidenceSettings":
        total = self.weight_ocr + self.weight_llm + self.weight_format + self.weight_reference
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"confidence weights must sum to 1.0 (got {total:.6f})")
        return self


class LoggingSettings(BaseModel):
    """Structured logging. Redaction is applied by construction, not by convention."""

    level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    console: bool = True
    file_name: str = "ssi_extractor.log"
    json_lines: bool = True
    redact: bool = True


class AuditSettings(BaseModel):
    """G4 immutable audit log."""

    enabled: bool = True
    file_name_template: str = "{document_id}.audit.jsonl"
    hash_chain: bool = True


class Settings(BaseSettings):
    """Root settings object."""

    model_config = SettingsConfigDict(
        env_prefix="SSI_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    paths: PathSettings = Field(default_factory=PathSettings)
    input_gate: InputGateSettings = Field(default_factory=InputGateSettings)
    ocr: OcrSettings = Field(default_factory=OcrSettings)
    privacy: PrivacySettings = Field(default_factory=PrivacySettings)
    llm: LlmSettings = Field(default_factory=LlmSettings)
    confidence: ConfidenceSettings = Field(default_factory=ConfidenceSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    audit: AuditSettings = Field(default_factory=AuditSettings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton."""
    return Settings()
