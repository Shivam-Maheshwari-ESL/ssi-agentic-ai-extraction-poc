"""Structured, redacted-by-construction logging to console and file.

Two handlers are installed: a human-readable console handler and a JSON-lines
file handler under ``logs/``. Both sit behind the same redaction filter, so a
raw account number cannot reach either one regardless of how the log call was
written. UTF-8 is forced on the file handler.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ssi_extractor.config.settings import Settings, get_settings
from ssi_extractor.observability.redaction import redact, redact_structure

__all__ = ["RedactingFilter", "bind_context", "configure_logging", "get_logger"]

_LOGGER_ROOT = "ssi_extractor"
_RESERVED_RECORD_KEYS = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {
    "message",
    "asctime",
    "taskName",
}


class RedactingFilter(logging.Filter):
    """Redacts the rendered message and every structured extra on the record."""

    def __init__(self, *, hash_length: int, enabled: bool) -> None:
        super().__init__()
        self._hash_length = hash_length
        self._enabled = enabled

    def filter(self, record: logging.LogRecord) -> bool:
        if not self._enabled:
            return True

        # Render once here so redaction sees the final text, including args.
        rendered = record.getMessage()
        record.msg = redact(rendered, hash_length=self._hash_length)
        record.args = None

        for key, value in list(record.__dict__.items()):
            if key in _RESERVED_RECORD_KEYS or key.startswith("_"):
                continue
            record.__dict__[key] = redact_structure(value, hash_length=self._hash_length)
        return True


class JsonLinesFormatter(logging.Formatter):
    """One JSON object per line, with structured extras preserved."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _RESERVED_RECORD_KEYS or key.startswith("_"):
                continue
            payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(settings: Settings | None = None) -> logging.Logger:
    """Install console and file handlers on the package logger. Idempotent."""
    settings = settings or get_settings()
    logger = logging.getLogger(_LOGGER_ROOT)
    logger.setLevel(settings.logging.level)
    logger.propagate = False

    if getattr(logger, "_ssi_configured", False):
        return logger

    redacting = RedactingFilter(
        hash_length=settings.privacy.log_hash_length,
        enabled=settings.logging.redact,
    )

    if settings.logging.console:
        console = logging.StreamHandler(stream=sys.stderr)
        console.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        console.addFilter(redacting)
        logger.addHandler(console)

    settings.paths.logs_dir.mkdir(parents=True, exist_ok=True)
    log_path: Path = settings.paths.logs_dir / settings.logging.file_name
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(
        JsonLinesFormatter() if settings.logging.json_lines else logging.Formatter()
    )
    file_handler.addFilter(redacting)
    logger.addHandler(file_handler)

    logger._ssi_configured = True  # type: ignore[attr-defined]
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the configured package root."""
    suffix = name.removeprefix(f"{_LOGGER_ROOT}.").removeprefix(_LOGGER_ROOT)
    return logging.getLogger(_LOGGER_ROOT if not suffix else f"{_LOGGER_ROOT}.{suffix.lstrip('.')}")


def bind_context(logger: logging.Logger, **context: Any) -> logging.LoggerAdapter[logging.Logger]:
    """Attach per-document context (document id, page, stage) to every record."""
    return logging.LoggerAdapter(logger, extra=context)
