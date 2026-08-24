"""Shared fixtures. Every test runs against an isolated temporary project root."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest

from ssi_extractor.config.settings import PathSettings, Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings rooted at a temp directory, so no test touches real project folders."""
    configured = Settings(paths=PathSettings(project_root=tmp_path))
    configured.paths.ensure_writable_dirs()
    return configured


@pytest.fixture(autouse=True)
def _clean_logging() -> Iterator[None]:
    """Detach package log handlers between tests so file handles do not leak."""
    import logging

    logger = logging.getLogger("ssi_extractor")
    original_handlers = list(logger.handlers)
    original_flag = getattr(logger, "_ssi_configured", False)
    yield
    for handler in list(logger.handlers):
        if handler not in original_handlers:
            handler.close()
            logger.removeHandler(handler)
    logger._ssi_configured = original_flag  # type: ignore[attr-defined]
