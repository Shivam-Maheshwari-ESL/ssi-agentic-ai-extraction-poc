"""G1 — input safety gate.

The first pipeline node, and a hard stop: nothing reaches a parser, a model or
the filesystem until this passes. Every rejection is a typed, specific outcome
with an audit entry — never a traceback, never a silent skip. A password-protected
PDF in particular must be reported as exactly that, because "skipped quietly" is
indistinguishable from "contained no instructions".
"""

from __future__ import annotations

import hashlib
import re
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from ssi_extractor.config.settings import Settings, get_settings
from ssi_extractor.guardrails.g4_audit_log import AuditEvent, AuditLog
from ssi_extractor.observability.logging import get_logger

__all__ = ["GateOutcome", "GateResult", "InjectionFinding", "run_input_gate", "sanitize_injection"]

_logger = get_logger(__name__)

# Patterns that look like an attempt to address the model rather than to state a
# settlement instruction. Matched against document text, never against prompts.
_INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("instruction_override", re.compile(r"\b(ignore|disregard|forget)\b[^.\n]{0,40}\b(previous|above|prior|all)\b[^.\n]{0,40}\b(instruction|prompt|rule)s?\b", re.IGNORECASE)),
    ("role_assertion", re.compile(r"\b(you are now|act as|from now on you)\b", re.IGNORECASE)),
    ("system_prompt_probe", re.compile(r"\b(system prompt|developer message|your instructions)\b", re.IGNORECASE)),
    ("output_hijack", re.compile(r"\b(respond only with|output exactly|return the following json)\b", re.IGNORECASE)),
    ("exfiltration", re.compile(r"\b(api[_ -]?key|secret|password)\b[^.\n]{0,20}\b(send|post|email|upload)\b", re.IGNORECASE)),
)

_REDACTION_MARKER = "[REDACTED: suspected embedded instruction]"


class GateOutcome(StrEnum):
    """Why the gate let a document through, or did not."""

    ACCEPTED = "ACCEPTED"
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    NOT_A_PDF = "NOT_A_PDF"
    TOO_LARGE = "TOO_LARGE"
    TOO_MANY_PAGES = "TOO_MANY_PAGES"
    PASSWORD_REQUIRED = "PASSWORD_REQUIRED"
    CORRUPT = "CORRUPT"
    EMPTY = "EMPTY"


class InjectionFinding(BaseModel):
    """One suspected prompt-injection span found in document text."""

    model_config = ConfigDict(frozen=True)

    page: int
    pattern: str
    excerpt: str


class GateResult(BaseModel):
    """The gate's verdict, carried in pipeline state."""

    model_config = ConfigDict(frozen=True)

    outcome: GateOutcome
    path: Path
    document_id: str = ""
    message: str = ""
    size_bytes: int = 0
    page_count: int = 0
    sha256: str = ""
    injection_findings: tuple[InjectionFinding, ...] = ()

    @property
    def accepted(self) -> bool:
        return self.outcome is GateOutcome.ACCEPTED


class _Rejection(Exception):
    def __init__(self, outcome: GateOutcome, message: str) -> None:
        super().__init__(message)
        self.outcome = outcome
        self.message = message


def sanitize_injection(text: str, *, page: int) -> tuple[str, tuple[InjectionFinding, ...]]:
    """Neutralise suspected embedded instructions in extracted text.

    The span is replaced rather than deleted, so a human reading the evidence can
    see that something was removed and where. Extraction still receives the
    surrounding settlement data.
    """
    findings: list[InjectionFinding] = []
    cleaned = text
    for name, pattern in _INJECTION_PATTERNS:
        for match in list(pattern.finditer(cleaned)):
            findings.append(
                InjectionFinding(page=page, pattern=name, excerpt=match.group(0)[:120])
            )
        cleaned = pattern.sub(_REDACTION_MARKER, cleaned)
    return cleaned, tuple(findings)


def _fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _document_id(path: Path, fingerprint: str) -> str:
    """Stable per-document id: readable stem plus a short content hash.

    Content-addressed so a re-run of the same file resumes its checkpoint and
    audit chain, while an edited file starts a new one.
    """
    stem = re.sub(r"[^0-9A-Za-z._-]+", "_", path.stem)[:60] or "document"
    return f"{stem}-{fingerprint[:10]}"


def _validate_file(path: Path, settings: Settings) -> tuple[int, str]:
    if not path.is_file():
        raise _Rejection(GateOutcome.FILE_NOT_FOUND, f"No such file: {path}")

    size = path.stat().st_size
    if size == 0:
        raise _Rejection(GateOutcome.EMPTY, "File is empty (0 bytes).")
    if size > settings.input_gate.max_file_bytes:
        raise _Rejection(
            GateOutcome.TOO_LARGE,
            f"File is {size} bytes, above the {settings.input_gate.max_file_bytes}-byte cap.",
        )

    with path.open("rb") as handle:
        header = handle.read(8)
    if not any(header.startswith(prefix) for prefix in settings.input_gate.allowed_magic_prefixes):
        raise _Rejection(
            GateOutcome.NOT_A_PDF,
            f"Magic bytes {header[:5]!r} are not a PDF header; the extension is not trusted.",
        )
    return size, _fingerprint(path)


def _inspect_pdf(path: Path, settings: Settings) -> int:
    """Open the PDF defensively and return its page count."""
    import fitz

    try:
        document = fitz.open(path)
    except Exception as exc:
        raise _Rejection(GateOutcome.CORRUPT, f"PDF could not be opened: {exc}") from exc

    try:
        if document.needs_pass:
            raise _Rejection(
                GateOutcome.PASSWORD_REQUIRED,
                "PDF is encrypted and requires a password; it was not processed. "
                "Supply the password or provide a decrypted copy.",
            )
        page_count = document.page_count
        if page_count == 0:
            raise _Rejection(GateOutcome.CORRUPT, "PDF reports zero pages.")
        if page_count > settings.input_gate.max_pages:
            raise _Rejection(
                GateOutcome.TOO_MANY_PAGES,
                f"PDF has {page_count} pages, above the {settings.input_gate.max_pages}-page cap.",
            )
        # Touch the first page so a structurally broken body fails here, inside
        # the gate, rather than midway through the pipeline.
        document.load_page(0)
        return page_count
    finally:
        document.close()


def run_input_gate(
    path: Path | str,
    *,
    settings: Settings | None = None,
    audit: AuditLog | None = None,
) -> GateResult:
    """Validate one input file. Never raises for a bad document — it reports."""
    settings = settings or get_settings()
    path = Path(path)

    try:
        size, fingerprint = _validate_file(path, settings)
        page_count = _inspect_pdf(path, settings)
    except _Rejection as rejection:
        result = GateResult(
            outcome=rejection.outcome,
            path=path,
            document_id=re.sub(r"[^0-9A-Za-z._-]+", "_", path.stem)[:60] or "document",
            message=rejection.message,
            size_bytes=path.stat().st_size if path.is_file() else 0,
        )
        _logger.error(
            "G1 rejected %s: %s",
            path.name,
            rejection.message,
            extra={"outcome": rejection.outcome.value, "document": path.name},
        )
        if audit is not None:
            audit.record(
                AuditEvent.DOCUMENT_REJECTED,
                stage="G1",
                outcome=rejection.outcome.value,
                detail={"message": rejection.message, "file": path.name},
            )
        return result

    result = GateResult(
        outcome=GateOutcome.ACCEPTED,
        path=path,
        document_id=_document_id(path, fingerprint),
        message="accepted",
        size_bytes=size,
        page_count=page_count,
        sha256=fingerprint,
    )
    _logger.info(
        "G1 accepted %s (%s pages, %s bytes).",
        path.name,
        page_count,
        size,
        extra={"document_id": result.document_id, "sha256_prefix": fingerprint[:12]},
    )
    if audit is not None:
        audit.record(
            AuditEvent.DOCUMENT_ACCEPTED,
            stage="G1",
            outcome=GateOutcome.ACCEPTED.value,
            detail={"pages": page_count, "size_bytes": size, "sha256": fingerprint},
        )
    return result
