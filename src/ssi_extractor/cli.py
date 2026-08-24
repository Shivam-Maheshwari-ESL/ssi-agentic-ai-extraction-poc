"""CLI: ``input_pdf/*.pdf`` -> ``output_json/<name>.json``.

Deliberately plain ``argparse`` plus ``rich`` for progress — no extra dependency,
and every option maps to a settings field so nothing is configurable only here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ssi_extractor.config.settings import Settings, get_settings
from ssi_extractor.graph.builder import build_pipeline, run_document
from ssi_extractor.llm.port import LlmPort
from ssi_extractor.observability.logging import configure_logging, get_logger
from ssi_extractor.stages.ocr import OcrEngine
from ssi_extractor.validators.reference_data import ReferenceData

__all__ = ["main"]

_logger = get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ssi-extract",
        description=(
            "Extract settlement instructions from banking PDFs into JSON, with per-field "
            "validation, confidence and page citations."
        ),
    )
    parser.add_argument("pdfs", nargs="*", type=Path, help="Specific PDFs to process.")
    parser.add_argument("--input-dir", type=Path, default=None, help="Directory of PDFs to process.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Where to write JSON output.")
    parser.add_argument(
        "--masked-export",
        action="store_true",
        help="Also write a PII-masked copy of each output.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help=(
            "Run only the deterministic stages (gate, classification, OCR, chunking, "
            "schema discovery). Useful when the model endpoint is unreachable."
        ),
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Skip OCR; process native text layers only.",
    )
    parser.add_argument("--log-level", default=None, help="DEBUG, INFO, WARNING, ERROR.")
    return parser


def _resolve_inputs(arguments: argparse.Namespace, settings: Settings) -> list[Path]:
    if arguments.pdfs:
        return [path for path in arguments.pdfs if path.suffix.lower() == ".pdf"]
    directory = arguments.input_dir or settings.paths.input_dir
    return sorted(path for path in Path(directory).glob("*.pdf"))


def _build_llm(no_llm: bool) -> LlmPort | None:
    """Construct the provider, or report why extraction will be skipped.

    A missing or unreachable provider is not a crash: the deterministic stages
    still run and the output records that extraction did not happen, which is far
    more useful for diagnosis than a traceback.
    """
    if no_llm:
        _logger.warning("--no-llm set: deterministic stages only, no extraction.")
        return None

    from ssi_extractor.llm import build_llm

    try:
        return build_llm()
    except Exception as exc:
        _logger.error(
            "LLM provider unavailable (%s: %s). Continuing with deterministic stages only.",
            type(exc).__name__,
            exc,
        )
        return None


def main(argv: list[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    settings = get_settings()

    if arguments.output_dir is not None:
        settings = settings.model_copy(
            update={
                "paths": settings.paths.model_copy(
                    update={"output_dir": Path(arguments.output_dir).resolve()}
                )
            }
        )
    if arguments.log_level:
        settings = settings.model_copy(
            update={"logging": settings.logging.model_copy(update={"level": arguments.log_level.upper()})}
        )

    settings.paths.ensure_writable_dirs()
    configure_logging(settings)

    pdfs = _resolve_inputs(arguments, settings)
    if not pdfs:
        _logger.error("No PDFs found. Place files in %s or pass paths.", settings.paths.input_dir)
        return 2

    llm = _build_llm(arguments.no_llm)
    engine = OcrEngine()
    reference = ReferenceData.load(settings=settings)
    pipeline = build_pipeline(llm=llm, settings=settings, ocr_engine=engine, reference=reference)

    from rich.console import Console
    from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

    console = Console(stderr=False)
    failures = 0

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Extracting", total=len(pdfs))
        for pdf in pdfs:
            progress.update(task, description=f"Extracting {pdf.name}")
            try:
                state = run_document(
                    pdf,
                    settings=settings,
                    masked_export=arguments.masked_export,
                    compiled=pipeline,
                )
            except Exception as exc:
                failures += 1
                _logger.exception("Unhandled failure on %s: %s", pdf.name, exc)
                progress.advance(task)
                continue

            if state.rejected:
                failures += 1
                console.print(f"[red]REJECTED[/] {pdf.name}: {state.rejection_reason}")
            else:
                console.print(f"[green]OK[/] {state.summary()}")
                if state.output_path:
                    console.print(f"       -> {state.output_path}")
            progress.advance(task)

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
