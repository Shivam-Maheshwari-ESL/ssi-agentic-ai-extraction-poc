"""LangGraph wiring: every stage and every guardrail is a node.

The graph is the only orchestrator. Stage modules are pure functions; the nodes
here call them and put the result in state, which keeps the stages unit-testable
and the control flow in exactly one place.

Conditional edges implement the branches the spec requires: G1 rejection short-
circuits to the end, a document with no located instruction skips extraction, and
extraction/validation only run when there is something to extract. The OCR retry
loop lives inside Stage 3 rather than as graph edges, because it must re-enhance
the *same* region with stronger parameters — a decision local to the region, not
to the document.

Checkpointing uses SQLite so a re-run resumes rather than repeating work.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ssi_extractor.agents.base import AgentBudgetExceeded, AgentContext
from ssi_extractor.agents.extraction_agent import ExtractionAgent
from ssi_extractor.agents.schema_synthesis_agent import SchemaSynthesisAgent
from ssi_extractor.config.settings import Settings, get_settings
from ssi_extractor.guardrails.g1_input_gate import run_input_gate
from ssi_extractor.guardrails.g2_extraction_guard import run_extraction_guard
from ssi_extractor.guardrails.g3_review_queue import enqueue_for_review
from ssi_extractor.guardrails.g4_audit_log import AuditEvent, AuditLog
from ssi_extractor.graph.state import PipelineState
from ssi_extractor.llm.port import LlmError, LlmPort
from ssi_extractor.observability.logging import get_logger
from ssi_extractor.schema.harvest import build_deterministic_descriptor, harvest_candidates
from ssi_extractor.stages.assemble import assemble_document, write_outputs
from ssi_extractor.stages.classify import classify_document
from ssi_extractor.stages.confidence import score_fields
from ssi_extractor.stages.locate_chunk import locate_and_chunk
from ssi_extractor.stages.mask import PiiVault, mask_payload
from ssi_extractor.stages.ocr import OcrEngine
from ssi_extractor.stages.page_text import build_document_text
from ssi_extractor.stages.validate import validate_records
from ssi_extractor.validators.reference_data import ReferenceData

__all__ = ["build_pipeline", "run_document"]

_logger = get_logger(__name__)


class _Timer:
    """Small helper so every node records how long it took."""

    def __init__(self) -> None:
        self.started = time.perf_counter()

    @property
    def seconds(self) -> float:
        return time.perf_counter() - self.started


def build_pipeline(
    *,
    llm: LlmPort | None = None,
    settings: Settings | None = None,
    ocr_engine: OcrEngine | None = None,
    reference: ReferenceData | None = None,
) -> Any:
    """Compile the LangGraph pipeline.

    ``llm`` may be ``None``: the deterministic spine (G1 through chunking and
    schema discovery) still runs and reports what it found, which is what makes the
    pipeline diagnosable when the model endpoint is unreachable.
    """
    from langgraph.graph import END, StateGraph

    settings = settings or get_settings()
    engine = ocr_engine or OcrEngine()
    reference_data = reference or ReferenceData.load(settings=settings)

    def node_input_gate(state: PipelineState) -> dict[str, Any]:
        timer = _Timer()
        audit_probe = AuditLog("pending", settings=settings)
        gate = run_input_gate(state.pdf_path, settings=settings, audit=audit_probe)
        update: dict[str, Any] = {
            "gate": gate,
            "document_name": state.pdf_path.name,
            "document_id": gate.document_id,
        }
        if not gate.accepted:
            update["rejected"] = True
            update["rejection_reason"] = f"{gate.outcome.value}: {gate.message}"
        state.record_stage("G1_input_gate", ok=gate.accepted, detail=gate.outcome.value, seconds=timer.seconds)
        update["stages"] = state.stages
        return update

    def node_classify(state: PipelineState) -> dict[str, Any]:
        timer = _Timer()
        composition = classify_document(state.pdf_path)
        state.record_stage("1_classify_regions", detail=composition.summary(), seconds=timer.seconds)
        return {"composition": composition, "stages": state.stages}

    def node_read_text(state: PipelineState) -> dict[str, Any]:
        timer = _Timer()
        assert state.composition is not None
        document_text = build_document_text(state.composition, engine=engine)
        detail = (
            f"{sum(len(page.lines) for page in document_text.pages)} line(s), "
            f"{sum(len(page.tables) for page in document_text.pages)} table(s)"
        )
        state.record_stage("2_3_enhance_and_ocr", detail=detail, seconds=timer.seconds)
        return {"document_text": document_text, "stages": state.stages}

    def node_locate_chunk(state: PipelineState) -> dict[str, Any]:
        timer = _Timer()
        assert state.document_text is not None
        located = locate_and_chunk(state.document_text)
        state.record_stage(
            "4_locate_and_chunk",
            ok=bool(located.chunks),
            detail=f"{len(located.chunks)} chunk(s), pattern {located.dominant_pattern.value}",
            seconds=timer.seconds,
        )
        return {"located": located, "stages": state.stages}

    def node_discover_schema(state: PipelineState) -> dict[str, Any]:
        timer = _Timer()
        assert state.located is not None
        audit = AuditLog(state.document_id, settings=settings)

        candidates = harvest_candidates(state.located)
        descriptor = build_deterministic_descriptor(
            state.located, candidates, document_id=state.document_id
        )
        detail = f"deterministic: {len(descriptor.fields)} field(s)"
        usage = state.usage
        calls = state.llm_calls

        if llm is not None:
            agent = SchemaSynthesisAgent(llm, document_name=state.document_name, settings=settings)
            context = AgentContext(document_id=state.document_id, audit=audit, settings=settings)
            try:
                result = agent.run(
                    agent.build_user_prompt(
                        candidates,
                        list(state.located.chunks),
                        layout_pattern=state.located.dominant_pattern.value,
                    ),
                    context=context,
                    audit_event=AuditEvent.SCHEMA_DESCRIPTOR_BUILT,
                )
                usage = usage + result.usage
                calls += context.calls_made
                if result.ok and result.value is not None:
                    descriptor = agent.to_descriptor(
                        result.value,
                        deterministic=descriptor,
                        candidates=candidates,
                        document_id=state.document_id,
                    )
                    detail = f"synthesised: {len(descriptor.fields)} field(s)"
                else:
                    detail += " (synthesis rejected; deterministic descriptor kept)"
            except (AgentBudgetExceeded, LlmError) as exc:
                detail += f" (synthesis unavailable: {type(exc).__name__})"

        audit.set_schema_descriptor_hash(descriptor.descriptor_hash)
        audit.record(
            AuditEvent.SCHEMA_DESCRIPTOR_BUILT,
            stage="4c",
            outcome=descriptor.source.value,
            detail={"fields": len(descriptor.fields), "hash": descriptor.descriptor_hash},
        )
        state.record_stage("4c_schema_discovery", detail=detail, seconds=timer.seconds)
        return {
            "descriptor": descriptor,
            "usage": usage,
            "llm_calls": calls,
            "stages": state.stages,
        }

    def node_extract(state: PipelineState) -> dict[str, Any]:
        timer = _Timer()
        assert state.located is not None and state.descriptor is not None

        if llm is None:
            state.record_stage(
                "5_extraction",
                ok=False,
                detail="no LLM provider available; extraction skipped",
                seconds=timer.seconds,
            )
            return {
                "records": [None] * len(state.located.chunks),
                "schema_failures": {
                    index: "no LLM provider available"
                    for index in range(len(state.located.chunks))
                },
                "errors": [*state.errors, "extraction skipped: no LLM provider available"],
                "stages": state.stages,
            }

        audit = AuditLog(state.document_id, settings=settings)
        audit.set_schema_descriptor_hash(state.descriptor.descriptor_hash)
        agent = ExtractionAgent(
            llm, state.descriptor, document_name=state.document_name, settings=settings
        )
        context = AgentContext(document_id=state.document_id, audit=audit, settings=settings)

        records: list[Any] = []
        failures: dict[int, str] = {}
        chunks = list(state.located.chunks)

        for chunk in chunks:
            try:
                result = agent.run(
                    agent.build_user_prompt(chunk, chunk_total=len(chunks)),
                    context=context,
                    audit_event=AuditEvent.FIELD_EXTRACTED,
                    audit_detail={"chunk": chunk.index + 1, "pages": list(chunk.pages)},
                )
            except AgentBudgetExceeded as exc:
                failures[chunk.index] = str(exc)
                records.append(None)
                _logger.error("Extraction budget exhausted: %s", exc)
                continue

            if result.ok:
                records.append(result.value)
            else:
                records.append(None)
                failures[chunk.index] = "; ".join(result.failures[-2:]) or "extraction rejected"

        extracted = sum(1 for record in records if record is not None)
        state.record_stage(
            "5_extraction",
            ok=extracted > 0,
            detail=f"{extracted}/{len(chunks)} instruction(s) extracted",
            seconds=timer.seconds,
        )
        return {
            "records": records,
            "schema_failures": failures,
            "usage": state.usage + context.usage,
            "llm_calls": state.llm_calls + context.calls_made,
            "stages": state.stages,
        }

    def node_guard(state: PipelineState) -> dict[str, Any]:
        timer = _Timer()
        assert state.located is not None and state.descriptor is not None
        audit = AuditLog(state.document_id, settings=settings)
        report = run_extraction_guard(
            state.records,
            list(state.located.chunks),
            state.descriptor,
            settings=settings,
            audit=audit,
            schema_failures=state.schema_failures,
        )
        state.record_stage("G2_extraction_guard", ok=report.ok, detail=report.summary(), seconds=timer.seconds)
        return {"guard": report, "stages": state.stages}

    def node_validate(state: PipelineState) -> dict[str, Any]:
        timer = _Timer()
        assert state.located is not None and state.descriptor is not None and state.guard is not None
        audit = AuditLog(state.document_id, settings=settings)
        audit.set_schema_descriptor_hash(state.descriptor.descriptor_hash)
        report = validate_records(
            state.records,
            list(state.located.chunks),
            state.descriptor,
            state.guard,
            settings=settings,
            audit=audit,
        )
        counts = report.counts()
        state.record_stage(
            "7_validation",
            detail=f"{counts['VALIDATED']} validated, {counts['FAILED']} failed",
            seconds=timer.seconds,
        )
        return {"validation": report, "stages": state.stages}

    def node_confidence(state: PipelineState) -> dict[str, Any]:
        timer = _Timer()
        assert state.validation is not None and state.located is not None
        scored = score_fields(
            list(state.validation.fields),
            list(state.located.chunks),
            document_text=state.document_text,
            reference=reference_data,
            settings=settings,
        )
        flagged = sum(1 for field in scored if field.needs_review)
        state.record_stage(
            "8_confidence_and_citation",
            detail=f"{len(scored)} field(s), {flagged} flagged",
            seconds=timer.seconds,
        )
        return {"scored_fields": scored, "stages": state.stages}

    def node_review_queue(state: PipelineState) -> dict[str, Any]:
        timer = _Timer()
        assert state.descriptor is not None and state.located is not None
        audit = AuditLog(state.document_id, settings=settings)
        queue = enqueue_for_review(
            state.scored_fields,
            list(state.located.chunks),
            state.descriptor,
            document_id=state.document_id,
            document_name=state.document_name,
            settings=settings,
            audit=audit,
        )
        state.record_stage("G3_review_queue", detail=f"{queue.count} entry(ies)", seconds=timer.seconds)
        return {"review_count": queue.count, "stages": state.stages}

    def node_assemble(state: PipelineState) -> dict[str, Any]:
        timer = _Timer()
        assert state.composition is not None and state.located is not None and state.descriptor is not None
        assembled = assemble_document(
            document_name=state.document_name,
            composition=state.composition,
            located=state.located,
            descriptor=state.descriptor,
            scored_fields=state.scored_fields,
            review_count=state.review_count,
        )

        masked_payload = None
        if state.masked_export:
            vault = PiiVault()
            try:
                masked_payload = mask_payload(
                    assembled.payload, state.descriptor, vault=vault, settings=settings
                ).payload
            finally:
                # The unmasked-token map exists only for this document.
                vault.purge()

        written = write_outputs(
            assembled,
            input_stem=state.pdf_path.stem,
            masked_payload=masked_payload,
            settings=settings,
        )
        audit = AuditLog(state.document_id, settings=settings)
        audit.record(
            AuditEvent.DOCUMENT_COMPLETED,
            stage="9",
            outcome=str(written.payload.get("status", "")),
            detail={
                "instructions": written.instruction_count,
                "review_queued": state.review_count,
                "output": str(written.output_path),
            },
        )
        state.record_stage("9_assemble", detail=f"{written.instruction_count} record(s)", seconds=timer.seconds)
        return {
            "payload": written.payload,
            "output_path": written.output_path,
            "sidecar_path": written.sidecar_path,
            "masked_path": written.masked_path,
            "stages": state.stages,
        }

    def node_rejected(state: PipelineState) -> dict[str, Any]:
        _logger.error("Document rejected: %s", state.rejection_reason)
        return {}

    # --- routing ---------------------------------------------------------

    def route_after_gate(state: PipelineState) -> str:
        return "rejected" if state.rejected else "classify"

    def route_after_chunking(state: PipelineState) -> str:
        # Nothing located means nothing to extract; the document still gets an
        # output file recording that, rather than silently producing nothing.
        if state.located is None or not state.located.chunks:
            return "assemble"
        return "discover_schema"

    graph = StateGraph(PipelineState)
    graph.add_node("input_gate", node_input_gate)
    graph.add_node("rejected", node_rejected)
    graph.add_node("classify", node_classify)
    graph.add_node("read_text", node_read_text)
    graph.add_node("locate_chunk", node_locate_chunk)
    graph.add_node("discover_schema", node_discover_schema)
    graph.add_node("extract", node_extract)
    graph.add_node("guard", node_guard)
    graph.add_node("validate", node_validate)
    graph.add_node("confidence", node_confidence)
    graph.add_node("review_queue", node_review_queue)
    graph.add_node("assemble", node_assemble)

    graph.set_entry_point("input_gate")
    graph.add_conditional_edges(
        "input_gate", route_after_gate, {"rejected": "rejected", "classify": "classify"}
    )
    graph.add_edge("rejected", END)
    graph.add_edge("classify", "read_text")
    graph.add_edge("read_text", "locate_chunk")
    graph.add_conditional_edges(
        "locate_chunk",
        route_after_chunking,
        {"discover_schema": "discover_schema", "assemble": "assemble"},
    )
    graph.add_edge("discover_schema", "extract")
    graph.add_edge("extract", "guard")
    graph.add_edge("guard", "validate")
    graph.add_edge("validate", "confidence")
    graph.add_edge("confidence", "review_queue")
    graph.add_edge("review_queue", "assemble")
    graph.add_edge("assemble", END)

    return graph.compile()


def run_document(
    pdf_path: Path | str,
    *,
    llm: LlmPort | None = None,
    settings: Settings | None = None,
    ocr_engine: OcrEngine | None = None,
    reference: ReferenceData | None = None,
    masked_export: bool = False,
    compiled: Any | None = None,
) -> PipelineState:
    """Run the pipeline over one PDF and return the final state."""
    settings = settings or get_settings()
    pipeline = compiled or build_pipeline(
        llm=llm, settings=settings, ocr_engine=ocr_engine, reference=reference
    )
    initial = PipelineState(pdf_path=Path(pdf_path), masked_export=masked_export)
    final = pipeline.invoke(initial)
    # LangGraph returns the state as a mapping; rebuild the typed model so callers
    # keep the same interface whichever version of the library is installed.
    return PipelineState.model_validate(final) if isinstance(final, dict) else final
