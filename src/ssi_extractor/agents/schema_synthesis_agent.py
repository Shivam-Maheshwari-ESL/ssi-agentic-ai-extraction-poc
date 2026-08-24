"""Stage 4c — the schema synthesis agent.

Deterministic harvesting recovers the fields whose structure is visible in the
geometry: table columns, punctuated key:value pairs, SWIFT tags. It cannot always
recover fields whose label and value arrived fused in a single OCR box, or fields
a prose sentence states without a label at all. This agent closes that gap: it
reads the labels and a sample of chunk text and proposes the document's field
list, groups and repeating unit.

Three properties make it safe to have a model decide the shape:

* It proposes *structure*, never values. No extracted value depends on it.
* Its answer is validated by code and rejected on any violation, and the
  deterministic descriptor remains available, so an unreachable or misbehaving
  model degrades the schema rather than failing the document.
* The resulting descriptor is hashed into the audit log, so the shape that
  produced an output can always be recovered.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ssi_extractor.agents.base import UNTRUSTED_PREAMBLE, BaseAgent
from ssi_extractor.config.settings import Settings
from ssi_extractor.llm.port import LlmPort
from ssi_extractor.schema.descriptor import (
    Cardinality,
    DescriptorSource,
    FieldDescriptor,
    FieldKind,
    RepeatingUnit,
    SchemaDescriptor,
    slugify,
)
from ssi_extractor.schema.harvest import CandidateField
from ssi_extractor.schema.kinds import infer_kind
from ssi_extractor.stages.locate_chunk import InstructionChunk

__all__ = ["SchemaSynthesisAgent", "SynthesisedField", "SynthesisedSchema"]

_MAX_FIELDS = 80
_MAX_SAMPLE_CHARS = 2600

_SYSTEM_PROMPT = """You determine the field structure of a settlement-instruction document.

{untrusted_preamble}

You are given the labels a deterministic parser already recovered, and a sample of the document's own text. Return the complete list of fields one instruction in this document contains.

Rules:
1. Propose fields, never values. Do not return any account number, code or name as a field name.
2. Include every field the deterministic parser found, keeping its label, unless the label is plainly a fragment of another label.
3. Add fields that are visible in the sample text but missing from the parser's list — most often because a label and its value were captured as one line. Use the document's own wording for the label.
4. Do not invent fields the document does not contain, and do not add fields because they are common in settlement instructions. If the document has no ISIN, there is no ISIN field.
5. Group fields the way the document groups them, using the document's own section or header names. Leave group empty when the document has no grouping.
6. Set "kind" from the shape of the values that field holds, using only the listed kinds.
7. Set "cardinality" to MULTI only when one instruction genuinely holds several values for that field.
8. In "repeating_unit_description", say in one sentence what constitutes a single instruction in this document.

Return only the JSON object required by the schema."""

_USER_PROMPT = """Document: {document_name}
Layout pattern: {layout_pattern}
Instructions located: {chunk_count}

Fields the deterministic parser recovered (label — group — kind — sample values):
{candidates}

Sample of the document's own text:
{sample}"""

_ALLOWED_KINDS = ", ".join(kind.value for kind in FieldKind)


class SynthesisedField(BaseModel):
    """One field proposed by the agent."""

    model_config = ConfigDict(extra="forbid")

    label: str
    group: str = ""
    kind: str = FieldKind.UNKNOWN.value
    cardinality: str = Cardinality.SINGLE.value

    def to_descriptor(self, sample_values: tuple[str, ...]) -> FieldDescriptor:
        """Convert to a descriptor field, re-deriving hints from observed values.

        The agent's ``kind`` is accepted only as a proposal: hints and cardinality
        come from the values the document actually contains, so a confident but
        wrong kind cannot loosen a downstream length or checksum check.
        """
        inference = infer_kind(list(sample_values), label=self.label)
        try:
            kind = FieldKind(self.kind)
        except ValueError:
            kind = inference.kind

        # A checksum-confirmed kind outranks the agent's opinion; otherwise the
        # agent's reading of an ambiguous field is kept.
        if inference.confidence >= 0.8 and inference.kind is not FieldKind.UNKNOWN:
            kind = inference.kind

        groups = tuple(part.strip() for part in self.group.split("/") if part.strip())
        return FieldDescriptor(
            name=slugify(self.label),
            label=self.label.strip(),
            group_path=groups,
            kind=kind,
            kind_confidence=max(inference.confidence, 0.4),
            cardinality=(
                Cardinality.MULTI
                if self.cardinality.upper() == Cardinality.MULTI.value
                else inference.cardinality
            ),
            hints=inference.hints,
            source_pattern="schema_synthesis",
        )


class SynthesisedSchema(BaseModel):
    """The agent's proposed document schema."""

    model_config = ConfigDict(extra="forbid")

    fields: list[SynthesisedField] = Field(default_factory=list)
    repeating_unit_description: str = ""


class SchemaSynthesisAgent(BaseAgent[SynthesisedSchema]):
    """Proposes the document's field structure from labels and sample text."""

    prompt_version_setting = "synthesis_prompt_version"

    def __init__(
        self,
        llm: LlmPort,
        *,
        document_name: str,
        settings: Settings | None = None,
    ) -> None:
        super().__init__(llm, settings=settings)
        self._document_name = document_name

    @property
    def name(self) -> str:
        return "schema_synthesis_agent"

    def system_prompt(self) -> str:
        return _SYSTEM_PROMPT.format(untrusted_preamble=UNTRUSTED_PREAMBLE)

    def json_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "fields": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string"},
                            "group": {"type": "string"},
                            "kind": {"type": "string", "enum": [kind.value for kind in FieldKind]},
                            "cardinality": {"type": "string", "enum": ["SINGLE", "MULTI"]},
                        },
                        "required": ["label", "group", "kind", "cardinality"],
                        "additionalProperties": False,
                    },
                },
                "repeating_unit_description": {"type": "string"},
            },
            "required": ["fields", "repeating_unit_description"],
            "additionalProperties": False,
        }

    def parse(self, payload: dict[str, Any]) -> SynthesisedSchema:
        """Validate the proposal — reject rather than repair."""
        try:
            schema = SynthesisedSchema.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(f"schema proposal did not validate: {exc.error_count()} error(s)") from exc

        if not schema.fields:
            raise ValueError("schema proposal contained no fields")
        if len(schema.fields) > _MAX_FIELDS:
            raise ValueError(
                f"schema proposal contained {len(schema.fields)} fields, above the cap of {_MAX_FIELDS}"
            )

        seen: set[tuple[str, str]] = set()
        for field in schema.fields:
            if not field.label.strip():
                raise ValueError("a proposed field had an empty label")
            if not slugify(field.label):
                raise ValueError(f"proposed label {field.label!r} does not yield a usable name")
            key = (field.group.strip().lower(), slugify(field.label))
            if key in seen:
                raise ValueError(f"proposed field {field.label!r} is duplicated within its group")
            seen.add(key)
        return schema

    def build_user_prompt(
        self,
        candidates: list[CandidateField],
        chunks: list[InstructionChunk],
        *,
        layout_pattern: str,
    ) -> str:
        """Render the proposal request from labels and a bounded text sample.

        Only a sample is sent: enough text to reveal the structure, never the whole
        document. That keeps this call cheap and bounded, and keeps the pipeline's
        promise that the whole document is never handed to a model.
        """
        if candidates:
            candidate_lines = "\n".join(
                f"- {candidate.label} — {'/'.join(candidate.group_path) or '(no group)'} — "
                f"{(candidate.kind_hint.value if candidate.kind_hint else 'unknown')} — "
                f"samples: {', '.join(candidate.sample_values[:3]) or '(none)'}"
                for candidate in candidates
            )
        else:
            candidate_lines = "- (the deterministic parser recovered no labels)"

        sample_parts: list[str] = []
        budget = _MAX_SAMPLE_CHARS
        for chunk in chunks[:3]:
            text = chunk.text[:budget]
            sample_parts.append(f"[chunk {chunk.index + 1}, page {chunk.page_label}]\n{text}")
            budget -= len(text)
            if budget <= 0:
                break

        return _USER_PROMPT.format(
            document_name=self._document_name,
            layout_pattern=layout_pattern,
            chunk_count=len(chunks),
            candidates=candidate_lines,
            sample=self.wrap_untrusted("\n\n".join(sample_parts)),
        )

    def to_descriptor(
        self,
        proposal: SynthesisedSchema,
        *,
        deterministic: SchemaDescriptor,
        candidates: list[CandidateField],
        document_id: str,
    ) -> SchemaDescriptor:
        """Turn an accepted proposal into a descriptor, keeping observed evidence.

        Sample values and page numbers come from the deterministic harvest wherever
        a proposed field matches one, so citations and hints stay grounded in what
        was actually seen on the page.
        """
        by_name = {candidate.name: candidate for candidate in candidates}
        fields: list[FieldDescriptor] = []
        used_paths: set[str] = set()

        for proposed in proposal.fields:
            name = slugify(proposed.label)
            candidate = by_name.get(name)
            samples = candidate.sample_values if candidate else ()
            field = proposed.to_descriptor(samples)
            if candidate is not None:
                field = field.model_copy(update={"pages": candidate.pages})

            path = field.path
            if path in used_paths:
                continue
            used_paths.add(path)
            fields.append(field)

        # Anything the deterministic harvest found but the agent dropped is kept:
        # losing a field that was demonstrably on the page would be a regression,
        # and an extra NOT_APPLICABLE field costs nothing.
        #
        # Matching is by *name* as well as path, because the agent commonly places
        # a field the harvester found at the top level inside a group (or the
        # reverse). Keying on the full path alone would then emit both, and the
        # same printed value would be reported twice.
        used_names = {field.name for field in fields}
        for field in deterministic.fields:
            if field.path in used_paths or field.name in used_names:
                continue
            fields.append(field)
            used_paths.add(field.path)
            used_names.add(field.name)

        unit = deterministic.repeating_unit
        if proposal.repeating_unit_description.strip():
            unit = unit.model_copy(
                update={"description": proposal.repeating_unit_description.strip()}
            )

        return SchemaDescriptor(
            document_id=document_id,
            source=DescriptorSource.SYNTHESISED,
            fields=tuple(fields),
            repeating_unit=unit,
            notes=(*deterministic.notes, f"{len(proposal.fields)} field(s) proposed by synthesis"),
        )
