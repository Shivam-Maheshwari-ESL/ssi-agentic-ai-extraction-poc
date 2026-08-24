# ARCHITECTURE

Stage-by-stage design and the reasoning behind each decision. Read `PROGRESS.md` first for
current status; this file explains *why* the code is shaped the way it is, so a change does
not undo a deliberate choice.

---

## 1. Governing constraints

| Constraint | Where it shows up |
|---|---|
| **Python 3.14 only** — no second runtime | RapidOCR instead of PaddleOCR; every dependency dry-run verified |
| **Fully generic** — field names, count and nesting vary per document | Runtime schema discovery; validators dispatched by inferred *kind*, never by field name |
| **Not tuned to the samples** | Every heuristic is geometric, structural or kind-based; no sample wording appears in code |
| **Deterministic where possible** | LLM used for exactly three jobs: schema synthesis, extraction, adjudication |
| **Privacy by default** | Local processing; external vision fallback off; masking before anything leaves |
| **Guardrails are stages, not middleware** | G1–G4 are LangGraph nodes |
| **One concern per folder** | `agents/`, `schema/`, `stages/`, `guardrails/`, `validators/`, `llm/`, `utils/`, `observability/`, `config/`, `graph/`, `prompts/` |

---

## 2. Graph

`graph/builder.py` compiles one `StateGraph` over `graph/state.PipelineState`. Stage modules
are **pure functions**; nodes call them and place results in state. That keeps stages unit
testable and control flow in exactly one place.

```
input_gate ──(rejected)──▶ rejected ──▶ END
     │
     └─▶ classify ─▶ read_text ─▶ locate_chunk ──(no chunks)──▶ assemble ─▶ END
                                        │
                                        └─▶ discover_schema ─▶ extract ─▶ guard
                                            ─▶ validate ─▶ confidence ─▶ review_queue
                                            ─▶ assemble ─▶ END
```

Conditional edges: G1 rejection short-circuits; a document with no located instruction still
produces an output file recording that fact rather than silently nothing.

**The OCR retry loop is deliberately inside Stage 3, not the graph.** It must re-enhance *the
same region* with stronger parameters — a decision local to a region, not to a document.
Modelling it as graph edges would force region state into document state.

---

## 3. The generic schema engine

Nothing downstream may reference a literal field name. Four cooperating parts:

1. **Harvest** (`schema/harvest.py`) — table header bands become group paths; `key: value`
   pairs; SWIFT tags via a tag→meaning map; prose/form label-value adjacency. Each candidate
   carries **the sample values actually observed**.
2. **Kind inference** (`schema/kinds.py`) — decides what a value *is* from its shape, not its
   label: IBAN (mod-97), ISIN (check digit), LEI (mod-97-10), BIC (ISO 9362), CFI, sort code,
   percentage, currency, country (**name detector first** — names collide with CFI/BIC by
   length), date, account, enum, org/person name, free text. Labels are a weak tie-breaker
   only, and never override a checksum-confirmed kind. Confidence scales with how many samples
   passed, so one populated cell cannot claim the certainty of a full column.
3. **Synthesis agent** (`agents/schema_synthesis_agent.py`) — proposes fields, groups and the
   repeating unit from labels plus a **bounded** text sample. It proposes *structure, never
   values*; its answer is code-validated and rejected on any violation; the deterministic
   descriptor is always available as fallback. Its kind proposals are overridden by
   checksum-confirmed inference.
4. **Runtime model + prompt** (`schema/model_builder.py`, `schema/strict_schema.py`,
   `prompts/extraction.py`) — one descriptor drives the Pydantic model, the provider-strict
   JSON schema and the prompt's field catalogue.

Descriptors **union-merge** across chunks so the record array stays homogeneous: a field
absent from one instruction is emitted `NOT_APPLICABLE`, never omitted. Merging keeps the
*weaker* length constraint on conflict, because two markets legitimately use different account
lengths and keeping the stricter value would manufacture failures.

The descriptor is hashed (`descriptor_hash`) and recorded in G4, so any output can be traced
to the exact shape that produced it.

---

## 4. Stages

### G1 — input safety gate
Magic-byte check before any parse, size and page caps, `needs_pass` → **typed
`PASSWORD_REQUIRED` rejection** (never a silent skip), corrupt/zero-page rejection, SHA-256
fingerprint, and a **content-addressed document id** (`stem-hash10`) so a re-run resumes the
same audit chain while an edited file starts a new one. A prompt-injection pre-filter replaces
suspected embedded instructions with a marker and records the spans — replaced rather than
deleted, so a reviewer can see something was removed. Rejections return a typed result; the
gate never raises for a bad document.

### Stage 1 — region/layer classifier
Classification is **per layer within a region**, not per page. Text blocks are intersected
with image boxes: a native block overlapping an image yields two layers in one region — the
"typed text over a scanned stamp" case. `nativeTextPages / scannedPages / mixedPages` derive
from this composition, so one stamp makes a page `mixed`, not `scanned`. Calling that page
scanned would push good text through OCR and lose characters; calling it native would drop the
stamp.

### Stage 2 — enhancement
Operations are **conditional on measured metrics** (blur variance, skew, noise, contrast,
inversion), not applied unconditionally: binarising a clean 300-DPI render destroys detail.
Two tiers — `STANDARD`, and `AGGRESSIVE` only on retry (adds illumination normalisation and
adaptive threshold). Never runs on native-text layers.

### Stage 3 — OCR
RapidOCR (PP-OCR via onnxruntime): word text, confidence and bbox. Orientation trial
0/90/180/270 picking the most confident read, which handles rotated and upside-down scans; it
stops early once a read clears tier-1 rather than always paying for four passes. Below tier-1:
re-enhance `AGGRESSIVE` and re-OCR **once**. Only if that still fails, and only if
`privacy.external_fallback_enabled`, does the region become a vision-fallback candidate. Native
layers bypass OCR with confidence pinned at 1.0.

### Page text and table reconstruction (`stages/page_text.py`)
The subtlest module in the build; see `PROGRESS.md` §6.8 for the five root causes.

- **Fragments, not lines, are the unit.** Many PDFs emit each cell as its own text block, and
  OCR has no line structure at all.
- **Prose** uses rows clustered by vertical overlap.
- **Tables** use dual-axis ink bands: column corridors from a **projection profile counted in
  rows** (a column must carry content in ≥15% of rows, so one full-width banner cannot erase a
  boundary) and row bands from horizontal corridors.
- **The table body** is the longest contiguous run of rows whose fragments do **not straddle**
  column boundaries. Letterhead and contact blocks straddle; table cells do not. This
  separates the table from page furniture **without reading a word**, so it holds in any
  language and for scans.
- **Header group bands** straddle by design (`BENEFICIARY` over four columns) and are
  recovered by walking upward from the body.
- **Wrapped cell lines** (a row whose first column is empty) are merged into the row above —
  applied to the body only, because across the header boundary it would fuse letterhead into
  the column labels.

### Stage 4 — locate, classify, chunk
Layout classification per page: `TABLE_ROW`, `SECTION_BLOCK`, `NARRATIVE`, `FORM_KEY_VALUE`,
`SWIFT_MESSAGE`, `MULTI_COLUMN`, `AMENDMENT`, `UNRECOGNISED`. Then the matching cutter.

- **Relevance is scored on structural and kind signals**, not English keywords, so a non-English
  document still locates. Identifier kinds (BIC/IBAN/account/…) are strong evidence; front
  matter shape subtracts. Skipped regions are *recorded with a reason*, never silently dropped.
- **Chunk text is verbatim region text.** Rebuilding it from parsed pairs discarded every line
  the parser could not split and was pure data loss (`PROGRESS.md` §6.12).
- **Section headings stay in the text** — a `GBP` band is the settlement currency, not decoration.
- **Sections merge into one instruction unless they repeat labels.** Label repetition, not
  section count, is what actually indicates repetition.
- **Page-break stitching**: a row whose continuation appears on the next page with empty
  leading columns becomes one chunk citing both pages.
- **Amendment detection** (multilingual token set) marks unstated fields `NOT_APPLICABLE`,
  never `FAILED` — otherwise every unmentioned field in a delta document is a false negative.
- **The chunker fixes `instructionCount`, not the model.** This is what stops a 13-row table
  yielding 11 records.
- Only located chunks ever reach a model. The whole document never does.

### Stage 5 — extraction agent
One chunk → the runtime-built model. No tools, no retrieval: the only capability is returning
JSON. Document text is wrapped in `<untrusted_document_text>` under an instruction-precedence
preamble. Per-chunk call cap and output-token cap. Sees **unmasked** values deliberately — a
tokenised account number cannot be copied character-exactly, which is why masking is Stage 6.

### G2 — extraction guard
Four checks: schema (validate-and-retry in the agent, verdict recorded here), **hallucination**
(every populated value must appear in its own source chunk, tolerant of OCR noise via folded
sliding-window similarity), duplicate/conflict across instructions sharing an anchor-kind
identity, and document-level reference drift. Drift needs ≥6 identifier samples before a ratio
means anything, and reports `INDETERMINATE` when reference data is absent.

### Stage 6 — PII masking
Recognisers registered **per field kind** from the discovered schema, so masking follows
whatever fields the document turned out to have. Identifiers are **format-preserving** (a
10-digit account masks to a different 10-digit string), so a masked export still exercises
length and charset checks. Names/addresses are tokenised. The token↔value vault is in-memory
and purged when the document finishes: reversibility is needed *within* a run, and persisting
the map would recreate the exposure masking exists to prevent. The primary JSON stays
unmasked as the source of truth.

### Stage 7 — validation
Three levels, all dispatched by kind:
- **value** — format, checksum, length, charset, ISO membership, reference lookup;
- **field** — does the value's own shape agree with the kind this field holds (a BIC in an
  account field is a *field-assignment* error, not a value error);
- **chunk** — required-kind coverage for the repeating unit; amendments exempt by design.

A field never becomes `VALIDATED` on a failed check. Absence is `NOT_APPLICABLE`, not failure.
Ambiguity (e.g. a genuinely ambiguous day/month order) is neither pass nor fail: confidence is
reduced and the field is queued — that population is what the 7b adjudicator is for.

### G3 — review queue
Append-only JSONL per document, one entry per `FAILED` or sub-floor field, carrying value,
status, confidence **breakdown**, evidence, page and reasons. Supersede by reference, never
overwrite.

### Stage 8 — confidence and citation
`blended = w_ocr·ocr + w_llm·model + w_fmt·format + w_ref·reference`, weights in config
(validated to sum to 1.0), then scaled by field-level kind agreement, then reduced if evidence
cannot be traced to the chunk, then capped for `FAILED`. **Native layers pin OCR confidence at
1.0** so a mixed row is not penalised for one scanned sub-field. **Empty reference data scores
neutrally** — an unknown BIC must not look wrong when no directory is loaded.

Page resolution is a three-tier ladder: pages whose text actually contains the value → the
model's citation *where it agrees with the chunk it was given* → the chunk's pages. Tier one is
what makes a stitched row cite both pages.

### Stage 9 — assembler
Fixed metadata + `settlementInstructionRecords` following the discovered structure, plus a
`<name>.schema.json` sidecar describing that structure — a consumer needs it to interpret a
shape that varies per document. `rowAnalysis` states layout, pages, source (native vs OCR with
mean confidence), stitching, amendment status and counts.

### G4 — audit log
Write-once JSONL, one entry per decision, each carrying `prev_hash` (tamper evidence), model
id, prompt version, descriptor hash, stage, outcome, timestamp. Values appear only as salted
hashes plus a non-identifying shape summary (`len=10 classes=digit`). fsync per entry, so a
crash cannot lose the record of a decision that already took effect. `verify_chain()` reports
the first entry that breaks.

---

## 5. Cross-cutting

**Redaction by construction.** The logging filter redacts the *rendered* message and every
structured extra, so a `logger.info(f"... {account}")` written months from now is still
covered. Tokens are `<KIND:hash>` — stable (logs stay correlatable) and one-way.

**Provider abstraction.** `llm/port.py` exposes exactly one method: `complete_json`. Agents
never touch a vendor SDK. The Azure adapter negotiates `temperature` away when the deployment
rejects it, and retries only genuinely transient errors so a prompt or schema mistake fails
fast instead of hiding behind retries.

**Agent base.** Owns the call budget, audit stamping, untrusted-input delimiting and the
reject-and-retry policy, quoting the specific validation error back to the model. Repairing a
bad response silently is never correct — the repair would not appear in the audit trail.

---

## 6. Deliberate non-choices

| Rejected | Why |
|---|---|
| Fixed Pydantic contract from the sample's field names | Field set is not known before the document is read |
| Per-institution hand-authored schema files | Same objection: assumes the field set in advance |
| Second Python runtime for PaddleOCR | Explicitly rejected; RapidOCR runs the same models on 3.14 |
| Embedding-based relevance filtering | No embeddings deployment exists on the Azure resource |
| Seeding reference data from the samples | Would look accurate on these two files and wrong everywhere else |
| Trusting the reference JSONs as ground truth | They contain corrupted BICs and a wrong page-type count |
| Key-by-key comparison against the reference | Different shape by design; comparison is semantic |
| LLM deciding instruction count | The chunker decides; the model was observed producing 11 for 13 rows in the earlier POC |
