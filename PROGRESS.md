# PROGRESS

**Last updated:** 2026-08-21, end of session 1.
**Purpose:** hand this project to another machine or another coding assistant with no
questions needed. Read this file, then `ARCHITECTURE.md`, then `TROUBLESHOOTING.md`.

---

## 1. State in one paragraph

The pipeline runs **end to end** on both sample documents through a LangGraph graph: G1 input
gate → region/layer classification → image enhancement → RapidOCR with a single retry →
geometric table reconstruction → SSI location and chunking → runtime schema discovery
(deterministic harvest + synthesis agent) → extraction agent → G2 extraction guard → PII
masking → three-level validation → G3 review queue → blended confidence with page citations →
assembler, with a G4 hash-chained audit log written throughout. 19 unit tests pass. The
remaining work is listed in §7 and is mostly *additional* capability (vision fallback,
adjudicator, fixtures), not repair of what exists.

---

## 2. Verified results (final run, 2026-08-21 18:32)

Command: `python -m ssi_extractor.cli --masked-export --input-dir input_pdf`
(2 documents, ~4 minutes wall clock including masking).

### Inversis — native-text wide table, 12 columns, 1 page

| Metric | Value |
|---|---|
| Instructions located | **13 / 13** (matches reference) |
| Fields per instruction | 12 discovered |
| Field outcomes | 109 VALIDATED, 5 FAILED, 42 NOT_APPLICABLE (156 total) |
| Queued for review (G3) | 34 |
| G2 findings | 1 × CONFLICTING_INSTRUCTION, reference-drift verdict `OK` |
| Semantic recall vs reference | **0.981** (51 of 52 scored facts) |
| Semantic precision | 0.637 |
| Countries | 13 / 13 correct |
| PSET BICs | 13 / 13 correct, **including `IJSDJPJT` where the reference has the corrupted `JSDPJPT`** |

The single "missing" fact is `COUNTRY: 'AU'`: the document prints `AUSTRALIAN`, which is not
an ISO country name, so the comparator cannot canonicalise it. The extraction is faithful to
the page. Most "extra" facts are the 13 occurrences of `INVLESMMXXX`, which the reference
records as the corrupted `INVLESMXXX` — our value is correct.

### Vida Bank — image-only key:value form letter, 1 page

| Metric | Value |
|---|---|
| Stage 1 classification | `SCANNED` (0 native chars, 1 image at 81.6% page area) |
| OCR | 24 words, mean confidence **0.847**, cleared tier-1 on the **first** attempt → **no vision fallback** |
| Instructions located | 1 / 1 |
| Fields discovered | 12 (deterministic harvest 10 + synthesis) |
| Field outcomes | 8 VALIDATED, 4 FAILED |
| Semantic recall | 0.800 (an earlier identical run scored **1.000**; see §4 on run-to-run variance) |
| Values recovered | IBAN, sort code, account number, currency `GBP`, effective date `2025-04-25`, custody codes |

**A guardrail firing for real:** in this run the model transposed the SWIFT code
`BARCGB22` → `BARCBG22`. G2's hallucination check caught it (`HALLUCINATED_VALUE x2`,
the value does not appear in the source chunk), validation marked it `FAILED`, and G3 queued
it. The comparator independently flagged it as a near-miss at 0.88 similarity. This is the
single most important behaviour in the system and it works.

### Ground truth caveat — read before trusting any score
`../input output data/{inversis-json.txt, vida.txt}` are **contract references, not ground
truth**. Verified defects: `INVLESMXXX` (should be `INVLESMMXXX`, all 13 rows),
`CITIICATT` (`CITICATT`), `CITIIE1XXX` (`CITIIEA1XXX`), `JSDPJPT` (`IJSDJPJT`), a France tax
rate of `15%` that the PDF shows as `0%`; `vida.txt` claims `scannedPages: 0` for an
image-only PDF; leaves carry **no `page` key**. Precision below 1.0 is therefore partly the
reference being wrong.

---

## 3. What works, module by module

| Component | File | Status |
|---|---|---|
| Settings (all thresholds/weights, env-overridable) | `config/settings.py` | done |
| Credential loader (Spring `${VAR:default}`) | `config/credentials.py` | done, tested |
| Redaction + JSONL logging | `observability/{redaction,logging}.py` | done, tested |
| G4 audit log (hash chain, resume-aware, PII hashed) | `guardrails/g4_audit_log.py` | done, tested |
| G1 input gate (magic bytes, caps, `PASSWORD_REQUIRED`, injection sanitiser) | `guardrails/g1_input_gate.py` | done |
| Stage 1 region/layer classifier | `stages/classify.py` | done |
| Stage 2 enhancement (2 tiers, condition-driven) | `stages/enhance.py` | done |
| Stage 3 OCR + single retry + orientation trial | `stages/ocr.py` | done |
| Page text + geometric table reconstruction | `stages/page_text.py` | done — the hardest part, see §6 |
| Stage 4 locate/classify/chunk/stitch | `stages/locate_chunk.py` | done |
| Schema descriptor, kind inference, model builder, strict schema | `schema/*` | done |
| Harvesting + deterministic descriptor | `schema/harvest.py` | done |
| Format validators and checksums | `validators/formats.py` | done |
| Validator registry (kind-dispatched) | `validators/registry.py` | done |
| Reference data loaders (empty by default, neutral) | `validators/reference_data.py` | done |
| LLM port + Azure adapter + factory | `llm/*` | done |
| Agent base (budget, audit, delimiting, reject-and-retry) | `agents/base.py` | done |
| Extraction agent | `agents/extraction_agent.py` | done |
| Schema synthesis agent | `agents/schema_synthesis_agent.py` | done |
| G2 extraction guard | `guardrails/g2_extraction_guard.py` | done |
| Stage 6 PII masking (format-preserving, in-memory vault) | `stages/mask.py` | done |
| Stage 7 validation (value/field/chunk levels) | `stages/validate.py` | done |
| G3 review queue | `guardrails/g3_review_queue.py` | done |
| Stage 8 confidence + citation | `stages/confidence.py` | done |
| Stage 9 assembler + schema sidecar | `stages/assemble.py` | done |
| LangGraph state, nodes, routing | `graph/{state,builder}.py` | done |
| CLI | `cli.py` | done |
| Semantic comparator | `tests/compare_semantic.py` | done |

---

## 4. Known issues (open, with root cause where known)

1. **Run-to-run variance in extraction.** Two identical Vida runs scored recall 1.000 and
   0.800. The deployment (`gpt-5.4-mini`) rejects a non-default `temperature`, so the adapter
   drops it and the model runs at its default sampling. Mitigations already in place: the
   chunker fixes `instructionCount` (never the model), and G2 catches invented values.
   *Fix direction:* multi-pass self-consistency for identifier-kind fields, or a deployment
   that accepts `temperature=0`.
2. **OCR character confusion is not yet repaired.** `BO01` reads as `BOO1` (letter O vs
   zero). `stages/ocr_repair.py` — position-aware repair restricted to digit-only segments —
   is specified in `ARCHITECTURE.md` but **not implemented**.
3. **Vision fallback (3b) is not implemented.** `agents/vision_fallback_agent.py` does not
   exist. The OCR retry path that decides when it would be needed *is* implemented and
   currently never escalates on these samples (correct behaviour).
4. **Adjudicator agent (7b / 4b) is not implemented.** Validation marks ambiguous values and
   queues them; nothing arbitrates yet. `agents/adjudicator_agent.py` is the file to add.
5. **Presidio needs a spaCy model.** Without it, masking falls back to kind-driven rules:
   identifiers are masked format-preservingly, free-text names/addresses are tokenised
   wholesale. The engine is now checked *locally* before construction, because Presidio
   otherwise tries to download from github and blocks 15 s per attempt.
6. **The "Securities non Eligible in T2S" banner row** merges into the following data row in
   the Inversis table, producing one row whose country cell reads
   `AUSTRIA Securities non`. Cosmetic; the values are correct.
7. **Group band text can split across columns** (`GLOBAL CUSTODIAN` → groups `GLOBAL` and
   `CUSTODIAN`). Cosmetic naming only.
8. **Duplicate concept fields can still appear** when the synthesis agent nests a field the
   harvester found flat. Dedupe by name and by identical value sets is in place; a residual
   case remains (`custody.euroclear` vs `euroclear` was fixed, similar cases may exist).
9. **No checkpointer is attached yet.** `graph/builder.py` calls `graph.compile()` without
   `checkpointer=`; `langgraph-checkpoint-sqlite` is installed and `.checkpoints/` exists, so
   resume is a small change, not a design change.
10. **`audit/pending.audit.jsonl`** accumulates gate entries written before a document id is
    known. Harmless but untidy; pass the resolved id through instead.

---

## 5. Environment hazards that cost real time

- **The venv must not live in a cloud-synced folder.** With it inside Dropbox, bare
  `python -c "print('hello')"` took **22 s** after installing dependencies, and 4-second test
  runs timed out at 300 s.
- **VPN is required.** Off-VPN: `403 Access denied due to Virtual Network/Firewall rules`.
- **`paddlepaddle` has no Python 3.14 distribution.** PaddleOCR/PP-Structure cannot be used;
  RapidOCR runs the same PP-OCR models via onnxruntime. Do not "fix" this by adding a second
  Python runtime — that was explicitly rejected.
- **First OCR call loads ONNX models** (seconds). The engine is constructed once per run and
  shared; keep it that way.
- **Full-document runs take minutes**, mostly model latency (one call per instruction). Use
  `--no-llm` while working on the deterministic stages.

---

## 6. Bugs already found and fixed — do not re-introduce

Each of these was diagnosed from a real failure. The reasoning matters more than the diff.

1. **BIC regex was 6+1+2 instead of ISO 9362's 4+2+2(+3).** Accepted 9- and 12-character
   strings, rejected real 11-character BICs.
2. **Account numbers matched the phone pattern.** `^\+?[\d\s().-]{7,}\d$` matches a bare
   10-digit account. PHONE now requires a `+` prefix or a phone-shaped label.
3. **Ordinary words satisfied code checks.** `normalise_identifier()` upper-cases, so
   `"Create"` became `"CREATE"` and passed the 6-letter CFI pattern. `_reject_prose()` now
   disqualifies any value containing a lower-case letter for BIC/CFI/ISIN/LEI.
4. **Country *names* collided with code formats by length** — `CANADA`/`FRANCE` (6 letters)
   inferred as CFI, `NETHERLANDS` (11) as BIC. A name-only detector now runs **first**. This
   also silently penalised those fields' confidence during validation.
5. **`schwifty`'s registry made kind inference pathologically slow.** Registry lookup is now
   opt-in (`check_bic(..., use_registry=True)`), used only by Stage 7.
6. **`cv2.HoughLinesP` returns (N,1,4) or (N,4) depending on the build** → unpack error in
   deskew. Use `np.asarray(lines).reshape(-1, 4)`.
7. **Column splitting never fired on form lines.** The threshold used the *median* gap, which
   for `"Account Name   Barclays"` already *is* the column gap. Use a low percentile.
8. **Table reconstruction — five successive root causes** (the hardest problem in the build):
   - `page.find_tables()` returns **0 tables** for the Inversis PDF: it is whitespace-aligned
     with no ruling lines, so geometric reconstruction is mandatory.
   - The PDF emits **each cell as its own text block**, so `get_text("words")` block/line
     indices gave 209 single-cell "lines" → rows must be clustered geometrically.
   - Clustering rows *first* fused tall multi-line cells into neighbours → tables use
     **dual-axis ink bands** (column corridors + row bands) over raw fragments; only prose
     uses clustered rows.
   - Zero-occupancy column corridors do not survive a real page: one full-width banner covers
     every corridor → columns come from a **projection profile counted in rows** (a column
     must carry content in ≥15% of rows).
   - Front matter still landed inside the table → the body is the longest contiguous run of
     rows whose fragments **do not straddle** column boundaries; header *group bands* straddle
     by design and are recovered by walking upward.
9. **Repeated column headers collapsed two columns into one field** (two `Swift Address`
   columns under different bands). Repeats are numbered.
10. **A label hint could contradict the values** (`"Account Name"` → ACCOUNT_NUMBER for a
    column of institution names). An identifier hint is rejected when no value has a digit.
11. **Form sections were split into separate instructions.** A CASH block and a CUSTODY block
    describe one instruction. Sections merge unless they *repeat* labels.
12. **Form chunk text was rebuilt from parsed pairs, discarding unparsed lines.** OCR
    routinely returns a label and its value as one box, so `Account Name VidaBankLimited`,
    `Effective 25/04/2025` and the `GBP` band never reached the model. Chunks now carry
    **verbatim** region text; parsed pairs are kept separately for harvesting. This was pure
    data loss and is the most important fix in the list.
13. **Section headings were stripped from chunk text.** A heading is often a value (`GBP` =
    settlement currency), so headings stay in the text.
14. **Presidio blocked for minutes** trying to download a spaCy model. Availability is now
    checked locally first.

---

## 7. Next steps, in priority order

1. **Attach the SQLite checkpointer** in `graph/builder.py` (`compile(checkpointer=...)`) and
   thread a thread-id per document id, delivering the resume requirement.
2. **`stages/ocr_repair.py`** — position-aware `0/O`, `1/l`, `5/S`, `8/B` repair applied
   **only** inside segments whose inferred kind is digit-only. Wire into `stages/ocr.py`
   after `_assemble_text`. Fixes `BOO1` → `BO01`.
3. **`agents/adjudicator_agent.py`** (7b) — arbitrate `ValidationOutcome.AMBIGUOUS` fields and
   the 4b unrecognised-layout call. `stages/validate.py` already marks the population, and
   `LayoutPattern.UNRECOGNISED` already routes.
4. **`agents/vision_fallback_agent.py`** (3b) — masked-input vision escalation for regions
   still below `ocr.confidence_floor` after the retry. `OcrOutcome.needs_vision_fallback` is
   already computed. Must be exercised by a deliberately illegible fixture, or it becomes
   dead code.
5. **`tests/generate_fixtures.py`** — synthetic PDFs via PyMuPDF + OpenCV covering every
   declared combination: narrative letter, one-block-per-market, raw MT540/541 text,
   amendment/delta, multi-page table split across a page break, two-column letterhead, dense
   multi-instruction page, nested sub-accounts, non-English labels, rotated/skewed/watermarked
   scans, text-over-image combo regions, phone-camera capture, low-res fax.
6. **The mandated tests**: golden-file, malformed-PDF rejection, hallucination detection,
   mixed-region extraction, page-break-spanning table, narrative letter, amendment/delta,
   password-protected rejection, plus **retry-not-escalation** (assert the vision-fallback
   counter stays at zero) and **schema-genericity** (two fixtures with entirely different
   field names extract correctly with no code change).
7. **Cross-field rule engine** — `validators/cross_field.py` with declarative predicates over
   *kinds* in `config/rules/*.yaml` (e.g. a depository BIC's country characters agree with the
   settlement-country field). Deliberately not keyed to field names.
8. **Multi-pass self-consistency** for identifier-kind fields, to address issue §4.1.
9. **Milestone 2**: FastAPI service, Celery + Redis fan-out, at-rest encryption. All three are
   designed-for seams — stages are pure functions, the graph is the only orchestrator.

---

## 8. Test and probe inventory

```bash
pytest -q                                    # 19 unit tests, no network
python scripts/check_llm_connectivity.py     # credentials + endpoint + strict output
python scripts/probe_stage1.py               # G1 + region/layer classification
python scripts/probe_ocr.py                  # OCR the image-only sample, show the retry path
python scripts/probe_grid.py <pdf>            # table reconstruction + chunk cutting
python scripts/probe_bands.py <pdf>           # column/row band diagnostics
python scripts/probe_kv.py                    # key:value detection on the form sample
python scripts/probe_descriptor.py            # harvest -> descriptor -> strict schema
python scripts/probe_chunks.py                # chunk inventory for every input PDF
python tests/compare_semantic.py <ref> <out>  # semantic scoring
```

Debug in this order: **bands → grid → chunks → descriptor → prompt → output**. If a value is
wrong, check whether the field exists in the descriptor, then whether the value reached the
chunk text, then what the prompt said about it.

---

## 9. Sample data

`../input output data/` holds `inversis_ssi_document.pdf` + `inversis-json.txt` and
`vida_bank.pdf` + `vida.txt`. `../SSI Sample Documents/` holds degraded variants:
`_scan_40%/70%`, `_blurred`, and password-protected copies (password in `Password.txt`).
Both source documents are **single-page**; there is no multi-page, narrative, SWIFT-text or
amendment sample, which is why §7.5 (synthetic fixtures) matters.

Copies of the two source PDFs are already in `input_pdf/`.
