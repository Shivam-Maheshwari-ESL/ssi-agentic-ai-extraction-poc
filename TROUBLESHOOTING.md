# TROUBLESHOOTING

Observed failure modes with their causes and fixes, plus how to debug each stage. Every entry
here was hit for real during development.

---

## 1. Setup and environment

### `403 Access denied due to Virtual Network/Firewall rules`
The Azure endpoint is behind a network ACL. **Connect the corporate VPN.** Verify with
`python scripts/check_llm_connectivity.py` — it prints the resolved endpoint and deployment
(never the key). The deterministic stages work without it: `python -m ssi_extractor.cli --no-llm`.

### Everything is inexplicably slow (20 s to start Python, tests timing out)
The virtual environment is inside a cloud-synced folder (Dropbox/OneDrive/Drive) and the sync
client is indexing several hundred megabytes of newly installed wheels. Move the venv outside
the synced tree:

```bash
python -m venv C:/venvs/ssi
source C:/venvs/ssi/Scripts/activate
python -m pip install -r requirements.txt
```

### `ERROR: No matching distribution found for paddlepaddle`
Expected. `paddlepaddle` has no Python 3.14 build, so PaddleOCR/PP-Structure cannot be used.
RapidOCR runs the same PP-OCR models through onnxruntime. **Do not** solve this with a second
Python version — that approach was explicitly rejected. Before adding any dependency:

```bash
python -m pip install --dry-run <package>     # no --no-deps: transitive gaps must surface
```

### A run stalls for minutes right before the output is written
Presidio is trying to download a spaCy model from github (15 s per connection attempt). This
is fixed in `stages/mask.py` — model availability is now checked locally first — so if you see
it again, that check has been removed or bypassed. Either install the model
(`python -m spacy download en_core_web_lg`) or leave it out and accept kind-driven masking.

### `ModuleNotFoundError: No module named 'ssi_extractor'`
Either install the package (`pip install -e .`) or set `PYTHONPATH=src`. When running a script
under `tests/` directly, both paths are needed:

```bash
PYTHONPATH=src python tests/compare_semantic.py <ref> <out>
```

Note that `PYTHONPATH="$(pwd)/src"` on Windows Git Bash can fail on paths containing spaces;
the relative `PYTHONPATH=src` form works.

---

## 2. Extraction quality

### A field is missing from the output entirely
The field is not in the descriptor. Check, in order:

```bash
python scripts/probe_descriptor.py     # is the field discovered at all?
python scripts/probe_grid.py <pdf>     # did the value reach the chunk text?
```

If the value is in the chunk text but no field exists, the harvest missed it and the synthesis
agent did not propose it. If the value is *not* in the chunk text, the chunker dropped it —
that is the higher-severity bug (see `PROGRESS.md` §6.12 for the previous instance).

### A value is wrong but plausible
G2 should have caught it. Check `audit/<document-id>.audit.jsonl` for the
`EXTRACTION_GUARD_VERDICT` entry and `review_queue/<document-id>.review.jsonl` for the field.
A real observed case: the model returned `BARCBG22` for `BARCGB22`; G2 flagged
`HALLUCINATED_VALUE`, validation marked it `FAILED`, G3 queued it. If a wrong value is
**not** flagged, the hallucination check's similarity tolerance is too loose — see
`utils/text.substring_present(min_ratio=…)`.

### Results differ between two identical runs
Known: the `gpt-5.4-mini` deployment rejects a non-default `temperature`, so the adapter drops
it and the model samples at its default. The structural guarantees still hold (the chunker
fixes instruction count; G2 catches invented values). See `PROGRESS.md` §4.1.

### Every identifier in a document fails validation
Read the G2 drift verdict. `DRIFT` means ≥50% of identifiers failed their format checks, which
usually indicates the wrong document type or systemic OCR corruption rather than many
individual errors. Check `documentAnalysis` for the page composition and the OCR mean
confidence in `rowAnalysis`.

### A value is classified as the wrong kind
Reproduce in isolation — this is the fastest debugging loop in the project:

```bash
PYTHONPATH=src python -c "
from ssi_extractor.schema.kinds import infer_kind
print(infer_kind(['<value>'], label='<label>'))"
```

Detector **order** matters (`schema/kinds.py: _DETECTORS`): country *names* must be checked
before CFI and BIC, because `FRANCE` is six upper-case letters (CFI) and `NETHERLANDS` is
eleven characters (BIC). If you add a detector, add it in specific-to-general position and
re-run the matrix in `PROGRESS.md` §6.

---

## 3. Table and layout problems

### `tables=0` for a document that obviously has a table
`page.find_tables()` only finds *ruled* tables and returns nothing for whitespace-aligned ones.
Reconstruction should take over. Diagnose the geometry:

```bash
python scripts/probe_bands.py <pdf>     # how many column corridors and row bands?
```

- **`columns: 1`** — a full-width element (banner, title, contact block) is covering every
  corridor. The projection profile counts *rows*, so raise the threshold in `_column_bounds`
  only as a last resort; first check whether the page really has aligned columns.
- **`bands` far more than the visible rows** — multi-line cells split into separate bands;
  `_merge_wrapped_rows` should rejoin them (a continuation row has an empty first column).
- **`bands` far fewer than the visible rows** — `_MIN_ROW_CORRIDOR` is too large for tightly
  set text.

### Front matter appears as an instruction
The tabular span detector uses **straddling**: table cells sit inside one column, letterhead
crosses boundaries. If front matter is being included, check `_straddles_columns`' margin and
whether the front matter happens to align with columns.

### Header row is wrong or missing
`_find_header_span` works downwards to the first row carrying digits, then upwards over
label-like rows. A header row containing a digit (a year in a banner) breaks the downward
search — that exact case produced a wrong header during development.

### One instruction is split into two records (or two merged into one)
For tables: `_stitch_page_breaks` / `_is_continuation`. For forms: the section-merge rule
(sections merge unless they repeat labels). Verify with `python scripts/probe_chunks.py`.

---

## 4. Guardrails

### A document is rejected and you expected it to process
`GateOutcome` names the reason: `NOT_A_PDF` (magic bytes, extension not trusted),
`PASSWORD_REQUIRED`, `TOO_LARGE`, `TOO_MANY_PAGES`, `CORRUPT`, `EMPTY`. Caps are configurable
(`SSI_INPUT_GATE__MAX_PAGES`, `SSI_INPUT_GATE__MAX_FILE_BYTES`).

### The review queue is empty but fields clearly failed
G3 runs before the assembler in the graph. Check `SSI_CONFIDENCE__REVIEW_FLOOR` and that
`node_review_queue` is still on the path in `graph/builder.py`.

### `verify_chain()` reports a broken chain
Either the file was edited, or a resumed run started a new chain segment (`AuditLog`
continues an existing log and logs a warning if the tail is unreadable). Resumed logs are
expected to verify as a chain **per segment**.

### Raw PII appears in a log
That is a defect — redaction is by construction. Check that `configure_logging()` ran (the CLI
calls it) and that the handler still carries `RedactingFilter`. Add a pattern to
`observability/redaction.py: REDACTION_PATTERNS` for a shape it does not yet recognise. The
BIC pattern is ISO 9362 (8 or 11 characters, never 9 or 10) — a previous bug missed real BICs
because the pattern was wrong.

---

## 5. Comparison and scoring

### Recall looks bad but the values look right
Check whether the reference is at fault. `../input output data/*.txt` contain corrupted BICs
(`INVLESMXXX` for `INVLESMMXXX`, `JSDPJPT` for `IJSDJPJT`), a wrong tax rate, and
`scannedPages: 0` for an image-only PDF. The comparator lists near-misses with a similarity
score precisely so a one-character difference is visible and attributable.

### Precision looks bad
Common causes, in order: the reference omits fields the document contains (it is not
exhaustive); the schema contains duplicate concept fields (dedupe in `schema/harvest.py`
`_deduplicate_same_concept` and the name-based skip in the synthesis agent); the comparator's
atomiser is over-generating facts from inside long strings (only digit-bearing tokens are
harvested, and a spaced IBAN's groups are marked consumed).

### The comparator reports a country as missing
Country canonicalisation resolves alpha-2, alpha-3 and ISO names, but a document may print an
adjective (`AUSTRALIAN`) that is not an ISO name. The extraction is faithful to the page; the
comparator simply cannot canonicalise it.

---

## 6. Debug order that works

**bands → grid → chunks → descriptor → prompt → output.**

```bash
python scripts/probe_bands.py <pdf>        # geometry
python scripts/probe_grid.py <pdf>         # table + chunk cutting
python scripts/probe_chunks.py             # chunk inventory
python scripts/probe_descriptor.py         # discovered schema + strict JSON schema
python -m ssi_extractor.cli --no-llm       # whole deterministic spine, no tokens spent
python -m ssi_extractor.cli               # full run
```

Then read, in this order: `logs/ssi_extractor.log` (stage timings and decisions),
`output_json/<name>.schema.json` (what the pipeline thought the document contained),
`review_queue/<name>.review.jsonl` (what it was unsure about, with confidence breakdowns),
`audit/<name>.audit.jsonl` (how each value was produced, model and prompt version included).
