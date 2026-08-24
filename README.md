# SSI Extraction Pipeline

Hybrid agentic pipeline that reads a banking **SSI (Standard Settlement Instructions)** PDF —
native text, scanned, photographed, or regions where those overlap — and extracts every
settlement instruction into JSON where **each leaf carries `value`, `status`, `confidence`,
`evidence` and `page`**.

Deterministic code does everything that can be deterministic. An LLM is called only where
genuine reasoning is required: discovering the document's field structure, mapping a chunk to
those fields, and adjudicating ambiguity. Orchestration is **LangGraph** end to end.

The output schema is **discovered per document at runtime**. Field names, field count,
grouping and nesting are read off the page, so a document from a different institution
produces correctly structured output with **no code change**.

---

## 1. Requirements

| | |
|---|---|
| Python | **3.14+** (developed and verified on 3.14.6) — a hard constraint, see `requirements.txt` |
| OS | Windows, macOS or Linux (developed on Windows 11) |
| Network | Corporate **VPN required** to reach the Azure OpenAI endpoint |
| Credentials | `AzureOpeapiKeys.txt` in the project root |
| Disk | ~1.5 GB for dependencies (onnxruntime + OCR models + spaCy stack) |

No Tesseract, no PaddlePaddle and no GPU are required. OCR runs the PP-OCR models through
onnxruntime on CPU.

---

## 2. Setup

```bash
# 1. create and activate a virtual environment (Python 3.14)
python -m venv .venv
# Windows (Git Bash):     source .venv/Scripts/activate
# Windows (PowerShell):   .venv\Scripts\Activate.ps1
# macOS / Linux:          source .venv/bin/activate

# 2. install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 3. install the package itself (editable, gives you the `ssi-extract` command)
python -m pip install -e ".[dev]"

# 4. OPTIONAL — better PII masking of free-text names and addresses.
#    Skip it if your network blocks the download; masking still works.
python -m spacy download en_core_web_lg
```

> **Put the virtual environment outside any cloud-synced folder** (Dropbox, OneDrive,
> Google Drive). Installing ~500 MB of wheels into a synced folder made interpreter
> start-up take 22 seconds on the development machine. See `TROUBLESHOOTING.md`.

### Credentials

`AzureOpeapiKeys.txt` lives in the project root and is a Spring-style YAML fragment:

```yaml
azure:
   openai:
     api-key: ${AZURE_OPENAI_API_KEY:<inline-default>}
     endpoint: ${AZURE_OPENAI_ENDPOINT:https://<resource>.openai.azure.com/}
     deployment-name: ${AZURE_OPENAI_DEPLOYMENT_NAME:gpt-5.4-mini}
     api-version: ${AZURE_OPENAI_API_VERSION:2024-10-21}
```

Each `${VAR:default}` resolves with Spring semantics: **the environment variable when set,
otherwise the inline default**. So the file works as-is, and any value can be overridden from
the environment without editing it. Point elsewhere with `SSI_AZURE_CREDENTIALS_FILE`.
The file is gitignored, and the resolved key is `SecretStr`-wrapped and covered by the log
redaction filter, so it cannot reach a log, trace or audit entry.

Verify credentials and reachability before a run:

```bash
python scripts/check_llm_connectivity.py
```

Expected on success: `payload: {'ok': True, 'note': 'reachable'}` … `OK`.
`403 Access denied due to Virtual Network/Firewall rules` means the VPN is not connected.

---

## 3. Running

```bash
# process every PDF in input_pdf/ -> output_json/
python -m ssi_extractor.cli

# or, after `pip install -e .`
ssi-extract

# specific files, plus a PII-masked copy of each output
python -m ssi_extractor.cli --masked-export input_pdf/my_document.pdf

# deterministic stages only: gate, classification, OCR, chunking, schema discovery.
# No model calls. Use this to diagnose without spending tokens or needing the VPN.
python -m ssi_extractor.cli --no-llm

# native text layers only, skipping OCR
python -m ssi_extractor.cli --no-ocr

python -m ssi_extractor.cli --help
```

### Folders

| Folder | Contents |
|---|---|
| `input_pdf/` | PDFs to process |
| `output_json/` | `<name>.json`, `<name>.schema.json` (the discovered schema), `<name>.masked.json` |
| `logs/` | `ssi_extractor.log` — JSON lines, PII-redacted by construction |
| `audit/` | `<document-id>.audit.jsonl` — G4 hash-chained, write-once audit trail |
| `review_queue/` | `<document-id>.review.jsonl` — G3 fields needing human judgement |
| `config/reference/` | Optional golden-set data (empty by default) |
| `.checkpoints/` | LangGraph checkpoints for resume |

### Output shape

Metadata is fixed; the record body follows the document's own discovered structure.

```json
{
  "documentName": "example.pdf",
  "status": "COMPLETED",
  "documentAnalysis": "1 page(s) | 1 native | 0 scanned | 0 mixed | Instructions: 13 | ...",
  "pageCount": 1, "nativeTextPages": 1, "scannedPages": 0, "mixedPages": 0,
  "instructionCount": 13,
  "settlementInstructionRecords": [
    {
      "rowAnalysis": "Instruction 1 of 13 | Layout: TABLE_ROW | Page(s): 1 | Source: native text | ...",
      "country":  { "value": "AUSTRALIAN", "status": "VALIDATED", "confidence": 0.87,
                    "evidence": "AUSTRALIAN", "page": [1] },
      "beneficiary": {
        "bic_code": { "value": "INVLESMMXXX", "status": "VALIDATED", "confidence": 0.91,
                      "evidence": "INVLESMMXXX", "page": [1] }
      }
    }
  ],
  "extractionSummary": { "fieldsValidated": 109, "fieldsFailed": 5,
                         "fieldsNotApplicable": 42, "fieldsQueuedForReview": 34,
                         "schemaDescriptorHash": "f38efe…" }
}
```

`status` is one of `VALIDATED`, `FAILED`, `NOT_APPLICABLE`. `NOT_APPLICABLE` means the
document does not state the field — normal, and never a failure. `page` is a list because a
value stitched from regions spanning several pages cites all of them.

Because the shape varies per document, `<name>.schema.json` describes it: every field's path,
label, inferred kind, cardinality, PII flag, validator hints and how it was discovered.

---

## 4. Configuration

Every threshold, weight, cap and toggle lives in `src/ssi_extractor/config/settings.py`. All
are overridable from the environment with the `SSI_` prefix and `__` for nesting:

```bash
export SSI_OCR__CONFIDENCE_TIER1=0.75            # OCR retry threshold
export SSI_CONFIDENCE__REVIEW_FLOOR=0.80         # human-review floor
export SSI_PRIVACY__EXTERNAL_FALLBACK_ENABLED=true   # allow the vision fallback
export SSI_LLM__PROVIDER=anthropic               # switch provider
export SSI_LLM__MAX_CALLS_PER_DOCUMENT=200       # cost cap
export SSI_LOGGING__LEVEL=DEBUG
```

Privacy defaults: processing is local, the external vision fallback is **off**, and nothing
is sent externally without PII masking first.

---

## 5. Testing

```bash
pytest -q                # unit tests; `llm`-marked tests excluded by default
pytest -q -m llm         # tests that make real provider calls (needs VPN)

# semantic comparison against a reference JSON of a different shape
python tests/compare_semantic.py "<reference>.txt" output_json/<name>.json
```

`tests/compare_semantic.py` compares **meaning, not structure**: both sides are reduced to
`(inferred kind, canonical value)` facts, so differing key names and nesting do not count as
errors, while a dropped character does. It reports recall, precision and per-kind counts, and
flags near-misses that indicate an altered character on either side.

---

## 6. Documentation map

| File | Purpose |
|---|---|
| `README.md` | this file — setup and running |
| `PROGRESS.md` | current status, verified metrics, what works, what is left, next steps |
| `ARCHITECTURE.md` | stage-by-stage design and why each decision was made |
| `TROUBLESHOOTING.md` | known failure modes, environment hazards, debugging recipes |
| `Claude.md` | engineering persona/instructions plus the decision and change log |

---

## 7. Status

Working end to end on native-text tables and scanned key:value forms, with all four
guardrails active. See `PROGRESS.md` for measured results and the remaining work
(vision fallback, adjudicator agent, OCR digit repair, synthetic fixture suite, FastAPI and
Celery scale-out).
