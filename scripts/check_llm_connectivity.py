"""Smoke-test the configured LLM provider: credentials, reachability, structured output.

Run before a pipeline run when the environment is uncertain (VPN, rotated key,
changed deployment). Prints a one-line verdict and never prints the key.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ssi_extractor.config.credentials import load_azure_credentials  # noqa: E402
from ssi_extractor.llm import LlmError, build_llm  # noqa: E402

SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}, "note": {"type": "string"}},
    "required": ["ok", "note"],
    "additionalProperties": False,
}


def main() -> int:
    credentials = load_azure_credentials()
    print(f"credentials: {credentials.masked_summary}")

    try:
        llm = build_llm()
    except Exception as exc:
        print(f"FAIL adapter construction: {type(exc).__name__}: {exc}")
        return 2

    print(f"model_id: {llm.model_id}")

    try:
        response = llm.complete_json(
            system_prompt="You return JSON only.",
            user_prompt='Return exactly {"ok": true, "note": "reachable"}.',
            json_schema=SCHEMA,
            schema_name="connectivity_check",
            max_output_tokens=2000,
        )
    except LlmError as exc:
        print(f"FAIL provider call: {exc}")
        return 3
    except Exception as exc:
        print(f"FAIL unexpected: {type(exc).__name__}: {exc}")
        return 4

    print(f"payload: {response.payload}")
    print(f"tokens: prompt={response.usage.prompt_tokens} completion={response.usage.completion_tokens}")
    print("OK" if response.payload.get("ok") else "FAIL payload did not confirm")
    return 0 if response.payload.get("ok") else 5


if __name__ == "__main__":
    raise SystemExit(main())
