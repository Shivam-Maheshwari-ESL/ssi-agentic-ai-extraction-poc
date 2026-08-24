"""Turn a runtime Pydantic model into a provider-strict JSON schema.

Structured-output modes are stricter than JSON Schema generally: no ``default``,
no sibling keys next to ``$ref``, ``additionalProperties: false`` everywhere, and
every property listed as required. Pydantic's own schema does not satisfy those
constraints, so it is transformed here rather than each agent hand-writing one.

Making every property required is safe *because* every leaf defaults to a
populated ``NOT_APPLICABLE`` object: the model is obliged to say something about
each field, which is exactly the contract ("every field carries a value, a status,
a confidence and an evidence citation") and removes the ambiguity between "the
document did not state this" and "the model forgot".
"""

from __future__ import annotations

import copy
from typing import Any

from pydantic import BaseModel

__all__ = ["to_strict_schema"]

_DROPPED_KEYS = frozenset(
    {
        "default",
        "examples",
        "exclusiveMaximum",
        "exclusiveMinimum",
        "format",
        "maximum",
        "minimum",
        "maxItems",
        "minItems",
        "maxLength",
        "minLength",
        "pattern",
        "title",
    }
)


def _inline_refs(node: Any, definitions: dict[str, Any], depth: int = 0) -> Any:
    """Inline ``$ref`` targets.

    Providers reject sibling keys beside ``$ref`` and some reject ``$defs``
    entirely, so references are resolved into place. Depth is bounded because a
    self-referential model would otherwise expand forever.
    """
    if depth > 24:
        return {"type": "string"}

    if isinstance(node, dict):
        if "$ref" in node:
            ref = node["$ref"]
            name = ref.rsplit("/", 1)[-1]
            target = definitions.get(name, {})
            resolved = _inline_refs(copy.deepcopy(target), definitions, depth + 1)
            extra = {key: value for key, value in node.items() if key != "$ref"}
            if isinstance(resolved, dict):
                resolved.update(_strip(extra))
            return resolved
        return {key: _inline_refs(value, definitions, depth + 1) for key, value in node.items()}

    if isinstance(node, list):
        return [_inline_refs(item, definitions, depth + 1) for item in node]
    return node


def _strip(node: Any) -> Any:
    """Remove keywords the strict mode rejects, recursively."""
    if isinstance(node, dict):
        return {
            key: _strip(value)
            for key, value in node.items()
            if key not in _DROPPED_KEYS
        }
    if isinstance(node, list):
        return [_strip(item) for item in node]
    return node


def _enforce_objects(node: Any) -> Any:
    """Close every object and mark all of its properties required."""
    if isinstance(node, dict):
        node = {key: _enforce_objects(value) for key, value in node.items()}

        if node.get("type") == "object" or "properties" in node:
            properties = node.get("properties", {})
            node["type"] = "object"
            node["properties"] = properties
            node["required"] = list(properties.keys())
            node["additionalProperties"] = False

        # anyOf/oneOf branches with a null member are how Pydantic renders an
        # optional; strict mode accepts the union, but each branch still needs to
        # obey the object rules, which the recursion above has already applied.
        return node

    if isinstance(node, list):
        return [_enforce_objects(item) for item in node]
    return node


def to_strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Produce a strict JSON schema for ``model``."""
    raw = model.model_json_schema(ref_template="#/$defs/{model}")
    definitions = raw.pop("$defs", {})
    inlined = _inline_refs(raw, definitions)
    stripped = _strip(inlined)
    strict = _enforce_objects(stripped)

    strict.pop("$defs", None)
    strict["type"] = "object"
    strict.setdefault("properties", {})
    strict["required"] = list(strict["properties"].keys())
    strict["additionalProperties"] = False
    return strict
