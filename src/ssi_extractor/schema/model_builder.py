"""Build a Pydantic model at runtime from a discovered ``SchemaDescriptor``.

The extraction agent must return a shape the pipeline can enforce, but the shape
is not known until the document has been read. So the model is constructed per
document with ``create_model``: groups become nested models, leaves are always
``ExtractedField``, and a ``MULTI`` field becomes a list of them.

``extra="forbid"`` on every generated model is deliberate. G2 must be able to
reject a response that invents a field rather than silently coercing it, and a
forbidding model turns that into a validation error instead of a lost value.
"""

from __future__ import annotations

import keyword
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, create_model

from ssi_extractor.schema.descriptor import (
    Cardinality,
    FieldDescriptor,
    SchemaDescriptor,
    slugify,
)
from ssi_extractor.schema.leaf import ExtractedField

__all__ = ["build_record_model", "safe_model_field_name"]

_MODEL_CONFIG = ConfigDict(extra="forbid", populate_by_name=True)


def safe_model_field_name(name: str) -> str:
    """Make a discovered name usable as a Python attribute.

    Discovered names come from document labels, so they can collide with Python
    keywords or start with a digit. The JSON key is preserved via an alias, so the
    output shape still reflects the document.
    """
    candidate = slugify(name)
    if not candidate:
        candidate = "field"
    if keyword.iskeyword(candidate) or keyword.issoftkeyword(candidate):
        candidate = f"{candidate}_"
    if candidate[0].isdigit():
        candidate = f"f_{candidate}"
    return candidate


def _class_name(parts: tuple[str, ...], suffix: str) -> str:
    """A readable, deterministic model class name for debugging and error messages."""
    joined = "".join(part.title().replace("_", "") for part in (slugify(part) for part in parts))
    return f"{joined or 'Root'}{suffix}"


def _leaf_annotation(field: FieldDescriptor) -> tuple[Any, Any]:
    """Annotation and default for one leaf.

    Every leaf is optional at the model level and defaults to a
    ``NOT_APPLICABLE`` field: a document that does not state a value must produce
    a populated ``NOT_APPLICABLE`` leaf, never a missing key, so the record array
    stays homogeneous across instructions.
    """
    description = f"{field.label} ({field.kind.value})"
    if field.cardinality is Cardinality.MULTI:
        annotation = Annotated[
            list[ExtractedField],
            Field(default_factory=list, description=description, alias=field.name),
        ]
        return annotation, Field(default_factory=list, description=description, alias=field.name)

    return (
        ExtractedField,
        Field(
            default_factory=ExtractedField.not_applicable,
            description=description,
            alias=field.name,
        ),
    )


def _build_group_model(
    group_path: tuple[str, ...],
    fields: list[FieldDescriptor],
    children: dict[str, dict[str, Any]],
    tree: dict[tuple[str, ...], Any],
) -> type[BaseModel]:
    """Recursively build the model for one group level."""
    definitions: dict[str, tuple[Any, Any]] = {}

    for field in fields:
        annotation, default = _leaf_annotation(field)
        definitions[safe_model_field_name(field.name)] = (annotation, default)

    for child_name, child_content in children.items():
        child_path = (*group_path, child_name)
        child_model = _build_group_model(
            child_path,
            child_content["fields"],
            child_content["children"],
            tree,
        )
        definitions[safe_model_field_name(child_name)] = (
            child_model,
            Field(default_factory=child_model, description=child_name, alias=slugify(child_name)),
        )

    model = create_model(
        _class_name(group_path, "Group" if group_path else "Record"),
        __config__=_MODEL_CONFIG,
        **definitions,
    )
    tree[group_path] = model
    return model


def _group_index(descriptor: SchemaDescriptor) -> dict[str, Any]:
    """Index the flat field list into a nested {fields, children} structure."""
    root: dict[str, Any] = {"fields": [], "children": {}}
    for field in descriptor.fields:
        node = root
        for part in field.group_path:
            node = node["children"].setdefault(part, {"fields": [], "children": {}})
        node["fields"].append(field)
    return root


def build_record_model(descriptor: SchemaDescriptor) -> type[BaseModel]:
    """Build the per-instruction record model for a descriptor.

    The returned model is what the extraction agent is asked to fill and what G2
    validates its response against — one descriptor, one model, no hand-written
    contract anywhere.
    """
    if descriptor.is_empty:
        raise ValueError("cannot build a record model from an empty schema descriptor")

    index = _group_index(descriptor)
    built: dict[tuple[str, ...], Any] = {}
    return _build_group_model((), index["fields"], index["children"], built)
