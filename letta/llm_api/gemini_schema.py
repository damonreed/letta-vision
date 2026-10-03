"""Rewrite JSON Schema so Gemini function declarations accept it.

Gemini turns a JSON Schema ``type`` array into ``any_of``. An ``array`` member
with no ``items`` schema fails validation as::

    parameters.properties[value].any_of[5].items: missing field

``items`` must itself have a ``type``. An ``items`` schema that is only
``anyOf`` fails the same way. ``additionalProperties`` and ``default`` are
outside the OpenAPI subset Gemini accepts.
"""

from __future__ import annotations

import copy
from typing import Any

# Same subset the Google client already drops. ``$ref`` is resolved before this runs
# on the native Gemini path; dropping a leftover ref is safer than forwarding it.
_UNSUPPORTED_KEYS = (
    "additionalProperties",
    "default",
    "exclusiveMaximum",
    "exclusiveMinimum",
    "$schema",
    "const",
    "$ref",
    "$defs",
    "definitions",
)

# Item types for an "any JSON" array. Each becomes its own array branch so
# ``items`` always has a concrete type. ``integer`` stays distinct from ``number``
# because both appear in published tool schemas.
_JSON_ARRAY_ITEM_TYPES = ("string", "number", "integer", "boolean", "object")


def sanitize_openai_tools_for_gemini(tools: list[dict] | None) -> list[dict] | None:
    """Sanitize Chat Completions tool dicts in place-safe copies."""
    if not tools:
        return tools
    sanitized = copy.deepcopy(tools)
    for tool in sanitized:
        function = tool.get("function") if isinstance(tool, dict) else None
        if isinstance(function, dict) and isinstance(function.get("parameters"), dict):
            function["parameters"] = sanitize_json_schema_for_gemini(function["parameters"])
    return sanitized


def sanitize_json_schema_for_gemini(schema: Any) -> Any:
    """Return a Gemini-safe copy of a JSON Schema node."""
    if isinstance(schema, list):
        return [sanitize_json_schema_for_gemini(item) for item in schema]
    if not isinstance(schema, dict):
        return schema

    node = {key: value for key, value in schema.items() if key not in _UNSUPPORTED_KEYS}

    for key in ("properties",):
        value = node.get(key)
        if isinstance(value, dict):
            node[key] = {name: sanitize_json_schema_for_gemini(prop) for name, prop in value.items()}

    if isinstance(node.get("items"), (dict, list)):
        node["items"] = sanitize_json_schema_for_gemini(node["items"])

    for key in ("anyOf", "oneOf", "allOf", "prefixItems"):
        value = node.get(key)
        if isinstance(value, list):
            node[key] = [sanitize_json_schema_for_gemini(item) if isinstance(item, dict) else item for item in value]

    if isinstance(node.get("type"), list) and "array" in node["type"]:
        node = _expand_type_union(node)

    if node.get("type") == "array" and not _items_have_concrete_type(node.get("items")):
        node = _expand_untyped_array(node)

    if node.get("type") == "string" and "format" in node and node["format"] not in ("enum", "date-time"):
        del node["format"]

    return node


def _items_have_concrete_type(items: Any) -> bool:
    return isinstance(items, dict) and isinstance(items.get("type"), str) and bool(items["type"])


def _array_branch(item_type: str) -> dict[str, Any]:
    return {"type": "array", "items": {"type": item_type}}


def _expand_type_union(node: dict[str, Any]) -> dict[str, Any]:
    """Turn ``type: [..., "array", ...]`` into ``anyOf`` with typed array items."""
    types = [type_name for type_name in node.pop("type") if isinstance(type_name, str)]
    branches: list[dict[str, Any]] = []
    for type_name in types:
        if type_name == "array":
            branches.extend(_array_branch(item_type) for item_type in _JSON_ARRAY_ITEM_TYPES)
        else:
            branches.append({"type": type_name})
    existing = node.get("anyOf")
    if isinstance(existing, list):
        branches.extend(existing)
    node["anyOf"] = branches
    return node


def _expand_untyped_array(node: dict[str, Any]) -> dict[str, Any]:
    """Replace an array whose ``items`` schema has no ``type``.

    Gemini reports that shape as ``items: missing field``. Hoist each item type
    to its own array branch instead of leaving ``anyOf`` inside ``items``.
    """
    item_types: list[str] = []
    items = node.get("items")
    if isinstance(items, dict):
        options = items.get("anyOf") if isinstance(items.get("anyOf"), list) else []
        if isinstance(items.get("type"), list):
            options = [*options, *({"type": type_name} for type_name in items["type"])]
        for option in options:
            if not isinstance(option, dict):
                continue
            item_type = option.get("type")
            if isinstance(item_type, str) and item_type not in ("array", "null") and item_type not in item_types:
                item_types.append(item_type)
    if not item_types:
        item_types = list(_JSON_ARRAY_ITEM_TYPES)

    expanded: dict[str, Any] = {"anyOf": [_array_branch(item_type) for item_type in item_types]}
    if "description" in node:
        expanded["description"] = node["description"]
    return expanded
