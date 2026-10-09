"""Accumulate OpenRouter `reasoning_details` stream deltas into complete items."""

from typing import Any

# String fields that arrive split across deltas for the same item.
_CONCAT_FIELDS = ("text", "summary", "data")


def merge_reasoning_details(acc: list[dict[str, Any]], deltas: Any) -> None:
    """Fold one chunk's `reasoning_details` into `acc` in place.

    Deltas for the same item share `type` and `index`; their string fields concatenate.
    Items without an index are kept as separate entries.
    """
    if not isinstance(deltas, list):
        return
    for item in deltas:
        if not isinstance(item, dict):
            continue
        index = item.get("index")
        existing = None
        if index is not None:
            existing = next((d for d in acc if d.get("index") == index and d.get("type") == item.get("type")), None)
        if existing is None:
            acc.append(dict(item))
            continue
        for key, value in item.items():
            if value is None:
                continue
            if key in _CONCAT_FIELDS and isinstance(value, str) and isinstance(existing.get(key), str):
                existing[key] += value
            elif existing.get(key) is None:
                existing[key] = value
