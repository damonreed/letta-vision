"""Helpers for cursor datetimes passed through URL query strings."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

# x-www-form-urlencoded query parsing decodes '+' as space; repair "T... 00:00" offsets.
_ISO_OFFSET_SPACE_RE = re.compile(r"^(?P<head>.+T\d{2}:\d{2}:\d{2}(?:\.\d+)?) (?P<tz>\d{2}:\d{2})$")


def repair_iso_datetime_query_cursor(value: str) -> str:
    """Repair UTC offsets where '+' was decoded as a space in a query string."""
    text = value.strip()
    match = _ISO_OFFSET_SPACE_RE.match(text)
    if match:
        return f"{match.group('head')}+{match.group('tz')}"
    return text


def parse_cursor_datetime(value: Optional[str]) -> Optional[datetime]:
    if value is None:
        return None
    return datetime.fromisoformat(repair_iso_datetime_query_cursor(value))
