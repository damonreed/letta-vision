"""Recover DeepSeek V4 / V4.1 tool calls leaked into assistant text.

DeepSeek-V4.1 invokes tools by writing a block like::

    <｜DSML｜ calls>
    <｜DSML｜ invoke name="compose_image">
    <｜DSML｜ parameter name="prompt" string="true">...</｜DSML｜ parameter>
    </｜DSML｜ invoke>
    </｜DSML｜ calls>

V4 used the same shape without the space in the tag name (``<｜DSML｜tool_calls>``).
OpenRouter often leaves this markup in ``message.content`` and omits ``tool_calls``.
The model also drifts: spaces around ``=``, a value stuffed into the ``string``
attribute, ``</parameter>`` instead of the DSML close, and a tail of ``</function>``.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from letta.log import get_logger
from letta.schemas.letta_message_content import TextContent
from letta.schemas.openai.chat_completion_response import FunctionCall, ToolCall

logger = get_logger(__name__)

_BAR = r"(?:｜|\|)"
_DSML_MARK = re.compile(rf"<{_BAR}\s*DSML\s*{_BAR}")
_INVOKE_OPEN = re.compile(rf"<{_BAR}\s*DSML\s*{_BAR}\s*invoke\b([^>]*)>", re.IGNORECASE)
_INVOKE_CLOSE = re.compile(rf"</{_BAR}\s*DSML\s*{_BAR}\s*invoke\s*>", re.IGNORECASE)
_PARAM_OPEN = re.compile(rf"<(?:{_BAR}\s*DSML\s*{_BAR}\s*)?parameter\b([^>]*)>", re.IGNORECASE)
_PARAM_CLOSE = re.compile(rf"</(?:{_BAR}\s*DSML\s*{_BAR}\s*)?parameter\s*>", re.IGNORECASE)
_NAME_ATTR = re.compile(r"""\bname\s*=\s*["']([^"']*)["']""")
_STRING_ATTR = re.compile(r"""\bstring\s*=\s*["']([^"']*)["']""")
_TOOL_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:-]*$")
_CLOSE_TAG = re.compile(r"</[^>\n]{1,120}>")
_MARKERS = ("<｜DSML｜", "<|DSML|")


def contains_dsml(text: str | None) -> bool:
    return bool(text) and _DSML_MARK.search(text) is not None


def is_dsml_only(text: str | None) -> bool:
    """True when the text is DSML markup with no prose left once the markup is removed."""
    if not contains_dsml(text):
        return False
    return not strip_dsml_markup(text).strip()


def strip_dsml_markup(text: str) -> str:
    """Drop DSML tool-call blocks, including a trailing run of closing tags."""
    if not text or not contains_dsml(text):
        return text
    mark = _DSML_MARK.search(text)
    assert mark is not None
    closes = list(_INVOKE_CLOSE.finditer(text, mark.start()))
    if not closes:
        return text[: mark.start()].rstrip()
    end = closes[-1].end()
    tail = text[end:]
    if _is_close_spam(tail):
        return text[: mark.start()].rstrip()
    return (text[: mark.start()] + tail).strip()


def parse_dsml_tool_calls(text: str) -> list[dict[str, Any]]:
    """Return ``{"name", "arguments"}`` dicts for each closed, named invoke."""
    if not contains_dsml(text):
        return []
    calls: list[dict[str, Any]] = []
    for opening in _INVOKE_OPEN.finditer(text):
        close = _INVOKE_CLOSE.search(text, opening.end())
        if close is None:
            break
        name_match = _NAME_ATTR.search(opening.group(1) or "")
        name = name_match.group(1).strip() if name_match else ""
        if not _TOOL_NAME.match(name):
            continue
        arguments = _parse_parameters(text[opening.end() : close.start()])
        if not arguments:
            continue
        calls.append({"name": name, "arguments": arguments})
    return calls


def apply_dsml_recovery(
    content: list | None,
    tool_calls: list | None,
    extra_markup: str = "",
) -> tuple[list | None, list]:
    """Remove DSML from text parts. Parse it into tool calls only when none were provided."""
    existing = list(tool_calls or [])
    parts = list(content or [])
    recovered: list[dict[str, Any]] = []
    saw_markup = bool(extra_markup and contains_dsml(extra_markup))
    new_parts: list = []

    for part in parts:
        text = part.text if isinstance(part, TextContent) else None
        if text is not None and contains_dsml(text):
            saw_markup = True
            recovered.extend(parse_dsml_tool_calls(text))
            cleaned = strip_dsml_markup(text).strip()
            if cleaned:
                new_parts.append(part.model_copy(update={"text": cleaned}))
            continue
        new_parts.append(part)

    if extra_markup and contains_dsml(extra_markup):
        recovered.extend(parse_dsml_tool_calls(extra_markup))
        residue = strip_dsml_markup(extra_markup).strip()
        if residue:
            new_parts.append(TextContent(text=residue))

    if recovered and not existing:
        existing = [_to_tool_call(call) for call in recovered]
        logger.info(
            "Recovered %d DSML tool call(s) from assistant content: %s",
            len(existing),
            [call.function.name for call in existing],
        )
    elif saw_markup and existing:
        logger.info("Stripped DSML markup already represented by %d native tool call(s)", len(existing))
    elif saw_markup:
        logger.warning("DSML markup in assistant content did not parse into a tool call")

    if content is None and not new_parts:
        return None, existing
    return new_parts, existing


class DsmlStreamSplitter:
    """Keep DSML tool markup out of streamed assistant text.

    Chunks before the marker are returned as visible text. From the marker on,
    text is held in ``markup`` for recovery after the stream ends. A trailing
    prefix of the marker is held back so a tag split across chunks is not shown.
    """

    def __init__(self) -> None:
        self._hold = ""
        self._in_markup = False
        self._markup: list[str] = []

    def feed(self, chunk: str) -> str:
        if not chunk:
            return ""
        if self._in_markup:
            self._markup.append(chunk)
            return ""
        text = self._hold + chunk
        self._hold = ""
        idx = _earliest_marker(text)
        if idx is not None:
            self._in_markup = True
            self._markup.append(text[idx:])
            return text[:idx]
        hold = _marker_prefix_suffix(text)
        if hold:
            self._hold = hold
            return text[: len(text) - len(hold)]
        return text

    def flush_visible(self) -> str:
        """Return held-back text that never became a DSML marker."""
        if self._in_markup:
            return ""
        held = self._hold
        self._hold = ""
        return held

    @property
    def markup(self) -> str:
        return "".join(self._markup)


def _parse_parameters(body: str) -> dict[str, Any] | None:
    opens = list(_PARAM_OPEN.finditer(body))
    if not opens:
        return None
    arguments: dict[str, Any] = {}
    for index, opening in enumerate(opens):
        close = _PARAM_CLOSE.search(body, opening.end())
        if close is None:
            return None
        next_open = opens[index + 1] if index + 1 < len(opens) else None
        if next_open is not None and next_open.start() < close.start():
            gap = body[opening.end() : next_open.start()]
            if not gap.strip():
                continue
        name_match = _NAME_ATTR.search(opening.group(1) or "")
        if name_match is None:
            continue
        raw_body = _CLOSE_TAG.sub("", body[opening.end() : close.start()])
        raw_body = re.sub(r"<[^>\n]{1,120}>", "", raw_body).strip()
        string_attr = _STRING_ATTR.search(opening.group(1) or "")
        arguments[name_match.group(1).strip()] = _coerce_parameter(
            raw_body, string_attr.group(1) if string_attr else None
        )
    return arguments or None


def _coerce_parameter(body: str, string_attr: str | None) -> Any:
    if string_attr == "true":
        return body
    if string_attr == "false":
        return _parse_json_or_raw(body)
    if string_attr and not body:
        # V4.1 sometimes writes ``string="3:4"`` and leaves the element empty.
        return _parse_json_or_raw(string_attr)
    if body:
        return _parse_json_or_raw(body)
    return string_attr or ""


def _parse_json_or_raw(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    if value[0] in "[{-" or value[0].isdigit() or value in ("true", "false", "null"):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _is_close_spam(tail: str) -> bool:
    """True when the tail is only closing tags, whitespace, and a tag cut off at the end."""
    without_tags = _CLOSE_TAG.sub("", tail)
    # A length-capped reply often ends mid-tag: ``</function`` with no ``>``.
    without_tags = re.sub(r"</[^>\n]*", "", without_tags)
    return without_tags.strip() == ""


def _to_tool_call(call: dict[str, Any]) -> ToolCall:
    return ToolCall(
        id=f"call-{uuid.uuid4().hex[:24]}",
        function=FunctionCall(
            name=call["name"],
            arguments=json.dumps(call["arguments"], ensure_ascii=False),
        ),
    )


def _earliest_marker(text: str) -> int | None:
    indexes = [text.find(marker) for marker in _MARKERS]
    found = [index for index in indexes if index >= 0]
    return min(found) if found else None


def _marker_prefix_suffix(text: str) -> str:
    max_hold = 0
    for marker in _MARKERS:
        for length in range(1, len(marker)):
            if text.endswith(marker[:length]):
                max_hold = max(max_hold, length)
    return text[-max_hold:] if max_hold else ""
