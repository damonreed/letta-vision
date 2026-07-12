"""Helpers for in-band thinking XML tags in reasoning/assistant text."""

from __future__ import annotations

# In-band end-of-reasoning markers some providers/models emit inside reasoning_content
# (notably Aion via OpenRouter), followed by the visible reply still on the reasoning channel.
_THINKING_CLOSE_TAGS = (
    "</thinking>",
    "</think>",
)


def split_reasoning_at_thinking_close(text: str) -> tuple[str, str | None]:
    """Split reasoning text if it contains an in-band thinking close tag.

    Returns ``(reasoning, response_or_none)``. When a close tag is found, reasoning is
    everything before the tag (tags stripped) and response is everything after.
    """
    if not text:
        return text, None

    lower = text.lower()
    best_idx: int | None = None
    best_len = 0
    for tag in _THINKING_CLOSE_TAGS:
        idx = lower.find(tag.lower())
        if idx < 0:
            continue
        if best_idx is None or idx < best_idx:
            best_idx = idx
            best_len = len(tag)

    if best_idx is None:
        return text, None

    reasoning = text[:best_idx].rstrip()
    response = text[best_idx + best_len :].lstrip("\n")
    # Also drop a matching open tag if present at the start of reasoning
    for open_tag in ("<thinking>", "<think>"):
        if reasoning.lower().startswith(open_tag):
            reasoning = reasoning[len(open_tag) :].lstrip("\n")
            break
    return reasoning, response if response else None


class ThinkingCloseSplitBuffer:
    """Streaming buffer that splits reasoning deltas at in-band thinking close tags."""

    def __init__(self) -> None:
        self.closed = False
        self._holdback = ""

    def feed(self, chunk: str) -> tuple[str, str]:
        """Return ``(reasoning_delta, response_delta)``; either side may be empty."""
        if not chunk:
            return "", ""
        if self.closed:
            return "", chunk

        text = self._holdback + chunk
        self._holdback = ""

        lower = text.lower()
        best_idx: int | None = None
        best_len = 0
        for tag in _THINKING_CLOSE_TAGS:
            idx = lower.find(tag.lower())
            if idx < 0:
                continue
            if best_idx is None or idx < best_idx:
                best_idx = idx
                best_len = len(tag)

        if best_idx is not None:
            self.closed = True
            reasoning = text[:best_idx]
            response = text[best_idx + best_len :]
            return reasoning, response

        hold = _possible_close_tag_suffix(text)
        if hold:
            self._holdback = hold
            return text[: len(text) - len(hold)], ""
        return text, ""

    def flush(self) -> str:
        """Emit any held-back suffix as reasoning (no close tag was completed)."""
        held = self._holdback
        self._holdback = ""
        return held


def _possible_close_tag_suffix(text: str) -> str:
    """Hold back a trailing prefix of any close tag so tags spanning chunks are detected."""
    lower = text.lower()
    max_hold = 0
    for tag in _THINKING_CLOSE_TAGS:
        tag_l = tag.lower()
        for n in range(1, len(tag_l)):
            if lower.endswith(tag_l[:n]):
                max_hold = max(max_hold, n)
    return text[-max_hold:] if max_hold else ""
