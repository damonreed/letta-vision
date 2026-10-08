"""Shared helpers for provider error detection/mapping.

Keep these utilities free of heavy imports to avoid circular dependencies between
LLM clients (provider-specific) and streaming interfaces.
"""

from __future__ import annotations

import re


def is_ambiguous_provider_rejection_message(msg: str) -> bool:
    """True for multi-cause upstream rejections that are not definitive overflow.

    OpenRouter sometimes forwards vague Sail Research / similar 400s like:
    "Possible causes: input exceeds the model's maximum context length, or the
    request contains invalid parameters."

    Matching the bare "maximum context length" substring would falsely trigger
    compaction. Treat these as ordinary bad requests instead.
    """
    if not msg:
        return False
    lower = msg.lower()
    if "or the request contains invalid parameters" in lower:
        return True
    return "possible causes:" in lower and "invalid parameters" in lower


def is_context_window_overflow_message(msg: str) -> bool:
    """Best-effort detection for context window overflow errors.

    Different providers (and even different API surfaces within the same provider)
    may phrase context-window errors differently. We centralize the heuristic so
    all layers (clients, streaming interfaces, agent loops) behave consistently.

    Ambiguous multi-cause rejections must not count as overflow — see
    ``is_ambiguous_provider_rejection_message``.
    """
    if not msg:
        return False

    if is_ambiguous_provider_rejection_message(msg):
        return False

    lower = msg.lower()
    return (
        "exceeds the context window" in lower
        or "this model's maximum context length is" in lower
        or "context_length_exceeded" in lower
        or "input tokens exceed the configured limit" in lower
        or "exceeds the maximum allowed input length" in lower
        # Classic OpenAI accounting form (also covered by the "this model's..."
        # pattern above; kept for slightly different phrasings).
        or ("maximum context length" in lower and "however" in lower and "resulted in" in lower)
    )


def is_openrouter_image_payload_limit_message(msg: str) -> bool:
    """OpenRouter rejects requests when cumulative in-context image bytes exceed ~30MB."""
    lower = msg.lower()
    return "downloaded image content cannot exceed" in lower or "image content cannot exceed 30mb" in lower


def openrouter_image_payload_limit_user_message() -> str:
    return (
        "OpenRouter rejected this request because the total in-context image payload exceeds "
        "its 30MB limit. Try compressing images, sending fewer images per turn, or starting a "
        "new conversation for a fresh image budget."
    )


def is_insufficient_credits_message(msg: str) -> bool:
    """Best-effort detection for insufficient credits/quota/billing errors.

    BYOK users on OpenRouter, OpenAI, etc. may exhaust their credits mid-stream
    or get rejected pre-flight. We detect these so they map to 402 instead of 400/500.
    """
    lower = msg.lower()
    return (
        "insufficient credits" in lower
        or "requires more credits" in lower
        or "add more credits" in lower
        or "exceeded your current quota" in lower
        or "you've exceeded your budget" in lower
        or ("billing" in lower and "hard limit" in lower)
        or "can only afford" in lower
    )


_OPENROUTER_PROVIDER_NAME_RE = re.compile(
    r"""['"]provider_name['"]\s*:\s*['"]([^'"]+)['"]""",
    re.IGNORECASE,
)


def extract_openrouter_provider_name(error: Exception | str | dict | None) -> str | None:
    """Extract upstream OpenRouter provider_name from an error body, dict, or message."""
    if error is None:
        return None

    if isinstance(error, dict):
        return _provider_name_from_mapping(error)

    if isinstance(error, Exception):
        body = getattr(error, "body", None)
        if isinstance(body, dict):
            name = _provider_name_from_mapping(body)
            if name:
                return name
        return _provider_name_from_text(str(error))

    if isinstance(error, str):
        return _provider_name_from_text(error)

    return None


def _provider_name_from_mapping(data: dict) -> str | None:
    error_data = data.get("error", data)
    if not isinstance(error_data, dict):
        return None
    metadata = error_data.get("metadata", {})
    if not isinstance(metadata, dict):
        return None
    name = metadata.get("provider_name")
    return name if isinstance(name, str) and name.strip() else None


def _provider_name_from_text(text: str) -> str | None:
    match = _OPENROUTER_PROVIDER_NAME_RE.search(text)
    if not match:
        return None
    name = match.group(1).strip()
    return name or None
