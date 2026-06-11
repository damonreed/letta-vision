"""MiniMax OpenAI-compatible API helpers (reasoning_split, reasoning_details, vision)."""

from __future__ import annotations

import re
from typing import Any

from letta.schemas.llm_config import LLMConfig

# Models that echo thinking in assistant text while also streaming reasoning_content.
_THINK_TAG = "think"
_THINKING_BLOCK_PATTERNS = (
    re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE),
    re.compile(rf"<{_THINK_TAG}>.*?</{_THINK_TAG}>\s*", re.DOTALL | re.IGNORECASE),
)

# MiniMax M3 vision accepts low/default/high — not OpenAI's "auto".
_MINIMAX_IMAGE_DETAIL_VALUES = frozenset({"low", "default", "high"})


def is_minimax_openai_compatible(llm_config: LLMConfig) -> bool:
    """True when requests go to MiniMax's OpenAI-compatible chat completions API."""
    model = (llm_config.model or "").lower()
    endpoint = (llm_config.model_endpoint or "").lower()
    handle = (llm_config.handle or "").lower()
    if "minimax" in model:
        return True
    if "minimax.io" in endpoint:
        return True
    if "minimax" in handle:
        return True
    return False


def strip_duplicate_thinking_from_assistant_text(text: str) -> str:
    """Remove inline thinking tags when reasoning was already extracted separately."""
    if not text:
        return text
    stripped = text
    for pattern in _THINKING_BLOCK_PATTERNS:
        stripped = pattern.sub("", stripped)
    return stripped


def extract_reasoning_from_message_data(message_data: dict) -> str | None:
    """Pull thinking text from MiniMax/OpenAI-compatible message payloads."""
    reasoning = message_data.get("reasoning_content") or message_data.get("reasoning")
    if reasoning:
        return reasoning

    details = message_data.get("reasoning_details")
    if not details:
        return None

    parts: list[str] = []
    for item in details:
        if isinstance(item, dict):
            text = item.get("text")
            if text:
                parts.append(text)
        elif isinstance(item, str):
            parts.append(item)
    return "\n".join(parts) if parts else None


def minimax_image_detail_for_request(detail: str | None) -> str:
    """Map OpenAI-style image detail to a value MiniMax accepts."""
    if detail in _MINIMAX_IMAGE_DETAIL_VALUES:
        return detail
    return "default"


def _normalize_image_detail_in_content_part(part: dict[str, Any]) -> None:
    part_type = part.get("type")
    if part_type == "image_url":
        image_url = part.get("image_url")
        if isinstance(image_url, dict):
            image_url["detail"] = minimax_image_detail_for_request(image_url.get("detail"))
    elif part_type == "input_image":
        part["detail"] = minimax_image_detail_for_request(part.get("detail"))


def _normalize_image_detail_in_message_list(messages: list[Any]) -> None:
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict):
                _normalize_image_detail_in_content_part(part)


def normalize_minimax_openai_request_images(request_data: dict) -> None:
    """Rewrite image detail fields so MiniMax does not reject OpenAI 'auto'."""
    if "messages" in request_data:
        messages = request_data.get("messages")
        if isinstance(messages, list):
            _normalize_image_detail_in_message_list(messages)
    if "input" in request_data:
        inp = request_data.get("input")
        if isinstance(inp, list):
            _normalize_image_detail_in_message_list(inp)


def apply_minimax_openai_request_extras(request_data: dict, llm_config: LLMConfig) -> None:
    """Enable separated reasoning fields per MiniMax interleaved-thinking docs."""
    if not is_minimax_openai_compatible(llm_config):
        return
    normalize_minimax_openai_request_images(request_data)
    if not llm_config.enable_reasoner:
        return
    extra = dict(request_data.get("extra_body") or {})
    extra["reasoning_split"] = True
    request_data["extra_body"] = extra
