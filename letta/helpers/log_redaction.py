"""
Redact large or sensitive payloads before logging, tracing, or OTEL events.

Vision agents embed base64 image data in LLM requests; logging raw request_data
will fill disks and leak image bytes into log aggregators.
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any

from letta.services.mcp.tool_result_formatter import redact_base64_in_text

_DATA_URL_RE = re.compile(r"data:image/[^;]+;base64,[A-Za-z0-9+/=\s]+", re.IGNORECASE)
_B64_FIELD_RE = re.compile(r'"(?:data|image|image_url|url)"\s*:\s*"[A-Za-z0-9+/=]{200,}"', re.IGNORECASE)

# Keys whose string values are usually secrets or huge blobs when logging LLM payloads.
_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "authorization",
        "access_token",
        "refresh_token",
        "client_secret",
    }
)

# Keys commonly holding base64 image bytes in OpenAI / Letta message shapes.
_IMAGE_DATA_KEYS = frozenset(
    {
        "data",
        "image",
        "image_url",
        "input_image",
        "source",
    }
)

_DEFAULT_MAX_STR = 400
_BASE64_MIN_LEN = 256


def _looks_like_base64(s: str) -> bool:
    if len(s) < _BASE64_MIN_LEN:
        return False
    sample = s[:2000]
    if sample.startswith("data:image"):
        return True
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=\n\r")
    return all(c in allowed for c in sample)


def _omit_blob(label: str, length: int) -> str:
    return f"[{label} omitted, {length} chars]"


def redact_value_for_log(value: Any, *, max_str_len: int = _DEFAULT_MAX_STR) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if value.startswith("data:image") and ";base64," in value:
            return _omit_blob("image data URL", len(value))
        if _looks_like_base64(value):
            return _omit_blob("base64", len(value))
        if len(value) > max_str_len * 8:
            return redact_base64_in_text(value[:max_str_len]) + f"… [truncated, {len(value)} chars total]"
        return redact_base64_in_text(value)
    if isinstance(value, list):
        return [redact_value_for_log(v, max_str_len=max_str_len) for v in value]
    if isinstance(value, dict):
        return redact_dict_for_log(value, max_str_len=max_str_len)
    return value


def redact_dict_for_log(data: dict, *, max_str_len: int = _DEFAULT_MAX_STR) -> dict:
    out: dict = {}
    for key, val in data.items():
        key_l = str(key).lower()
        if key_l in _SENSITIVE_KEYS and isinstance(val, str):
            out[key] = "[redacted]"
            continue
        if key_l == "url" and isinstance(val, str) and val.startswith("data:image"):
            out[key] = _omit_blob("image data URL", len(val))
            continue
        if key_l in _IMAGE_DATA_KEYS:
            if isinstance(val, str) and (val.startswith("data:image") or _looks_like_base64(val)):
                out[key] = _omit_blob("image payload", len(val))
                continue
            if isinstance(val, dict):
                out[key] = redact_dict_for_log(val, max_str_len=max_str_len)
                continue
        out[key] = redact_value_for_log(val, max_str_len=max_str_len)
    return out


def redact_llm_payload_for_log(payload: Any) -> Any:
    """Deep-copy and redact an LLM request/response dict for logs, traces, and OTEL."""
    if payload is None:
        return None
    try:
        cloned = copy.deepcopy(payload)
    except Exception:
        cloned = payload
    return redact_value_for_log(cloned)


def safe_log_json(payload: Any, *, max_str_len: int = _DEFAULT_MAX_STR) -> str:
    """JSON-serialize an LLM payload with base64 and large strings redacted."""
    try:
        redacted = redact_llm_payload_for_log(payload)
        return json.dumps(redacted, default=str)
    except Exception as e:
        return json.dumps({"error": f"log serialization failed: {e}"})
