"""
Curated vision-capability registry for LLM models.

Single source of truth for supports_vision flags and README documentation.
"""

from __future__ import annotations

import fnmatch

from letta.settings import settings

# (provider label, model id or pattern) — patterns use shell-style globs (*)
VISION_CAPABLE_MODELS: list[tuple[str, str]] = [
    ("OpenRouter", "moonshotai/kimi-k2.6"),
    ("OpenRouter", "moonshotai/kimi-k2.5"),
    ("OpenAI", "gpt-4o*"),
    ("OpenAI", "gpt-4.1*"),
    ("OpenAI", "o1*"),
    ("OpenAI", "o3*"),
    ("OpenAI", "o4*"),
    ("Anthropic", "claude-opus-4-*"),
    ("Anthropic", "claude-sonnet-4-*"),
    ("Anthropic", "claude-haiku-4-*"),
    ("Google", "gemini-2.5-pro*"),
    ("Google", "gemini-2.5-flash*"),
]

_EXTRA_VISION_MODELS: set[str] | None = None


def _load_extra_vision_models() -> set[str]:
    global _EXTRA_VISION_MODELS
    if _EXTRA_VISION_MODELS is not None:
        return _EXTRA_VISION_MODELS
    raw = (settings.vision_models_extra or "").strip()
    if not raw:
        _EXTRA_VISION_MODELS = set()
    else:
        _EXTRA_VISION_MODELS = {part.strip() for part in raw.split(",") if part.strip()}
    return _EXTRA_VISION_MODELS


def _model_identifiers(model: str, handle: str | None = None) -> list[str]:
    ids = [model]
    if handle:
        ids.append(handle)
        if "/" in handle:
            ids.append(handle.split("/", 1)[-1])
    return ids


def _matches_pattern(model_id: str, pattern: str) -> bool:
    if "*" in pattern or "?" in pattern:
        return fnmatch.fnmatch(model_id, pattern)
    return model_id == pattern


def model_supports_vision(model: str, handle: str | None = None) -> bool:
    """Return True if the model is flagged as vision-capable in the registry."""
    identifiers = _model_identifiers(model, handle)
    extra = _load_extra_vision_models()
    for ident in identifiers:
        if ident in extra:
            return True
        for _, pattern in VISION_CAPABLE_MODELS:
            if _matches_pattern(ident, pattern):
                return True
    return False


def registry_table_rows() -> list[tuple[str, str]]:
    """Rows for operator documentation (provider, model id/pattern)."""
    return list(VISION_CAPABLE_MODELS)


def merge_provider_preferences(llm_config, extra_body: dict) -> dict:
    """Merge OpenRouter provider preferences from LLMConfig into extra_body."""
    prefs = getattr(llm_config, "provider_preferences", None)
    if not prefs:
        return extra_body
    merged = dict(extra_body)
    existing = merged.get("provider")
    if isinstance(existing, dict) and isinstance(prefs, dict):
        merged["provider"] = {**existing, **prefs}
    else:
        merged["provider"] = prefs
    return merged
