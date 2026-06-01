"""
Curated vision-capability registry for LLM models.

Single source of truth for supports_vision flags and README documentation.

There are 12 registry rows (FR §3.1). A live catalog may show ~30 entries with
supports_vision=true because dated variants expand via globs (e.g. six gpt-4o*
snapshots). Over-inclusion is worse than under-inclusion: a false positive lets
the client attach images and the provider may silently degrade.

Patterns are intentionally conservative:
- o3-mini* is excluded (text/reasoning-only on the API; o3 and o3-pro have vision)
- gpt-4 / claude-3 / llama must not match (see tests/test_vision_capability.py)
"""

from __future__ import annotations

import fnmatch
import json
import os
from pathlib import Path

from letta.settings import settings

# (provider label, model id or pattern) — patterns use shell-style globs (*)
VISION_CAPABLE_MODELS: list[tuple[str, str]] = [
    # OpenRouter — empirically validated (smoke test)
    ("OpenRouter", "moonshotai/kimi-k2.6"),
    ("OpenRouter", "moonshotai/kimi-k2.5"),
    # Moonshot AI direct (OpenAI-compatible BYOK; model ids omit org prefix)
    ("Moonshot", "kimi-k2.6"),
    ("Moonshot", "kimi-k2.5"),
    ("Moonshot", "kimi-k2*"),
    ("Moonshot", "*vision-preview*"),
    # OpenAI — FR §3.1 families; globs expand dated catalog entries only
    ("OpenAI", "gpt-4o*"),
    ("OpenAI", "gpt-4.1*"),
    ("OpenAI", "o1"),
    ("OpenAI", "o1-*"),
    ("OpenAI", "o3"),
    ("OpenAI", "o3-pro*"),
    ("OpenAI", "o3-2025-*"),
    ("OpenAI", "o4-mini*"),
    # Anthropic — Claude 4 multimodal line
    ("Anthropic", "claude-opus-4-*"),
    ("Anthropic", "claude-sonnet-4-*"),
    ("Anthropic", "claude-haiku-4-*"),
    # Google — when provider is configured
    ("Google", "gemini-2.5-pro*"),
    ("Google", "gemini-2.5-flash*"),
    # MiniMax M3 — natively multimodal (OpenAI-compatible BYOK and native provider)
    ("Minimax", "MiniMax-M3"),
    ("Minimax", "minimax-m3"),
    ("MiniMax", "MiniMax-M3*"),
    ("MiniMax", "minimax-m3*"),
]

_EXTRA_VISION_MODELS: set[str] | None = None
_BRIDGE_VISION_OVERRIDES: dict[str, bool] | None = None
_BRIDGE_OVERRIDES_MTIME: float | None = None


def _normalize_model_basename(model_id: str) -> str:
    return model_id.rsplit("/", 1)[-1].lower().replace("_", "-")


def _load_bridge_vision_overrides() -> dict[str, bool]:
    """Manual vision flags from letta-vision-client (shared model_overrides.json)."""
    global _BRIDGE_VISION_OVERRIDES, _BRIDGE_OVERRIDES_MTIME

    path = Path(os.environ.get("MODEL_OVERRIDES_PATH", "/data/shared/model_overrides.json"))
    mtime = path.stat().st_mtime if path.exists() else 0.0
    if _BRIDGE_VISION_OVERRIDES is not None and _BRIDGE_OVERRIDES_MTIME == mtime:
        return _BRIDGE_VISION_OVERRIDES

    overrides: dict[str, bool] = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            raw = data.get("vision") if isinstance(data, dict) else None
            if isinstance(raw, dict):
                for key, value in raw.items():
                    if isinstance(key, str) and key.strip():
                        overrides[key.strip()] = bool(value)
        except (json.JSONDecodeError, OSError):
            pass

    _BRIDGE_VISION_OVERRIDES = overrides
    _BRIDGE_OVERRIDES_MTIME = mtime
    return overrides


def _bridge_override_for_identifiers(identifiers: list[str], overrides: dict[str, bool]) -> bool | None:
    if not overrides:
        return None
    for ident in identifiers:
        if ident in overrides:
            return overrides[ident]
        ident_lower = ident.lower()
        ident_base = _normalize_model_basename(ident)
        for key, value in overrides.items():
            if key.lower() == ident_lower:
                return value
            if _normalize_model_basename(key) == ident_base:
                return value
    return None


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
    seen: set[str] = set()
    ordered: list[str] = []

    def add(value: str | None) -> None:
        if not value or value in seen:
            return
        seen.add(value)
        ordered.append(value)
        add(_normalize_model_basename(value))
        if "/" in value:
            add(value.split("/", 1)[-1])

    add(model)
    if handle:
        add(handle)
    return ordered


def _matches_pattern(model_id: str, pattern: str) -> bool:
    model_id_lower = model_id.lower()
    pattern_lower = pattern.lower()
    if "*" in pattern_lower or "?" in pattern_lower:
        return fnmatch.fnmatch(model_id_lower, pattern_lower)
    return model_id_lower == pattern_lower


def model_supports_vision(model: str, handle: str | None = None) -> bool:
    """Return True if the model is flagged as vision-capable in the registry."""
    identifiers = _model_identifiers(model, handle)
    bridge_override = _bridge_override_for_identifiers(identifiers, _load_bridge_vision_overrides())
    if bridge_override is not None:
        return bridge_override

    extra = {x.lower() for x in _load_extra_vision_models()}
    for ident in identifiers:
        if ident.lower() in extra:
            return True
        ident_lower = ident.lower()
        for _, pattern in VISION_CAPABLE_MODELS:
            if _matches_pattern(ident_lower, pattern):
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
