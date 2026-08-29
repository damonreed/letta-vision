"""
Vision-capability resolution for LLM models.

Precedence (see model_supports_vision):
1. Manual override (model_overrides.json from letta-vision-client)
2. OpenRouter catalog cache (architecture.input_modalities) for openrouter/* handles
3. Persisted provider_models.supports_vision (positive only) when the OpenRouter cache is cold
4. Curated registry globs + LETTA_VISION_MODELS_EXTRA for BYOK / non-OpenRouter paths

The curated registry (FR §3.1) remains authoritative for openai-proxy, Moonshot BYOK,
Ollama, etc. OpenRouter base-provider handles use the OpenRouter /v1/models catalog
at sync time — over-inclusion is worse than under-inclusion for those paths too, but
the upstream catalog is the practical source of truth for hundreds of routed models.
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
    ("OpenRouter", "moonshotai/kimi-k3"),
    ("OpenRouter", "z-ai/glm-5.3-flash"),
    # Z.AI direct / BYOK (OpenAI-compatible; model ids omit org prefix)
    ("Z.AI", "glm-5.3-flash"),
    # Moonshot AI direct (OpenAI-compatible BYOK; model ids omit org prefix)
    ("Moonshot", "kimi-k3"),
    ("Moonshot", "kimi-k3*"),
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

# (provider label, model id or pattern, max image parts per request)
# Empirically measured serving-side caps: these providers silently drop image parts
# beyond the cap instead of erroring. kimi-k2.6 via OpenRouter (Parasail and Novita,
# probed 2026-06-11 with 12 numbered images) keeps only the FIRST 8 parts, so the
# newest images vanish unless we demote older ones first. k2.5 shares the serving
# stack and is assumed to have the same cap.
MODEL_MAX_IMAGE_PARTS: list[tuple[str, str, int]] = [
    ("OpenRouter", "moonshotai/kimi-k2*", 8),
    ("Moonshot", "kimi-k2*", 8),
]


def model_max_image_parts(model: str, handle: str | None = None) -> int | None:
    """Max image parts the model reliably receives per request (None = no known cap)."""
    for ident in _model_identifiers(model, handle):
        ident_lower = ident.lower()
        for _, pattern, cap in MODEL_MAX_IMAGE_PARTS:
            if _matches_pattern(ident_lower, pattern):
                return cap
    return None


_EXTRA_VISION_MODELS: set[str] | None = None
_BRIDGE_VISION_OVERRIDES: dict[str, bool] | None = None
_BRIDGE_OVERRIDES_MTIME: float | None = None
_OPENROUTER_VISION_BY_MODEL_ID: dict[str, bool] = {}


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


def _is_openrouter_handle(handle: str | None) -> bool:
    if not handle:
        return False
    return handle.lower().startswith("openrouter/")


def refresh_openrouter_vision_cache(models: list[dict]) -> None:
    """Rebuild in-memory OpenRouter vision cache from a /v1/models list response."""
    from letta.schemas.providers.openrouter import OpenRouterProvider

    global _OPENROUTER_VISION_BY_MODEL_ID
    updated: dict[str, bool] = {}
    for model in models:
        model_id = model.get("id")
        if not isinstance(model_id, str) or not model_id.strip():
            continue
        updated[model_id] = OpenRouterProvider.model_has_image_input(model)
    _OPENROUTER_VISION_BY_MODEL_ID = updated


def warm_openrouter_vision_cache_from_db(model_id: str, supports_vision: bool | None) -> None:
    """Seed or update one OpenRouter model id in the vision cache (e.g. after DB sync)."""
    if not model_id or supports_vision is None:
        return
    _OPENROUTER_VISION_BY_MODEL_ID[model_id] = bool(supports_vision)


def warm_openrouter_vision_cache_from_db_rows(rows: list[tuple[str, bool | None]]) -> None:
    """Bulk-load persisted OpenRouter vision flags (e.g. after startup DB sync)."""
    for model_id, supports_vision in rows:
        warm_openrouter_vision_cache_from_db(model_id, supports_vision)


def openrouter_model_supports_vision(model_id: str) -> bool | None:
    """Return cached OpenRouter vision flag for a model id, or None if unknown."""
    if model_id in _OPENROUTER_VISION_BY_MODEL_ID:
        return _OPENROUTER_VISION_BY_MODEL_ID[model_id]
    return None


def _registry_supports_vision(identifiers: list[str]) -> bool:
    extra = {x.lower() for x in _load_extra_vision_models()}
    for ident in identifiers:
        if ident.lower() in extra:
            return True
        ident_lower = ident.lower()
        for _, pattern in VISION_CAPABLE_MODELS:
            if _matches_pattern(ident_lower, pattern):
                return True
    return False


def _openrouter_cache_supports_vision(model: str, handle: str | None, identifiers: list[str]) -> bool | None:
    if not any(_is_openrouter_handle(ident) for ident in identifiers):
        return None
    for ident in identifiers:
        if ident in _OPENROUTER_VISION_BY_MODEL_ID:
            return _OPENROUTER_VISION_BY_MODEL_ID[ident]
    cached = openrouter_model_supports_vision(model)
    if cached is not None:
        return cached
    return None


def model_supports_vision(model: str, handle: str | None = None, db_flag: bool | None = None) -> bool:
    """Return True if the model accepts image content blocks for the configured handle.

    OpenRouter cache misses do not consult registry globs (over-inclusion is worse
    than under-inclusion for routed catalogs). Callers that have a persisted
    provider_models.supports_vision value should pass it as db_flag so a cold
    cache cannot persist false onto a newly assigned agent.
    """
    identifiers = _model_identifiers(model, handle)
    bridge_override = _bridge_override_for_identifiers(identifiers, _load_bridge_vision_overrides())
    if bridge_override is not None:
        return bridge_override

    openrouter_cached = _openrouter_cache_supports_vision(model, handle, identifiers)
    if openrouter_cached is not None:
        return openrouter_cached

    # Positive DB catalog flag only. LLMConfig.supports_vision defaults to False, so a
    # missing/unknown row must not block BYOK registry matches.
    if db_flag:
        return True

    if any(_is_openrouter_handle(ident) for ident in identifiers):
        return False

    return _registry_supports_vision(identifiers)


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
