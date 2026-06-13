import json

import pytest

from letta.llm_api import model_registry as reg
from letta.llm_api.model_registry import (
    model_supports_vision,
    openrouter_model_supports_vision,
    refresh_openrouter_vision_cache,
)
from letta.schemas.enums import ProviderCategory
from letta.schemas.providers.openrouter import OpenRouterProvider


def _or_models():
    return [
        {
            "id": "moonshotai/kimi-k2.6",
            "architecture": {"input_modalities": ["text", "image"]},
        },
        {
            "id": "deepseek/deepseek-v4-pro",
            "architecture": {"input_modalities": ["text"]},
        },
        {
            "id": "google/gemma-4-31b-it",
            "architecture": {"modality": "text+image->text"},
        },
    ]


@pytest.fixture(autouse=True)
def _reset_openrouter_cache():
    reg._OPENROUTER_VISION_BY_MODEL_ID.clear()
    yield
    reg._OPENROUTER_VISION_BY_MODEL_ID.clear()


def test_model_has_image_input_from_input_modalities():
    model = {"architecture": {"input_modalities": ["text", "image"]}}
    assert OpenRouterProvider.model_has_image_input(model)


def test_model_has_image_input_text_only():
    model = {"architecture": {"input_modalities": ["text"]}}
    assert not OpenRouterProvider.model_has_image_input(model)


def test_model_has_image_input_modality_fallback():
    model = {"architecture": {"modality": "text+image->text", "input_modalities": ["text"]}}
    assert OpenRouterProvider.model_has_image_input(model)


def test_refresh_openrouter_vision_cache():
    refresh_openrouter_vision_cache(_or_models())
    assert openrouter_model_supports_vision("moonshotai/kimi-k2.6") is True
    assert openrouter_model_supports_vision("deepseek/deepseek-v4-pro") is False
    assert openrouter_model_supports_vision("google/gemma-4-31b-it") is True
    assert openrouter_model_supports_vision("unknown/model") is None


def test_openrouter_deepseek_v4_pro_cached_false():
    refresh_openrouter_vision_cache(_or_models())
    assert not model_supports_vision(
        "deepseek/deepseek-v4-pro",
        handle="openrouter/deepseek/deepseek-v4-pro",
    )


def test_openrouter_kimi_k26_cached_true():
    refresh_openrouter_vision_cache(_or_models())
    assert model_supports_vision(
        "moonshotai/kimi-k2.6",
        handle="openrouter/moonshotai/kimi-k2.6",
    )


def test_openrouter_without_cache_defaults_false_not_registry():
    """Registry would match kimi, but openrouter/* without cache must return false."""
    assert not model_supports_vision(
        "moonshotai/kimi-k2.6",
        handle="openrouter/moonshotai/kimi-k2.6",
    )


def test_non_openrouter_handle_uses_registry():
    assert model_supports_vision(
        "moonshotai/kimi-k2.6",
        handle="openai-proxy/moonshotai/kimi-k2.6",
    )


def test_manual_override_wins_over_openrouter_cache(tmp_path, monkeypatch):
    refresh_openrouter_vision_cache(_or_models())
    overrides = tmp_path / "model_overrides.json"
    overrides.write_text(
        json.dumps({"vision": {"openrouter/deepseek/deepseek-v4-pro": True}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("MODEL_OVERRIDES_PATH", str(overrides))
    reg._BRIDGE_VISION_OVERRIDES = None
    assert model_supports_vision(
        "deepseek/deepseek-v4-pro",
        handle="openrouter/deepseek/deepseek-v4-pro",
    )
    reg._BRIDGE_VISION_OVERRIDES = None


def test_manual_override_can_disable_image_capable_or_model(tmp_path, monkeypatch):
    refresh_openrouter_vision_cache(_or_models())
    overrides = tmp_path / "model_overrides.json"
    overrides.write_text(
        json.dumps({"vision": {"openrouter/moonshotai/kimi-k2.6": False}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("MODEL_OVERRIDES_PATH", str(overrides))
    reg._BRIDGE_VISION_OVERRIDES = None
    assert not model_supports_vision(
        "moonshotai/kimi-k2.6",
        handle="openrouter/moonshotai/kimi-k2.6",
    )
    reg._BRIDGE_VISION_OVERRIDES = None


@pytest.mark.asyncio
async def test_list_llm_models_async_stamps_supports_vision(monkeypatch):
    provider = OpenRouterProvider(
        name="openrouter",
        provider_category=ProviderCategory.base,
        base_url="https://openrouter.ai/api/v1",
    )

    async def mock_list(*_args, **_kwargs):
        return {"data": _or_models()}

    monkeypatch.setattr("letta.llm_api.openai.openai_get_model_list_async", mock_list)

    configs = await provider.list_llm_models_async()
    by_model = {c.model: c.supports_vision for c in configs}
    assert by_model["moonshotai/kimi-k2.6"] is True
    assert by_model["deepseek/deepseek-v4-pro"] is False
    assert by_model["google/gemma-4-31b-it"] is True
    assert openrouter_model_supports_vision("deepseek/deepseek-v4-pro") is False
