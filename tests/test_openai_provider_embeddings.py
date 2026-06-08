import pytest

from letta.schemas.enums import ProviderCategory
from letta.schemas.providers.openai import OpenAIProvider
from letta.schemas.providers.openrouter import OpenRouterProvider


@pytest.mark.asyncio
async def test_self_hosted_embedding_discovery_from_models_api(monkeypatch):
    provider = OpenAIProvider(
        name="llama-cpp-embed",
        provider_category=ProviderCategory.byok,
        base_url="http://host.docker.internal:8089/v1",
    )

    async def mock_get_models():
        return [
            {
                "id": "bge-m3-q4_k_m.gguf",
                "meta": {"n_embd": 1024, "n_ctx": 8192},
            }
        ]

    monkeypatch.setattr(provider, "_get_models_async", mock_get_models)

    embeddings = await provider.list_embedding_models_async()
    assert len(embeddings) == 1
    assert embeddings[0].embedding_model == "bge-m3-q4_k_m.gguf"
    assert embeddings[0].embedding_dim == 1024
    assert embeddings[0].handle == "llama-cpp-embed/bge-m3-q4_k_m.gguf"


@pytest.mark.asyncio
async def test_self_hosted_chat_provider_has_no_fake_openai_embeddings(monkeypatch):
    provider = OpenAIProvider(
        name="llama-cpp-chat",
        provider_category=ProviderCategory.byok,
        base_url="http://host.docker.internal:8088/v1",
    )

    async def mock_get_models():
        return [
            {
                "id": "gemma-4-26B-A4B-it-Q4_K_M.gguf",
                "meta": {"n_embd": 2816, "n_ctx": 65536},
                "capabilities": ["completion", "multimodal"],
            }
        ]

    monkeypatch.setattr(provider, "_get_models_async", mock_get_models)

    embeddings = await provider.list_embedding_models_async()
    assert embeddings == []


@pytest.mark.asyncio
async def test_self_hosted_llm_sync_skips_embedding_models(monkeypatch):
    provider = OpenAIProvider(
        name="llama-cpp-embed",
        provider_category=ProviderCategory.byok,
        base_url="http://host.docker.internal:8089/v1",
    )

    data = [{"id": "bge-m3-q4_k_m.gguf", "meta": {"n_embd": 1024, "n_ctx": 8192}}]
    llms = await provider._list_llm_models(data)
    assert llms == []


@pytest.mark.asyncio
async def test_openrouter_gateway_has_no_fake_openai_embeddings(monkeypatch):
    provider = OpenRouterProvider(
        name="openrouter",
        provider_category=ProviderCategory.base,
        base_url="https://openrouter.ai/api/v1",
    )

    async def mock_list(*_args, **_kwargs):
        return {"data": []}

    monkeypatch.setattr("letta.llm_api.openai.openai_get_model_list_async", mock_list)

    embeddings = await provider.list_embedding_models_async()
    assert embeddings == []


@pytest.mark.asyncio
async def test_openrouter_discovers_embedding_models_from_api(monkeypatch):
    provider = OpenRouterProvider(
        name="openrouter",
        provider_category=ProviderCategory.base,
        base_url="https://openrouter.ai/api/v1",
    )

    async def mock_list(*_args, **_kwargs):
        return {
            "data": [
                {
                    "id": "openai/text-embedding-3-small",
                    "architecture": {"output_modalities": ["embeddings"]},
                },
                {
                    "id": "baai/bge-m3",
                    "architecture": {"output_modalities": ["embeddings"]},
                },
                {
                    "id": "anthropic/claude-sonnet-4",
                    "architecture": {"output_modalities": ["text"]},
                },
            ]
        }

    monkeypatch.setattr("letta.llm_api.openai.openai_get_model_list_async", mock_list)

    embeddings = await provider.list_embedding_models_async()
    handles = {e.handle for e in embeddings}
    assert handles == {"openrouter/openai/text-embedding-3-small", "openrouter/baai/bge-m3"}
    by_model = {e.embedding_model: e for e in embeddings}
    assert by_model["openai/text-embedding-3-small"].embedding_dim == 1536
    assert by_model["openai/text-embedding-3-small"].embedding_endpoint_type == "openrouter"
    assert by_model["baai/bge-m3"].embedding_dim == 1024


@pytest.mark.asyncio
async def test_official_openai_keeps_static_embedding_catalog():
    provider = OpenAIProvider(
        name="openai",
        provider_category=ProviderCategory.base,
        base_url="https://api.openai.com/v1",
    )

    embeddings = await provider.list_embedding_models_async()
    handles = {e.handle for e in embeddings}
    assert "openai/text-embedding-3-small" in handles
    assert "openai/text-embedding-ada-002" in handles
