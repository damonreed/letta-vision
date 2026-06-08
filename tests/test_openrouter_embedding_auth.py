import pytest

from letta.llm_api.openai_client import OpenAIClient
from letta.schemas.embedding_config import EmbeddingConfig
from letta.settings import model_settings


def _prepare_embedding_kwargs(embedding_config: EmbeddingConfig) -> dict:
    client = OpenAIClient(actor=None)
    return client._prepare_client_kwargs_embedding(embedding_config)


@pytest.mark.parametrize(
    "embedding_config,env_key,expected_api_key",
    [
        (
            EmbeddingConfig(
                embedding_endpoint_type="openrouter",
                embedding_endpoint="https://openrouter.ai/api/v1",
                embedding_model="openai/text-embedding-3-small",
                embedding_dim=1536,
                handle="openrouter/openai/text-embedding-3-small",
            ),
            "or-test-key",
            "or-test-key",
        ),
        (
            EmbeddingConfig(
                embedding_endpoint_type="openai",
                embedding_endpoint="https://openrouter.ai/api/v1",
                embedding_model="text-embedding-3-small",
                embedding_dim=1536,
                handle="openrouter/text-embedding-3-small",
            ),
            "or-handle-key",
            "or-handle-key",
        ),
        (
            EmbeddingConfig(
                embedding_endpoint_type="openai",
                embedding_endpoint="http://host.docker.internal:8089/v1",
                embedding_model="bge-m3-q4_k_m.gguf",
                embedding_dim=1024,
                handle="llama-cpp-embed/bge-m3-q4_k_m.gguf",
            ),
            None,
            "DUMMY_API_KEY",
        ),
    ],
)
def test_prepare_client_kwargs_embedding_api_key(embedding_config, env_key, expected_api_key, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    if env_key:
        monkeypatch.setenv("OPENROUTER_API_KEY", env_key)

    kwargs = _prepare_embedding_kwargs(embedding_config)
    assert kwargs["api_key"] == expected_api_key

    if "openrouter.ai" in (embedding_config.embedding_endpoint or ""):
        headers = kwargs.get("default_headers") or {}
        if model_settings.openrouter_referer:
            assert headers.get("HTTP-Referer") == model_settings.openrouter_referer
        if model_settings.openrouter_title:
            assert headers.get("X-Title") == model_settings.openrouter_title
