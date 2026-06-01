"""BYOK OpenAI-compatible providers include provider slug in model handles."""

from letta.schemas.enums import ProviderCategory, ProviderType
from letta.schemas.providers.openai import OpenAIProvider


def test_byok_openai_proxy_handle_includes_provider_slug():
    provider = OpenAIProvider(
        name="siliconflow",
        provider_category=ProviderCategory.byok,
        base_url="https://api.siliconflow.com/v1",
    )
    assert provider.get_handle("moonshotai/Kimi-K2.6", base_name="openai-proxy") == (
        "openai-proxy/siliconflow/moonshotai/Kimi-K2.6"
    )


def test_byok_moonshot_handle_includes_slug():
    provider = OpenAIProvider(
        name="Moonshot AI",
        provider_category=ProviderCategory.byok,
        base_url="https://api.moonshot.ai/v1",
    )
    assert provider.get_handle("kimi-k2.6", base_name="openai-proxy") == "openai-proxy/moonshot-ai/kimi-k2.6"


def test_base_openai_proxy_skips_slug():
    provider = OpenAIProvider(
        name="openai",
        provider_category=ProviderCategory.base,
        base_url="https://api.openai.com/v1",
    )
    assert provider.get_handle("gpt-4o") == "openai/gpt-4o"
