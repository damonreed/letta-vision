from letta.llm_api.openai_client import _openai_client_timeout
from letta.settings import settings


def test_default_request_timeout_is_300():
    assert settings.llm_request_timeout_seconds == 300.0


def test_openai_client_timeout_uses_settings():
    timeout = _openai_client_timeout(streaming=False)
    assert timeout.read == settings.llm_request_timeout_seconds


def test_retry_on_timeout_defaults_false():
    assert settings.llm_retry_on_timeout is False
