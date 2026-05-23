import json

import pytest

from letta.services.context_window_calculator.message_payload_for_token_estimate import (
    ESTIMATED_TOKENS_PER_IMAGE,
    openai_content_block_to_plaintext,
    strip_images_from_api_messages_for_token_estimate,
)
from letta.services.context_window_calculator.token_counter import ApproxTokenCounter


def _huge_data_url() -> str:
    return "data:image/png;base64," + ("A" * 500_000)


@pytest.mark.asyncio
async def test_approx_counter_ignores_base64_payload_size():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {"type": "image_url", "image_url": {"url": _huge_data_url(), "detail": "auto"}},
            ],
        }
    ]
    counter = ApproxTokenCounter()
    raw_tokens = counter._approx_token_count(json.dumps(messages))
    estimated = await counter.count_message_tokens(messages)

    assert estimated < 10_000
    assert estimated > ESTIMATED_TOKENS_PER_IMAGE
    assert raw_tokens > 100_000


def test_strip_images_counts_images():
    messages = [
        {
            "role": "tool",
            "content": [
                {"type": "text", "text": "screenshot"},
                {"type": "image_url", "image_url": {"url": _huge_data_url()}},
            ],
        }
    ]
    redacted, image_count = strip_images_from_api_messages_for_token_estimate(messages)
    assert image_count == 1
    assert _huge_data_url() not in json.dumps(redacted)
    assert "[image]" in json.dumps(redacted)


def test_openai_content_block_to_plaintext_omits_images():
    block = {"type": "image_url", "image_url": {"url": _huge_data_url()}}
    assert openai_content_block_to_plaintext(block) == "[Image omitted]"
    assert _huge_data_url() not in openai_content_block_to_plaintext(block)
