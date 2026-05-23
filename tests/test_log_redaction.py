import json

from letta.helpers.log_redaction import redact_llm_payload_for_log, safe_log_json


def test_redacts_openai_vision_message():
    huge = "A" * 50_000
    payload = {
        "model": "openai-proxy/Qwen/Qwen-Image",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{huge}"}},
                ],
            }
        ],
    }
    out = safe_log_json(payload)
    assert huge not in out
    assert "omitted" in out.lower()


def test_redacts_nested_source_base64():
    huge = "B" * 10_000
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": huge},
                    }
                ],
            }
        ]
    }
    redacted = redact_llm_payload_for_log(payload)
    dumped = json.dumps(redacted)
    assert huge not in dumped
