"""
Multi-turn image context persistence (v0.4.0).

Live tests require a running Letta server with OpenRouter configured (OPENROUTER_API_KEY)
and are marked integration. Serialization guards run offline.
"""

import base64
import os
from pathlib import Path
from typing import Any, List

import pytest
from letta_client import Letta
from letta_client.types import AgentState
from letta_client.types.agents import AssistantMessage
from letta_client.types.agents.image_content_param import ImageContentParam, SourceBase64Image
from letta_client.types.agents.text_content_param import TextContentParam

from letta.schemas.enums import MessageRole
from letta.schemas.letta_message_content import ImageContent, LettaImage, TextContent
from letta.schemas.message import Message

pytestmark = pytest.mark.integration

KIMI_MODEL = "moonshotai/kimi-k2.6"
DATA_DIR = Path(__file__).parent / "data"
ANT_IMAGE_PATH = DATA_DIR / "Camponotus_flavomarginatus_ant.jpg"

# Minimal 1x1 red PNG (differs visually from the ant photo for two-image compare tests).
RED_PIXEL_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)

DEGRADATION_MARKERS = (
    "no image",
    "cannot see",
    "can't see",
    "don't see an image",
    "no attached image",
    "image was mentioned",
    "unable to view",
    "i cannot view images",
    "image is not visible",
    "i don't have access to",
    "as a text-based",
    "please upload",
    "re-send",
    "resend",
)


def _openrouter_configured() -> bool:
    return bool(os.getenv("OPENROUTER_API_KEY"))


def _load_ant_image_b64() -> str:
    with open(ANT_IMAGE_PATH, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def _assistant_text(messages: List[Any]) -> str:
    parts = []
    for msg in messages:
        if isinstance(msg, AssistantMessage):
            content = msg.content
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    text = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
                    if text:
                        parts.append(text)
    return "\n".join(parts)


def _assert_no_degradation(text: str) -> None:
    blob = text.lower()
    assert text.strip(), "Expected non-empty assistant response"
    for marker in DEGRADATION_MARKERS:
        assert marker not in blob, f"Degradation marker {marker!r} in response: {text[:300]}"


def _user_content_has_image_url(content) -> bool:
    if isinstance(content, list):
        return any(
            (p.get("type") == "image_url" if isinstance(p, dict) else getattr(p, "type", None) == "image_url")
            for p in content
        )
    return False


@pytest.fixture(scope="function")
def vision_agent(client: Letta) -> AgentState:
    send_message_tool = client.tools.list(name="send_message").items[0]
    agent = client.agents.create(
        name="vision-persistence-test",
        agent_type="letta_v1_agent",
        include_base_tools=False,
        tool_ids=[send_message_tool.id],
        model=KIMI_MODEL,
        embedding="openai/text-embedding-3-small",
    )
    yield agent


@pytest.fixture
def ant_image_message() -> List[dict]:
    data = _load_ant_image_b64()
    return [
        {
            "role": "user",
            "content": [
                ImageContentParam(
                    type="image",
                    source=SourceBase64Image(type="base64", data=data, media_type="image/jpeg"),
                ),
                TextContentParam(
                    type="text",
                    text="What animal is in this photo? Describe its color briefly.",
                ),
            ],
        }
    ]


@pytest.fixture
def red_pixel_image_message() -> List[dict]:
    return [
        {
            "role": "user",
            "content": [
                ImageContentParam(
                    type="image",
                    source=SourceBase64Image(type="base64", data=RED_PIXEL_PNG_B64, media_type="image/png"),
                ),
                TextContentParam(
                    type="text",
                    text="This is the second image: a solid color square. Name the color you see.",
                ),
            ],
        }
    ]


def test_serialization_guard_three_turn_history():
    """Offline guard: every image-bearing user turn in history must reach the provider payload."""
    ant = ImageContent(
        source=LettaImage(file_id="file-1", data="YW50", media_type="image/jpeg"),
    )
    red = ImageContent(
        source=LettaImage(file_id="file-2", data=RED_PIXEL_PNG_B64, media_type="image/png"),
    )
    history = [
        Message(role=MessageRole.system, content=[TextContent(text="sys")]),
        Message(role=MessageRole.user, content=[TextContent(text="turn1"), ant]),
        Message(role=MessageRole.assistant, content=[TextContent(text="ant")]),
        Message(role=MessageRole.user, content=[TextContent(text="turn2 text only")]),
        Message(role=MessageRole.assistant, content=[TextContent(text="ok")]),
        Message(role=MessageRole.user, content=[TextContent(text="turn3"), red]),
    ]
    serialized = Message.to_openai_dicts_from_list(history)
    image_user_rows = [
        m for m in serialized if m["role"] == "user" and _user_content_has_image_url(m["content"])
    ]
    assert len(image_user_rows) == 2


@pytest.mark.skipif(not _openrouter_configured(), reason="OPENROUTER_API_KEY not set")
def test_image_recall_across_turns(
    disable_e2b_api_key: Any,
    client: Letta,
    vision_agent: AgentState,
    ant_image_message: List[dict],
) -> None:
    """Turn 1: ant image. Turn 2: text-only recall question."""
    client.agents.messages.create(agent_id=vision_agent.id, messages=ant_image_message)
    response = client.agents.messages.create(
        agent_id=vision_agent.id,
        messages=[
            {
                "role": "user",
                "content": (
                    "What color was the animal in the photo I sent earlier? "
                    "Answer in one short sentence using only the earlier image, not a guess."
                ),
            }
        ],
    )
    text = _assistant_text(response.messages)
    _assert_no_degradation(text)
    blob = text.lower()
    assert any(word in blob for word in ("red", "orange", "brown", "rust", "ant")), (
        f"Expected color recall from prior image, got: {text[:400]}"
    )


@pytest.mark.skipif(not _openrouter_configured(), reason="OPENROUTER_API_KEY not set")
def test_two_image_comparison_across_turns(
    disable_e2b_api_key: Any,
    client: Letta,
    vision_agent: AgentState,
    ant_image_message: List[dict],
    red_pixel_image_message: List[dict],
) -> None:
    """Turn 1: ant. Turn 2: red pixel. Turn 3: compare both."""
    client.agents.messages.create(agent_id=vision_agent.id, messages=ant_image_message)
    client.agents.messages.create(agent_id=vision_agent.id, messages=red_pixel_image_message)
    response = client.agents.messages.create(
        agent_id=vision_agent.id,
        messages=[
            {
                "role": "user",
                "content": (
                    "Compare the first image I sent (animal photo) with the second image (solid color). "
                    "State one concrete visual difference between them in one or two sentences."
                ),
            }
        ],
    )
    text = _assistant_text(response.messages)
    _assert_no_degradation(text)
    blob = text.lower()
    assert len(text) > 40
    # Should reference both kinds of content, not ask for re-upload
    assert any(
        kw in blob
        for kw in (
            "ant",
            "insect",
            "animal",
            "photo",
            "red",
            "solid",
            "square",
            "pixel",
            "color",
            "different",
        )
    ), f"Expected comparison across both images, got: {text[:400]}"


@pytest.mark.skipif(not _openrouter_configured(), reason="OPENROUTER_API_KEY not set")
def test_interleaved_text_and_images(
    disable_e2b_api_key: Any,
    client: Letta,
    vision_agent: AgentState,
    ant_image_message: List[dict],
    red_pixel_image_message: List[dict],
) -> None:
    """Text follow-up, then second image, then question relating both."""
    client.agents.messages.create(agent_id=vision_agent.id, messages=ant_image_message)
    client.agents.messages.create(
        agent_id=vision_agent.id,
        messages=[{"role": "user", "content": "Thanks. I may send another image next."}],
    )
    client.agents.messages.create(agent_id=vision_agent.id, messages=red_pixel_image_message)
    response = client.agents.messages.create(
        agent_id=vision_agent.id,
        messages=[
            {
                "role": "user",
                "content": (
                    "How does the solid-color second image differ from the animal in my first image? "
                    "Mention both in your answer."
                ),
            }
        ],
    )
    text = _assistant_text(response.messages)
    _assert_no_degradation(text)
    assert len(text) > 30
