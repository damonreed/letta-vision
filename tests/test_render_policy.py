from letta.schemas.enums import MessageRole
from letta.schemas.letta_message_content import ImageContent, LettaImage, TextContent
from letta.schemas.llm_config import LLMConfig
from letta.schemas.message import Message
from letta.services.vision.render_policy import (
    RenderTier,
    compute_image_render_decisions,
    find_image_needing_1mp_now,
)
from letta.settings import settings


def _llm_config() -> LLMConfig:
    return LLMConfig(
        model="openai/gpt-4o",
        model_endpoint_type="openai",
        model_endpoint="https://api.openai.com/v1",
        context_window=128000,
        handle="openai/gpt-4o",
    )


def _user_image_message(image_id: str) -> Message:
    return Message(
        role=MessageRole.user,
        content=[
            TextContent(text="what is this?"),
            ImageContent(source=LettaImage(file_id=image_id, media_type="image/png")),
        ],
    )


def test_find_image_needing_1mp_now_when_full_exceeds_budget():
    cap = settings.vision_context_byte_cap
    messages = [_user_image_message("img-big")]
    meta = {
        "img-big": {
            "file_size_full": cap + 1,
            "file_size_1mp": None,
            "object_url_1mp": None,
        }
    }
    assert find_image_needing_1mp_now(messages, _llm_config(), image_metadata=meta) == "img-big"


def test_find_image_needing_1mp_now_returns_none_when_1mp_baked():
    cap = settings.vision_context_byte_cap
    messages = [_user_image_message("img-big")]
    meta = {
        "img-big": {
            "file_size_full": cap + 1,
            "file_size_1mp": 500_000,
            "object_url_1mp": "sha256/abc_1mp",
        }
    }
    assert find_image_needing_1mp_now(messages, _llm_config(), image_metadata=meta) is None


def test_compute_render_decisions_one_mp_when_full_too_large():
    cap = settings.vision_context_byte_cap
    messages = [_user_image_message("img-big")]
    meta = {
        "img-big": {
            "file_size_full": cap + 1,
            "file_size_1mp": 500_000,
            "object_url_1mp": "sha256/abc_1mp",
        }
    }
    decisions = compute_image_render_decisions(messages, _llm_config(), image_metadata=meta)
    assert decisions["img-big"] == RenderTier.ONE_MP


def test_compute_render_decisions_text_when_1mp_missing():
    cap = settings.vision_context_byte_cap
    messages = [_user_image_message("img-big")]
    meta = {
        "img-big": {
            "file_size_full": cap + 1,
            "file_size_1mp": None,
            "object_url_1mp": None,
        }
    }
    decisions = compute_image_render_decisions(messages, _llm_config(), image_metadata=meta)
    assert decisions["img-big"] == RenderTier.TEXT


def test_compute_render_decisions_tool_return_since_last_user_gets_full():
    img = "img-gen"
    messages = [
        Message(role=MessageRole.user, content=[TextContent(text="generate a scene")]),
        _tool_return_generate_image_message(img),
    ]
    meta = {
        img: {
            "file_size_full": 300_000,
            "file_size_1mp": None,
            "object_url_full": "sha256/gen",
        }
    }
    decisions = compute_image_render_decisions(messages, _llm_config(), image_metadata=meta)
    assert decisions[img] == RenderTier.FULL


def _tool_return_generate_image_message(image_id: str) -> Message:
    from letta.schemas.message import ToolReturn

    return Message(
        role=MessageRole.tool,
        tool_returns=[
            ToolReturn(
                tool_call_id="functions.generate_image:1",
                status="success",
                func_response=[
                    TextContent(text='{"status":"ok"}'),
                    ImageContent(source=LettaImage(file_id=image_id, media_type="image/png")),
                ],
            )
        ],
    )


def _kimi_config() -> LLMConfig:
    return LLMConfig(
        model="moonshotai/kimi-k2.6",
        model_endpoint_type="openai",
        model_endpoint="https://openrouter.ai/api/v1",
        context_window=256000,
        handle="openrouter/moonshotai/kimi-k2.6",
    )


def _small_meta(image_ids: list[str]) -> dict:
    return {
        img: {
            "file_size_full": 200_000,
            "file_size_1mp": 150_000,
            "object_url_full": f"sha256/{img}",
            "object_url_1mp": f"sha256/{img}_1mp",
        }
        for img in image_ids
    }


def test_model_max_image_parts_registry():
    from letta.llm_api.model_registry import model_max_image_parts

    assert model_max_image_parts("moonshotai/kimi-k2.6", handle="openrouter/moonshotai/kimi-k2.6") == 8
    assert model_max_image_parts("kimi-k2.5") == 8
    assert model_max_image_parts("openai/gpt-4o", handle="openai/gpt-4o") is None


def test_compute_render_decisions_part_count_cap_demotes_oldest():
    """kimi providers silently drop image parts after the first 8; newest must win."""
    image_ids = [f"img-{i}" for i in range(1, 11)]  # img-1 oldest .. img-10 newest
    messages = [Message(role=MessageRole.user, content=[TextContent(text="go")])]
    for img in image_ids:
        messages.append(_tool_return_generate_image_message(img))

    decisions = compute_image_render_decisions(messages, _kimi_config(), image_metadata=_small_meta(image_ids))

    for img in image_ids[2:]:  # newest 8
        assert decisions[img] != RenderTier.TEXT, img
    for img in image_ids[:2]:  # oldest 2
        assert decisions[img] == RenderTier.TEXT, img


def test_compute_render_decisions_part_count_cap_counts_duplicate_parts():
    """An image referenced from two tool returns consumes two parts of the cap."""
    image_ids = [f"img-{i}" for i in range(1, 10)]  # 9 ids; img-9 newest appears twice
    messages = [Message(role=MessageRole.user, content=[TextContent(text="go")])]
    for img in image_ids:
        messages.append(_tool_return_generate_image_message(img))
    messages.append(_tool_return_generate_image_message("img-9"))  # e.g. image_fetch of same id

    decisions = compute_image_render_decisions(messages, _kimi_config(), image_metadata=_small_meta(image_ids))

    # 10 parts total; img-9 takes 2, img-8..img-3 take 6 -> img-2 and img-1 demote
    assert decisions["img-9"] != RenderTier.TEXT
    for img in [f"img-{i}" for i in range(3, 9)]:
        assert decisions[img] != RenderTier.TEXT, img
    assert decisions["img-2"] == RenderTier.TEXT
    assert decisions["img-1"] == RenderTier.TEXT


def test_compute_render_decisions_no_part_cap_for_uncapped_models():
    image_ids = [f"img-{i}" for i in range(1, 13)]
    messages = [Message(role=MessageRole.user, content=[TextContent(text="go")])]
    for img in image_ids:
        messages.append(_tool_return_generate_image_message(img))

    decisions = compute_image_render_decisions(messages, _llm_config(), image_metadata=_small_meta(image_ids))
    assert all(tier != RenderTier.TEXT for tier in decisions.values())


def test_find_image_needing_1mp_now_respects_part_count_cap():
    """No 1MP bake for images that the count cap will demote to text anyway."""
    cap = settings.vision_context_byte_cap
    image_ids = [f"img-{i}" for i in range(1, 10)]  # 9 ids, img-9 newest
    messages = [Message(role=MessageRole.user, content=[TextContent(text="go")])]
    for img in image_ids:
        messages.append(_tool_return_generate_image_message(img))

    # Oldest image is current-turn-sized over budget but count-capped out -> no bake.
    meta = _small_meta(image_ids)
    meta["img-1"] = {"file_size_full": cap + 1, "file_size_1mp": None, "object_url_1mp": None}
    assert find_image_needing_1mp_now(messages, _kimi_config(), image_metadata=meta) is None


def test_compute_render_decisions_tool_return_image_in_walk_on_later_turn():
    img = "img-gen-persist"
    meta = {
        img: {
            "file_size_full": 300_000,
            "file_size_1mp": 200_000,
            "object_url_1mp": "sha256/gen-persist_1mp",
            "object_url_full": "sha256/gen-persist",
        }
    }
    messages = [
        Message(role=MessageRole.user, content=[TextContent(text="generate scene 1")]),
        _tool_return_generate_image_message(img),
        Message(role=MessageRole.user, content=[TextContent(text="what do you see in that image?")]),
        Message(role=MessageRole.assistant, content=[TextContent(text="working on it")]),
    ]
    decisions = compute_image_render_decisions(messages, _llm_config(), image_metadata=meta)
    assert img in decisions
    assert decisions[img] == RenderTier.ONE_MP
