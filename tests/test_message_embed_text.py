import json

import pytest

from letta.embeddings.message_embed_text import (
    collect_letta_image_ids_from_message,
    inject_image_caption_gists,
)
from letta.schemas.enums import MessageRole
from letta.schemas.letta_message_content import ImageContent, LettaImage, TextContent
from letta.schemas.message import Message


def test_collect_letta_image_ids_dedupes_and_preserves_order():
    msg = Message(
        role=MessageRole.user,
        content=[
            ImageContent(source=LettaImage(file_id="image-aaa", media_type="image/png")),
            ImageContent(source=LettaImage(file_id="image-bbb", media_type="image/png")),
            ImageContent(source=LettaImage(file_id="image-aaa", media_type="image/png")),
        ],
    )
    assert collect_letta_image_ids_from_message(msg) == ["image-aaa", "image-bbb"]


def test_inject_image_caption_gists_into_user_json():
    base = json.dumps({"content": "Describe this."})
    out = inject_image_caption_gists(base, ["Person in dark turtleneck before monitors."])
    parsed = json.loads(out)
    assert parsed["content"] == "Describe this."
    assert parsed["image_captions"] == ["Person in dark turtleneck before monitors."]


def test_inject_image_caption_gists_empty_base():
    out = inject_image_caption_gists("", ["Only an image caption."])
    assert json.loads(out) == {"image_captions": ["Only an image caption."]}


@pytest.mark.asyncio
async def test_build_message_embed_text_v1_skips_captions(monkeypatch):
    from letta.embeddings.message_embed_text import build_message_embed_text

    async def _fake_load(image_ids, actor):
        return ["should not appear"]

    monkeypatch.setattr(
        "letta.embeddings.message_embed_text.load_caption_gists_for_images",
        _fake_load,
    )

    msg = Message(
        role=MessageRole.user,
        content=[
            TextContent(text="hello"),
            ImageContent(source=LettaImage(file_id="image-abc", media_type="image/png")),
        ],
    )
    text = await build_message_embed_text(msg, actor=None, include_image_captions=False)
    parsed = json.loads(text)
    assert parsed["content"] == "hello"
    assert "image_captions" not in parsed


@pytest.mark.asyncio
async def test_build_message_embed_text_v2_includes_captions(monkeypatch):
    from letta.embeddings.message_embed_text import build_message_embed_text
    from letta.services.message_manager import MessageManager

    async def _fake_load(image_ids, actor):
        assert image_ids == ["image-abc"]
        return ["Short caption gist."]

    monkeypatch.setattr(
        "letta.embeddings.message_embed_text.load_caption_gists_for_images",
        _fake_load,
    )

    msg = Message(
        role=MessageRole.user,
        content=[
            TextContent(text="hello"),
            ImageContent(source=LettaImage(file_id="image-abc", media_type="image/png")),
        ],
    )
    mgr = MessageManager()
    text = await build_message_embed_text(
        msg,
        actor=None,
        include_image_captions=True,
        base_extractor=mgr._extract_message_text,
    )
    parsed = json.loads(text)
    assert parsed["content"] == "hello"
    assert parsed["image_captions"] == ["Short caption gist."]
