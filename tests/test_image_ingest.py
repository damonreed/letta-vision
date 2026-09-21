import base64

import pytest

from letta.schemas.enums import MessageRole
from letta.schemas.letta_message_content import Base64Image, ImageContent, TextContent
from letta.schemas.message import Message, ToolReturn
from letta.schemas.letta_message_content import LettaImage
from letta.services.image_ingest import (
    _apply_generated_text_if_blank,
    _extract_assistant_text,
    _merge_caption_fields,
    _parse_caption_json,
    _probe_image_dimensions,
    _text_is_populated,
    convert_historic_images_in_message,
    ingest_images_in_message,
)


@pytest.mark.asyncio
async def test_ingest_images_in_user_message_replaces_base64(monkeypatch):
    async def fake_ingest(data, media_type, actor, **kwargs):
        assert kwargs.get("provenance") == "uploaded"
        return "image-test-1"

    monkeypatch.setattr("letta.services.image_ingest.ingest_image_sync", fake_ingest)

    msg = Message(
        role=MessageRole.user,
        content=[
            TextContent(text="what is this?"),
            ImageContent(source=Base64Image(media_type="image/png", data=base64.b64encode(b"png").decode())),
        ],
    )
    image_ids = await ingest_images_in_message(msg, actor=None)

    assert image_ids == ["image-test-1"]
    assert msg.content[1].source.type == "letta"
    assert msg.content[1].source.file_id == "image-test-1"
    assert msg.content[1].source.data is None


@pytest.mark.asyncio
async def test_ingest_images_in_tool_return_uses_generated_provenance(monkeypatch):
    async def fake_ingest(data, media_type, actor, **kwargs):
        assert kwargs.get("provenance") == "generated"
        return "image-test-2"

    monkeypatch.setattr("letta.services.image_ingest.ingest_image_sync", fake_ingest)

    msg = Message(
        role=MessageRole.tool,
        content=[TextContent(text="generated image")],
        tool_returns=[
            ToolReturn(
                status="success",
                func_response=[
                    TextContent(text="here is the render"),
                    ImageContent(source=Base64Image(media_type="image/png", data=base64.b64encode(b"png").decode())),
                ],
            )
        ],
    )
    image_ids = await ingest_images_in_message(msg, actor=None)

    assert image_ids == ["image-test-2"]
    func_response = msg.tool_returns[0].func_response
    assert func_response[1].source.type == "letta"
    assert func_response[1].source.file_id == "image-test-2"


@pytest.mark.asyncio
async def test_ingest_images_in_tool_return_ingests_all_images(monkeypatch):
    call_count = 0

    async def fake_ingest(data, media_type, actor, **kwargs):
        nonlocal call_count
        call_count += 1
        return f"image-test-{call_count}"

    monkeypatch.setattr("letta.services.image_ingest.ingest_image_sync", fake_ingest)

    shared_prefix = "A" * 128
    msg = Message(
        role=MessageRole.tool,
        content=[TextContent(text="generated images")],
        tool_returns=[
            ToolReturn(
                status="success",
                func_response=[
                    TextContent(text="two renders"),
                    ImageContent(
                        source=Base64Image(
                            media_type="image/jpeg",
                            data=shared_prefix + "111",
                        )
                    ),
                    ImageContent(
                        source=Base64Image(
                            media_type="image/jpeg",
                            data=shared_prefix + "222",
                        )
                    ),
                ],
            )
        ],
    )
    image_ids = await ingest_images_in_message(msg, actor=None)

    assert image_ids == ["image-test-1", "image-test-2"]
    func_response = msg.tool_returns[0].func_response
    assert func_response[1].source.file_id == "image-test-1"
    assert func_response[2].source.file_id == "image-test-2"


@pytest.mark.asyncio
async def test_ingest_images_skips_fetch_image_tool_returns(monkeypatch):
    ingest_called = False

    async def fake_ingest(data, media_type, actor, **kwargs):
        nonlocal ingest_called
        ingest_called = True
        return "image-should-not-create"

    monkeypatch.setattr("letta.services.image_ingest.ingest_image_sync", fake_ingest)

    msg = Message(
        role=MessageRole.tool,
        name="fetch_image",
        content=[TextContent(text="fetch_image result")],
        tool_returns=[
            ToolReturn(
                status="success",
                func_response=[
                    TextContent(text="summary"),
                    ImageContent(
                        source=Base64Image(media_type="image/png", data=base64.b64encode(b"png").decode(), detail="high")
                    ),
                ],
            )
        ],
    )
    image_ids = await ingest_images_in_message(msg, actor=None)

    assert image_ids == []
    assert ingest_called is False
    func_response = msg.tool_returns[0].func_response
    assert func_response[1].source.type == "base64"
    assert func_response[1].source.data is not None


@pytest.mark.asyncio
async def test_convert_historic_images_converts_letta_inline_data(monkeypatch):
    async def fake_ingest(data, media_type, actor, **kwargs):
        return "image-historic-1"

    monkeypatch.setattr("letta.services.image_ingest.ingest_image_sync", fake_ingest)

    msg = Message(
        role=MessageRole.user,
        content=[
            ImageContent(
                source=LettaImage(
                    file_id="image-old",
                    data=base64.b64encode(b"png").decode(),
                    media_type="image/png",
                )
            )
        ],
    )
    image_ids, changed = await convert_historic_images_in_message(msg, actor=None)

    assert changed is True
    assert image_ids == ["image-historic-1"]
    assert msg.content[0].source.file_id == "image-historic-1"
    assert msg.content[0].source.data is None


@pytest.mark.asyncio
async def test_convert_historic_images_converts_fetch_image_tool_return(monkeypatch):
    async def fake_ingest(data, media_type, actor, **kwargs):
        return "image-historic-2"

    monkeypatch.setattr("letta.services.image_ingest.ingest_image_sync", fake_ingest)

    msg = Message(
        role=MessageRole.tool,
        name="fetch_image",
        content=[TextContent(text="fetch_image result")],
        tool_returns=[
            ToolReturn(
                status="success",
                func_response=[
                    ImageContent(source=Base64Image(media_type="image/png", data=base64.b64encode(b"png").decode())),
                ],
            )
        ],
    )
    image_ids, changed = await convert_historic_images_in_message(msg, actor=None)

    assert changed is True
    assert image_ids == ["image-historic-2"]


@pytest.mark.asyncio
async def test_convert_historic_strips_fetch_image_letta_inline_data_without_reingest(monkeypatch):
    ingest_called = False

    async def fake_ingest(data, media_type, actor, **kwargs):
        nonlocal ingest_called
        ingest_called = True
        return "image-should-not-create"

    monkeypatch.setattr("letta.services.image_ingest.ingest_image_sync", fake_ingest)

    msg = Message(
        role=MessageRole.tool,
        name="fetch_image",
        tool_returns=[
            ToolReturn(
                status="success",
                func_response=[
                    ImageContent(
                        source=LettaImage(
                            file_id="image-existing",
                            data=base64.b64encode(b"png").decode(),
                            media_type="image/png",
                        )
                    ),
                ],
            )
        ],
    )
    image_ids, changed = await convert_historic_images_in_message(msg, actor=None)

    assert changed is True
    assert image_ids == ["image-existing"]
    assert ingest_called is False
    assert msg.tool_returns[0].func_response[0].source.data is None


def test_parse_caption_json_plain_object():
    parsed = _parse_caption_json(
        '{"caption": "Red apple", "description": "A red apple on wood", "details": "Close-up fruit"}'
    )
    assert parsed["caption"] == "Red apple"
    assert parsed["description"] == "A red apple on wood"
    assert parsed["details"] == "Close-up fruit"


def test_parse_caption_json_markdown_fence():
    parsed = _parse_caption_json(
        '```json\n{"caption": "Cat", "description": "Orange tabby cat", "details": "Sitting on sofa"}\n```'
    )
    assert parsed["caption"] == "Cat"
    assert parsed["description"] == "Orange tabby cat"


def test_extract_assistant_text_openai_shape():
    text = _extract_assistant_text({"choices": [{"message": {"content": "OK"}}]})
    assert text == "OK"


def test_extract_assistant_text_minimax_reasoning_split_fallback():
    text = _extract_assistant_text(
        {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "reasoning_content": '{"caption": "Lantern street", "description": "Night market"}',
                    }
                }
            ]
        }
    )
    assert "Lantern street" in text


def test_text_is_populated_rejects_blank():
    assert _text_is_populated("kept caption") is True
    assert _text_is_populated(None) is False
    assert _text_is_populated("") is False
    assert _text_is_populated("   \n") is False


def test_merge_caption_fields_preserves_populated_and_fills_blanks():
    existing = {"caption": "Keep me", "description": None, "details": "  "}
    generated = {"caption": "New cap", "description": "New desc", "details": "New details"}
    merged = _merge_caption_fields(existing, generated)
    assert merged["caption"] == "Keep me"
    assert merged["description"] == "New desc"
    assert merged["details"] == "New details"


def test_merge_caption_fields_all_populated_ignores_generated():
    existing = {"caption": "A", "description": "B", "details": "C"}
    generated = {"caption": "X", "description": "Y", "details": "Z"}
    assert _merge_caption_fields(existing, generated) == existing


def test_apply_generated_text_if_blank_skips_populated_fields():
    from types import SimpleNamespace

    row = SimpleNamespace(caption="User caption", description=None, details="  ")
    generated = {"caption": "VLM cap", "description": "VLM desc", "details": "VLM details"}
    applied = _apply_generated_text_if_blank(row, generated)
    assert row.caption == "User caption"
    assert row.description == "VLM desc"
    assert row.details == "VLM details"
    assert applied == {
        "caption": "User caption",
        "description": "VLM desc",
        "details": "VLM details",
    }


def test_apply_generated_text_if_blank_does_not_touch_fully_populated_row():
    from types import SimpleNamespace

    row = SimpleNamespace(caption="A", description="B", details="C")
    applied = _apply_generated_text_if_blank(row, {"caption": "X", "description": "Y", "details": "Z"})
    assert row.caption == "A"
    assert row.description == "B"
    assert row.details == "C"
    assert applied == {"caption": "A", "description": "B", "details": "C"}


@pytest.mark.asyncio
async def test_resolve_enrichment_captions_skips_vlm_when_all_populated(monkeypatch):
    from types import SimpleNamespace

    from letta.services.image_ingest import _resolve_enrichment_captions

    called = False

    async def fake_generate(*args, **kwargs):
        nonlocal called
        called = True
        return {"caption": "X", "description": "Y", "details": "Z"}

    monkeypatch.setattr("letta.services.image_ingest._generate_three_tier_captions", fake_generate)
    image = SimpleNamespace(
        id="image-1",
        media_type="image/png",
        caption="Existing caption",
        description="Existing description",
        details="Existing details",
    )
    captions = await _resolve_enrichment_captions(image, b"raw", actor=None)
    assert called is False
    assert captions == {
        "caption": "Existing caption",
        "description": "Existing description",
        "details": "Existing details",
    }


@pytest.mark.asyncio
async def test_resolve_enrichment_captions_fills_only_blank_fields(monkeypatch):
    from types import SimpleNamespace

    from letta.services.image_ingest import _resolve_enrichment_captions

    generated = {"caption": "Generated cap", "description": "Generated desc", "details": "Generated details"}

    async def fake_generate(*args, **kwargs):
        return generated

    monkeypatch.setattr("letta.services.image_ingest._generate_three_tier_captions", fake_generate)
    image = SimpleNamespace(
        id="image-2",
        media_type="image/png",
        caption="User caption",
        description=None,
        details="",
    )
    captions = await _resolve_enrichment_captions(image, b"raw", actor=None)
    assert captions == generated


def test_probe_image_dimensions_reads_png_size():
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (2048, 1365), color=(10, 20, 30)).save(buf, format="PNG")
    width, height = _probe_image_dimensions(buf.getvalue())
    assert (width, height) == (2048, 1365)


def test_probe_image_dimensions_undecodable_returns_none():
    assert _probe_image_dimensions(b"not-an-image") == (None, None)


@pytest.mark.asyncio
async def test_ingest_image_sync_records_probed_dimensions(monkeypatch):
    from io import BytesIO

    from PIL import Image

    from letta.orm.image import ImageRecord
    from letta.services.image_ingest import ingest_image_sync

    buf = BytesIO()
    Image.new("RGB", (2496, 1664), color=(1, 2, 3)).save(buf, format="PNG")
    raw = buf.getvalue()
    captured = {}

    class FakeStore:
        def content_hash(self, data):
            return "hash-test"

        def wire_byte_size(self, data):
            return len(data)

        async def put_bytes(self, content_hash, data, suffix=""):
            return f"obj/{content_hash}"

    class FakeManager:
        def new_image_id(self):
            return "image-dim-1"

        async def get_by_hash_async(self, content_hash, actor):
            return None

        async def create_record_async(self, record, actor):
            captured["record"] = record

    monkeypatch.setattr("letta.services.image_ingest.get_object_store_client", lambda: FakeStore())
    monkeypatch.setattr("letta.services.image_ingest.ImageManager", FakeManager)

    class FakeActor:
        organization_id = "org-test"

    image_id = await ingest_image_sync(raw, "image/png", actor=FakeActor(), provenance="generated")
    assert image_id == "image-dim-1"
    record = captured["record"]
    assert isinstance(record, ImageRecord)
    assert record.width == 2496
    assert record.height == 1664
