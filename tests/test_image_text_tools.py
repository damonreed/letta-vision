import pytest

from letta.services.image_text import (
    _apply_insert,
    _apply_str_replace,
    edit_image_text,
    format_image_llm_reference,
    format_image_text_block,
    get_image_text,
)


class _Image:
    def __init__(self):
        self.id = "image-abc"
        self.media_type = "image/png"
        self.file_size_full = 42
        self.caption = "Short"
        self.description = "Search text"
        self.details = "Line one\nLine two"
        self.object_url_full = "images/sha256/test"


@pytest.mark.asyncio
async def test_get_image_text_all_fields(monkeypatch):
    class _Mgr:
        async def get_by_id_async(self, image_id, actor):
            assert image_id == "image-abc"
            return _Image()

    monkeypatch.setattr("letta.services.image_text.ImageManager", lambda: _Mgr())

    result = await get_image_text("abc", actor=None)
    assert result == {
        "handle": "image-abc",
        "caption": "Short",
        "description": "Search text",
        "details": "Line one\nLine two",
    }


@pytest.mark.asyncio
async def test_get_image_text_single_field(monkeypatch):
    class _Mgr:
        async def get_by_id_async(self, image_id, actor):
            return _Image()

    monkeypatch.setattr("letta.services.image_text.ImageManager", lambda: _Mgr())

    result = await get_image_text("image-abc", actor=None, field="description")
    assert result == "Search text"


def test_format_image_llm_reference_omits_empty_tiers():
    text = format_image_llm_reference("abc", caption=None, description="  ")
    assert text == "Image ID: image-abc (image_fetch, image_get_text, image_edit_text)"

    text = format_image_llm_reference(
        "image-xyz",
        caption="Short",
        description="Longer search text",
    )
    assert "Image ID: image-xyz" in text
    assert "Caption: Short" in text
    assert "Description: Longer search text" in text


def test_format_image_text_block_includes_all_tiers():
    text = format_image_text_block(_Image())
    assert "Caption: Short" in text
    assert "Description: Search text" in text
    assert "Details: Line one" in text
    assert "42 bytes" in text


def test_apply_str_replace_unique_match():
    assert _apply_str_replace("alpha beta", "beta", "gamma", "caption") == "alpha gamma"


def test_apply_str_replace_requires_unique_match():
    with pytest.raises(ValueError, match="Multiple occurrences"):
        _apply_str_replace("aa", "a", "b", "caption")


def test_apply_insert_appends_by_default():
    assert _apply_insert("line one", "line two", -1, "details") == "line one\nline two"


@pytest.mark.asyncio
async def test_edit_image_text_reembeds(monkeypatch):
    image = _Image()
    reembedded = []

    class _Mgr:
        async def get_by_id_async(self, image_id, actor):
            return image

        async def update_text_fields_async(self, image_id, actor, *, fields):
            for name, value in fields.items():
                setattr(image, name, value)
            return image

    async def _reembed(image_id, actor):
        reembedded.append(image_id)

    monkeypatch.setattr("letta.services.image_text.ImageManager", lambda: _Mgr())
    monkeypatch.setattr("letta.services.image_text._reembed_after_text_edit", _reembed)

    result = await edit_image_text(
        "abc",
        "description",
        "str_replace",
        actor=None,
        old_string="Search",
        new_string="Updated search",
    )
    assert result == "Updated search text"
    assert image.description == "Updated search text"
    assert reembedded == ["image-abc"]


@pytest.mark.asyncio
async def test_edit_image_text_multiple_fields_reembeds_once(monkeypatch):
    image = _Image()
    updates = []
    reembedded = []

    class _Mgr:
        async def get_by_id_async(self, image_id, actor):
            return image

        async def update_text_fields_async(self, image_id, actor, *, fields):
            updates.append(dict(fields))
            for name, value in fields.items():
                setattr(image, name, value)
            return image

    async def _reembed(image_id, actor):
        reembedded.append(image_id)

    monkeypatch.setattr("letta.services.image_text.ImageManager", lambda: _Mgr())
    monkeypatch.setattr("letta.services.image_text._reembed_after_text_edit", _reembed)

    result = await edit_image_text(
        "image-abc",
        actor=None,
        edits=[
            {"field": "caption", "command": "set", "new_string": "New caption"},
            {"field": "description", "command": "str_replace", "old_string": "Search", "new_string": "Index"},
            {"field": "details", "command": "insert", "insert_text": "Line three"},
        ],
    )
    assert result == {
        "handle": "image-abc",
        "fields": {
            "caption": "New caption",
            "description": "Index text",
            "details": "Line one\nLine two\nLine three",
        },
    }
    assert len(updates) == 1
    assert updates[0]["caption"] == "New caption"
    assert updates[0]["description"] == "Index text"
    assert updates[0]["details"] == "Line one\nLine two\nLine three"
    assert reembedded == ["image-abc"]
    assert image.caption == "New caption"


@pytest.mark.asyncio
async def test_edit_image_text_same_field_edits_compose(monkeypatch):
    image = _Image()

    class _Mgr:
        async def get_by_id_async(self, image_id, actor):
            return image

        async def update_text_fields_async(self, image_id, actor, *, fields):
            for name, value in fields.items():
                setattr(image, name, value)
            return image

    async def _reembed(image_id, actor):
        return None

    monkeypatch.setattr("letta.services.image_text.ImageManager", lambda: _Mgr())
    monkeypatch.setattr("letta.services.image_text._reembed_after_text_edit", _reembed)

    result = await edit_image_text(
        "abc",
        actor=None,
        edits=[
            {"field": "details", "command": "str_replace", "old_string": "Line two", "new_string": "Line 2"},
            {"field": "details", "command": "insert", "insert_text": "Line 3", "insert_line": -1},
        ],
    )
    assert result["fields"] == {"details": "Line one\nLine 2\nLine 3"}


@pytest.mark.asyncio
async def test_edit_image_text_does_not_write_when_a_later_edit_fails(monkeypatch):
    image = _Image()
    updates = []

    class _Mgr:
        async def get_by_id_async(self, image_id, actor):
            return image

        async def update_text_fields_async(self, image_id, actor, *, fields):
            updates.append(fields)
            return image

    monkeypatch.setattr("letta.services.image_text.ImageManager", lambda: _Mgr())

    with pytest.raises(ValueError, match="Failed to edit `description`"):
        await edit_image_text(
            "image-abc",
            actor=None,
            edits=[
                {"field": "caption", "command": "set", "new_string": "ok"},
                {"field": "description", "command": "str_replace", "old_string": "missing", "new_string": "x"},
            ],
        )
    assert updates == []
    assert image.caption == "Short"


@pytest.mark.asyncio
async def test_edit_image_text_rejects_mixed_single_and_batch_args():
    with pytest.raises(ValueError, match="Do not pass both"):
        await edit_image_text(
            "image-abc",
            "caption",
            "set",
            actor=None,
            new_string="x",
            edits=[{"field": "description", "command": "set", "new_string": "y"}],
        )


def test_image_edit_text_schema_exposes_edits():
    from letta.functions.function_sets.base import image_edit_text as image_edit_text_tool
    from letta.functions.schema_generator import generate_schema

    schema = generate_schema(image_edit_text_tool)
    params = schema["parameters"]
    assert "field" not in params["required"]
    assert "command" not in params["required"]
    assert params["required"] == ["handle"]
    edits = params["properties"]["edits"]
    assert edits["type"] == "array"
    item = edits["items"]
    assert item["properties"]["field"]["enum"] == ["caption", "description", "details"]
    assert item["properties"]["command"]["enum"] == ["str_replace", "insert", "set"]
    assert set(item["required"]) == {"field", "command"}
