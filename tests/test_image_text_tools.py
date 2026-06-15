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

        async def update_text_field_async(self, image_id, actor, *, field, value):
            setattr(image, field, value)
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
