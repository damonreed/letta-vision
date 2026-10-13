import pytest

from letta.services.tool_executor.sandbox_image_files import (
    image_handles_in_code,
    load_sandbox_image_files,
    sandbox_write_paths,
)

AFTER = "image-8faad61f-fbe3-4702-9877-0a8546e43ccf"
BEFORE = "image-4c992962-d45a-4856-8afd-dc37018cb36b"
PNG = b"\x89PNG\r\n\x1a\n"


def test_no_handle_mounts_nothing():
    code = "print('hello')\nfrom PIL import Image\nImage.new('RGB', (8, 8))"
    assert image_handles_in_code(code) == []
    assert sandbox_write_paths(code) == []


def test_handle_mounts_at_images_path():
    code = f"""
from PIL import Image
after = Image.open("/images/{AFTER}")
before = Image.open("/images/{BEFORE}")
"""
    assert image_handles_in_code(code) == [AFTER, BEFORE]
    assert sandbox_write_paths(code) == [f"/images/{AFTER}", f"/images/{BEFORE}"]


def test_extension_path_is_written_too():
    code = f'Image.open("/images/{AFTER}.png")'
    assert sandbox_write_paths(code) == [f"/images/{AFTER}", f"/images/{AFTER}.png"]


def test_handle_mentioned_once_is_deduped():
    code = f'a = "{AFTER}"\nb = "{AFTER}"\n'
    assert image_handles_in_code(code) == [AFTER]
    assert sandbox_write_paths(code) == [f"/images/{AFTER}"]


@pytest.mark.asyncio
async def test_load_fetches_each_handle_once(monkeypatch):
    fetches = []

    class _Image:
        object_url_full = "sha256/after"

    class _Mgr:
        async def get_by_id_async(self, image_id, actor):
            assert image_id == AFTER
            return _Image()

    class _Store:
        async def get_bytes(self, key):
            fetches.append(key)
            return PNG

    monkeypatch.setattr("letta.services.tool_executor.sandbox_image_files.ImageManager", lambda: _Mgr())
    monkeypatch.setattr(
        "letta.services.tool_executor.sandbox_image_files.get_object_store_client",
        lambda: _Store(),
    )

    code = f'Image.open("/images/{AFTER}.png")\nprint("{AFTER}")'
    files = await load_sandbox_image_files(code, actor=object())

    assert fetches == ["sha256/after"]
    assert files == [(f"/images/{AFTER}", PNG), (f"/images/{AFTER}.png", PNG)]


@pytest.mark.asyncio
async def test_missing_image_raises(monkeypatch):
    class _Mgr:
        async def get_by_id_async(self, image_id, actor):
            return None

    monkeypatch.setattr("letta.services.tool_executor.sandbox_image_files.ImageManager", lambda: _Mgr())

    with pytest.raises(ValueError, match=AFTER):
        await load_sandbox_image_files(f'open("/images/{AFTER}")', actor=object())


@pytest.mark.asyncio
async def test_handles_require_actor():
    with pytest.raises(ValueError, match="actor"):
        await load_sandbox_image_files(f'open("/images/{AFTER}")', actor=None)


@pytest.mark.asyncio
async def test_no_handles_skips_actor():
    assert await load_sandbox_image_files("print(1)", actor=None) == []
