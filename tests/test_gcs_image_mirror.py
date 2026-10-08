import pytest

from letta.services.object_store import gcs_image_mirror as gim


class _FakeMirror:
    def __init__(self):
        self.puts = []
        self._exists = set()

    def gs_uri(self, image_id: str) -> str:
        return f"gs://bucket/{image_id}"

    async def put_image(self, image_id: str, data: bytes, content_type: str) -> str:
        self.puts.append((image_id, data, content_type))
        self._exists.add(image_id)
        return self.gs_uri(image_id)

    async def exists(self, image_id: str) -> bool:
        return image_id in self._exists

    async def signed_get_url(self, image_id: str) -> str:
        return f"https://signed.example/{image_id}"


@pytest.mark.asyncio
async def test_ensure_mirrored_copies_from_minio_when_missing(monkeypatch):
    mirror = _FakeMirror()
    store_calls = []

    class _Store:
        async def get_bytes(self, key: str) -> bytes:
            store_calls.append(key)
            return b"png-bytes"

    monkeypatch.setattr(gim, "get_gcs_image_mirror", lambda: mirror)
    monkeypatch.setattr(
        "letta.services.object_store.client.get_object_store_client",
        lambda: _Store(),
    )

    url = await gim.ensure_mirrored_and_sign(
        "image-abc",
        minio_key="sha256/deadbeef",
        content_type="image/png",
    )

    assert url == "https://signed.example/image-abc"
    assert store_calls == ["sha256/deadbeef"]
    assert mirror.puts == [("image-abc", b"png-bytes", "image/png")]

    # Second call should not re-fetch MinIO.
    url2 = await gim.ensure_mirrored_and_sign(
        "image-abc",
        minio_key="sha256/deadbeef",
        content_type="image/png",
    )
    assert url2 == url
    assert store_calls == ["sha256/deadbeef"]


@pytest.mark.asyncio
async def test_mirror_image_bytes_noop_when_unconfigured(monkeypatch):
    monkeypatch.setattr(gim, "get_gcs_image_mirror", lambda: None)
    assert await gim.mirror_image_bytes("image-x", b"x", "image/png") is None
