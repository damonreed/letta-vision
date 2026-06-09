"""Image derivative utilities (1MP resize, wire-byte sizing)."""

from __future__ import annotations

import io
from typing import Tuple

from letta.log import get_logger
from letta.orm.image import ImageRecord
from letta.schemas.user import User as PydanticUser
from letta.services.image_manager import ImageManager
from letta.services.object_store.client import ObjectStoreClient, get_object_store_client

logger = get_logger(__name__)

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore


def generate_1mp_derivative(raw: bytes, media_type: str = "image/jpeg") -> Tuple[bytes, str, int]:
    """Resize to ~1MP longest edge; return bytes, media_type, wire-byte size."""
    if Image is None:
        raise RuntimeError("Pillow is required for image derivatives")

    img = Image.open(io.BytesIO(raw))
    img = img.convert("RGB") if img.mode not in ("RGB", "L") else img
    w, h = img.size
    target_pixels = 1_000_000
    scale = (target_pixels / (w * h)) ** 0.5
    if scale < 1.0:
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    out_type = "image/jpeg"
    img.save(buf, format="JPEG", quality=85, optimize=True)
    out_bytes = buf.getvalue()
    return out_bytes, out_type, ObjectStoreClient.wire_byte_size(out_bytes)


async def generate_1mp_now(image_id: str, actor: PydanticUser) -> int:
    """Create and persist a 1MP derivative for render-policy use (idempotent)."""
    manager = ImageManager()
    image = await manager.get_by_id_async(image_id, actor)
    if not image:
        raise ValueError(f"Image {image_id} not found")
    if image.object_url_1mp and image.file_size_1mp:
        return image.file_size_1mp

    store = get_object_store_client()
    raw = await store.get_bytes(image.object_url_full)
    onemp_bytes, _, onemp_wire = generate_1mp_derivative(raw, image.media_type)
    onemp_key = await store.put_bytes(image.content_hash, onemp_bytes, suffix="_1mp")

    from letta.server.db import db_registry

    async with db_registry.async_session() as session:
        row = await ImageRecord.read_async(db_session=session, identifier=image_id, actor=actor)
        row.object_url_1mp = onemp_key
        row.file_size_1mp = onemp_wire
        await row.update_async(session, actor=actor)

    logger.info("Generated on-demand 1MP derivative for %s (%s wire bytes)", image_id, onemp_wire)
    return onemp_wire
