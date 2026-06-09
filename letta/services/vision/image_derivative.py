"""Image derivative utilities (1MP resize, wire-byte sizing)."""

from __future__ import annotations

import io
from typing import Tuple

from letta.services.object_store.client import ObjectStoreClient

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
