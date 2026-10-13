"""Mount object-store images into a run_code sandbox by handle.

The sandbox has no database and no object store. The agent passes a handle by
writing it in the code. Before the code runs, the server loads those bytes and
writes them at ``/images/<handle>``. Pillow opens that path without an extension.
"""

from __future__ import annotations

import re
from typing import Optional

from letta.schemas.user import User
from letta.services.image_manager import ImageManager
from letta.services.object_store.client import get_object_store_client

# image-<uuid>, the form stored on image records and printed in tool results.
_HANDLE_RE = re.compile(
    r"image-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
# Optional suffix when the code opens /images/<handle>.png (or similar).
_EXPLICIT_PATH_RE = re.compile(
    r"/images/(image-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})"
    r"(\.[A-Za-z0-9]+)?"
)

SANDBOX_IMAGE_DIR = "/images"


def sandbox_path(handle: str) -> str:
    return f"{SANDBOX_IMAGE_DIR}/{handle}"


def image_handles_in_code(code: str) -> list[str]:
    """Unique handles in first-seen order."""
    seen: set[str] = set()
    handles: list[str] = []
    for match in _HANDLE_RE.finditer(code or ""):
        handle = match.group(0)
        if handle not in seen:
            seen.add(handle)
            handles.append(handle)
    return handles


def sandbox_write_paths(code: str) -> list[str]:
    """Paths to write for handles named in ``code``.

    Every handle is written at ``/images/<handle>``. A literal
    ``/images/<handle>.<ext>`` is written at that exact path as well, so
    ``Image.open`` with an extension still finds the file.
    """
    handles = image_handles_in_code(code)
    if not handles:
        return []
    paths: list[str] = []
    seen: set[str] = set()
    for handle in handles:
        path = sandbox_path(handle)
        seen.add(path)
        paths.append(path)
    for match in _EXPLICIT_PATH_RE.finditer(code or ""):
        path = match.group(0)
        if path not in seen:
            seen.add(path)
            paths.append(path)
    return paths


async def load_sandbox_image_files(code: str, actor: Optional[User]) -> list[tuple[str, bytes]]:
    """Load bytes for every image handle named in ``code``.

    Returns ``(sandbox_path, bytes)`` pairs. Empty when the code names no handle.
    """
    paths = sandbox_write_paths(code)
    if not paths:
        return []
    if actor is None:
        raise ValueError("Cannot load image handles for run_code without an actor")

    mgr = ImageManager()
    records: dict[str, str] = {}
    ordered_handles: list[str] = []
    for path in paths:
        handle = _HANDLE_RE.search(path)
        if handle is None:
            continue
        handle_id = handle.group(0)
        if handle_id in records:
            continue
        image = await mgr.get_by_id_async(handle_id, actor)
        if image is None:
            raise ValueError(f"Image not found: {handle_id}")
        if not image.object_url_full:
            raise ValueError(f"Image {handle_id} has no stored bytes")
        records[handle_id] = image.object_url_full
        ordered_handles.append(handle_id)

    store = get_object_store_client()
    cache: dict[str, bytes] = {}
    for handle_id in ordered_handles:
        cache[handle_id] = await store.get_bytes(records[handle_id])

    files: list[tuple[str, bytes]] = []
    for path in paths:
        handle = _HANDLE_RE.search(path)
        if handle is None:
            continue
        files.append((path, cache[handle.group(0)]))
    return files
