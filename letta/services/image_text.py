"""Image text tier read/edit helpers for agent tools."""

from __future__ import annotations

from typing import Literal, Optional, Union

from letta.constants import CORE_MEMORY_LINE_NUMBER_WARNING, MEMORY_TOOLS_LINE_NUMBER_PREFIX_REGEX
from letta.schemas.image import PydanticImage
from letta.schemas.user import User as PydanticUser
from letta.services.image_manager import ImageManager

ImageTextField = Literal["caption", "description", "details"]
IMAGE_TEXT_FIELDS = ("caption", "description", "details")


def normalize_image_handle(handle: str) -> str:
    """Accept bare uuid or image-<uuid> handles from recall."""
    cleaned = (handle or "").strip()
    if not cleaned:
        return cleaned
    if cleaned.startswith("image-"):
        return cleaned
    return f"image-{cleaned}"


def _field_value(image: PydanticImage, field: ImageTextField) -> str:
    return getattr(image, field) or ""


def format_image_text_block(image: PydanticImage) -> str:
    """Human-readable summary of all three text tiers plus file metadata."""
    media_type = image.media_type or "image/jpeg"
    byte_size = image.file_size_full or 0
    lines = [
        f"Caption: {_field_value(image, 'caption') or '(none)'}",
        f"Description: {_field_value(image, 'description') or '(none)'}",
        f"Details: {_field_value(image, 'details') or '(none)'}",
        f"({media_type}, {byte_size} bytes)",
    ]
    return "\n".join(lines)


def _text_fields_dict(image: PydanticImage) -> dict[str, Optional[str]]:
    return {field: getattr(image, field) for field in IMAGE_TEXT_FIELDS}


async def get_image_text(
    handle: str,
    actor: PydanticUser,
    field: Optional[ImageTextField] = None,
) -> Union[str, dict]:
    image_id = normalize_image_handle(handle)
    image = await ImageManager().get_by_id_async(image_id, actor)
    if not image:
        return f"Image {handle} not found."
    if field is None:
        return {"handle": image_id, **_text_fields_dict(image)}
    return _field_value(image, field)


def _reject_line_number_artifacts(text: str, param_name: str) -> None:
    if MEMORY_TOOLS_LINE_NUMBER_PREFIX_REGEX.search(text):
        raise ValueError(
            f"{param_name} contains a line number prefix, which is not allowed. "
            "Do not include line numbers when calling image text tools."
        )
    if CORE_MEMORY_LINE_NUMBER_WARNING in text:
        raise ValueError(
            f"{param_name} contains a line number warning, which is not allowed. "
            "Do not include line number information when calling image text tools."
        )


def _apply_str_replace(current_value: str, old_string: str, new_string: str, field: ImageTextField) -> str:
    _reject_line_number_artifacts(old_string, "old_string")
    _reject_line_number_artifacts(new_string, "new_string")

    old_string = str(old_string).expandtabs()
    new_string = str(new_string).expandtabs()
    current_value = str(current_value).expandtabs()

    occurrences = current_value.count(old_string)
    if occurrences == 0:
        raise ValueError(
            f"No replacement was performed, old_string `{old_string}` did not appear verbatim in image field `{field}`."
        )
    if occurrences > 1:
        lines = [idx + 1 for idx, line in enumerate(current_value.split("\n")) if old_string in line]
        raise ValueError(
            f"No replacement was performed. Multiple occurrences of old_string `{old_string}` in lines {lines}. "
            "Please ensure it is unique."
        )
    return current_value.replace(old_string, new_string)


def _apply_insert(current_value: str, insert_text: str, insert_line: int, field: ImageTextField) -> str:
    _reject_line_number_artifacts(insert_text, "insert_text")

    current_value = str(current_value).expandtabs()
    insert_text = str(insert_text).expandtabs()
    current_value_lines = current_value.split("\n")
    n_lines = len(current_value_lines)

    if insert_line == -1:
        insert_line = n_lines
    elif insert_line < 0 or insert_line > n_lines:
        raise ValueError(
            f"Invalid `insert_line` parameter: {insert_line}. It should be within the range of lines "
            f"of the image field `{field}`: {[0, n_lines]}, or -1 to append to the end."
        )

    insert_text_lines = insert_text.split("\n")
    new_value_lines = current_value_lines[:insert_line] + insert_text_lines + current_value_lines[insert_line:]
    return "\n".join(new_value_lines)


async def _reembed_after_text_edit(image_id: str, actor: PydanticUser) -> None:
    from letta.services.image_ingest import reembed_image_embedding_only

    await reembed_image_embedding_only(image_id, actor)


async def edit_image_text(
    handle: str,
    field: ImageTextField,
    command: Literal["str_replace", "insert", "set"],
    actor: PydanticUser,
    *,
    old_string: Optional[str] = None,
    new_string: Optional[str] = None,
    insert_text: Optional[str] = None,
    insert_line: int = -1,
) -> str:
    if field not in IMAGE_TEXT_FIELDS:
        raise ValueError(f"Invalid field `{field}`. Must be one of: {', '.join(IMAGE_TEXT_FIELDS)}.")

    image_id = normalize_image_handle(handle)
    mgr = ImageManager()
    image = await mgr.get_by_id_async(image_id, actor)
    if not image:
        raise ValueError(f"Image {handle} not found.")

    current_value = _field_value(image, field)

    if command == "str_replace":
        if old_string is None:
            raise ValueError("old_string is required for str_replace command")
        if new_string is None:
            raise ValueError("new_string is required for str_replace command")
        new_value = _apply_str_replace(current_value, old_string, new_string, field)
    elif command == "insert":
        if insert_text is None:
            raise ValueError("insert_text is required for insert command")
        new_value = _apply_insert(current_value, insert_text, insert_line, field)
    elif command == "set":
        if new_string is None:
            raise ValueError("new_string is required for set command")
        _reject_line_number_artifacts(new_string, "new_string")
        new_value = str(new_string).expandtabs()
    else:
        raise ValueError(
            f"Unknown command `{command}`. Supported commands: str_replace, insert, set."
        )

    updated = await mgr.update_text_field_async(image_id, actor, field=field, value=new_value)
    if not updated:
        raise ValueError(f"Image {handle} not found.")

    await _reembed_after_text_edit(image_id, actor)
    return new_value
