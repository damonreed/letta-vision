"""Image ingest: sync store + background enrichment."""

from __future__ import annotations

import base64
import json
import re
from datetime import datetime, timezone
from typing import List, Literal, Optional, Union

from letta.embeddings.resolver import resolve_embedding_config_async
from letta.embeddings.util import prepare_vector_for_write
from letta.llm_api.llm_client import LLMClient
from letta.log import get_logger
from letta.orm.image import ImageRecord
from letta.schemas.enums import AgentType, MessageRole
from letta.schemas.letta_message_content import (
    Base64Image,
    ImageContent,
    ImageSourceType,
    LettaImage,
    MessageContentType,
    TextContent,
)
from letta.schemas.llm_config import LLMConfig
from letta.schemas.message import Message as PydanticMessage
from letta.schemas.user import User as PydanticUser
from letta.services.image_manager import ImageManager
from letta.services.object_store.client import get_object_store_client
from letta.services.vision.image_derivative import generate_1mp_derivative
from letta.settings import settings
from letta.utils import fire_and_forget

logger = get_logger(__name__)

ImageProvenance = Literal["uploaded", "generated"]

_CAPTION_FALLBACK = {
    "caption": "Image attached to conversation.",
    "description": "An image was shared in this conversation.",
    "details": "Image content available via fetch_image.",
}

_CAPTION_PROMPT = (
    "Describe this image. Respond with a single JSON object only (no markdown fences) "
    "with exactly these keys:\n"
    '- "caption": 20-50 words, concise label.\n'
    '- "description": 100-200 words, literal content nouns for search indexing.\n'
    '- "details": 1500-2000 words, thorough literal description structured as a prompt-ready '
    "image-generation spec.\n"
    "\n"
    'Inside the "details" string value only, include these exact section headings in order:\n'
    "MEDIUM_AND_STYLE: rendering medium and style (oil painting, digital painting, photorealistic, "
    "screenshot, etc.); brushwork, surface quality, photographic characteristics; painterly, "
    "naturalistic, cinematic, glamour, or stylized.\n"
    "ASPECT_RATIO_AND_FRAMING: aspect ratio, camera angle, shot distance, orientation, subject placement.\n"
    "SUBJECT: every person or creature — apparent age, pose, posture, gesture, expression, gaze, body "
    "orientation, proportions, clothing, accessories, hair, skin texture, hands, feet, held or worn items; "
    "exact colors and materials; age, wear, asymmetry, awkwardness, imperfection.\n"
    "LIGHTING_AND_COLOR: light direction and quality, time of day, shadows, highlights, palette, contrast; "
    "painterly, natural, cinematic, glamour, or high-contrast.\n"
    "ENVIRONMENT_AND_PROPS: objects, architecture, plants, furniture, background; placement, materials, colors.\n"
    "SPATIAL_LAYOUT: spatial relations (left/right, foreground/background, above/below, near/far); enough "
    "geometry to reconstruct the scene.\n"
    "TEXT_AND_SYMBOLS: visible text, symbols, logos, UI, watermarks, emblems; reproduce text exactly.\n"
    "PRESERVE: unusual, awkward, asymmetrical, worn, handmade, aged, or non-standard choices a model might "
    "normalize away; what must NOT change; anti-normalization (no de-aging, skin smoothing, idealized "
    "proportions, added glamour/cinematic styling, converting worn/handmade items into new/fashion versions).\n"
    "\n"
    "Base every field on pixels only, not assumptions. Do not infer story, identity, emotion, or off-screen "
    "content. Describe exactly what is visible. The details field must be prompt-ready and preserve the "
    "image's specific imperfections and choices."
)


def _fallback_captions() -> dict:
    return dict(_CAPTION_FALLBACK)


def _extract_assistant_text(response: dict) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    text = ""
    if isinstance(content, str):
        text = content.strip()
    elif isinstance(content, list):
        text = "".join(
            part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"
        ).strip()
    if text:
        return text
    from letta.llm_api.minimax_openai import extract_reasoning_from_message_data

    reasoning = extract_reasoning_from_message_data(msg)
    return reasoning.strip() if reasoning else ""


def _parse_caption_json(text: str) -> dict:
    if not text:
        raise ValueError("empty caption response")

    stripped = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", stripped)
    if fence:
        stripped = fence.group(1).strip()

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        stripped = stripped[start : end + 1]

    parsed = json.loads(stripped)
    if not isinstance(parsed, dict):
        raise ValueError("caption JSON is not an object")

    return {
        "caption": (parsed.get("caption") or "").strip() or None,
        "description": (parsed.get("description") or "").strip() or None,
        "details": (parsed.get("details") or "").strip() or None,
    }


async def ingest_image_sync(
    data: bytes,
    media_type: str,
    actor: PydanticUser,
    *,
    provenance: ImageProvenance = "uploaded",
    generation_prompt: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> str:
    """Store image bytes and return image id (dedup by content hash).

    Enrichment is scheduled separately via ``link_image_to_message`` once the
    owning message id is known (FR §9 two-embed dance).
    """
    store = get_object_store_client()
    content_hash = store.content_hash(data)
    manager = ImageManager()

    existing = await manager.get_by_hash_async(content_hash, actor)
    if existing:
        return existing.id

    key = await store.put_bytes(content_hash, data)
    image_id = manager.new_image_id()
    record = ImageRecord(
        id=image_id,
        organization_id=actor.organization_id,
        content_hash=content_hash,
        object_url_full=key,
        media_type=media_type,
        width=width,
        height=height,
        file_size_full=store.wire_byte_size(data),
        provenance=provenance,
        generation_prompt=generation_prompt,
        enrichment_status="pending",
        enrichment_attempts=0,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        is_deleted=False,
    )
    await manager.create_record_async(record, actor)
    return image_id


def _provenance_for_message(message: PydanticMessage) -> ImageProvenance:
    return "uploaded" if message.role == MessageRole.user else "generated"


async def _ingest_base64_source(
    *,
    data: str,
    media_type: str,
    detail: Optional[str],
    actor: PydanticUser,
    provenance: ImageProvenance,
    generation_prompt: Optional[str] = None,
) -> LettaImage:
    raw = base64.standard_b64decode(data)
    image_id = await ingest_image_sync(
        raw,
        media_type or "image/png",
        actor,
        provenance=provenance,
        generation_prompt=generation_prompt,
    )
    return LettaImage(
        file_id=image_id,
        data=None,
        media_type=media_type or "image/png",
        detail=detail,
    )


async def _ingest_image_block(
    block: Union[ImageContent, dict],
    actor: PydanticUser,
    *,
    provenance: ImageProvenance,
    generation_prompt: Optional[str] = None,
) -> tuple[Optional[Union[ImageContent, dict]], Optional[str]]:
    """Ingest inline base64 image blocks; return updated block and image id."""
    if isinstance(block, ImageContent):
        source = block.source
        if source.type == ImageSourceType.letta:
            return block, getattr(source, "file_id", None)
        if source.type != ImageSourceType.base64:
            return block, None
        letta_source = await _ingest_base64_source(
            data=source.data,
            media_type=source.media_type,
            detail=getattr(source, "detail", None),
            actor=actor,
            provenance=provenance,
            generation_prompt=generation_prompt,
        )
        return ImageContent(source=letta_source), letta_source.file_id

    if not isinstance(block, dict) or block.get("type") != "image":
        return block, None

    source = block.get("source") or {}
    if source.get("type") == "letta" and source.get("file_id"):
        return block, source["file_id"]
    if source.get("type") != "base64" or not source.get("data"):
        return block, None

    letta_source = await _ingest_base64_source(
        data=source["data"],
        media_type=source.get("media_type") or "image/png",
        detail=source.get("detail"),
        actor=actor,
        provenance=provenance,
        generation_prompt=generation_prompt,
    )
    return (
        {
            "type": "image",
            "source": {
                "type": "letta",
                "file_id": letta_source.file_id,
                "media_type": letta_source.media_type,
                "data": None,
                "detail": letta_source.detail,
            },
        },
        letta_source.file_id,
    )


async def _ingest_migration_image_block(
    block: Union[ImageContent, dict],
    actor: PydanticUser,
    *,
    provenance: ImageProvenance,
    generation_prompt: Optional[str] = None,
) -> tuple[Optional[Union[ImageContent, dict]], Optional[str]]:
    """Convert historic inline bytes (base64 or letta+data) to LettaImage refs."""
    if isinstance(block, ImageContent):
        source = block.source
        if source.type == ImageSourceType.letta:
            inline_data = getattr(source, "data", None)
            existing_id = getattr(source, "file_id", None)
            if inline_data and existing_id:
                return (
                    ImageContent(
                        source=LettaImage(
                            file_id=existing_id,
                            data=None,
                            media_type=getattr(source, "media_type", None) or "image/png",
                            detail=getattr(source, "detail", None),
                        )
                    ),
                    existing_id,
                )
            if inline_data:
                letta_source = await _ingest_base64_source(
                    data=inline_data,
                    media_type=getattr(source, "media_type", None) or "image/png",
                    detail=getattr(source, "detail", None),
                    actor=actor,
                    provenance=provenance,
                    generation_prompt=generation_prompt,
                )
                return ImageContent(source=letta_source), letta_source.file_id
            return block, existing_id
    elif isinstance(block, dict) and block.get("type") == "image":
        source = block.get("source") or {}
        if source.get("type") == "letta" and source.get("data"):
            existing_id = source.get("file_id")
            if existing_id:
                return (
                    {
                        "type": "image",
                        "source": {
                            "type": "letta",
                            "file_id": existing_id,
                            "media_type": source.get("media_type") or "image/png",
                            "data": None,
                            "detail": source.get("detail"),
                        },
                    },
                    existing_id,
                )
            letta_source = await _ingest_base64_source(
                data=source["data"],
                media_type=source.get("media_type") or "image/png",
                detail=source.get("detail"),
                actor=actor,
                provenance=provenance,
                generation_prompt=generation_prompt,
            )
            return (
                {
                    "type": "image",
                    "source": {
                        "type": "letta",
                        "file_id": letta_source.file_id,
                        "media_type": letta_source.media_type,
                        "data": None,
                        "detail": letta_source.detail,
                    },
                },
                letta_source.file_id,
            )

    return await _ingest_image_block(
        block,
        actor,
        provenance=provenance,
        generation_prompt=generation_prompt,
    )


async def convert_historic_images_in_message(message: PydanticMessage, actor: PydanticUser) -> tuple[List[str], bool]:
    """Part 1 migration: convert all inline image bytes, including tool returns."""
    provenance = _provenance_for_message(message)
    generation_prompt = None
    image_ids: List[str] = []
    changed = False

    if isinstance(message.content, list):
        updated_content = []
        for block in message.content:
            if (isinstance(block, ImageContent) and block.type == MessageContentType.image) or (
                isinstance(block, dict) and block.get("type") == "image"
            ):
                new_block, image_id = await _ingest_migration_image_block(
                    block,
                    actor,
                    provenance=provenance,
                    generation_prompt=generation_prompt,
                )
                if new_block != block:
                    changed = True
                updated_content.append(new_block)
                if image_id:
                    image_ids.append(image_id)
            else:
                updated_content.append(block)
        message.content = updated_content

    if message.tool_returns:
        for tool_return in message.tool_returns:
            func_response = tool_return.func_response
            if not isinstance(func_response, list):
                continue
            updated_parts = []
            for part in func_response:
                if (isinstance(part, ImageContent) and part.type == MessageContentType.image) or (
                    isinstance(part, dict) and part.get("type") == "image"
                ):
                    new_part, image_id = await _ingest_migration_image_block(
                        part,
                        actor,
                        provenance=provenance,
                        generation_prompt=generation_prompt,
                    )
                    if new_part != part:
                        changed = True
                    updated_parts.append(new_part)
                    if image_id:
                        image_ids.append(image_id)
                else:
                    updated_parts.append(part)
            tool_return.func_response = updated_parts

    return image_ids, changed


async def ingest_images_in_message(message: PydanticMessage, actor: PydanticUser) -> List[str]:
    """Persist inline image bytes as image records; replace with LettaImage refs."""
    provenance = _provenance_for_message(message)
    generation_prompt = None
    image_ids: List[str] = []

    if isinstance(message.content, list):
        updated_content = []
        for block in message.content:
            if (isinstance(block, ImageContent) and block.type == MessageContentType.image) or (
                isinstance(block, dict) and block.get("type") == "image"
            ):
                new_block, image_id = await _ingest_image_block(
                    block,
                    actor,
                    provenance=provenance,
                    generation_prompt=generation_prompt,
                )
                updated_content.append(new_block)
                if image_id:
                    image_ids.append(image_id)
            else:
                updated_content.append(block)
        message.content = updated_content

    # fetch_image already references persisted images; re-ingesting strips inline bytes.
    if message.tool_returns and message.name not in ("fetch_image", "image_fetch"):
        for tool_return in message.tool_returns:
            func_response = tool_return.func_response
            if not isinstance(func_response, list):
                continue
            updated_parts = []
            for part in func_response:
                if (isinstance(part, ImageContent) and part.type == MessageContentType.image) or (
                    isinstance(part, dict) and part.get("type") == "image"
                ):
                    new_part, image_id = await _ingest_image_block(
                        part,
                        actor,
                        provenance=provenance,
                        generation_prompt=generation_prompt,
                    )
                    updated_parts.append(new_part)
                    if image_id:
                        image_ids.append(image_id)
                else:
                    updated_parts.append(part)
            tool_return.func_response = updated_parts

    return image_ids


async def link_image_to_message(image_id: str, message_id: str, actor: PydanticUser) -> None:
    """Run or finalize enrichment and push the message re-embed when ready."""
    manager = ImageManager()
    image = await manager.get_by_id_async(image_id, actor)
    if not image:
        return

    from letta.services.message_manager import MessageManager

    msg_mgr = MessageManager()

    if image.enrichment_status == "complete":
        msg = await msg_mgr.get_message_by_id_async(message_id, actor)
        if msg and msg.agent_id:
            await msg_mgr._embed_messages_background(
                [msg],
                actor,
                msg.agent_id,
                embedding_version=2,
            )
        return

    if image.enrichment_status == "failed":
        msg = await msg_mgr.get_message_by_id_async(message_id, actor)
        if msg and msg.agent_id:
            await msg_mgr._embed_messages_background(
                [msg],
                actor,
                msg.agent_id,
                embedding_version=1,
            )
        return

    await enrich_image_background(image_id, actor, message_id=message_id)


def schedule_image_enrichment_for_message(
    message: PydanticMessage,
    actor: PydanticUser,
    image_ids: List[str],
) -> None:
    """Fire-and-forget enrichment + message re-embed for each linked image."""
    if not message.id:
        return
    seen = set()
    for image_id in image_ids:
        if not image_id or image_id in seen:
            continue
        seen.add(image_id)
        fire_and_forget(
            link_image_to_message(image_id, message.id, actor),
            task_name=f"enrich_image_{image_id}_for_{message.id}",
        )


async def schedule_image_re_enrichment(image_id: str, actor: PydanticUser) -> None:
    """Mark pending and queue a forced background re-enrichment run."""
    await _set_enrichment_pending(image_id, actor, reset_attempts=True)
    fire_and_forget(
        enrich_image_background(image_id, actor, force=True),
        task_name=f"re_enrich_image_{image_id}",
    )


async def _set_enrichment_pending(
    image_id: str,
    actor: PydanticUser,
    *,
    reset_attempts: bool = False,
) -> None:
    from letta.server.db import db_registry

    async with db_registry.async_session() as session:
        row = await ImageRecord.read_async(db_session=session, identifier=image_id, actor=actor)
        row.enrichment_status = "pending"
        row.error_message = None
        if reset_attempts:
            row.enrichment_attempts = 0
        await row.update_async(session, actor=actor)


def _captions_from_image(image) -> dict:
    return {
        "caption": image.caption,
        "description": image.description,
        "details": image.details,
    }


async def reembed_image_embedding_only(image_id: str, actor: PydanticUser) -> None:
    """Re-pixel-embed from stored full-resolution bytes; leave captions and 1MP unchanged.

    Used for embedding-space migration (e.g. preview→GA) when metadata is already complete.
    """
    from letta.server.db import db_registry

    manager = ImageManager()
    store = get_object_store_client()
    image = await manager.get_by_id_async(image_id, actor)
    if not image:
        raise ValueError(f"Image not found: {image_id}")
    if not image.object_url_full:
        raise ValueError(f"Image {image_id} has no full object URL")

    raw = await store.get_bytes(image.object_url_full)
    media_type = image.media_type or "image/jpeg"
    captions = _captions_from_image(image)

    embedding_config = await resolve_embedding_config_async(actor=actor)
    llm_client = LLMClient.create(embedding_config.embedding_endpoint_type, actor=actor)
    prepared = await _embed_image_vector(
        llm_client,
        raw,
        media_type,
        captions,
        embedding_config,
    )

    async with db_registry.async_session() as session:
        row = await ImageRecord.read_async(db_session=session, identifier=image_id, actor=actor)
        row.embedding = prepared
        row.embedding_config = embedding_config.model_dump()
        row.embedding_space_id = embedding_config.embedding_space_id
        row.enrichment_status = "complete"
        row.error_message = None
        await row.update_async(session, actor=actor)


async def enrich_image_background(
    image_id: str,
    actor: PydanticUser,
    *,
    message_id: Optional[str] = None,
    force: bool = False,
) -> None:
    """Background: 1MP derivative, captions, pixel embed, message re-embed push."""
    manager = ImageManager()
    store = get_object_store_client()
    try:
        image = await manager.get_by_id_async(image_id, actor)
        if not image:
            return
        if image.enrichment_status == "complete" and not force:
            return
        if force:
            await _set_enrichment_pending(image_id, actor, reset_attempts=True)

        raw = await store.get_bytes(image.object_url_full)
        if image.object_url_1mp and image.file_size_1mp:
            onemp_key = image.object_url_1mp
            onemp_wire = image.file_size_1mp
            embed_bytes = await store.get_bytes(onemp_key)
            embed_media_type = "image/jpeg"
        else:
            embed_bytes, embed_media_type, onemp_wire = generate_1mp_derivative(raw, image.media_type)
            onemp_key = await store.put_bytes(image.content_hash, embed_bytes, suffix="_1mp")

        captions = await _generate_three_tier_captions(raw, image.media_type, actor)
        embedding_config = await resolve_embedding_config_async(actor=actor)
        llm_client = LLMClient.create(embedding_config.embedding_endpoint_type, actor=actor)

        from letta.server.db import db_registry

        async with db_registry.async_session() as session:
            row = await ImageRecord.read_async(db_session=session, identifier=image_id, actor=actor)
            row.object_url_1mp = onemp_key
            row.file_size_1mp = onemp_wire
            row.caption = captions.get("caption")
            row.description = captions.get("description")
            row.details = captions.get("details")
            await row.update_async(session, actor=actor)

        prepared = await _embed_image_vector(
            llm_client,
            embed_bytes,
            embed_media_type,
            captions,
            embedding_config,
        )

        async with db_registry.async_session() as session:
            row = await ImageRecord.read_async(db_session=session, identifier=image_id, actor=actor)
            row.embedding = prepared
            row.embedding_config = embedding_config.model_dump()
            row.embedding_space_id = embedding_config.embedding_space_id
            row.enrichment_status = "complete"
            row.error_message = None
            await row.update_async(session, actor=actor)

        if message_id and (captions.get("caption") or captions.get("description")):
            from letta.services.message_manager import MessageManager

            msg_mgr = MessageManager()
            msg = await msg_mgr.get_message_by_id_async(message_id, actor)
            await msg_mgr._embed_messages_background(
                [msg],
                actor,
                msg.agent_id,
                embedding_version=2,
            )
    except Exception as e:
        logger.error("Image enrichment failed for %s: %s", image_id, e, exc_info=True)
        await _mark_enrichment_failed(image_id, actor, str(e), message_id=message_id)


async def _mark_enrichment_failed(image_id: str, actor: PydanticUser, error: str, *, message_id: Optional[str] = None) -> None:
    from letta.server.db import db_registry

    async with db_registry.async_session() as session:
        row = await ImageRecord.read_async(db_session=session, identifier=image_id, actor=actor)
        row.enrichment_attempts = (row.enrichment_attempts or 0) + 1
        if row.enrichment_attempts >= settings.image_enrichment_max_attempts:
            row.enrichment_status = "failed"
            row.error_message = error
        await row.update_async(session, actor=actor)

    if message_id:
        from letta.services.message_manager import MessageManager

        msg_mgr = MessageManager()
        msg = await msg_mgr.get_message_by_id_async(message_id, actor)
        await msg_mgr._embed_messages_background([msg], actor, msg.agent_id, embedding_version=1)


async def _embed_image_vector(
    llm_client: LLMClient,
    embed_bytes: bytes,
    embed_media_type: str,
    captions: dict,
    embedding_config,
) -> list:
    """Pixel embed from caller-supplied image bytes; fall back to caption text if the API rejects image input."""
    b64 = base64.standard_b64encode(embed_bytes).decode("ascii")
    image_input = [
        {"type": "image_url", "image_url": {"url": f"data:{embed_media_type};base64,{b64}"}}
    ]
    try:
        vectors = await llm_client.request_image_embeddings(image_input, embedding_config)
        return prepare_vector_for_write(vectors[0], embedding_config)
    except Exception as pixel_err:
        text = "\n\n".join(
            part for part in (captions.get("caption"), captions.get("description")) if part
        )
        if not text:
            raise pixel_err
        logger.warning("Image pixel embed failed, using caption text fallback: %s", pixel_err)
        vectors = await llm_client.request_embeddings(
            [text],
            embedding_config,
            input_type_override="search_document",
        )
        return prepare_vector_for_write(vectors[0], embedding_config)


async def _generate_three_tier_captions(raw: bytes, media_type: str, actor: PydanticUser) -> dict:
    """Single structured VLM call for caption (20-50 words), description (100-200 words), details (1500-2000 words)."""
    handle = settings.image_caption_model_handle or settings.default_llm_handle
    if not handle:
        return _fallback_captions()

    from letta.services.provider_manager import ProviderManager

    pm = ProviderManager()
    llm_config = await pm.get_llm_config_from_handle(handle, actor)
    caption_config = LLMConfig(**llm_config.model_dump())
    caption_config.put_inner_thoughts_in_kwargs = False
    caption_config.enable_reasoner = False

    client = LLMClient.create(caption_config.model_endpoint_type, actor=actor)
    b64 = base64.standard_b64encode(raw).decode("ascii")
    message = PydanticMessage(
        role=MessageRole.user,
        content=[
            TextContent(text=_CAPTION_PROMPT),
            ImageContent(
                source=Base64Image(
                    media_type=media_type or "image/png",
                    data=b64,
                    detail="high",
                )
            ),
        ],
    )

    request_data = client.build_request_data(
        agent_type=AgentType.letta_v1_agent,
        messages=[message],
        llm_config=caption_config,
        tools=[],
        system=None,
    )
    # Details tier targets 1500-2000 words; allow ample completion budget.
    request_data["max_tokens"] = 8192

    response = await client.request_async(request_data, caption_config)
    text = _extract_assistant_text(response)
    captions = _parse_caption_json(text)
    if not (captions.get("caption") or captions.get("description")):
        raise ValueError(f"caption JSON missing required fields: {text[:500]}")
    return captions
