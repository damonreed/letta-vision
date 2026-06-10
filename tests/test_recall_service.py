from letta.schemas.enums import MessageRole
from letta.schemas.letta_message_content import TextContent
from letta.schemas.message import Message
from letta.services.message_manager import MessageManager
from letta.schemas.letta_message_content import ImageContent, LettaImage, TextContent
from letta.services.recall.recall_service import (
    RecallHit,
    _apply_diversity_cap,
    _dedup_image_message_hits,
    _file_archive_lexical_sql,
    _image_recall_snippet,
    _recall_snippet,
    _snippet_for_display,
    _source_passage_lexical_sql,
    finalize_recall_hits,
    format_recall_hit,
)


def test_snippet_for_display_unwraps_user_json():
    wrapped = MessageManager()._extract_message_text(
        Message(role=MessageRole.user, content=[TextContent(text="cerulean-lighthouse-42")])
    )
    assert _snippet_for_display(wrapped) == "cerulean-lighthouse-42"


def test_recall_snippet_reads_message_content_not_legacy_text_column():
    class _Row:
        def to_pydantic(self):
            return Message(
                role=MessageRole.user,
                content=[TextContent(text="Lyra prefers oat milk in coffee")],
            )

    assert _recall_snippet("message", _Row()) == "Lyra prefers oat milk in coffee"


def test_image_recall_snippet_prefers_description_and_falls_back():
    class _Image:
        id = "image-abc"
        description = None
        caption = "Flat lay of everyday carry items"
        details = None
        generation_prompt = None
        provenance = "uploaded"
        media_type = "image/jpeg"
        enrichment_status = "complete"

    assert _image_recall_snippet(_Image()) == "Flat lay of everyday carry items"

    class _Bare:
        id = "image-xyz"
        description = None
        caption = None
        details = None
        generation_prompt = None
        provenance = "uploaded"
        media_type = "image/png"
        enrichment_status = "pending"

    snippet = _image_recall_snippet(_Bare())
    assert "image_fetch(image-xyz)" in snippet
    assert "pending" in snippet


def test_recall_snippet_message_with_image_only_content():
    class _Row:
        def to_pydantic(self):
            return Message(
                role=MessageRole.user,
                content=[
                    TextContent(text=""),
                    ImageContent(
                        source=LettaImage(file_id="image-111", media_type="image/jpeg", data=None),
                    ),
                ],
            )

    snippet = _recall_snippet("message", _Row())
    assert "image-111" in snippet
    assert "image_fetch" in snippet


def test_file_archive_lexical_sql_avoids_ambiguous_null_agent_param():
    scoped = _file_archive_lexical_sql(with_agent_filter=True)
    assert "sa.agent_id = :agent_id" in scoped
    assert "IS NULL" not in scoped

    unscoped = _file_archive_lexical_sql(with_agent_filter=False)
    assert "sa.agent_id" not in unscoped


def test_source_passage_lexical_sql_scopes_to_agent_when_requested():
    scoped = _source_passage_lexical_sql(with_agent_filter=True)
    assert "sources_agents" in scoped
    assert "sa.agent_id = :agent_id" in scoped
    assert "sp.file_name" in scoped

    unscoped = _source_passage_lexical_sql(with_agent_filter=False)
    assert "sources_agents" not in unscoped
    assert "file_name" in unscoped


def test_format_recall_hit_labels_file_passages_with_filename():
    hit = format_recall_hit(
        RecallHit(
            layer="file",
            snippet="Victoria\nDamian",
            handle="passage-abc",
            score=0.0328,
            reasons=["lexical"],
            filename="villains.txt",
        )
    )
    assert hit.startswith("[file] handle=passage-abc filename=villains.txt score=0.0328")
    assert "Victoria" in hit


def test_dedup_image_message_hits_drops_redundant_message():
    hits = [
        RecallHit(
            layer="image",
            snippet="Portrait description",
            handle="image-111",
            score=0.02,
            reasons=["vector"],
        ),
        RecallHit(
            layer="message",
            snippet="Here is the portrait",
            handle="message-222",
            score=0.015,
            reasons=["vector"],
            linked_image_ids=["image-111"],
        ),
    ]
    filtered = _dedup_image_message_hits(hits)
    assert len(filtered) == 1
    assert filtered[0].layer == "image"
    assert "message" in filtered[0].reasons


def test_apply_diversity_cap_limits_hits_per_file():
    hits = [
        RecallHit(layer="file", snippet="a", handle="p1", score=0.05, reasons=["vector"], source_group="doc.md"),
        RecallHit(layer="file", snippet="b", handle="p2", score=0.04, reasons=["vector"], source_group="doc.md"),
        RecallHit(layer="file", snippet="c", handle="p3", score=0.03, reasons=["vector"], source_group="doc.md"),
        RecallHit(layer="file", snippet="d", handle="p4", score=0.02, reasons=["vector"], source_group="doc.md"),
        RecallHit(layer="archival", snippet="e", handle="a1", score=0.01, reasons=["vector"]),
    ]
    capped = _apply_diversity_cap(hits, limit=10, per_source_cap=3)
    assert len(capped) == 4
    assert sum(1 for hit in capped if hit.source_group == "doc.md") == 3


def test_finalize_recall_hits_applies_dedup_then_cap():
    hits = [
        RecallHit(layer="image", snippet="img", handle="image-1", score=0.05, reasons=["vector"]),
        RecallHit(
            layer="message",
            snippet="msg",
            handle="message-1",
            score=0.04,
            reasons=["vector"],
            linked_image_ids=["image-1"],
        ),
        RecallHit(layer="file", snippet="1", handle="p1", score=0.03, reasons=["vector"], source_group="x.md"),
        RecallHit(layer="file", snippet="2", handle="p2", score=0.02, reasons=["vector"], source_group="x.md"),
        RecallHit(layer="file", snippet="3", handle="p3", score=0.01, reasons=["vector"], source_group="x.md"),
    ]
    final = finalize_recall_hits(hits, limit=5)
    assert len(final) == 4
    assert all(hit.layer != "message" for hit in final)


def test_format_recall_hit_labels_file_archives_with_filename():
    hit = format_recall_hit(
        RecallHit(
            layer="file_archive",
            snippet="[Victoria's character] Victoria is a villain and wears black leather.",
            handle="file_archive-abc",
            score=0.0318,
            reasons=["vector"],
            filename="v060-test/villains.txt",
        )
    )
    assert hit.startswith(
        "[file_archive] handle=file_archive-abc filename=v060-test/villains.txt score=0.0318"
    )
