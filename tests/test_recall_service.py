from letta.schemas.enums import MessageRole
from letta.schemas.letta_message_content import TextContent
from letta.schemas.message import Message
from letta.services.message_manager import MessageManager
from letta.schemas.letta_message_content import ImageContent, LettaImage, TextContent
from letta.services.recall.recall_service import (
    RecallHit,
    _file_archive_lexical_sql,
    _image_recall_snippet,
    _recall_snippet,
    _snippet_for_display,
    _source_passage_lexical_sql,
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
    assert "fetch_image(image-xyz)" in snippet
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
    assert "fetch_image" in snippet


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
