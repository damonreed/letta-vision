from letta.schemas.enums import MessageRole
from letta.schemas.letta_message_content import TextContent
from letta.schemas.message import Message
from letta.services.message_manager import MessageManager
from letta.services.recall.recall_service import _recall_snippet, _snippet_for_display


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
