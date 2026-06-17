"""File body text edit helpers for agent tools."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

from letta.constants import CORE_MEMORY_LINE_NUMBER_WARNING, MEMORY_TOOLS_LINE_NUMBER_PREFIX_REGEX
from letta.log import get_logger
from letta.schemas.enums import FileProcessingStatus
from letta.schemas.user import User as PydanticUser
from letta.services.agent_open_files_manager import AgentOpenFilesManager
from letta.services.file_manager import FileManager
from letta.utils import safe_create_task

if TYPE_CHECKING:
    from letta.schemas.agent import AgentState

logger = get_logger(__name__)

TEXT_EDITABLE_MIME_TYPES = frozenset(
    {
        "text/plain",
        "text/markdown",
        "text/x-markdown",
    }
)
TEXT_EDITABLE_EXTENSIONS = frozenset({".txt", ".md", ".markdown", ".text"})

FileEditCommand = Literal["str_replace", "insert", "set"]


def is_text_editable_file(*, file_type: Optional[str], file_name: Optional[str]) -> bool:
    mime = (file_type or "").lower()
    if mime in TEXT_EDITABLE_MIME_TYPES:
        return True
    name = (file_name or "").lower()
    suffix = Path(name).suffix
    return suffix in TEXT_EDITABLE_EXTENSIONS


def _reject_line_number_artifacts(text: str, param_name: str) -> None:
    if MEMORY_TOOLS_LINE_NUMBER_PREFIX_REGEX.search(text):
        raise ValueError(
            f"{param_name} contains a line number prefix, which is not allowed. "
            "Do not include line numbers when calling file text tools."
        )
    if CORE_MEMORY_LINE_NUMBER_WARNING in text:
        raise ValueError(
            f"{param_name} contains a line number warning, which is not allowed. "
            "Do not include line number information when calling file text tools."
        )


def _apply_str_replace(current_value: str, old_string: str, new_string: str) -> str:
    _reject_line_number_artifacts(old_string, "old_string")
    _reject_line_number_artifacts(new_string, "new_string")

    old_string = str(old_string).expandtabs()
    new_string = str(new_string).expandtabs()
    current_value = str(current_value).expandtabs()

    occurrences = current_value.count(old_string)
    if occurrences == 0:
        raise ValueError(
            f"No replacement was performed, old_string `{old_string}` did not appear verbatim in the file."
        )
    if occurrences > 1:
        lines = [idx + 1 for idx, line in enumerate(current_value.split("\n")) if old_string in line]
        raise ValueError(
            f"No replacement was performed. Multiple occurrences of old_string `{old_string}` in lines {lines}. "
            "Please ensure it is unique."
        )
    return current_value.replace(old_string, new_string)


def _apply_insert(current_value: str, insert_text: str, insert_line: int) -> str:
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
            f"of the file: {[0, n_lines]}, or -1 to append to the end."
        )

    insert_text_lines = insert_text.split("\n")
    new_value_lines = current_value_lines[:insert_line] + insert_text_lines + current_value_lines[insert_line:]
    return "\n".join(new_value_lines)


async def schedule_file_text_reingest(
    *,
    file_id: str,
    source_id: str,
    agent_state: "AgentState",
    actor: PydanticUser,
) -> None:
    from letta.helpers.pinecone_utils import should_use_pinecone
    from letta.services.file_processor.embedder.openai_embedder import OpenAIEmbedder
    from letta.services.file_processor.embedder.pinecone_embedder import PineconeEmbedder
    from letta.services.file_processor.file_processor import FileProcessor
    from letta.services.file_processor.parser.markitdown_parser import MarkitdownFileParser
    from letta.services.file_processor.parser.mistral_parser import MistralFileParser
    from letta.services.passage_manager import PassageManager
    from letta.services.source_manager import SourceManager
    from letta.settings import settings

    file_manager = FileManager()
    passage_manager = PassageManager()
    source_manager = SourceManager()

    async def _run_reingest() -> None:
        try:
            await file_manager.update_file_status(
                file_id=file_id,
                actor=actor,
                processing_status=FileProcessingStatus.PARSING,
                enforce_state_transitions=False,
            )

            existing_passages = await passage_manager.list_passages_by_file_id_async(file_id=file_id, actor=actor)
            if existing_passages:
                await passage_manager.delete_source_passages_async(actor=actor, passages=existing_passages)

            file_metadata = await file_manager.get_file_by_id(file_id=file_id, actor=actor, include_content=True)
            if file_metadata is None or file_metadata.content is None:
                logger.warning("file_text_reingest skipped: no content for file_id=%s", file_id)
                return

            agent_states = await source_manager.list_attached_agents(source_id=source_id, actor=actor)
            if not any(agent.id == agent_state.id for agent in agent_states):
                agent_states = list(agent_states) + [agent_state]

            from letta.embeddings.resolver import resolve_deployment_embedding_config_async

            embedding_config = await resolve_deployment_embedding_config_async(actor)

            if settings.mistral_api_key:
                file_parser = MistralFileParser()
            else:
                file_parser = MarkitdownFileParser()

            if should_use_pinecone():
                embedder = PineconeEmbedder(embedding_config=embedding_config)
            else:
                embedder = OpenAIEmbedder(embedding_config=embedding_config)

            file_processor = FileProcessor(file_parser=file_parser, embedder=embedder, actor=actor)
            await file_processor.process_imported_file(file_metadata=file_metadata, source_id=source_id)
        except Exception:
            logger.exception("file_text_reingest failed for file_id=%s", file_id)
            await file_manager.update_file_status(
                file_id=file_id,
                actor=actor,
                processing_status=FileProcessingStatus.ERROR,
                error_message="File text re-ingest failed after edit",
                enforce_state_transitions=False,
            )

    safe_create_task(_run_reingest(), label=f"file_edit_text_reingest:{file_id}")


async def edit_file_text(
    *,
    file_id: str,
    command: FileEditCommand,
    actor: PydanticUser,
    agent_state: "AgentState",
    old_string: Optional[str] = None,
    new_string: Optional[str] = None,
    insert_text: Optional[str] = None,
    insert_line: int = -1,
) -> dict:
    file_manager = FileManager()
    file_meta = await file_manager.get_file_by_id(file_id=file_id, actor=actor, include_content=True)
    if file_meta is None:
        raise ValueError(f"File {file_id} not found.")

    editable_name = file_meta.original_file_name or file_meta.file_name
    if not is_text_editable_file(file_type=file_meta.file_type, file_name=editable_name):
        raise ValueError(
            f"File {file_id} is not a text-editable file (type={file_meta.file_type!r}). "
            "Only plain-text files (.txt, .md) can be edited with file_edit_text."
        )

    current_value = file_meta.content or ""

    if command == "str_replace":
        if old_string is None:
            raise ValueError("old_string is required for str_replace command")
        if new_string is None:
            raise ValueError("new_string is required for str_replace command")
        new_value = _apply_str_replace(current_value, old_string, new_string)
    elif command == "insert":
        if insert_text is None:
            raise ValueError("insert_text is required for insert command")
        new_value = _apply_insert(current_value, insert_text, insert_line)
    elif command == "set":
        if new_string is None:
            raise ValueError("new_string is required for set command")
        _reject_line_number_artifacts(new_string, "new_string")
        new_value = str(new_string).expandtabs()
    else:
        raise ValueError(f"Unknown command `{command}`. Supported commands: str_replace, insert, set.")

    await file_manager.upsert_file_content(file_id=file_id, text=new_value, actor=actor)
    await AgentOpenFilesManager().clamp_cursors_for_file(
        file_id=file_id,
        max_char=len(new_value),
        actor=actor,
    )

    if not file_meta.source_id:
        raise ValueError(f"File {file_id} has no source_id; cannot re-ingest passages.")

    await schedule_file_text_reingest(
        file_id=file_id,
        source_id=file_meta.source_id,
        agent_state=agent_state,
        actor=actor,
    )

    return {
        "status": "success",
        "file_id": file_id,
        "char_count": len(new_value),
        "command": command,
        "processing_status": FileProcessingStatus.PARSING.value,
    }
