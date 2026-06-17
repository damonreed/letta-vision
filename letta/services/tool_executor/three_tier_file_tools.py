import mimetypes
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from letta.services.agent_manager import AgentManager

from letta.log import get_logger
from letta.orm.file_core_block import DEFAULT_FILE_CORE_CHAR_LIMIT
from letta.schemas.agent import AgentState
from letta.schemas.enums import FileProcessingStatus
from letta.schemas.file import FileMetadata as PydanticFileMetadata
from letta.schemas.user import User
from letta.services.agent_open_files_manager import AgentOpenFilesManager
from letta.services.file_archive_manager import FileArchiveManager
from letta.services.file_core_block_manager import FileCoreBlockManager
from letta.services.file_manager import FileManager
from letta.services.files.char_page_reader import CharPageReader
from letta.services.source_manager import SourceManager
from letta.utils import safe_create_task, sanitize_filename

logger = get_logger(__name__)


class ThreeTierFileTools:
    """Handlers for the three-tier filesystem tool surface."""

    def __init__(self, actor: User, agent_manager: Optional["AgentManager"] = None):
        from letta.services.agent_manager import AgentManager

        self.actor = actor
        self.agent_manager = agent_manager or AgentManager()
        self.open_files_manager = AgentOpenFilesManager()
        self.core_manager = FileCoreBlockManager()
        self.archive_manager = FileArchiveManager()
        self.file_manager = FileManager()
        self.source_manager = SourceManager()
        self.conversation_id: Optional[str] = None

    def _page_size(self, agent_state: AgentState) -> int:
        limit = agent_state.per_file_view_window_char_limit
        return max(1, int(limit) if limit is not None else 10_000)

    async def _resolve_file_id(self, agent_state: AgentState, file_id: str) -> str:
        from letta.services.files_agents_manager import FileAgentManager

        return await FileAgentManager().resolve_file_id_for_agent(
            agent_id=agent_state.id,
            file_id=file_id,
            actor=self.actor,
        )

    def _agent_has_folder(self, agent_state: AgentState, folder_id: str) -> bool:
        # AgentState exposes attached folders as `sources` (folder_ids is only on create/update schemas).
        return any(source.id == folder_id for source in (agent_state.sources or []))

    def _normalize_text_file_name(self, file_name: str) -> str:
        name = sanitize_filename(file_name.strip())
        if not name:
            raise ValueError("file_name must not be empty")
        if not Path(name).suffix:
            name = f"{name}.txt"
        return name

    async def _schedule_text_file_ingest(
        self,
        *,
        folder_id: str,
        file_metadata: PydanticFileMetadata,
        content: str,
        agent_state: AgentState,
    ) -> None:
        from letta.helpers.pinecone_utils import should_use_pinecone
        from letta.services.file_processor.embedder.openai_embedder import OpenAIEmbedder
        from letta.services.file_processor.embedder.pinecone_embedder import PineconeEmbedder
        from letta.services.file_processor.file_processor import FileProcessor
        from letta.services.file_processor.parser.markitdown_parser import MarkitdownFileParser
        from letta.services.file_processor.parser.mistral_parser import MistralFileParser
        from letta.settings import settings

        folder = await self.source_manager.get_source_by_id(source_id=folder_id, actor=self.actor)
        agent_states = await self.source_manager.list_attached_agents(source_id=folder_id, actor=self.actor)
        if not any(agent.id == agent_state.id for agent in agent_states):
            agent_states = list(agent_states) + [agent_state]

        content_bytes = content.encode("utf-8")
        from letta.embeddings.resolver import resolve_deployment_embedding_config_async

        embedding_config = await resolve_deployment_embedding_config_async(self.actor)

        if settings.mistral_api_key:
            file_parser = MistralFileParser()
        else:
            file_parser = MarkitdownFileParser()

        if should_use_pinecone():
            embedder = PineconeEmbedder(embedding_config=embedding_config)
        else:
            embedder = OpenAIEmbedder(embedding_config=embedding_config)

        file_processor = FileProcessor(file_parser=file_parser, embedder=embedder, actor=self.actor)

        async def _run_ingest() -> None:
            await file_processor.process(
                agent_states=agent_states,
                source_id=folder_id,
                content=content_bytes,
                file_metadata=file_metadata,
            )

        safe_create_task(_run_ingest(), label=f"file_add_ingest:{file_metadata.id}")

    async def file_add(
        self,
        agent_state: AgentState,
        folder_id: str,
        file_name: str,
        content: str,
        headline: Optional[str] = None,
    ) -> dict:
        folder = await self.source_manager.get_source_by_id(source_id=folder_id, actor=self.actor)
        original_filename = self._normalize_text_file_name(file_name)
        folder_attached = self._agent_has_folder(agent_state, folder_id)

        if not folder_attached:
            await self.agent_manager.attach_source_async(
                agent_id=agent_state.id,
                source_id=folder_id,
                actor=self.actor,
            )
            folder_attached = True

        existing_file = await self.file_manager.get_file_by_original_name_and_source(
            original_filename=original_filename,
            source_id=folder_id,
            actor=self.actor,
        )
        if existing_file:
            await self.file_manager.delete_file(file_id=existing_file.id, actor=self.actor)

        unique_filename = await self.file_manager.generate_unique_filename(
            original_filename=original_filename,
            source=folder,
            organization_id=self.actor.organization_id,
        )
        content_bytes = content.encode("utf-8")
        mime_type = mimetypes.guess_type(original_filename)[0] or "text/plain"

        file_metadata = PydanticFileMetadata(
            source_id=folder_id,
            file_name=unique_filename,
            original_file_name=original_filename,
            file_path=None,
            file_type=mime_type,
            file_size=len(content_bytes),
            processing_status=FileProcessingStatus.PARSING,
        )
        file_metadata = await self.file_manager.create_file(file_metadata, actor=self.actor, text=content)

        summary = (headline or "").strip()
        if not summary:
            preview = content.strip().replace("\n", " ")[:200]
            summary = preview or original_filename
        summary = summary[:DEFAULT_FILE_CORE_CHAR_LIMIT]

        await self.core_manager.get_or_create(
            file_id=file_metadata.id,
            organization_id=self.actor.organization_id,
            actor=self.actor,
            default_summary=summary,
        )

        await self._schedule_text_file_ingest(
            folder_id=folder_id,
            file_metadata=file_metadata,
            content=content,
            agent_state=agent_state,
        )

        logger.info(
            "file_add created file_id=%s folder_id=%s name=%s bytes=%s",
            file_metadata.id,
            folder_id,
            unique_filename,
            len(content_bytes),
        )

        return {
            "status": "success",
            "file_id": file_metadata.id,
            "file_name": unique_filename,
            "original_file_name": original_filename,
            "folder_id": folder_id,
            "folder_attached": folder_attached,
            "processing_status": file_metadata.processing_status.value,
            "char_count": len(content),
        }

    async def open_file(self, agent_state: AgentState, file_id: str) -> dict:
        file_id = await self._resolve_file_id(agent_state, file_id)
        open_row, evicted = await self.open_files_manager.open_file(
            agent_id=agent_state.id,
            file_id=file_id,
            actor=self.actor,
            max_files_open=agent_state.max_files_open,
        )
        core = await self.core_manager.get_or_create(
            file_id=file_id,
            organization_id=self.actor.organization_id,
            actor=self.actor,
        )
        file_meta = await self.file_manager.get_file_by_id(file_id=file_id, actor=self.actor, include_content=True)
        content = file_meta.content or ""
        page_size = self._page_size(agent_state)
        reader = CharPageReader(content, page_size)
        return {
            "status": "success",
            "file_id": file_id,
            "headline": core.summary,
            "total_chars": reader.total_chars,
            "total_pages": reader.total_pages,
            "page_size_chars": page_size,
            "cursor_char": open_row.cursor_char,
            "evicted_file_ids": evicted,
        }

    async def close_file(self, agent_state: AgentState, file_id: str) -> dict:
        file_id = await self._resolve_file_id(agent_state, file_id)
        closed = await self.open_files_manager.close_file(
            agent_id=agent_state.id, file_id=file_id, actor=self.actor
        )
        return {"status": "success" if closed else "not_open", "file_id": file_id}

    async def _read_page_at_cursor(self, agent_state: AgentState, file_id: str, cursor: int) -> dict:
        file_meta = await self.file_manager.get_file_by_id(file_id=file_id, actor=self.actor, include_content=True)
        page_size = self._page_size(agent_state)
        reader = CharPageReader(file_meta.content or "", page_size)
        content, start, end, next_cursor = reader.read_page(cursor)
        page_number = reader.page_number_for_cursor(start)
        return {
            "file_id": file_id,
            "char_range": [start, end],
            "page_number": page_number,
            "total_pages": reader.total_pages,
            "page_size_chars": page_size,
            "content": content,
            "next_cursor_char": next_cursor,
        }

    async def file_read_page(self, agent_state: AgentState, file_id: str) -> dict:
        file_id = await self._resolve_file_id(agent_state, file_id)
        open_row = await self.open_files_manager.get_open_file(
            agent_id=agent_state.id, file_id=file_id, actor=self.actor
        )
        if open_row is None:
            raise ValueError(f"File {file_id} is not open. Call open_file first.")
        result = await self._read_page_at_cursor(agent_state, file_id, open_row.cursor_char)
        await self.open_files_manager.update_cursor(
            agent_id=agent_state.id,
            file_id=file_id,
            cursor_char=result["next_cursor_char"],
            actor=self.actor,
        )
        del result["next_cursor_char"]
        return result

    async def file_read_next_page(self, agent_state: AgentState, file_id: str) -> dict:
        file_id = await self._resolve_file_id(agent_state, file_id)
        open_row = await self.open_files_manager.get_open_file(
            agent_id=agent_state.id, file_id=file_id, actor=self.actor
        )
        if open_row is None:
            raise ValueError(f"File {file_id} is not open. Call open_file first.")
        file_meta = await self.file_manager.get_file_by_id(file_id=file_id, actor=self.actor, include_content=True)
        page_size = self._page_size(agent_state)
        reader = CharPageReader(file_meta.content or "", page_size)
        next_cursor = reader.next_page_cursor(open_row.cursor_char)
        result = await self._read_page_at_cursor(agent_state, file_id, next_cursor)
        await self.open_files_manager.update_cursor(
            agent_id=agent_state.id, file_id=file_id, cursor_char=result["next_cursor_char"], actor=self.actor
        )
        del result["next_cursor_char"]
        return result

    async def file_read_prev_page(self, agent_state: AgentState, file_id: str) -> dict:
        file_id = await self._resolve_file_id(agent_state, file_id)
        open_row = await self.open_files_manager.get_open_file(
            agent_id=agent_state.id, file_id=file_id, actor=self.actor
        )
        if open_row is None:
            raise ValueError(f"File {file_id} is not open. Call open_file first.")
        file_meta = await self.file_manager.get_file_by_id(file_id=file_id, actor=self.actor, include_content=True)
        page_size = self._page_size(agent_state)
        reader = CharPageReader(file_meta.content or "", page_size)
        prev_cursor = reader.prev_page_cursor(open_row.cursor_char)
        result = await self._read_page_at_cursor(agent_state, file_id, prev_cursor)
        await self.open_files_manager.update_cursor(
            agent_id=agent_state.id, file_id=file_id, cursor_char=prev_cursor, actor=self.actor
        )
        if "next_cursor_char" in result:
            del result["next_cursor_char"]
        return result

    async def file_read_range(self, agent_state: AgentState, file_id: str, start_char: int, end_char: int) -> dict:
        file_id = await self._resolve_file_id(agent_state, file_id)
        file_meta = await self.file_manager.get_file_by_id(file_id=file_id, actor=self.actor, include_content=True)
        page_size = self._page_size(agent_state)
        limit = page_size * 2
        reader = CharPageReader(file_meta.content or "", page_size)
        content, start, end = reader.read_range(start_char, end_char, limit)
        return {"file_id": file_id, "char_range": [start, end], "content": content}

    async def file_grep(self, agent_state: AgentState, file_id: str, pattern: str, max_hits: int = 20) -> dict:
        file_id = await self._resolve_file_id(agent_state, file_id)
        file_meta = await self.file_manager.get_file_by_id(file_id=file_id, actor=self.actor, include_content=True)
        text = file_meta.content or ""
        regex = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
        hits: List[Dict[str, Any]] = []
        char_offset = 0
        for line_num, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                snippet_start = max(0, char_offset - 100)
                snippet_end = min(len(text), char_offset + len(line) + 100)
                hits.append(
                    {
                        "char_offset": char_offset,
                        "line_number": line_num,
                        "snippet": text[snippet_start:snippet_end],
                    }
                )
                if len(hits) >= max_hits:
                    break
            char_offset += len(line) + 1
        return {"file_id": file_id, "hits": hits}

    async def file_edit_text(
        self,
        agent_state: AgentState,
        file_id: str,
        command: str,
        old_string: Optional[str] = None,
        new_string: Optional[str] = None,
        insert_text: Optional[str] = None,
        insert_line: int = -1,
    ) -> dict:
        from letta.services.file_text import edit_file_text

        file_id = await self._resolve_file_id(agent_state, file_id)
        return await edit_file_text(
            file_id=file_id,
            command=command,
            actor=self.actor,
            agent_state=agent_state,
            old_string=old_string,
            new_string=new_string,
            insert_text=insert_text,
            insert_line=insert_line,
        )

    async def update_file_headline(self, agent_state: AgentState, file_id: str, new_summary: str) -> dict:
        file_id = await self._resolve_file_id(agent_state, file_id)
        previous = await self.core_manager.get(file_id=file_id, actor=self.actor)
        updated = await self.core_manager.update(
            file_id=file_id,
            new_summary=new_summary,
            agent_id=agent_state.id,
            actor=self.actor,
        )
        return {
            "status": "success",
            "file_id": file_id,
            "previous_summary": previous.summary if previous else "",
            "new_summary": updated.summary,
            "version": updated.version,
        }

    async def write_file_note(
        self,
        agent_state: AgentState,
        file_id: str,
        title: str,
        content: str,
        tags: Optional[List[str]] = None,
    ) -> dict:
        file_id = await self._resolve_file_id(agent_state, file_id)
        archive = await self.archive_manager.write_file_archive(
            file_id=file_id,
            title=title,
            content=content,
            tags=tags,
            author_agent_id=agent_state.id,
            source_conversation_id=self.conversation_id,
            actor=self.actor,
        )
        return {
            "status": "success",
            "archive_id": archive.id,
            "file_id": file_id,
            "title": archive.title,
            "tags": archive.tags,
        }

    async def file_notes_search(
        self,
        agent_state: AgentState,
        query: str,
        file_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 10,
    ) -> dict:
        results = await self.archive_manager.search_file_archives(
            query=query,
            agent_id=agent_state.id,
            actor=self.actor,
            file_id=file_id,
            tags=tags,
            limit=limit,
        )
        return {
            "results": [
                {
                    "archive_id": r.id,
                    "file_id": r.file_id,
                    "file_name": r.file_name,
                    "title": r.title,
                    "content": r.content,
                    "tags": r.tags,
                    "author_agent_id": r.author_agent_id,
                    "source_conversation_id": r.source_conversation_id,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in results
            ]
        }
