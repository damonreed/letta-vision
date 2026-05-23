import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from letta.services.agent_manager import AgentManager

from letta.services.agent_open_files_manager import AgentOpenFilesManager
from letta.services.file_archive_manager import FileArchiveManager
from letta.services.file_core_block_manager import FileCoreBlockManager
from letta.services.file_manager import FileManager
from letta.services.files.char_page_reader import CharPageReader
from letta.schemas.agent import AgentState
from letta.schemas.user import User


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

    async def update_file_core(self, agent_state: AgentState, file_id: str, new_summary: str) -> dict:
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

    async def write_archive(
        self,
        agent_state: AgentState,
        file_id: str,
        title: str,
        content: str,
        tags: Optional[List[str]] = None,
    ) -> dict:
        file_id = await self._resolve_file_id(agent_state, file_id)
        archive = await self.archive_manager.write_archive(
            file_id=file_id,
            title=title,
            content=content,
            tags=tags,
            author_agent_id=agent_state.id,
            source_conversation_id=self.conversation_id,
            embedding_config=agent_state.embedding_config,
            actor=self.actor,
        )
        return {
            "status": "success",
            "archive_id": archive.id,
            "file_id": file_id,
            "title": archive.title,
            "tags": archive.tags,
        }

    async def search_archives(
        self,
        agent_state: AgentState,
        query: str,
        file_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 10,
    ) -> dict:
        results = await self.archive_manager.search_archives(
            query=query,
            agent_id=agent_state.id,
            embedding_config=agent_state.embedding_config,
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
