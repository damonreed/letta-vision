from typing import TYPE_CHECKING, List, Literal, Optional

if TYPE_CHECKING:
    from letta.schemas.agent import AgentState


async def file_add(
    agent_state: "AgentState",
    folder_id: str,
    file_name: str,
    content: str,
    headline: Optional[str] = None,
) -> dict:
    """
    Create a plain-text file in a folder, store the given content, and ingest it for search.

    Args:
        folder_id (str): The folder (source) ID to store the file in.
        file_name (str): Filename (e.g. notes.txt). A .txt suffix is added when missing.
        content (str): Full text body to write into the file.
        headline (Optional[str]): Optional short file headline for directory listings.

    Returns:
        dict: Created file metadata (file_id, file_name, folder_id, processing_status).
    """
    raise NotImplementedError("Tool not implemented. Please contact the Letta team.")


async def add_text_file(
    agent_state: "AgentState",
    folder_id: str,
    file_name: str,
    content: str,
    headline: Optional[str] = None,
) -> dict:
    """
    Deprecated alias for file_add.

    Args:
        folder_id: The folder (source) ID to store the file in.
        file_name: Filename (e.g. notes.txt). A .txt suffix is added when missing.
        content: Full text body to write into the file.
        headline: Optional short file headline for directory listings.

    Returns:
        Created file metadata (file_id, file_name, folder_id, processing_status).
    """
    raise NotImplementedError("Tool not implemented. Please contact the Letta team.")


async def attach_folder(agent_state: "AgentState", folder_id: str) -> dict:
    """
    Attach a folder of files to the agent.

    Args:
        folder_id (str): The folder (source) ID to attach.

    Returns:
        dict: Status of the attach operation.
    """
    raise NotImplementedError("Tool not implemented. Please contact the Letta team.")


async def detach_folder(agent_state: "AgentState", folder_id: str) -> dict:
    """
    Detach a folder from the agent.

    Args:
        folder_id (str): The folder (source) ID to detach.

    Returns:
        dict: Status of the detach operation.
    """
    raise NotImplementedError("Tool not implemented. Please contact the Letta team.")


async def open_file(agent_state: "AgentState", file_id: str) -> dict:
    """
    Open a file: attach its file headline to context without loading content.

    Args:
        file_id (str): The file ID to open.

    Returns:
        dict: Open file metadata including cursor position.
    """
    raise NotImplementedError("Tool not implemented. Please contact the Letta team.")


async def close_file(agent_state: "AgentState", file_id: str) -> dict:
    """
    Close a file and detach its file headline from context.

    Args:
        file_id (str): The file ID to close.

    Returns:
        dict: Status of the close operation.
    """
    raise NotImplementedError("Tool not implemented. Please contact the Letta team.")


async def files_list_open(agent_state: "AgentState") -> dict:
    """
    List currently open files with cursor position, page numbers, and total size.

    Args:
        agent_state: Current agent state (injected by runtime).

    Returns:
        dict: Open files with cursor_char, total_chars, page_number, and total_pages.
    """
    raise NotImplementedError("Tool not implemented. Please contact the Letta team.")


async def file_read_page(agent_state: "AgentState", file_id: str) -> dict:
    """
    Read the current page at the cursor and advance.

    Args:
        file_id (str): The file ID to read.

    Returns:
        dict: Page content, character range, and updated cursor.
    """
    raise NotImplementedError("Tool not implemented. Please contact the Letta team.")


async def file_read_next_page(agent_state: "AgentState", file_id: str) -> dict:
    """
    Navigate to and read the next page.

    Args:
        file_id (str): The file ID to read.

    Returns:
        dict: Page content, character range, and updated cursor.
    """
    raise NotImplementedError("Tool not implemented. Please contact the Letta team.")


async def file_read_prev_page(agent_state: "AgentState", file_id: str) -> dict:
    """
    Navigate to and read the previous page.

    Args:
        file_id (str): The file ID to read.

    Returns:
        dict: Page content, character range, and updated cursor.
    """
    raise NotImplementedError("Tool not implemented. Please contact the Letta team.")


async def file_read_range(agent_state: "AgentState", file_id: str, start_char: int, end_char: int) -> dict:
    """
    Read a specific character range without updating the cursor.

    Args:
        file_id (str): The file ID to read.
        start_char (int): Start character offset (inclusive).
        end_char (int): End character offset (exclusive).

    Returns:
        dict: Requested content and character range.
    """
    raise NotImplementedError("Tool not implemented. Please contact the Letta team.")


async def file_edit_text(
    agent_state: "AgentState",
    file_id: str,
    command: Literal["str_replace", "insert", "set"],
    old_string: Optional[str] = None,
    new_string: Optional[str] = None,
    insert_text: Optional[str] = None,
    insert_line: int = -1,
) -> dict:
    """
    Edit the body text of a plain-text file (.txt, .md). Re-ingests passages for search after each edit.

    Commands (same semantics as image_edit_text and the memory tool):
        str_replace — replace old_string with new_string (must match exactly once)
        insert — insert insert_text after insert_line (-1 appends)
        set — replace the entire file body with new_string

    File body edits are shared across agents — the file is the source of truth for all readers.
    Re-read with file_read_page after editing if you are paging through the file.

    Args:
        file_id (str): The file ID to edit.
        command (str): Edit operation to perform.
        old_string (Optional[str]): Text to replace (str_replace only).
        new_string (Optional[str]): Replacement text (str_replace, set).
        insert_text (Optional[str]): Text to insert (insert only).
        insert_line (int): Line index for insert (-1 appends).

    Returns:
        dict: Status, file_id, char_count, command, and processing_status while re-ingest runs.
    """
    raise NotImplementedError("Tool not implemented. Please contact the Letta team.")


async def file_grep(agent_state: "AgentState", file_id: str, pattern: str, max_hits: int = 20) -> dict:
    """
    Search within a single file for a pattern.

    Args:
        file_id (str): The file ID to search.
        pattern (str): Regex or literal pattern to match.
        max_hits (int): Maximum number of hits to return.

    Returns:
        dict: Matching lines with character offsets.
    """
    raise NotImplementedError("Tool not implemented. Please contact the Letta team.")


async def update_file_headline(agent_state: "AgentState", file_id: str, new_summary: str) -> dict:
    """
    Update the shared file headline (a few sentences describing what the file is).

    Args:
        file_id (str): The file ID whose headline to update.
        new_summary (str): New short headline text (shared across agents; shown in directory listings).

    Returns:
        dict: Updated file headline block metadata.
    """
    raise NotImplementedError("Tool not implemented. Please contact the Letta team.")


async def write_file_note(
    agent_state: "AgentState", file_id: str, title: str, content: str, tags: Optional[List[str]] = None
) -> dict:
    """
    Write a file reading note linked to a file.

    Args:
        file_id (str): The file this note is about.
        title (str): Short title naming the topical focus.
        content (str): Note body (1-8000 characters).
        tags (Optional[List[str]]): Optional tags for later search.

    Returns:
        dict: Created file note metadata including stored tags.
    """
    raise NotImplementedError("Tool not implemented. Please contact the Letta team.")


async def file_notes_search(
    agent_state: "AgentState",
    query: str,
    file_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    limit: int = 10,
) -> dict:
    """
    Hybrid search over file reading notes.

    Args:
        query (str): Natural-language search query.
        file_id (Optional[str]): Limit search to one file's notes.
        tags (Optional[List[str]]): Filter by note tags.
        limit (int): Maximum results to return.

    Returns:
        dict: Ranked file note hits with provenance metadata.
    """
    raise NotImplementedError("Tool not implemented. Please contact the Letta team.")


async def write_file_archive(
    agent_state: "AgentState", file_id: str, title: str, content: str, tags: Optional[List[str]] = None
) -> dict:
    """
    Deprecated alias for write_file_note.

    Args:
        file_id: The file this note is about.
        title: Short title naming the topical focus.
        content: Note body (1-8000 characters).
        tags: Optional tags for later search.

    Returns:
        Created file note metadata including stored tags.
    """
    raise NotImplementedError("Tool not implemented. Please contact the Letta team.")


async def file_archives_search(
    agent_state: "AgentState",
    query: str,
    file_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    limit: int = 10,
) -> dict:
    """
    Deprecated alias for file_notes_search.

    Args:
        query: Natural-language search query.
        file_id: Limit search to one file's notes.
        tags: Filter by note tags.
        limit: Maximum results to return.

    Returns:
        Ranked file note hits with provenance metadata.
    """
    raise NotImplementedError("Tool not implemented. Please contact the Letta team.")


async def file_contents_search(agent_state: "AgentState", query: str, limit: int = 5) -> dict:
    """
    Hybrid search over ingested file passages (folder RAG).

    Args:
        query (str): Natural-language search query.
        limit (int): Maximum passages to return.

    Returns:
        dict: Matching passage excerpts with file_id, passage_id, and score.
    """
    raise NotImplementedError("Tool not implemented. Please contact the Letta team.")


async def search_file_archives(
    agent_state: "AgentState",
    query: str,
    file_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    limit: int = 10,
) -> dict:
    """
    Deprecated alias for file_notes_search.

    Args:
        query: Natural-language search query.
        file_id: Limit search to one file's notes.
        tags: Filter by note tags.
        limit: Maximum results to return.

    Returns:
        Ranked file archive hits with provenance metadata.
    """
    raise NotImplementedError("Tool not implemented. Please contact the Letta team.")


async def search_file_contents(agent_state: "AgentState", query: str, limit: int = 5) -> dict:
    """
    Deprecated alias for file_contents_search.

    Args:
        query: Natural-language search query.
        limit: Maximum passages to return.

    Returns:
        Matching passage excerpts with file_id, passage_id, and score.
    """
    raise NotImplementedError("Tool not implemented. Please contact the Letta team.")
