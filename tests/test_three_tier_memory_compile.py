"""Tests for three-tier memory prompt compilation."""

from letta.schemas.file_core_block import OpenFileCoreView
from letta.schemas.memory import Memory
from letta.schemas.block import Block


def test_open_files_section_in_compile():
    memory = Memory(
        blocks=[Block(label="persona", value="test", limit=2000)],
        open_file_cores=[
            OpenFileCoreView(
                file_id="file-abc",
                file_name="story.txt",
                source_id="source-xyz",
                summary="A stable headline.",
                cursor_char=4500,
                total_chars=12000,
            )
        ],
    )
    compiled = memory.compile(sources=[], page_size_chars=1000)
    assert "<open_files>" in compiled
    assert "A stable headline." in compiled
    assert 'cursor="4500/12000"' in compiled
    assert 'page="5/12"' in compiled
    assert 'name="story.txt"' in compiled


def test_open_files_section_without_page_size_omits_page_attr():
    memory = Memory(
        blocks=[Block(label="persona", value="test", limit=2000)],
        open_file_cores=[
            OpenFileCoreView(
                file_id="file-abc",
                file_name="story.txt",
                source_id="source-xyz",
                summary="A stable headline.",
                cursor_char=100,
                total_chars=500,
            )
        ],
    )
    compiled = memory.compile(sources=[])
    assert 'cursor="100/500"' in compiled
    assert 'page="' not in compiled


def test_directories_exclude_page_content():
    memory = Memory(
        blocks=[Block(label="persona", value="x", limit=2000)],
        file_blocks=[],
        open_file_cores=[],
    )

    class Source:
        id = "source-1"
        name = "docs"
        description = None
        instructions = None

    from letta.schemas.block import FileBlock

    memory.file_blocks = [
        FileBlock(
            label="chapter.txt",
            value="SECRET PAGE CONTENT",
            file_id="file-1",
            source_id="source-1",
            is_open=False,
            limit=5000,
        )
    ]
    memory.file_core_summaries = {"file-1": "A short few-sentence description of the chapter."}
    compiled = memory.compile(sources=[Source()])
    assert "SECRET PAGE CONTENT" not in compiled
    assert "chapter.txt" in compiled
    assert "A short few-sentence description of the chapter." in compiled
    assert '<file status="closed"' in compiled


def test_directories_include_open_file_headline():
    memory = Memory(
        blocks=[Block(label="persona", value="x", limit=2000)],
        file_blocks=[],
        open_file_cores=[
            OpenFileCoreView(
                file_id="file-open",
                file_name="open.txt",
                source_id="source-1",
                summary="Open file headline.",
                cursor_char=100,
            )
        ],
        file_core_summaries={"file-open": "Open file headline.", "file-closed": "Closed file headline."},
    )

    class Source:
        id = "source-1"
        name = "docs"
        description = None
        instructions = None

    from letta.schemas.block import FileBlock

    memory.file_blocks = [
        FileBlock(
            label="open.txt",
            value="",
            file_id="file-open",
            source_id="source-1",
            is_open=True,
            limit=5000,
        ),
        FileBlock(
            label="closed.txt",
            value="",
            file_id="file-closed",
            source_id="source-1",
            is_open=False,
            limit=5000,
        ),
    ]
    compiled = memory.compile(sources=[Source()])
    assert "Open file headline." in compiled
    assert "Closed file headline." in compiled
    assert '<file status="open"' in compiled
    assert '<file status="closed"' in compiled
