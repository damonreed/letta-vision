import pytest

from letta.constants import FILES_TOOLS
from letta.functions.functions import load_function_set
from letta.services.file_text import (
    _apply_insert,
    _apply_str_replace,
    edit_file_text,
    is_text_editable_file,
    schedule_file_text_reingest,
)
import letta.functions.function_sets.files as files_module


class _FileMeta:
    def __init__(self, *, file_type="text/plain", file_name="notes.txt", content="hello world", source_id="source-1"):
        self.id = "file-abc"
        self.file_type = file_type
        self.file_name = file_name
        self.original_file_name = file_name
        self.content = content
        self.source_id = source_id


class _AgentState:
    id = "agent-1"


def test_file_edit_text_in_files_tool_set():
    assert "file_edit_text" in FILES_TOOLS


def test_file_edit_text_function_schema():
    schemas = load_function_set(files_module)
    assert "file_edit_text" in schemas
    params = schemas["file_edit_text"]["json_schema"]["parameters"]["properties"]
    assert {"file_id", "command"}.issubset(params.keys())


def test_is_text_editable_file_accepts_plain_text_mime():
    assert is_text_editable_file(file_type="text/plain", file_name="data.bin") is True


def test_is_text_editable_file_accepts_md_extension():
    assert is_text_editable_file(file_type="application/octet-stream", file_name="readme.md") is True


def test_is_text_editable_file_rejects_pdf():
    assert is_text_editable_file(file_type="application/pdf", file_name="doc.pdf") is False


def test_apply_str_replace_unique_match():
    assert _apply_str_replace("alpha beta", "beta", "gamma") == "alpha gamma"


def test_apply_str_replace_requires_unique_match():
    with pytest.raises(ValueError, match="Multiple occurrences"):
        _apply_str_replace("aa", "a", "b")


def test_apply_insert_appends_by_default():
    assert _apply_insert("line one", "line two", -1) == "line one\nline two"


@pytest.mark.asyncio
async def test_edit_file_text_str_replace_persists_and_schedules_reingest(monkeypatch):
    file_meta = _FileMeta(content="alpha beta")
    upserted = []
    clamped = []
    reingest_calls = []

    class _FileManager:
        async def get_file_by_id(self, file_id, actor, *, include_content=False, strip_directory_prefix=False):
            return file_meta

        async def upsert_file_content(self, *, file_id, text, actor):
            upserted.append((file_id, text))
            file_meta.content = text
            return file_meta

    class _OpenFilesManager:
        async def clamp_cursors_for_file(self, *, file_id, max_char, actor):
            clamped.append((file_id, max_char))
            return 1

    async def _schedule_reingest(**kwargs):
        reingest_calls.append(kwargs)

    monkeypatch.setattr("letta.services.file_text.FileManager", lambda: _FileManager())
    monkeypatch.setattr("letta.services.file_text.AgentOpenFilesManager", lambda: _OpenFilesManager())
    monkeypatch.setattr("letta.services.file_text.schedule_file_text_reingest", _schedule_reingest)

    result = await edit_file_text(
        file_id="file-abc",
        command="str_replace",
        actor=None,
        agent_state=_AgentState(),
        old_string="beta",
        new_string="gamma",
    )

    assert result["status"] == "success"
    assert result["char_count"] == len("alpha gamma")
    assert upserted == [("file-abc", "alpha gamma")]
    assert clamped == [("file-abc", len("alpha gamma"))]
    assert reingest_calls[0]["file_id"] == "file-abc"


@pytest.mark.asyncio
async def test_edit_file_text_rejects_non_text_file(monkeypatch):
    class _FileManager:
        async def get_file_by_id(self, file_id, actor, *, include_content=False, strip_directory_prefix=False):
            return _FileMeta(file_type="application/pdf", file_name="doc.pdf", content="%PDF")

    monkeypatch.setattr("letta.services.file_text.FileManager", lambda: _FileManager())

    with pytest.raises(ValueError, match="not a text-editable file"):
        await edit_file_text(
            file_id="file-abc",
            command="set",
            actor=None,
            agent_state=_AgentState(),
            new_string="replacement",
        )


@pytest.mark.asyncio
async def test_schedule_file_text_reingest_deletes_passages_and_reprocesses(monkeypatch):
    deleted = []
    processed = []
    file_meta = _FileMeta(content="updated body")

    class _FileManager:
        async def update_file_status(self, **kwargs):
            return file_meta

        async def get_file_by_id(self, file_id, actor, *, include_content=False, strip_directory_prefix=False):
            return file_meta

    class _PassageManager:
        async def list_passages_by_file_id_async(self, file_id, actor):
            return [object()]

        async def delete_source_passages_async(self, actor, passages):
            deleted.append(passages)

    class _SourceManager:
        async def list_attached_agents(self, source_id, actor):
            return []

    class _FileProcessor:
        def __init__(self, **kwargs):
            pass

        async def process_imported_file(self, file_metadata, source_id):
            processed.append((file_metadata.content, source_id))
            return []

    monkeypatch.setattr("letta.services.file_text.FileManager", lambda: _FileManager())
    monkeypatch.setattr("letta.services.passage_manager.PassageManager", lambda: _PassageManager())
    monkeypatch.setattr("letta.services.source_manager.SourceManager", lambda: _SourceManager())
    monkeypatch.setattr("letta.services.file_processor.file_processor.FileProcessor", _FileProcessor)
    monkeypatch.setattr("letta.services.file_processor.parser.mistral_parser.MistralFileParser", lambda: object())
    monkeypatch.setattr("letta.services.file_processor.parser.markitdown_parser.MarkitdownFileParser", lambda: object())
    monkeypatch.setattr("letta.services.file_processor.embedder.openai_embedder.OpenAIEmbedder", lambda **kwargs: object())
    monkeypatch.setattr("letta.helpers.pinecone_utils.should_use_pinecone", lambda: False)
    async def _resolve_embedding(actor):
        return object()

    monkeypatch.setattr(
        "letta.embeddings.resolver.resolve_deployment_embedding_config_async",
        _resolve_embedding,
    )

    import letta.services.file_text as file_text_module

    pending = []

    def _capture_task(coro, label=None):
        pending.append(coro)

    monkeypatch.setattr(file_text_module, "safe_create_task", _capture_task)

    await schedule_file_text_reingest(
        file_id="file-abc",
        source_id="source-1",
        agent_state=_AgentState(),
        actor=None,
    )

    assert len(pending) == 1
    await pending[0]

    assert len(deleted) == 1
    assert processed == [("updated body", "source-1")]
