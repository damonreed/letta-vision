import json
from pathlib import Path

from letta.services.migration.historic_reembed import (
    Part2Checkpoint,
    TableCursor,
    _file_archive_embed_text,
    _passage_embed_text,
    load_part2_checkpoint,
    resolve_tables,
    save_part2_checkpoint,
)


class _Row:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def test_file_archive_embed_text_joins_title_and_content():
    row = _Row(title="Notes", content="Body text")
    assert _file_archive_embed_text(row) == "Notes\n\nBody text"


def test_passage_embed_text_strips():
    row = _Row(text="  hello  ")
    assert _passage_embed_text(row) == "hello"


def test_resolve_tables_all():
    assert resolve_tables("all") == ["archival_passages", "source_passages", "file_archives"]


def test_part2_checkpoint_roundtrip(tmp_path: Path):
    path = tmp_path / "part2.json"
    ckpt = Part2Checkpoint(
        organization_id="org-1",
        target_space_id="space-1",
        tables={"source_passages": TableCursor(last_id="p-1", processed=3, succeeded=3)},
    )
    save_part2_checkpoint(path, ckpt)
    loaded = load_part2_checkpoint(path)
    assert loaded is not None
    assert loaded.organization_id == "org-1"
    assert loaded.tables["source_passages"].last_id == "p-1"
    assert loaded.tables["source_passages"].processed == 3
    assert json.loads(path.read_text())["target_space_id"] == "space-1"
