# Release notes — v0.5.0 (letta-vision)

**Theme:** Three-tier filesystem memory — read content in history, stable headlines, searchable file reading notes.

## Highlights

- **Three-tier memory** — `file_core_blocks` headlines in `<directories>`; read pages via tool results; agent-written `file_archives` with semantic search.
- **Seven read/nav tools** — `CharPageReader` code-point paging, `file_grep`, cursor persistence in `agent_open_files`.
- **`add_text_file`** — agents create text files in attached folders with optional headline and async ingestion.
- **Live system refresh** — open/close/attach/detach/headline mutations recompile `<open_files>` and `<directories>` automatically.
- **Instruction rewrite** — `letta_v1.py` memory terminology, retrieval order, E2B sandbox docs; tool rename to `update_file_headline`, `write_file_archive`, `search_file_archives`.

## Upgrade from v0.4.0

```bash
git checkout v0.5.0
docker build -t letta-vision:v0.5.0 -t letta-vision:latest .
```

Pair with **letta-vision-client v0.5.0** and **letta-vision-deploy v0.5.0**.

Set `LETTA_VERSION=0.5.0` in deploy `.env` (health endpoint).

### Migration steps

1. Start server — Alembic migration `f1a2b3c4d5e6` runs automatically on container boot.
2. Optional backfill: `python scripts/backfill_file_core_blocks.py` (seeds headlines from legacy `visible_content`).
3. Refresh agent prompts: `python scripts/refresh_letta_v1_system_prompts.py` (conversation-scoped recompile).
4. Rebuild client stack: `docker compose up -d --build` from `letta-vision-deploy`.

Existing agents receive new file tools via `attach_missing_files_tools_async` on next sync.

## Verification

```bash
pytest tests/test_three_tier_memory_compile.py tests/test_char_page_reader.py \
  tests/test_add_text_file_tool.py tests/test_archive_tags.py tests/test_resolve_file_id.py
```

## Operator notes

- Phase 7 (drop `visible_content` columns, remove legacy tool names) deferred to v0.5.1.
- File headlines in `<directories>` cost system-prompt tokens every turn — keep them to a few sentences.
- Executor aliases (`update_file_core`, `write_archive`, `search_archives`) remain for pre-migration agents.

## Documentation

- [IMPLEMENTATION_REPORT_v0.5.0_three-tier-memory.md](IMPLEMENTATION_REPORT_v0.5.0_three-tier-memory.md) — full report for Ada.
- [CHANGELOG](../CHANGELOG.md) — complete change list.
