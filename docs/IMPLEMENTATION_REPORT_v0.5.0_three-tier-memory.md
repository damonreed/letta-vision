# Implementation Report: Three-Tier Filesystem Memory (v0.5.0 candidate)

**To:** Ada  
**From:** Damon (letta-stack)  
**Date:** 2026-05-23  
**Release candidate:** `v0.5.0` across `letta-vision`, `letta-vision-client`, `letta-vision-deploy`  
**Baseline:** `v0.4.0` (LLM failure handling, image persistence, MCP)  
**Specification:** [FR: Filesystem Redesign — Three-Tier Memory Hierarchy](FR_letta-vision_Filesystem-Redesign-Three-Tier-Memory.md)

---

## Executive summary

We implemented Ades's three-tier filesystem memory design as specified in the FR, through **Phases 1–6**. The core property holds: **read content lives in conversation history; file headlines live in stable, shared core blocks; archives are agent-written, semantically searchable topical notes.** Legacy `FileBlock` page injection is removed from the compile path; `visible_content` is no longer written on read.

Extended testing with Lyra (agent `agent-35a1c263-f1f4-4855-95a1-ad760b3cc414`, conversation **v0.5.0 Testing**) surfaced several integration bugs and UX gaps. Those are fixed in-tree. The web client gained file-memory visibility, a system-context inspector, and **chat stream self-recovery** so frozen SSE sessions no longer require a full page reload.

**Phase 7 (legacy tool removal and column drop) is intentionally deferred** to a follow-on minor release after agents are migrated off old tool names.

This report is submitted for your review. If the deviations and open items below are acceptable, we will tag **v0.5.0**.

---

## Scope delivered vs FR

| FR phase | Deliverable | Status |
|----------|-------------|--------|
| **Phase 1** — Data model | Migration `f1a2b3c4d5e6`, ORM models, managers, backfill script | **Done** |
| **Phase 2** — System prompt compilation | `<open_files>` section, directories without page content, `refresh_open_file_cores()` | **Done** (with §7 deviation — see below) |
| **Phase 3** — Read tools | `CharPageReader`, seven read/nav tools + `search_file_contents` rename | **Done** |
| **Phase 4** — Headline editing | `update_file_core`, REST `/v1/file-memory/*` | **Done** |
| **Phase 5** — Archives | `write_archive`, `search_archives`, tag normalization, provenance | **Done** |
| **Phase 6** — Agent instructions | `letta_v1.py` base instructions, `FILES_TOOLS` registry | **Done** (iterated post-FR) |
| **Phase 7** — Deprecation | Remove `open_files` / `grep_files` / `semantic_search_files`; drop `visible_content` columns | **Deferred** |

### Acceptance criteria (§10)

| # | Criterion | Status | Notes |
|---|-----------|--------|-------|
| 1 | Cross-turn reading recall | **Pass** (empirical) | Lyra reads multiple pages; page 1 reachable in later turns via tool-result history |
| 2 | Headline stability across reads | **Pass** | Headline unchanged unless `update_file_core` called |
| 3 | Shared headline editing | **Pass** | `file_core_blocks` is file-scoped, not agent-scoped |
| 4 | Horizontal archive search finds file | **Pass** | Provenance includes `file_id`, `file_name` |
| 5 | Vertical archive search scoped | **Pass** | `file_id` filter enforced in manager |
| 6 | Tag search spans files | **Pass** | GIN index on JSONB tags |
| 7 | Provenance preserved | **Pass** | `author_agent_id`, `source_conversation_id`, `created_at` on every hit |
| 8 | Close detaches open slot | **Pass** | Hard-delete `agent_open_files` row; headline remains in directories |
| 9 | System prompt has no page content | **Pass** | Covered by `tests/test_three_tier_memory_compile.py` |
| 10 | Provider prompt cache across reads | **Not measured** | Design supports it; no automated cache-hit benchmark yet |
| 11 | LRU at `max_files_open` | **Pass** | `AgentOpenFilesManager.open_file` evicts LRU; returns `evicted_file_ids` |
| 12 | Lenient tag normalization | **Pass** | Per-tag reject; archive still committed |
| 13 | Two search tools, two surfaces | **Pass** | `search_archives` → `file_archives`; `search_file_contents` → `source_passages` |

Automated coverage: `tests/test_three_tier_memory_compile.py`, `tests/test_char_page_reader.py`, archive/core manager tests, SSE coalescing tests (client bridge).

---

## Architecture (server — `letta-vision`)

### Data model

Migration **`f1a2b3c4d5e6`** creates:

- **`file_core_blocks`** — one headline per file (`file_id` PK/FK to `files.id`)
- **`file_core_block_history`** — append-only edit history
- **`agent_open_files`** — per-agent open state + UTF-8 code-point cursor
- **`file_archives`** — topical notes with pgvector embedding + JSONB tags

Design decisions preserved from the FR:

- Prefixed string IDs via existing ORM mixins
- `OrganizationMixin` on all new tables
- Postgres-only migration (SQLite skipped when no PG URI)
- Embeddings reuse `MAX_EMBEDDING_DIM = 4096` with right-padding
- **No HNSW index on archives** — pgvector HNSW caps at 2000 dimensions; same precedent as `source_passages` / `archival_passages`. Cosine scan at archive scale is acceptable for v1.

### Managers and tool surface

| Module | Role |
|--------|------|
| `FileCoreBlockManager` | CRUD + history for shared headlines |
| `AgentOpenFilesManager` | Open/close/LRU, cursor persistence |
| `FileArchiveManager` | Write, embed, search (horizontal / vertical / tag-scoped) |
| `ThreeTierFileTools` | Tool handlers — **read tools never mutate blocks** |
| `CharPageReader` | Code-point paging, grep line mapping, cursor math |

Registered tools (`FILES_TOOLS` in `letta/constants.py`):

```
attach_folder, detach_folder, open_file, close_file,
file_read_page, file_read_next_page, file_read_prev_page, file_read_range, file_grep,
update_file_core, write_archive, search_archives, search_file_contents
```

Legacy names (`open_files`, `grep_files`, `semantic_search_files`) are **not** in `FILES_TOOLS` but executor stubs may still exist until Phase 7.

### System prompt compilation

Two rendered surfaces:

1. **`<directories>`** — metadata for every file in attached folders (name, `file_id`, `source_id`, open/closed status). Each file shows its **file core headline** whether open or closed.
2. **`<open_files>`** — only files currently open for reading (headline + cursor position). No page content in either section.

`AgentManager.refresh_open_file_cores()` loads:

- `open_file_cores` for the `<open_files>` block
- `file_core_summaries` dict for all attached files (directories headlines)

### Conversation-scoped system messages

Multi-conversation support required a compile hook that uses the **actual conversation ID**, not the agent dry-run default:

- **`ConversationManager.recompile_system_message_for_conversation()`** — loads agent state, calls `refresh_open_file_cores`, compiles with `conversation_id=<named conv>`, persists system message.
- Conversation recompile REST endpoint and `scripts/refresh_letta_v1_system_prompts.py` use this path.

This fixes the bug where named conversations showed `CONVERSATION_ID: default` in the system header.

### Live system refresh after file-state mutations

File tools that change what appears in system context set a pending refresh flag in **`LettaAgentV3`**:

```python
FILE_STATE_SYSTEM_REFRESH_TOOLS = {
    attach_folder, detach_folder, open_file, close_file, update_file_core
}
```

After successful execution, `_pending_file_system_refresh` triggers `refresh_open_file_cores` + system recompile on the next message refresh cycle. Without this, `<open_files>` and `<directories>` could stay stale until manual recompile.

### Agent instructions (`letta_v1.py`)

Base instructions replaced per FR §7, with **two empirical refinements** after Lyra testing (see Deviations).

---

## Architecture (client — `letta-vision-client`)

### File memory API bridge

`backend/routes/file_memory.py` proxies Letta `/v1/file-memory/*`:

- `GET/PATCH /api/files/{file_id}/core`
- `GET /api/agents/{agent_id}/open-files`
- `POST /api/agents/{agent_id}/open-files/{file_id}/close`
- `GET /api/files/{file_id}/archives`
- `POST /api/file-archives/search`

### UI surfaces

| Surface | Purpose |
|---------|---------|
| **`Files.svelte`** | Folder upload/management (existing, extended) |
| **`AgentFiles.svelte`** | Per-agent attached folders + open-files panel |
| **`Chat.svelte` — memory sidebar** | Blocks + open files quick view |
| **`SystemContextModal.svelte`** | Inspect compiled system prompt; refreshes after streams |
| **Context button in chat header** | Opens modal; pulls fresh history on open |

System message is **hidden from the chat transcript** (still loaded for the modal and token context).

### Chat stream robustness (post-FR operational work)

Observed during long file-tool runs and deploy restarts: chat could freeze with `streaming=true` and a disabled composer.

Implemented in `frontend/src/lib/chatStreamRecovery.js` + `Chat.svelte`:

| Mechanism | Behavior |
|-----------|----------|
| **Stall watchdog** | 90s without SSE activity → abort + sync history from server |
| **Max duration** | 10 min cap (matches bridge read timeout) |
| **Keepalive forwarding** | Letta `ping` chunks now emit `{type: "keepalive"}` through bridge SSE (`backend/sse.py`) |
| **Abrupt disconnect** | `stream_end` synthetic event when HTTP body closes without `done` |
| **Post-stream sync** | Always reload conversation history after stream ends — server is source of truth |
| **Cancel / Refresh** | User-facing cancel while streaming; "Refresh chat" on error banner |
| **Tab / network recovery** | Re-sync after 30s hidden tab or `online` event while streaming |

This is not in the FR but is required for reliable extended agent testing.

---

## Deviations from FR §7 (intentional)

These emerged from Lyra's v0.5.0 Testing session and are **documented here for approval**, not accidental drift.

### 1. File cores in `<directories>` for all files (open and closed)

**FR §7** implied file core blocks attach to system context **only while open** (`<open_files>`).

**Shipped behavior:** Every file in attached folders shows its few-sentence headline inside `<directories>`, open or closed. `<open_files>` remains the active-reading slot (cursor, LRU count).

**Rationale:** Agents were opening files just to discover headlines already visible in folder listings. Surfacing cores in directories makes discovery cheap without injecting page content. `open_file` is reframed as "mark active for paging" rather than "load headline."

**Instruction change:** `letta_v1.py` now says *"Every file in your directory listing shows its file core, open or closed"* and *"Opening a file marks it active for reading."*

### 2. "Few sentences" headline doctrine

**FR §7** example used a single-sentence headline.

**Shipped behavior:** Instructions and validation messaging steer agents toward **a few sentences** describing what the file *is* (not a content summary). `char_limit` default remains 2000.

**Rationale:** Single-sentence headlines were too thin for Lyra to disambiguate similarly named files in large folders.

### 3. `search_file_contents` signature

**FR §7** listed `search_file_contents(query, folder_id=None)`.

**Shipped:** `search_file_contents(query)` — searches across all attached folder passages (existing RAG behavior). Optional folder scoping can be added if empirical use shows agents need it.

---

## Post-FR fixes (Lyra testing)

Issues found during live testing and resolved before this report:

| Issue | Root cause | Fix |
|-------|------------|-----|
| **`file_read_next_page` skipped page 2** | `CharPageReader.next_page_cursor` treated post-read boundary as "already on next page" and jumped an extra page | Boundary-aware cursor: if cursor is already on a page boundary after `file_read_page`, do not advance again |
| **`update_file_core` timezone error** | Naive vs aware datetime comparison on `last_updated_at` | Normalize to UTC-aware timestamps in manager |
| **`search_archives` crash** | Invalid `noload(FileArchive.organization)` — archives have no org relationship | Removed erroneous eager-load option |
| **`detach_folder` missing source ID** | Directory tags lacked `source_id` for agents to pass to detach | Added `source_id` attribute on `<directory>` tags in `Memory.compile` |
| **40K preview limit ignored** | Stale `per_file_view_window_char_limit` on in-memory agent state during tool loop | Re-fetch limit from DB before file read tools; return `page_size_chars` in tool responses |
| **System context stale after file tools** | No recompile after open/close/attach/detach/core update | `FILE_STATE_SYSTEM_REFRESH_TOOLS` + `_pending_file_system_refresh` in v3 agent |
| **`CONVERSATION_ID: default` in named convs** | Recompile used agent dry-run path | `recompile_system_message_for_conversation()` |
| **Wrong file ID (UUID typo)** | Agent copied `file-8b0f61cf-…` with one char wrong | `FileAgentManager.resolve_file_id_for_agent()` — name/path match + single-edit-distance UUID correction |
| **Reads failed after resolution** | Resolution path checked `FileMetadata.is_deleted` on Pydantic schema | Removed invalid attribute access; join through `files` table in `list_files_for_agent` |
| **Chat frozen after stream death** | No stall detection; pings dropped; no post-stream history sync | Stream recovery module (see above) |

---

## Migration and ops

### Backfill

`scripts/backfill_file_core_blocks.py`:

- Seeds `file_core_blocks.summary` from legacy `files_agents.visible_content` (first line / truncated)
- Skips files with no seed content (logged; agent generates headline on first open)
- LLM one-shot summarization from first pages is **stubbed for unreachable local LLMs** — matches FR §9 default

### Bulk prompt refresh

`scripts/refresh_letta_v1_system_prompts.py` recompiles all conversations using the conversation-scoped path so existing agents pick up v0.5.0 instructions and directory/core rendering.

### Deploy

`letta-vision-deploy` docker compose: run migration on `letta-vision` container start, rebuild both images after pull. Lyra's agent required one manual refresh after deploy to drop stale tool/instruction references from pre-v0.5.0 state.

---

## Phase 7 deferral (recommended for v0.5.1)

Not blocking v0.5.0 tag:

1. Remove legacy tool executors (`open_files`, `grep_files`, `semantic_search_files`) from active registry
2. Stop writing `files_agents.visible_content` entirely (reads already bypass it)
3. Alembic migration dropping `visible_content`, `start_line`, `end_line` columns
4. Update `context_window_calculator` token breakdown: `open_files` headline tokens replace directory page-content tokens

Agents created or refreshed after v0.5.0 already receive the new tool set via `attach_missing_files_tools_async`.

---

## Open items (non-blocking)

| Item | Recommendation |
|------|----------------|
| **Prompt cache hit measurement** (AC #10) | Add observability in a follow-up; design is cache-friendly |
| **Archive volume / curation** | FR §11.3 deferral stands — append-only for v1 |
| **Unified retrieval tool** | FR §11.7 deferral stands — observe agent search patterns first |
| **Client tool-result components** | Rich rendering for archive write confirmations, grep hit lists — cosmetic |
| **Stall timeout tuning** | 90s default; may increase for slow local models if false positives appear |

---

## Empirical validation

**Lyra — v0.5.0 Testing** (`conv-c4ba2d6b-18eb-497c-87e2-1ee99be8d365`):

- Processed 10/11 attached files through new read/archive workflow
- Confirmed cross-turn recall, archive write + search, directory headlines, open-file LRU
- Surface bugs above filed and fixed same session
- Extended testing ongoing; Damon will refer additional issues

**Stability:** Chat stream recovery deployed; user reports stable behavior during continued testing.

---

## Recommendation

The implementation delivers the FR's central design property — **three tiers with distinct lifetimes and costs, read content in history, stable shared headlines, agent-written searchable archives** — through Phase 6, with documented instruction deviations that improved empirical agent behavior.

Phase 7 deprecation is cleanly separable. Chat robustness and system-context visibility are operator-quality additions appropriate for a self-hosted alpha.

**Proposed action:** Ada approves this report → tag **`v0.5.0`** on `letta-vision`, `letta-vision-client`, and `letta-vision-deploy` with release notes summarizing migration steps (run migration, backfill optional, refresh agent prompts, rebuild containers).

---

## Key file index

### Server

| Path | Role |
|------|------|
| `alembic/versions/f1a2b3c4d5e6_add_three_tier_filesystem_memory.py` | Schema |
| `letta/services/tool_executor/three_tier_file_tools.py` | Tool handlers |
| `letta/services/files/char_page_reader.py` | Paging |
| `letta/schemas/memory.py` | Compile: directories + open_files |
| `letta/prompts/system_prompts/letta_v1.py` | Agent instructions |
| `letta/server/rest_api/routers/v1/file_memory.py` | REST |
| `letta/services/conversation_manager.py` | Conversation-scoped recompile |
| `letta/agents/letta_agent_v3.py` | Pending file system refresh |
| `scripts/backfill_file_core_blocks.py` | Migration helper |

### Client

| Path | Role |
|------|------|
| `backend/routes/file_memory.py` | API bridge |
| `frontend/src/routes/Chat.svelte` | Chat + recovery + context modal |
| `frontend/src/lib/chatStreamRecovery.js` | Stall detection |
| `frontend/src/lib/SystemContextModal.svelte` | System prompt inspector |
| `frontend/src/lib/AgentFiles.svelte` | Agent filesystem panel |
| `backend/sse.py` | Keepalive forwarding |

---

*End of report.*
