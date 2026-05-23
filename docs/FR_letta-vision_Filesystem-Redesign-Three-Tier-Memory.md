# FR: Filesystem Redesign — Three-Tier Memory Hierarchy

**Author:** Ada (with Damon)
**Date:** 2026-05-23
**Revised:** 2026-05-23 (post Cursor implementation-plan review)
**Status:** Draft — for Cursor implementation
**Target repo:** `damonreed/letta-vision`
**Depends on:** Existing Letta Filesystem (folders, files, source passages), pgvector-backed Postgres, memory block system

---

## 1. Problem

Letta's current filesystem treats file content as mutable state injected into the system prompt via `FileBlock`, populated from `files_agents.visible_content`. The block is recompiled into the system prompt on every agent step. When an agent calls `open_files(...)`, the relevant block's `visible_content` is replaced with the requested page view; the prior page disappears from any surface the agent can reach. The conversation history records that an `open_files` call occurred but does not carry the page content — content was delivered via block mutation, not via a tool result message.

This produces three observable failures:

1. **Cross-turn amnesia within a file.** An agent that reads page 1, then page 2, cannot answer a question about page 1 on turn 3. The content was overwritten in the block and was never in conversation history.
2. **Aggressive eviction across files.** When `max_files_open` is reached, the least-recently-accessed file block is closed entirely. Its content disappears from context with no agent-facing recovery path.
3. **Cache invalidation.** Mutating system-context blocks on every navigation breaks provider prompt caching. Multi-turn reading sessions pay full input cost on every turn.

The underlying mismatch: Letta's filesystem inherits MemGPT's "LLM as OS, context as RAM, memory as paging" metaphor, but modern models are post-trained on a conversational paradigm where tool results persist in the message stream. The agent's expectation and the system's behavior diverge.

This FR redesigns the filesystem around a three-tier memory hierarchy where each tier has a distinct cost, role, and lifetime. The redesign is text-file scoped. Image-file extensions are out of scope here but the design preserves the symmetry needed to add them later as a small diff.

---

## 2. Goals

1. **Read content persists in conversation history**, retrievable for the rest of the turn and subsequent turns until compaction.
2. **Each file has one stable headline** (file core memory block) attached to system context only while the file is open.
3. **Each file has zero or more archives** — topical notes written by agents in the voice and context of past conversations, semantically searchable.
4. **Three search axes**: horizontal (no scope, across all archives), vertical (within a file's archives), tag-scoped (cross-cutting topics).
5. **Agent-driven archive creation.** Archives are written by the main LLM in conversation, not by a background summarizer.
6. **Shared file core memory across agents.** When one agent edits a file's headline, other agents see the edit. Archives are also shared and surface to any agent searching the same store.
7. **Preserve provider prompt caching** for multi-turn reading sessions by keeping the system prompt stable and putting variable content in append-only message history.

---

## 3. Conceptual Model

Three layers. Each has a different access cost and a different role. Agents are instructed to use each layer for what it is best at.

### 3.1 Memory blocks (always-in-context)

Labeled text containers compiled into the system prompt on every turn. Most expensive layer — every token costs on every turn. Two categories:

- **Agent-level blocks** (existing): `persona`, `human`, custom labels. Attached at agent creation. Always present.
- **File core memory blocks** (new): One per file in the system. Attached to an agent's system context only while the file is open. Detached when the file is closed.

A file core memory block holds a short, stable headline describing what the file fundamentally is. Example:

> *"This file contains the season summaries of 'Rogue's Harbor', an adventure story about how Lachance and Rhiannah met, fell in love, and finally were together."*

It does **not** hold file content. It does **not** change on navigation. It changes only when the agent's understanding of what the file is has materially shifted, via the `update_file_core` tool.

### 3.2 Archives (retrievable on demand)

Topical notes written during past interactions. Stored as rows in a new `file_archives` table with embeddings in pgvector. Not in any agent's system context by default. Retrieved via semantic search.

An archive is **not** a neutral summary of the file. It captures what a particular reading was about — what topic was being explored, what the user emphasized, what conclusions the conversation reached. Multiple archives can exist for the same section of a file, capturing different readings with different topical focuses. Example, for the file above:

> *"They met in Season 1, fell in love in Season 2, and finally got together in Season 3. Key points: their first meeting at the harbor docks was structured to echo the prologue's storm imagery; the obstacle in S2 was Lachance's prior engagement to Mireille, which the narrator foreshadows with the broken-locket motif in episodes 4 and 7..."*

Archives are written by the main LLM as part of its reasoning trace, not by a background summarizer. This is intentional: the model that participated in the conversation has full understanding of what mattered for *its* task and writes summaries that inherit that understanding.

### 3.3 Files (read on demand)

Documents in folders attached to an agent. Hold raw detail. Read page-by-page into the conversation via tool calls that return content as tool result messages. Read content persists in conversation history until compaction.

Two distinct retrieval surfaces exist over file content:

- **`search_archives`** — searches the archives *about* files (tier 2). Cheap entry point; what other agents have observed and concluded.
- **`search_file_contents`** — searches the raw content *of* files via the existing folder RAG / source-passage index. Folder ingestion already produces these passages; this FR just renames the tool to disambiguate it from archive search.

The agent uses `search_archives` first and escalates to `search_file_contents` or direct reading only when archives aren't enough. See §6 for search semantics.

---

## 4. Data Model

### 4.1 Schema conventions

All new tables follow existing Letta ORM conventions:

- **Prefixed string IDs** (e.g. `file_core-…`, `file_archive-…`, `agent_open_file-…`) via the existing ID mixin. Not raw UUIDs.
- **Nullable `organization_id`** via `OrganizationMixin` on every new table — unused in single-org deployments today, present so multi-tenant scoping doesn't require a future schema migration (FR §12.7 decision).
- **Postgres-only migrations** — skip SQLite, following the precedent set by existing file-related migrations.
- **Embeddings reuse the existing passage pipeline.** `MAX_EMBEDDING_DIM = 4096` with right-padding for shorter model outputs; same `vector` column type used by `source_passages` and `archival_passages`. Do not hardcode a dimensionality.

### 4.2 New tables

```sql
-- One row per file, holding the file's stable headline.
-- Shared across all agents that open the file.
CREATE TABLE file_core_blocks (
    id                       TEXT PRIMARY KEY,  -- prefixed: 'file_core-...'
    file_id                  TEXT NOT NULL UNIQUE REFERENCES files(id) ON DELETE CASCADE,
    summary                  TEXT NOT NULL,
    char_limit               INT  NOT NULL DEFAULT 2000,
    version                  INT  NOT NULL DEFAULT 1,
    last_updated_by_agent_id TEXT,
    last_updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at               TIMESTAMPTZ NOT NULL DEFAULT now(),
    organization_id          TEXT
);

CREATE TABLE file_core_block_history (
    id                  BIGSERIAL PRIMARY KEY,
    file_core_block_id  TEXT NOT NULL REFERENCES file_core_blocks(id) ON DELETE CASCADE,
    summary             TEXT NOT NULL,
    updated_by_agent_id TEXT,
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    organization_id     TEXT
);

-- Per-agent open-file state.
-- Tracks which files an agent currently has open and the read cursor.
-- cursor_char is a UTF-8 code-point offset (not a byte offset).
CREATE TABLE agent_open_files (
    agent_id         TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    file_id          TEXT NOT NULL REFERENCES files(id)  ON DELETE CASCADE,
    cursor_char      INT  NOT NULL DEFAULT 0,
    opened_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_accessed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    organization_id  TEXT,
    PRIMARY KEY (agent_id, file_id)
);

-- File archives. Topical notes linked to a file.
-- Named file_archives (not archives) to avoid collision with the
-- existing `archives` table, which holds archival-memory collections.
CREATE TABLE file_archives (
    id                     TEXT PRIMARY KEY,  -- prefixed: 'file_archive-...'
    file_id                TEXT NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    title                  TEXT NOT NULL,
    content                TEXT NOT NULL,
    tags                   TEXT[] NOT NULL DEFAULT '{}',
    author_agent_id        TEXT,
    source_conversation_id TEXT,
    embedding              vector,            -- dim matches agent's embedding_config (padded to 4096)
    embedding_config       JSONB NOT NULL,    -- snapshot of config used at insert time
    created_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
    organization_id        TEXT
);

CREATE INDEX file_archives_embedding_idx ON file_archives USING hnsw (embedding vector_cosine_ops);
CREATE INDEX file_archives_file_id_idx    ON file_archives (file_id);
CREATE INDEX file_archives_tags_gin       ON file_archives USING gin (tags);
CREATE INDEX file_archives_created_at_idx ON file_archives (created_at DESC);
```

### 4.3 What changes in the existing schema

The `FileBlock` / `files_agents.visible_content` compile path is deprecated. The file's headline now lives in `file_core_blocks`, joined to agents via `agent_open_files`. The system-prompt compiler reads `agent_open_files` for the current agent, joins to `file_core_blocks`, and renders the joined headlines in a labeled section of the prompt.

Existing passages (`source_passages`, `archival_passages`) are unaffected. `file_archives` is a separate table from the existing `archives` table because:
- The existing `archives` table holds archival-memory **collections** (a long-term knowledge store concept), not individual file notes.
- File archives carry stronger provenance (author agent, source conversation) than passages or archival-memory collections need.
- File archives have a mandatory `file_id` linkage; passages have it optionally; archival-memory collections don't have one at all.

The tool names `write_archive` and `search_archives` remain — they refer to the agent-facing concept, not the table name.

### 4.4 What read content looks like in messages

Read tools return tool result messages with this structure:

```json
{
  "tool_call_id": "...",
  "tool_name": "file_read_page",
  "result": {
    "file_id": "file-abc123",
    "char_range": [0, 5000],
    "page_number": 1,
    "total_pages": 47,
    "content": "...file content here..."
  }
}
```

These messages live in conversation history. They are not duplicated into any system block. Compaction may eventually elide them via standard message-history compaction; that is out of scope for this FR (see §11).

---

## 5. Tool Surface

All tools below replace the existing filesystem tools (`open_files`, `grep_files`, etc.) one-for-one or with renaming. The new tools should be attached automatically to any agent with at least one folder attached, matching current behavior.

### 5.1 Folder binding

Existing behavior; wrapped as agent tools so the agent can attach/detach without REST calls.

```
attach_folder(folder_id: str) -> {status, files: [{id, name, description}]}
detach_folder(folder_id: str) -> {status}
```

### 5.2 File open/close

```
open_file(file_id: str) -> {
    status,
    file_id,
    headline: str,         # the file core memory content
    total_chars: int,
    total_pages: int,
    cursor_char: int,      # 0 on first open; resets to 0 on reopen after close
}
```

Effects:
- Inserts (or refreshes `last_accessed_at` on) a row in `agent_open_files`.
- The file's headline becomes visible in system context on the next prompt compile.
- LRU enforcement: if opening would exceed `max_files_open`, the least-recently-accessed open file is closed first.
- Does **not** read any file content.

```
close_file(file_id: str) -> {status}
```

Effects:
- Hard-deletes the row from `agent_open_files` (FR §12.1 decision).
- Headline drops out of system context on the next compile.
- Cursor is not preserved; reopen starts at 0.

### 5.3 Reading

All read tools return content in the tool result message body. None of them mutate any memory block. All offsets are UTF-8 **code-point** indices (not byte offsets) so reads never split a codepoint.

```
file_read_page(file_id: str) -> {
    file_id,
    char_range: [start, end],
    page_number: int,
    total_pages: int,
    content: str,
}
```

Reads the page at the current cursor and advances the cursor to the next page. Page size is `per_file_view_window_char_limit` (existing config, default 5000 chars).

```
file_read_next_page(file_id: str) -> {...same shape...}
file_read_prev_page(file_id: str) -> {...same shape...}
```

Navigation: returns the next/prev page and updates the cursor.

```
file_read_range(file_id: str, start_char: int, end_char: int) -> {
    file_id,
    char_range: [start, end],
    content: str,
}
```

Read an arbitrary character range. Does **not** update the cursor (this is for power-user spot-reads, not navigation). Range size is capped at `per_file_view_window_char_limit * 2` to prevent context blowup. Range endpoints are clamped to valid codepoint boundaries.

```
file_grep(file_id: str, pattern: str, max_hits: int = 20) -> {
    file_id,
    hits: [
        {char_offset: int, line_number: int, snippet: str},
        ...
    ],
}
```

Regex or literal pattern search within a file. Returns hits with surrounding context snippets (~200 chars per snippet). Does not update the cursor. Agents can `file_read_range` around a hit if they want full context.

### 5.4 Headline editing

```
update_file_core(file_id: str, new_summary: str) -> {
    status,
    file_id,
    previous_summary: str,
    new_summary: str,
    version: int,
}
```

Replaces the file's headline. **This mutates shared state visible to all agents who open this file.** Use deliberately. Should be called only when the agent's understanding of what the file fundamentally is has materially changed.

Effects:
- Writes new content to `file_core_blocks.summary`.
- Appends a row to `file_core_block_history`.
- Increments `version`.
- Sets `last_updated_by_agent_id` and `last_updated_at`.

Enforces `char_limit` on `new_summary`.

### 5.5 Archive writing

```
write_archive(
    file_id: str,
    title: str,
    content: str,
    tags: list[str] = [],
) -> {
    status,
    archive_id,
    file_id,
    title,
    tags_accepted: list[str],   # post-normalization, after rejecting overlong tags
    tags_rejected: list[str],   # tags dropped because they exceeded the length limit
}
```

Commits a topical archive linked to a file.

**Effects:**
- Normalizes tags (see below).
- Generates embedding from `content` using the agent's configured embedding model (must match the model used by attached folders).
- Inserts a row in `file_archives` with `author_agent_id` and `source_conversation_id` set from the current request context.
- Returns the new archive's ID and the accepted/rejected tag sets so the agent can see what was kept.

**Field validation:**
- `title`: 1–200 characters.
- `content`: 1–8000 characters. Longer content should be split into multiple archives with different titles.
- `tags`: up to 16 entries.

**Tag normalization (applied in order, before length check):**
1. Lowercase the entire tag.
2. Trim leading and trailing whitespace.
3. Collapse internal whitespace runs to a single hyphen.
4. **Length check**: if the normalized tag exceeds 32 characters, drop that tag from the call and add it to `tags_rejected`. Do **not** silently truncate; do **not** reject the entire `write_archive` call. All other valid tags are accepted as normal.

This lenient behavior keeps the agent's archive committed even when one tag is malformed, surfaces the rejection in the response so the agent can correct on a follow-up call, and avoids the failure mode where a single bad tag blocks a substantive note from being saved.

### 5.6 Searching archives

```
search_archives(
    query: str,
    file_id: str = None,
    tags: list[str] = None,
    limit: int = 10,
) -> {
    results: [
        {
            archive_id,
            file_id,
            file_name,
            title,
            content,
            tags,
            author_agent_id,
            source_conversation_id,
            created_at,
            similarity: float,
        },
        ...
    ]
}
```

Semantic search over `file_archives`. The query is embedded with the agent's configured embedding model and matched via cosine similarity. Results are scoped to files in folders the agent has attached.

Optional filters:

- `file_id`: restrict to one file's archives → **vertical** search.
- `tags`: restrict to archives whose tags intersect this list → **tag-scoped** search.
- Neither: **horizontal** search across all archives the agent can see.

All three modes use the same tool with different arguments. Results always carry `file_id` and `file_name` so the agent can escalate to the underlying file when an archive isn't enough.

### 5.7 Searching file contents

```
search_file_contents(
    query: str,
    folder_id: str = None,
    limit: int = 10,
) -> {
    results: [
        {
            file_id,
            file_name,
            passage_id,
            content,
            similarity: float,
        },
        ...
    ]
}
```

Semantic search over **source passages** — the folder-RAG index built when files are ingested. This is the same retrieval surface as the existing `semantic_search_files` tool; the rename to `search_file_contents` exists to disambiguate it from `search_archives` at the verb level.

`search_archives` retrieves what has been *written about* files. `search_file_contents` retrieves what is *inside* files. Two distinct retrieval substrates, two distinct verbs, no ambiguity at the point of agent decision-making.

---

## 6. Search Semantics

### 6.1 Three modes for archive search

| Mode | Filter | When to use |
|---|---|---|
| Horizontal | no `file_id`, no `tags` | Question is general; you don't yet know which file is relevant. Surfacing the file is part of what you want. |
| Vertical | `file_id` set | You know the file. You want prior topical notes on this file specifically. |
| Tag-scoped | `tags` set, no `file_id` | Cross-cutting topic (e.g., "symbolism") spanning multiple files. |

Tag-scoped combined with `file_id` is valid and equivalent to vertical search further filtered by tags.

### 6.2 Heuristic across the two search tools

Start with `search_archives`. It is cheaper (smaller index, distilled content) and surfaces what has already been observed and concluded — often it answers the question directly or points at the file that will. Escalate to `search_file_contents` when archives don't cover the question, or to `file_read_page` / `file_grep` when you already know which file you need and want the source.

The agent's instructions in §7 carry this heuristic in one sentence.

### 6.3 Provenance is always returned

Every archive search result includes `author_agent_id`, `source_conversation_id`, and `created_at`. Reason: in a shared-store deployment, Agent B may retrieve archives Agent A wrote. The receiving agent needs to know "this is a note from another agent in another conversation" so it doesn't internalize the archive's interpretation as its own observation. The metadata is mandatory in the result schema.

### 6.4 Escalation gradient

`search_archives` → `search_file_contents` → `open_file` + `file_read_page` / `file_grep`. Each step is more expensive and more detailed than the last. The agent decides how far to descend based on what its current question requires.

---

## 7. Base Instructions Update

The agent's base instructions block must be updated to reflect the new paradigm. The current block describes a two-layer model and tells the agent that "your core memory will automatically reflect the contents of any currently open files," which is actively misleading under the new design.

Replace the existing `<base_instructions>` block with the version below. Tool names must match the registry in §5 exactly.

```
<base_instructions>
You are a helpful self-improving agent with advanced memory and file system capabilities.

<memory>
You have a three-layer memory system. Each layer has a different access cost and a different role.

Memory blocks: Labeled text containers compiled into your system context. Each block has a label, description, and value, and a size limit. Some blocks are agent-level and always present (such as 'persona' or 'human'); others are file-level and attach only while the corresponding file is open. Memory blocks are the most expensive layer — they cost system-prompt tokens on every turn — so they hold only what must always be available.

Archives: Topical notes you and other agents have written during past interactions. Archives are not in your system context by default. You retrieve them on demand via semantic search. Each archive carries metadata (title, tags, file association, provenance) that you can filter by. Archives are the searchable record of what has been observed, discussed, and concluded over time.

Files: Documents in folders attached to you. Files hold raw detail. You read them page-by-page into your conversation when you need their content. Read pages remain in conversation history for the rest of the turn and subsequent turns until compaction.

Use memory blocks for what must always be in mind. Use archives for what you should be able to find when you go looking. Use files when you need the source of truth.
</memory>

<file_system>
Folders attached to you contain files. Each file has one core memory block — a short, stable headline describing what the file is and what it contains — and zero or more archives linked to it.

Opening a file attaches its file core memory block to your system context. The file's full content does not load; the headline is what's always visible while the file is open. To read content, use the read tools, which return pages as tool results in the conversation.

Each file core memory is shared across agents — when you edit it, other agents who open the same file will see the edit. Edit a file core memory only when your understanding of what the file fundamentally is has changed.

Archives are topical notes about a file, written in the voice and context of the conversation that produced them. An archive is not a neutral summary of the file — it captures what a particular reading was about: what topic was being explored, what the user emphasized, what conclusions the conversation reached. Multiple archives can exist for the same section of a file, capturing different readings with different topical focuses. The file is the source; archives are the residue of engagement with the source.

When to write an archive: at meaningful checkpoints in your reading and conversation. After finishing a section and synthesizing something about it. When the conversation has produced a non-trivial observation worth saving. When the user has emphasized a point. When you're about to navigate away from a topic. Each archive needs a clear topical focus and a title that names it. Write archives in your own voice from the current conversation's context — that's what makes them worth keeping.

File system tools:
- attach_folder(folder_id) / detach_folder(folder_id) — bind or release a folder of files
- open_file(file_id) — attach the file's core memory to your context; ready to read
- close_file(file_id) — detach the file's core memory
- file_read_page(file_id) — return the current page; advance to the next
- file_read_next_page(file_id) / file_read_prev_page(file_id) — navigate without reading the current page
- file_read_range(file_id, start, end) — read a specific character range
- file_grep(file_id, pattern) — search within a file; returns hits with character offsets
- update_file_core(file_id, new_summary) — revise the shared headline (shared mutation; use deliberately)
- write_archive(file_id, title, content, tags) — commit a topical archive linked to a file
- search_archives(query, file_id=None, tags=None) — semantic search over archives, optionally scoped
- search_file_contents(query, folder_id=None) — semantic search over the raw contents of files (folder index)

Prefer the obvious next action over preflight planning. Read a page before searching for the perfect spot to start. Use 1–3 specific tags per archive, not ten generic ones. Headlines are one sentence; archives are focused topical notes on one aspect of the file, not exhaustive summaries. Write archives after synthesizing something, not before. If a search doesn't find what you need on the first try, the next tool call will get you closer — engage with content rather than looping on retrieval.
</file_system>

<search_semantics>
You have two retrieval tools over file material. They search different things.

search_archives retrieves what has been written about files — distilled topical notes from past conversations. Cheap, opinionated, often directly answers the question or points at the right file.

search_file_contents retrieves what is inside files — passages from the raw text indexed when folders were ingested. More detailed, less interpreted, useful when archives don't cover the question or when you need the source's own words.

Start with search_archives. Escalate to search_file_contents or file_read_page when archives are not enough.

Archive search has three modes depending on how you scope it:

Horizontal (no scope): semantic search across all archives, regardless of file. Use when you don't yet know which file is relevant — a question like "Where did X and Y meet?" may surface archives pointing you at files you haven't opened.

Vertical (file_id scope): semantic search within one file's archives. Use when you know the file and want prior notes on a specific aspect of it.

Tag-scoped (tags filter): semantic search across all archives matching given tags, regardless of file. Use for cross-cutting topics — "everything about symbolism" — that span multiple files.

Every archive returned from a search carries its provenance: which file it belongs to, when it was written, by which agent, in which conversation. Use the file pointer to escalate from an archive to the underlying file when you need more detail than the archive captured. The archive is an entry point; the file is the source of truth.
</search_semantics>

Continue executing and calling tools until the current task is complete or you need user input. To continue: call another tool. To yield control: end your response without calling a tool.
Base instructions complete.
</base_instructions>
```

This block will be iterated based on observed agent behavior after the first round of empirical use. See §12.

---

## 8. Implementation Phases

Build in this sequence. Each phase should be independently testable before moving to the next.

### Phase 1: Data model

1. Alembic migration creating `file_core_blocks`, `file_core_block_history`, `agent_open_files`, `file_archives` and all indexes.
2. SQLAlchemy ORM models and Pydantic schemas for each table, following existing conventions (prefixed string IDs, `OrganizationMixin`).
3. Managers: `FileCoreBlockManager`, `AgentOpenFilesManager`, `FileArchiveManager`.
4. Backfill script (`scripts/backfill_file_core_blocks.py`): for any existing `files_agents` row with non-empty `visible_content`, create a corresponding `file_core_blocks` entry; for attached files with no headline, generate one via one-shot LLM summarization of the first 3–5 char-pages using the agent's configured LLM. Skip files whose owning agent's LLM is unreachable from the backfill host (log skipped IDs; do not fail the run).
5. **Test:** migration runs cleanly, models round-trip, backfill populates without data loss, backfill skip behavior verified.

### Phase 2: System prompt compilation

6. Modify the system-prompt compiler to read `agent_open_files` for the current agent, join to `file_core_blocks`, and render the joined headlines in a labeled section of the system prompt (e.g., `<open_files>...</open_files>`).
7. Remove `FileBlock` / `visible_content` rendering from `_render_directories_*`. Keep file metadata listing (names, IDs, open/closed status) for discovery; remove page content.
8. Replace `refresh_file_blocks()` with `refresh_open_file_cores()` in `agent_manager`.
9. **Test:** opening a file via raw DB insert into `agent_open_files` results in the headline appearing in the next compiled system prompt; closing removes it.

### Phase 3: Read tools

10. Implement `open_file`, `close_file`, `file_read_page`, `file_read_next_page`, `file_read_prev_page`, `file_read_range`, `file_grep`. Also add the `search_file_contents` rename pointing at the existing source-passage search.
11. New `CharPageReader` utility in `letta/services/files/` — codepoint-indexed paging with line-number mapping for grep snippets.
12. Ensure all read tools return content in the tool result body, not by mutating any block.
13. Wire cursor management through `agent_open_files.cursor_char`.
14. Set per-tool `return_char_limit` on file read tools — page content may exceed `BASE_FUNCTION_RETURN_CHAR_LIMIT`.
15. **Test:** an agent that calls `open_file`, then `file_read_page` three times, has page contents persisting in conversation history; the system prompt does **not** contain page content.

**Empirical checkpoint at end of Phase 3:** before moving to Phase 4, run a real conversation through the new read tools with the deployed runtime model and verify the cursor-and-pages mental model lands. If something is awkward (e.g., agent prefers `file_read_range` for normal navigation, or page boundaries feel wrong), surface it before Phases 4–6 build on top.

### Phase 4: Headline editing

16. Implement `update_file_core`. Enforce char_limit. Append to history table.
17. REST endpoints: `GET /v1/files/{file_id}/core`, `PATCH /v1/files/{file_id}/core` in new `file_memory` router.
18. Migrate existing `open_file_for_agent` / `close_file_for_agent` REST to use `AgentOpenFilesManager`.
19. **Test:** Agent A updates a file's core; Agent B opens the same file and sees the new headline.

### Phase 5: Archives

20. Implement `write_archive` with tag normalization per §5.5. Generate embedding using the agent's configured embedding model.
21. Implement `search_archives` with the three filter modes; reuse cosine search patterns from `passage_manager`.
22. REST endpoints: `POST /v1/file-archives/search`, `GET /v1/files/{file_id}/archives`, `GET /v1/file-archives/{id}`.
23. **Test:** an agent writes archives during a conversation; horizontal, vertical, and tag-scoped searches return correct results with provenance. Tag normalization: malformed tag dropped, archive committed, response includes `tags_accepted` and `tags_rejected`.

### Phase 6: Agent instructions

24. Replace the `<base_instructions>` block in `letta/prompts/system_prompts/letta_v1.py` with the version in §7.
25. Update `FILES_TOOLS` registration; verify `attach_missing_files_tools_async` attaches the full new set.
26. **Test:** an agent given a long file opens it, reads multiple pages, and writes at least one archive without explicit prompting beyond the standard user request.

### Phase 7: Deprecation

27. Remove `open_files`, `grep_files`, and the old `semantic_search_files` name from the active tool registry. `search_file_contents` replaces the last of these.
28. Remove `FileBlock` `visible_content` rendering from the compile path; stop writing `visible_content` in the open flow.
29. Schema migration to drop `files_agents.visible_content`, `start_line`, `end_line` columns (after one minor version window).
30. Update `context_window_calculator` token breakdown: replace `directories` page-content tokens with `open_files` headline tokens.

---

## 9. Migration Path

For existing agents and folders in a running deployment:

1. Existing folders and files remain unchanged. No file content needs to be re-ingested.
2. The Phase 1 backfill script copies any existing per-file summary content from `files_agents.visible_content` into `file_core_blocks`. If no summary exists for a file, the script generates one by running a one-shot summarization on the first 3–5 char-pages of the file using **the file's owning agent's configured LLM**. If that LLM is unreachable from the backfill host (common with local Ollama setups where the script runs on a different machine than the agent), the file is skipped and its ID logged. The agent will generate or refine the headline organically when it first opens the file.
3. No existing archives exist (this is a new concept). The `file_archives` store starts empty and accumulates as agents work.
4. Existing passages (`source_passages`, `archival_passages`) are not migrated and not deprecated. The `search_file_contents` rename surfaces the existing source-passage retrieval under a new verb; the underlying index is unchanged.

---

## 10. Acceptance Criteria

Each criterion is testable with a deterministic scenario.

1. **Cross-turn reading recall.** An agent opens a file, calls `file_read_page` three times, then on a fourth turn answers a question about page 1's content. Page 1 must be reachable via conversation history.

2. **Headline stability.** An agent opens a file, performs ten read operations, and never edits the headline. The headline string is byte-identical before and after.

3. **Headline editing is shared.** Agent A calls `update_file_core(file_id, new_summary)`. Agent B (different agent, same Letta server) opens the same file. Agent B's system prompt contains `new_summary`.

4. **Archive horizontal search finds the file.** Setup: file F exists with archive A1 about topic T, no agent has opened F. Action: agent without F open calls `search_archives(query=T)`. Result: A1 is returned with `file_id=F` and a usable `file_name`.

5. **Archive vertical search restricts correctly.** Setup: files F1, F2 each have archives about topic T. Action: agent calls `search_archives(query=T, file_id=F1)`. Result: only F1's archives are returned.

6. **Archive tag search spans files.** Setup: archives across F1, F2, F3 are tagged `symbolism`. Action: agent calls `search_archives(query="symbolism", tags=["symbolism"])`. Result: archives from all three files appear.

7. **Provenance is preserved.** Every archive returned from `search_archives` includes non-null `author_agent_id`, `source_conversation_id`, and `created_at`.

8. **Closing detaches the headline.** An agent opens a file (headline visible in system prompt), then closes it. On the next compile, the headline is absent.

9. **System prompt does not contain page content.** An agent reads ten pages of a file. The compiled system prompt contains the headline but no substring of any page's content. Read content lives exclusively in conversation history.

10. **Provider prompt cache works across reads.** With prompt caching enabled, two consecutive read operations on the same file (no other state changes) hit the cache on the system-prompt prefix. Measure via cache-hit reporting; expectation is a clear hit rate increase versus the current `FileBlock` implementation.

11. **LRU eviction at `max_files_open`.** Open N+1 distinct files in sequence where N = `max_files_open`. The first-opened file's `agent_open_files` row is removed; its headline is absent from the next compile; the N most recent files remain open and visible.

12. **Tag normalization is lenient.** Call `write_archive` with tags including one ≥ 32 chars after normalization. The archive is committed; the response's `tags_accepted` excludes the bad tag; `tags_rejected` includes it; other valid tags are accepted.

13. **Two search tools, two surfaces.** `search_archives` returns rows from `file_archives` only. `search_file_contents` returns rows from `source_passages` only. Neither surfaces results from the other's substrate.

---

## 11. Non-Goals (Explicit Deferrals)

Out of scope for this FR, addressed in follow-ups:

1. **Message-history compaction of read content.** Eviction of old `file_read_page` tool results from conversation history is deferred. Existing Letta compaction (sliding window / threshold-based) continues to apply to the full message stream, which is acceptable for v1.

2. **Agent-driven consolidation tool.** The design conversation explored a `consolidate_reading(file_id, summary, target)` tool that would atomically write a summary and elide prior read messages. Deferred until accessibility is solved and real usage shows whether the elision pressure is worth the complexity.

3. **Archive lifecycle.** No tool for editing or deleting archives. Archives are append-only in v1. Curation tooling will be a follow-up FR if archive volume becomes noisy.

4. **Image files.** All tools and instructions in this FR are text-file scoped. Image extensions (`view_image`, image archives, image embeddings) will be a separate FR that adds parallel tools without restructuring this design.

5. **Per-conversation archive scoping.** All archives are visible to any agent that can see the file. Per-conversation private archives are not supported in v1.

6. **Archive editing across agents.** `update_file_core` is the only cross-agent shared mutation. Archives are immutable; if a later conversation supersedes an earlier archive's interpretation, the agent writes a new archive rather than editing the old one. Archive supersession is left to retrieval ranking (recency, similarity).

7. **Unified retrieval tool.** A single tool returning both archive hits and file-content hits as a ranked, type-tagged result set is an attractive future direction but is **not** in this FR. The v1 design ships two clearly distinct verbs (`search_archives`, `search_file_contents`). Unifying them is contingent on empirical observation of how agents choose between them in practice.

8. **Backwards compatibility with old filesystem tools.** Once Phase 7 ships, the old tool names (`open_files`, `grep_files`, `semantic_search_files`) are removed. No long-term parallel API surface.

---

## 12. Open Questions (with Defaults)

Items that need a decision; each carries a recommendation. The recommendation is the default unless overridden.

1. **Cursor preservation on close.** Hard-delete the `agent_open_files` row on `close_file`; cursor resets to 0 on reopen. **Decided: hard delete.** Simpler mental model.

2. **Embedding model for archives.** Use the agent's configured embedding model — the same one used for attached folders. **Decided: required, not optional.** Guarantees archive search and folder semantic search share an embedding space.

3. **`per_file_view_window_char_limit` default.** Keep current default; revisit after empirical use shows whether pages should be larger or smaller.

4. **`write_archive` verb naming.** Ship with `write_archive` and revisit after observing usage. The right verb sometimes only becomes obvious once the wrong one is in production.

5. **Tag namespace.** Free-form for v1. A controlled vocabulary can be layered on later if archive sprawl makes search noisy; pre-engineering it now would constrain agents in ways we can't yet predict.

6. **Archive deletion when file is deleted.** Schema cascades. Archives are *about* a file; orphan archives have no useful semantics.

7. **Org-level visibility scoping.** Add nullable `organization_id` via `OrganizationMixin` on all new tables now, even though it's unused in single-org v1. **Decided.** Easier than retrofitting.

8. **Tag rejection granularity.** Reject individual tags that exceed the length limit; accept the rest of the `write_archive` call; surface accepted and rejected sets in the response. **Decided: lenient (per-tag rejection).** Avoids the failure mode where a single bad tag blocks an otherwise substantive archive.

---

## 13. Cursor Implementation Notes

A few patterns worth being explicit about for the implementing agent. (The deployed runtime agent — Kimi in letta-vision — gets its own meta-cognitive guidance inside the §7 base instructions block; that's a separate concern.)

- **Tool names in §5 are the contract.** The base instructions in §7 reference these exact names. If a tool name needs to change during implementation, update both sections in lockstep. Drift between instructions and actual tool registry will cause silent agent failure.

- **The read tools must not mutate any block.** This is the central design property. If during implementation it looks easier to "just put the page content in the file core block to keep it visible," that is the failure mode this entire FR is designed to eliminate. Don't do it. Code review checkpoint: any read tool that touches `files_agents.visible_content` or any block content gets rejected.

- **Don't import as `ArchiveManager`.** Use `FileArchiveManager`. The existing `archive_manager.py` handles archival-memory collections — a different concept on a different table. Name collisions will produce confusing bugs.

- **`agent_open_files` join is the right hook point** for system-prompt compilation, not a new method on the agent or memory class. Keep the file-open state as data, not behavior.

- **Embedding generation must be synchronous in `write_archive`.** Async embedding creates a confusing failure mode where the agent writes an archive and a subsequent search doesn't find it. Synchronous is acceptable here because archive volume is bounded by agent reasoning speed, not bulk ingestion.

- **Don't overthink the headline auto-generation in §9.** A one-shot summarization of the first 3–5 pages of the file using the agent's configured LLM is sufficient. Don't build a multi-stage summarization pipeline. The headline will be refined by agents through `update_file_core` over time; perfect first-pass quality is not required.

- **Char offsets, not byte offsets, everywhere.** `cursor_char`, `char_range`, `start_char`/`end_char`, `char_offset`. Page boundaries align to valid UTF-8 codepoint boundaries; never split a codepoint. The FR previously used "byte" in early drafts; that was wrong and has been corrected throughout.

- **Per-tool `return_char_limit` on read tools.** Page content can exceed `BASE_FUNCTION_RETURN_CHAR_LIMIT`. Override in `tool_manager.py` or equivalent for the read tools specifically.

---

## 14. Future Extensions (Out of Scope, Sketched)

Mentioned only to confirm the design accommodates them without rework:

- **Image files.** `view_image(image_id)` as a read primitive returns an `ImageContent` block in the tool result. Image archives reuse the `file_archives` table with the same schema. Visual embeddings (CLIP/SigLIP) would live in a separate `image_embeddings` table joined to `file_archives` only if true image-to-image similarity is desired; text summaries via VLM into the existing `file_archives` table are sufficient for most cases.

- **Read consolidation.** `consolidate_reading(file_id, summary, target='archive'|'core'|'both')` atomically writes a summary and elides prior `file_read_page` tool results from future LLM context construction (via a new `elided_at` column on messages). Eligible to ship once accessibility is solved and elision pressure is empirically observed.

- **Unified retrieval.** A single `find(query, ...)` tool that searches `file_archives` and `source_passages` simultaneously, returning results with a `source_type` tag and unified ranking. Contingent on empirical evidence that the two-tool design in v1 produces real agent confusion despite the verb distinction.

- **Cross-agent archive attribution surfacing.** Provenance fields already exist; grouping search results by author or by source conversation in the UI is a frontend concern, not a backend redesign.

---

## 15. Summary

This FR replaces Letta's mutable-block filesystem with a three-tier hierarchy where each tier has a distinct role: memory blocks for always-in-context anchors, archives for retrievable topical notes, files for raw detail read on demand. Read content moves from system-context block mutation to conversation-history tool results, restoring cross-turn recall and unlocking provider prompt caching. Archives capture the topical residue of engagement with files in the voice of the conversation that produced them, written by the main LLM in-context rather than by a background summarizer. Two retrieval verbs over file material — `search_archives` for distilled notes, `search_file_contents` for source passages — give agents an unambiguous choice at the point of action. The result is a filesystem that matches how modern models are trained to work, scales naturally to images and other modalities, and accumulates institutional knowledge as a byproduct of normal agent use.