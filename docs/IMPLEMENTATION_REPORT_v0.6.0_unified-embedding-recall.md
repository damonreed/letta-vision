# Implementation Report: Unified Embedding, Multimodal Image Memory & Unified Recall (v0.6.0)

**To:** Ada  
**From:** Damon (letta-stack)  
**Date:** 2026-06-09 (updated 2026-06-12)  
**Status:** **GA signed off** — uplift **executed and validated on sliver**; all recall layers coalescing correctly through `search_all`; post-uplift operational fixes shipped; `embedding_legacy_4096` dropped (`v062` + ORM)  
**Baseline:** `v0.5.0` (three-tier filesystem memory)  
**Specification:** [FR: Unified Embedding Space, Multimodal Image Memory & Unified Recall](FR_letta-vision_Unified-Embedding-Multimodal-Recall_v0.6.0-rc.md) (revision r2)  
**Uplift specification:** [FR: Historic Embedding Uplift & Corpus Conversion](FR_letta-vision_Historic-Embedding-Uplift_v0.6.0-GA.md)

---

## Executive summary

We implemented Ades's v0.6 FR through **Phases 1–9** of §14, with one deliberate architectural simplification beyond the FR text: **embedding is deployment-global** (`LETTA_DEFAULT_EMBEDDING_HANDLE`), not per-agent or per-folder. The resolver, native 768-dim pgvector storage, dual-column passage/archive migration, message and image vectors, object-store-backed image records, ingest/enrichment pipeline, tiered vision render policy, hybrid search tools, and client **Images** tab are all in-tree and deployed on sliver.

**Post-RC (2026-06-10)** we shipped the uplift FR as executable code and completed a cleanup pass:

- **Historic uplift CLI** (`scripts/historic_uplift.py`) — Part 1 base64→object conversion, batch image enrichment, Part 2 rolling re-embed (passages, file archives, messages v2 with caption gists), inventory/cost estimates, and historic tool-return byte strip.
- **Hybrid search refactor** — per-layer vector + lexical + RRF in `hybrid_search.py`; archival/file/content/image/message tools routed through it; **`recall` → `search_all`**, **`fetch_image` → `image_fetch`** with deprecated aliases.
- **Turbopuffer retirement** — `tpuf_client.py` and dual-write paths removed; all memory search is pgvector + `pg_trgm`.
- **Ref-only tool-return persistence** — new tool returns store `LettaImage` refs without inline bytes; historic strip job for pre-change rows.
- **Recall FR §6 post-processing** — image/message dedup and per-source diversity cap in `finalize_recall_hits`.
- **Vision fixes for MCP `generate_image`** — tool-return images participate in the byte-budget walk; metadata loaded for sizing; `fill_image_content_in_messages` extended to tool rows; TEXT tier injects `images.description`.
- **GA client (2026-06-11)** — uplift FR §12 Images inspector (list+detail, hybrid search, paginated browse, inline metadata edit, click-to-zoom); §13 shell (Chat default tab, mount-once, windowed history, conversation sidebar N+1 fix, deferred memory load).

**Uplift execution (2026-06-11, sliver):** Part 1 → enrich-pending → Part 2 completed across all five vector tables. Deployment embedding moved to GA `openrouter/google/gemini-embedding-2` @768 (space id `6490a4b17e06a258`). Live agent validation (Lyra) confirms all layers perform well individually and fuse correctly through `search_all`; new image ingest/enrichment/search paths solid.

**Post-uplift fixes (2026-06-11):** archival insert `embedding_space_id` persistence, tool-return serialization for archival insert/search, message uplift monotonic guard, MiniMax duplicate thinking strip, file-delete background recompile (client). See § Post-uplift validation.

**Recommendation for Ada:** v0.6.0 GA signed off (2026-06-12). Remaining documented gaps (recall filter params, tool lexical fallback, space-guard logging, filename-not-in-passage-embed-text) are v0.6.1 observability/content improvements, not blockers. Legacy column drop shipped with ORM mapping removal (`v062_drop_legacy_emb`).

---

## Scope delivered vs FR §14

| FR phase | Deliverable | Status |
|----------|-------------|--------|
| **Phase 1** — Embedding foundation | `EmbeddingConfig` extensions, resolver, provider client MRL/normalize/query override, padding removal | **Done** |
| **Phase 2** — Storage + guard | Alembic `v060`/`v061`, dual-column passages/archives, message vectors, HNSW + `pg_trgm`, atomic version guard | **Done** |
| **Phase 3** — Turbopuffer removal | Messages off tpuf; passage/file paths native | **Done** — tpuf client and dual-write removed (`643ea2418`) |
| **Phase 4** — Image records + object store | `images` table, `ImageManager`, S3/MinIO client, pixel embed path | **Done** |
| **Phase 5** — Ingest pipeline | Sync store, background 1MP + VLM captions + pixel embed, two-embed dance, retries | **Done** |
| **Phase 6** — Chat render policy | `LettaImage` refs, render walk, on-demand 1MP, serializer integration | **Done** — TEXT tier uses `images.description` + handle hint |
| **Phase 7** — Recall tool | Vector + lexical + RRF; `image_fetch` multimodal return | **Partial** — diversity cap + dedup done; `layers` / `time_range` / `source` filters still missing |
| **Phase 8** — Client Images tab | Server REST, client proxy, `Images.svelte` | **Done** — §12 inspector (2026-06-11): list+detail, search, pagination, inline edit |
| **Phase 8b** — Client shell polish | Tab order, Chat perf, mount-once | **Done** — §13 (2026-06-11) |
| **Phase 9** — Base instructions | `search_all` / `image_fetch` / layer tools in `letta_v1.py` | **Done** |

### Acceptance criteria (§15)

| # | Criterion | Status | Notes |
|---|-----------|--------|-------|
| 1 | Fresh agent embeds all corpus types at 768 with shared `embedding_space_id` | **Pass** | GA handle `gemini-embedding-2`; space id stamped on all create paths post-fix |
| 2 | No hardcoded embedding model; tpuf not required to boot/search | **Pass** | Tpuf removed; boot and all memory search paths are PG-native |
| 3 | Vectors 768, unit-length, no `np.pad` on write/query | **Pass** | `prepare_vector_for_write`; passage schema padding validator removed |
| 4 | Cross-space query returns zero vector hits + logs exclusion | **Partial** | `apply_embedding_space_guard` works; **no exclusion-count debug logging** |
| 5 | Image ingest: row, 1MP, captions, pixel embed, hash dedup | **Pass** | `image_ingest.py` + MinIO on sliver |
| 6 | Text query finds image via vector and/or trigram; `image_fetch` returns pixels | **Pass** | Empirical on sliver (Chat B + Images tab) |
| 7 | Long image-heavy chat: tiered render within wire-byte cap | **Pass** | Render walk + on-demand 1MP; TEXT tier uses description from DB |
| 8 | `search_all` fused deduped source-capped list | **Pass** | RRF fusion + `finalize_recall_hits` (dedup + per-source cap) |
| 9 | Enrichment failure: renderable image + text-only message embed | **Pass** | Failure path re-embeds message text |
| 10 | Client Images tab with metadata/actions | **Pass** | §12 inspector: 1MP rail+pane, hybrid search, inline caption/description/details, re-enrich, delete, click-to-zoom |
| 11 | HNSW + pg_trgm; dual-column passages/archives | **Pass** | `v060` + `v061` migrations |

Automated coverage includes: `test_render_policy.py`, `test_image_ingest.py`, `test_recall_service.py`, `test_fetch_image_and_recall_archives.py`, `test_openai_provider_embeddings.py`, embedding resolver tests, hybrid search tests.

---

## Amendments beyond FR r2 (documented for Ada)

These emerged during implementation and sliver validation. They should be treated as **approved deltas** unless Ada objects.

### A1. Deployment-global embedding (not per-agent / per-folder)

**FR §5** lists agent-level embedding override as resolver priority 1. **Shipped behavior:** `resolve_embedding_config` / `resolve_embedding_config_async` **ignore** `agent_state.embedding_config`. All runtime paths — archival insert, folder file ingest, file archive write/search, recall, message embed — use `LETTA_DEFAULT_EMBEDDING_HANDLE` via `resolve_deployment_embedding_config_async`.

**Rationale:** Per-resource embedding pickers caused dimension mismatches (768 vs 1024 vs 4096) and failed ingest on sliver. One deployment model matches FR §16 ("deployment dim is fixed") and extends it to **no per-agent model selection at all**.

**Client:** Removed embedding pickers from Agents and Files create flows (`letta-vision-client`).

**Residual:** `AgentState.embedding_config` column still populated at create (from deployment default) for API compat; it is not read at search/ingest time.

### A2. Recall tool output format

**Not in FR.** Tool return lines use a shared formatter:

```
[file] handle=passage-… filename=v060-test/villains.txt score=0.0328
Victoria
Damian

[file_archive] handle=file_archive-… filename=v060-test/villains.txt score=0.0318
[Victoria's character] Victoria is a villain and wears black leather.
```

- Folder-ingested passage layer labeled **`file`** (not `source`).
- **`filename=`** on both `file` and `file_archive` hits so agents distinguish file content from archival memory.

Implemented in `format_recall_hit()` (`recall_service.py`).

### A3. `image_fetch` message hydration

**FR §12** covers multimodal tool return. Additional fix: tool-return `LettaImage` refs with `file_id` but `data: null` are **hydrated from MinIO** on message read and before LLM calls (`image_hydration.py`, `message_manager.py`). `image_fetch` tool results are **not re-ingested** as new image records.

Empirical: Chat B shows fetched images in UI and passes pixels to the LLM after tool call.

### A4. Granular hybrid search tools (replacing monolithic `recall`)

**FR §12** describes one fused `recall` tool. **Shipped behavior:** layer-specific hybrid tools are primary (`archival_memory_search`, `file_archives_search`, `file_contents_search`, `conversation_search`, `image_search`); **`search_all`** is the optional cross-layer pass. Deprecated aliases (`recall`, `fetch_image`, `search_file_archives`, etc.) remain registered but marked deprecated in `DEPRECATED_LETTA_TOOLS` and excluded from default agent upserts.

Rationale: clearer agent routing, smaller tool schemas, and per-layer tuning (e.g. `file_contents_search` returns longer snippets).

### A5. Ref-only tool-return persistence

**Not in FR.** New messages persist tool-return images as **`LettaImage(file_id, data=null)`** only (`tool_return_storage.py`), matching chat content refs. Inline base64 is hydrated on read and at LLM request time under render policy. Historic rows with persisted bytes are cleaned by `historic_uplift.py strip-tool-returns`.

---

## Architecture (server — `letta-vision`)

### Embedding foundation

| Module | Role |
|--------|------|
| `letta/schemas/embedding_config.py` | `output_dimensionality`, `input_type`, `normalize`, `embedding_space_id`, `compute_space_id()` |
| `letta/embeddings/resolver.py` | Deployment-wide resolve; `validate_native_pg_embedding_config` |
| `letta/embeddings/util.py` | `prepare_vector_for_write`, `l2_normalize` |
| `letta/embeddings/query.py` | `embed_search_query`, `apply_embedding_space_guard` |
| `letta/embeddings/write.py` | `write_message_embedding_atomic` (conditional UPDATE) |
| `letta/embeddings/message_embed_text.py` | Message embed text; v2 caption gist injection |
| `letta/constants.py` | `DEPLOYMENT_EMBEDDING_DIM = 768`; `BASE_TOOLS`, `DEPRECATED_LETTA_TOOLS` |
| `letta/llm_api/openai_client.py` | `dimensions`, `input_type`, query override, image embeddings; tool-row image fill |

**Deploy default:** `openrouter/google/gemini-embedding-2` at 768-dim MRL, L2-normalized (GA; sliver migrated from preview during uplift).

### Storage (Alembic)

| Migration | Contents |
|-----------|----------|
| `v060_unified_embedding_multimodal_recall.py` | Passage dual-column; `embedding_space_id` backfill; message vector cols; `images` table; HNSW on 768 cols; `pg_trgm` GIN |
| `v061_file_archive_unified_embedding.py` | `file_archives` dual-column + trigram + HNSW |

Historic rows: `embedding_legacy_4096` untouched; new `embedding` NULL until uplift.

### Image memory

| Module | Role |
|--------|------|
| `letta/orm/image.py` | `images` table per FR §4.5 |
| `letta/services/image_manager.py` | CRUD, list, metadata update |
| `letta/services/image_ingest.py` | Sync hash-dedup store; background 1MP + VLM captions + pixel embed; `convert_historic_images_in_message` |
| `letta/services/object_store/client.py` | Content-addressed S3/MinIO; **wire-byte** size accounting |
| `letta/services/image_fetch.py` | `image_fetch` multimodal tool return |
| `letta/server/rest_api/routers/v1/images.py` | list/get/patch/delete/re-enrich/content/url |

**Caption VLM:** `settings.image_caption_model_handle` (sliver: `openrouter/minimax/minimax-m3` via compose).

### Vision render policy

| Module | Role |
|--------|------|
| `letta/services/vision/render_policy.py` | Byte-budget walk; current-turn includes tool returns after last user message; `supports_image_blocks_in_history` aligned with `model_supports_vision()` |
| `letta/services/vision/image_hydration.py` | `prepare_messages_for_vision_llm`; on-demand `generate_1mp_now`; tool-return hydration; TEXT tier description |
| `letta/services/vision/tool_return_storage.py` | Ref-only persistence; historic byte strip helpers |
| `letta/agents/letta_agent_v2.py`, `letta_agent_v3.py` | Hydration before `build_request_data`; no-truncate for MCP image tools |
| `letta/schemas/message.py` | `image_render_decisions` threaded into user and tool-return serialization |

**Wire bytes:** `file_size_full` / `file_size_1mp` stored as base64-encoded wire size (`object_store.wire_byte_size`). Cap: `settings.vision_context_byte_cap` (default 20 MiB).

**MCP image tools:** `generate_image`, `edit_image`, `compose_image` return pixels inline in the tool result; refs persist at rest and re-enter the byte-budget walk on later turns (not a tier bypass).

### Hybrid search & recall

| Module | Role |
|--------|------|
| `letta/services/recall/hybrid_search.py` | Per-layer vector + `pg_trgm` lexical + RRF (k=60); `search_all_hybrid` |
| `letta/services/recall/recall_service.py` | Hit types, `format_recall_hit`, `finalize_recall_hits` (dedup + diversity cap) |
| `letta/functions/function_sets/search_tools.py` | `search_all`, `image_fetch`, `image_search` stubs |
| `letta/services/tool_executor/core_tool_executor.py` | Hybrid search executors |

**Vector leg:** `archival_passages`, `source_passages` (layer `file`), `file_archives`, `messages`, `images` — all under space guard.

**Lexical leg:** `pg_trgm` `similarity()` on passage text, message text, image caption/description/details, file archive title/content.

**Post-fusion (FR §6):** `_dedup_image_message_hits` drops redundant message hits when the same image is already present; `_apply_diversity_cap` limits hits per `source_group`.

**Missing vs FR §12:** `layers` / `time_range` / `source` parameters; neighbor/offset handles for file chunks.

### Turbopuffer

| Path | Status |
|------|--------|
| All memory embed/search | **PG-native only** — tpuf client removed |
| Tool embed/search | **Still tpuf-dependent** when `embed_tools=True`; **no lexical fallback** when disabled |

### Agent tools

`BASE_TOOLS` includes `search_all`, `image_fetch`, `image_search`, `conversation_search`, `archival_memory_insert`, `archival_memory_search` (`letta/constants.py`). Deprecated names in `DEPRECATED_LETTA_TOOLS` (`recall`, `fetch_image`, …) kept for backward compat.

---

## Historic embedding uplift (implementation)

Per [FR: Historic Embedding Uplift](FR_letta-vision_Historic-Embedding-Uplift_v0.6.0-GA.md). Orchestrated by **`scripts/historic_uplift.py`**.

### CLI commands

| Command | Purpose |
|---------|---------|
| `inventory` | Part 1 scan stats + Part 2 row counts + cost estimate |
| `convert --dry-run` / `convert --i-have-a-snapshot` | Part 1: inline base64 → `LettaImage` refs + `images` rows |
| `enrich-pending [--dry-run]` | Batch 1MP + VLM captions + pixel embed for `pending`/`failed` images |
| `reembed [--dry-run] [--table all\|…]` | Part 2: rolling 768 re-embed with resumable checkpoints |
| `strip-tool-returns --dry-run` / `--i-have-a-snapshot` | Historic cleanup: remove persisted base64 from `tool_returns` |

All mutating commands require a Postgres snapshot flag and support `--checkpoint` / `--no-resume`.

### Modules

| Module | Role |
|--------|------|
| `letta/services/migration/image_base64_conversion.py` | Part 1 scan + convert via `convert_historic_images_in_message` |
| `letta/services/migration/enrich_pending_images.py` | Batch enrichment driver (concurrency + throttle) |
| `letta/services/migration/historic_reembed.py` | Part 2: `archival_passages`, `source_passages`, `file_archives`, `messages` |
| `letta/services/migration/tool_return_byte_strip.py` | Historic tool-return byte strip |
| `letta/services/migration/uplift_inventory.py` | Combined inventory + table sizes |
| `letta/services/migration/uplift_cost.py` | Embed + VLM cost estimates |
| `letta/services/migration/block_classifier.py` | Message content block classification for Part 1 scan |

### Part 2 message embed (v2)

Messages re-embed at **`embedding_version = 2`** using `build_message_embed_text(..., include_image_captions=True)` — caption **gists** (not full descriptions) folded into the JSON embed payload per the two-embed pattern. Requires Part 1 conversion + image enrichment so caption gists exist.

### Recommended execution order

1. `inventory` — baseline counts and cost
2. `convert --dry-run` → snapshot → `convert --i-have-a-snapshot`
3. `enrich-pending` — until pending/failed image count is zero
4. `reembed --dry-run` → `reembed` (all tables or per-table)
5. `strip-tool-returns` — if historic tool returns still carry inline bytes
6. Validate recall quality + HNSW performance → drop legacy 4096 column (shipped `v062`)

---

## Architecture (client — `letta-vision-client`)

| Surface | Purpose |
|---------|---------|
| **`App.svelte` / `stores.js`** | Tab order **Chat → Images → Files → Agents → MCP → Providers**; Chat default on load; Chat mount-once |
| **`Images.svelte`** | §12 inspector: 280px thumbnail rail (browse scroll-load or search hits w/ score overlay), 50/50 detail (preview+`meta-grid` \| inline editors), click-to-zoom |
| **`Chat.svelte`** | Windowed history (`limit=50`, load-older), deferred memory panel fetch |
| **`ConversationList.svelte`** | Sidebar from single `conversations.list` (no N+1 preview fetch) |
| **`backend/routes/images.py`** | Proxy `/v1/images/*`, `POST /search`, paginated list, content stream |
| **`backend/routes/messages.py`** | Windowed `GET /history` → `{ messages, has_more }`; `full=true` escape hatch |
| **`backend/routes/conversations.py`** | List w/o per-row message preview; `order_by=last_message_at` |
| **`Files.svelte`** | Folder upload; embedding picker **removed**; improved API error parsing |
| **`Agents.svelte`** | Embedding picker **removed** |
| **`frontend/src/lib/tools.js`** | Base tool list uses `search_all`, `image_fetch`, `image_search` |
| **`frontend/src/lib/toolResultImages.js`** | Tool-result image dedupe by `file_id`; content-proxy URLs |
| **`api.js`** | `parseApiError()`, `listImages({ limit, after… })`, `searchImages`, windowed `getHistory` |

---

## Architecture (deploy — `letta-vision-deploy`)

Sliver stack (`docker-compose.yml`):

| Variable | Purpose |
|----------|---------|
| `LETTA_DEFAULT_EMBEDDING_HANDLE` | `openrouter/google/gemini-embedding-2` |
| `LETTA_EMBED_ALL_MESSAGES` | `true` — background message embedding |
| `LETTA_OBJECT_STORE_URI` | `s3://letta-vision/images?endpoint=http://minio:9000` |
| `LETTA_IMAGE_CAPTION_MODEL_HANDLE` | `openrouter/minimax/minimax-m3` |
| MinIO service | Local object store; data at `/data/letta-vision/minio` on sliver |

Ports: API **8283**, client **8284**.

---

## Deviations from FR (intentional or pending)

| Topic | FR | Shipped / pending |
|-------|-----|-------------------|
| Agent embedding override | §5 priority 1 | **Removed** — deployment-global only (Amendment A1) |
| Per-folder embedding | Implicit per-source | **Removed** — same deployment handle |
| Turbopuffer | §7 remove from path | **Done** — memory paths tpuf-free |
| Tool search tpuf-less | §7 lexical fallback | **Missing** — raises if `embed_tools=False` |
| Recall signature | §12 filters + post-steps | **Partial** — dedup + diversity cap done; filter params missing |
| TEXT render tier | §10 description + handle | **Done** — description + `image_fetch` hint |
| Space guard logging | §8 exclusion counts | **Not implemented** |
| Recall return shape | §12 layer names | **`file`** + **`filename=`** (Amendment A2) |
| Monolithic `recall` | §12 single tool | **Split** — layer tools + `search_all` (Amendment A4) |

---

## Post-RC fixes (sliver validation)

### Integration (2026-06-09)

| Issue | Root cause | Fix |
|-------|------------|-----|
| **`image_fetch` empty in UI/LLM** | Re-ingest stripped `data`; no hydration on read | Skip ingest for `image_fetch`; hydrate from MinIO; inline base64 in tool return |
| **Folder ingest 4096 vs 768 error** | `Passage` schema `pad_embeddings` validator zero-padded to 4096 | Removed padding validator from `letta/schemas/passage.py` |
| **Per-agent/folder wrong embedding** | Stale UI pickers + resolver honored agent override | Deployment-global resolver; client embedding UI removed |
| **`recall` SQL crash** | asyncpg ambiguous type on nullable agent_id | Branch SQL on agent_id presence |
| **`recall` missed folder files** | Source passage vector/lexical not scoped to agent folders | Join `sources_agents` for `file` layer |
| **Agent misread recall hits** | Layer `source`, no filename | `file` layer + `filename=` in `format_recall_hit` |

### Cleanup & uplift code (2026-06-10)

| Change | Notes |
|--------|-------|
| Per-layer hybrid search | `hybrid_search.py`; RRF per layer; `search_all_hybrid` fuses all |
| Tpuf removal | Deleted `tpuf_client.py`, `turbopuffer_embedder.py`, dual-write in passage/archive managers |
| Tool renames | `search_all`, `image_fetch`, `image_search`; deprecated aliases in `DEPRECATED_LETTA_TOOLS` |
| Ref-only tool returns | Persist refs at rest; `strip-tool-returns` for history |
| Recall §6 filters | `_dedup_image_message_hits`, `_apply_diversity_cap` in `finalize_recall_hits` |
| TEXT tier hydration | `_text_for_demoted_image` loads `images.description` |
| Historic uplift CLI | Part 1 convert, enrich-pending, Part 2 reembed, inventory, strip-tool-returns |

### Vision / MCP (2026-06-10)

| Issue | Root cause | Fix |
|-------|------------|-----|
| **`generate_image` invisible to model** | Tool-return images excluded from render walk; no metadata for sizing; `fill_image_content` skipped tool rows | Current-turn tool returns in walk; load image metadata for budget; extend fill to tool rows by `tool_call_id`; pass render decisions through tool-return serialization |
| **Redundant `image_fetch` after generate** | Model saw TEXT placeholders instead of pixels | Unified vision model registry; hydration under standard byte cap (not always-FULL bypass) |

### GA client polish (2026-06-11)

| Change | Notes |
|--------|-------|
| Images §12 inspector | `Images.svelte` list+detail; `POST /v1/images/search`; paginated `GET /v1/images` (`after_created_at` + `after_id`); 1MP rail+pane; score overlays; inline metadata PATCH |
| Tab shell §13 | Chat default; nav order Chat-first; Chat mount-once |
| Chat perf | History window 50 + load-older; conversation sidebar 1 round-trip; memory on panel open |
| `ImageViewer` | Click zoomed image or backdrop to dismiss |
| File delete UX | Background `recompile_conversations_for_folder`; optimistic removal + deleting state on Files page |

### Post-uplift validation & fixes (2026-06-11)

Executed uplift on sliver; validated with Lyra live sessions (archival, file, image layers + `search_all`).

| Issue | Root cause | Fix | Commit |
|-------|------------|-----|--------|
| **`archival_memory_search` empty tool result** | Return dict not serialized through tool executor path | Explicit `{message, results}` return | `4a50303b3` |
| **New archival insert invisible to search** | `create_agent_passage_async` omitted `embedding_space_id` from ORM write despite `_prepare_passage_embedding_fields` | Pass `embedding_space_id` in `common_fields` (agent batch + source create too) | `f6f53581f` |
| **`archival_memory_insert` silent success** | Tool returned `None` | Return `{message, results: [{id, timestamp, tags}]}` | `f6f53581f` |
| **Message uplift writes skipped** | Monotonic `embedding_version` guard blocked v2 historic writes | Guard fix for uplift path | `7de811781` |
| **Legacy source upload invisible rows** | `DirectoryConnector` → deprecated `create_many_passages_async` skipped `_prepare_passage_embedding_fields` | Route through `create_many_source_passages_async` in `connectors.py` | GA sign-off fix |
| **MiniMax duplicate thinking in UI** | Same analysis in `reasoning_message` and inline think tags in assistant text | `strip_duplicate_thinking_from_assistant_text()` when reasoning extracted separately | `dc6ec9501` |
| **Reasoner models skip tool calls** | Kimi/MiniMax put tool intent in reasoning, finish without `tool_calls` | Operational: PATCH `enable_reasoner` per agent; documented in uplift FR §14.3 |
| **`file_contents_search` miss on new file** | Passage embed text had no query token (filename not in embed payload); vector rank low | Not an indexing bug — query/content mismatch; optional follow-up: prepend filename to passage embed text |
| **File delete hung** | Sync folder recompile blocked HTTP response | Background recompile + client optimistic UX | `d190cd0` (client) |
| **Historic tool-return base64 bloat** | `fetch_image` tool returns persisted raw bytes in `messages.tool_returns` | `tool_return_byte_strip.py` + `strip-tool-returns` CLI subcommand | uplift tooling |

**Sliver backfill:** one archival passage inserted before `f6f53581f` deploy had `embedding_space_id = NULL`; manually stamped to GA space. Post-deploy inserts stamp correctly.

**Live validation summary:**

| Surface | Result |
|---------|--------|
| `search_all` cross-layer fusion | **Pass** — dedup + diversity cap behave as designed |
| Per-layer hybrid tools | **Pass** — archival, file content, file archives, conversation, image |
| New image ingest → enrich → search | **Pass** — multiple Lyra image sessions |
| Archival insert → search | **Pass** — after embedding_space_id fix + backfill |
| Chat + vision + MCP images | **Pass** — render policy, hydration, tool-return pixels |

---

## Historic embedding uplift — GA gate

### What is already in place

1. **Target schema:** 768 `embedding` on passages, file archives, messages, images.
2. **Uplift tooling:** Resumable CLI with dry-run, checkpoints, inventory, and cost estimates.
3. **Space guard:** Queries only rank rows with matching `embedding_space_id`; uplift is a rolling fill — recall coverage increases monotonically.
4. **Single deployment model:** All backfill uses `LETTA_DEFAULT_EMBEDDING_HANDLE`.
5. **Idempotent writes:** Atomic `embedding_version` guard for message re-embed.
6. **Corpus execution on sliver:** Part 1, enrichment, and Part 2 **complete** under GA space `6490a4b17e06a258`.
7. **Post-uplift create-path fix:** New archival/source inserts stamp `embedding_space_id` (commit `f6f53581f`).

### What remains

1. **Optional v0.6.1 improvements** — see Pre-GA quality gaps below.

### Uplift operations runbook (§10.1)

- **Message uplift version:** Bump `UPLIFT_MESSAGE_EMBED_VERSION` before each space migration (currently `3`; predicate uses `MESSAGE_EMBED_VERSION = 2`). Without a bump, rows already at v3 fail the monotonic guard on re-run; steady-state re-runs unnecessarily re-embed v1 messages.
- **Part 2 checkpoint:** With default `--resume`, the checkpoint cursor overrides idempotency predicates; checkpoint is retained after completion; batch failures advance the cursor past failed rows. If `failed > 0`: re-run with `--no-resume`, delete `~/.letta/uplift_part2_checkpoint.json` after a clean run.

### Pre-GA quality gaps (v0.6.1 candidates)

| Gap | Impact | Recommendation |
|-----|--------|----------------|
| Recall filter params | Agents cannot narrow by layer/time/source | Small follow-up or document as v0.6.1 |
| Tool lexical fallback | Tool picker search fails tpuf-less | Small follow-up PR |
| Space guard logging | Harder to observe mixed-space during uplift | Add debug log with exclusion counts |
| Filename not in passage embed text | `file_contents_search` may miss filename-only queries | Prepend filename/headline to source passage embed input |

---

## Empirical validation (sliver)

| Test | Result |
|------|--------|
| Folder file ingest (`villains.txt`, `heroes.txt`) | **Pass** |
| `search_all("Victoria")` / layer tools | **Pass** — file passage + file archive note |
| Search output clarity | **Pass** — `file` + `filename=` formatting |
| `write_file_archive` on `villains.txt` | **Pass** |
| `image_fetch` in Chat B | **Pass** — UI + LLM receive pixels |
| Images tab | **Pass** — §12 inspector: search, paginated browse, inline metadata, full meta-grid |
| Chat tab load | **Pass** — mount-once; windowed history; sidebar without N+1 |
| Chat with vision model (Kimi K2.6, MiniMax M3) | **Pass** with render policy + hydration; reasoner toggle documented for tool reliability |
| Historic uplift (full corpus) | **Pass** — executed on sliver; GA space `6490a4b17e06a258` |
| Cross-layer `search_all` post-uplift | **Pass** — all five vector legs + lexical RRF + §6 post-processing |
| Lyra live sessions (archival + files + images) | **Pass** — end-to-end judged solid (2026-06-11) |
| `archival_memory_insert` → immediate search | **Pass** — after `embedding_space_id` fix |
| File delete | **Pass** — background recompile; no UI hang |

Agent `v060 test` on folder `v060-test`; Lyra `agent-35a1c263…` for post-uplift validation. Deployment embedding `openrouter/google/gemini-embedding-2` (GA).

---

## Open items

| Item | Recommendation |
|------|----------------|
| Recall filter params (`layers`, `time_range`, `source`) | v0.6.1 or document as limitation |
| Tool lexical fallback | §7 r2 — `ILIKE` on name/description when tpuf off |
| Space guard exclusion logging | Observability |
| Filename in passage embed text | v0.6.1 content improvement for file search |
| Virtualized chat scroll | Optional follow-up if load-older is annoying |
| Reasoner + tool reliability | Document per-agent `enable_reasoner` toggle; v1 PATCH quirk (`reasoning: true` vs `enable_reasoner`) |

---

## Recommendation to Ada

**v0.6.0 GA signed off (2026-06-12).** The implementation delivers Ades's core design on the full sliver corpus:

- One embedding space (768-dim gemini-embedding-2 GA, deployment-wide, space id `6490a4b17e06a258`)
- Native pgvector with space guard (no silent cross-model garbage)
- Images as first-class searchable records with object-store bytes and tiered chat render
- Per-layer hybrid search + optional `search_all` over passages, files, file reading notes, messages, and images — validated live with Lyra across all layers
- Reference-then-fetch via `image_fetch`; MCP tools deliver inline pixels
- Historic uplift executed: Part 1 conversion, enrichment, Part 2 re-embed
- Post-uplift operational fixes for archival insert/search, message uplift guard, vision/reasoning UX, and file delete

Remaining gaps (recall filter params, tool fallback, space-guard logging, filename-in-embed-text) are v0.6.1 polish items. Legacy 4096 column drop shipped at GA (`v062_drop_legacy_emb` + ORM).

---

## Key file index

### Server (`letta-vision`)

| Path | Role |
|------|------|
| `scripts/historic_uplift.py` | Uplift CLI orchestrator |
| `letta/services/migration/image_base64_conversion.py` | Part 1 base64 → refs |
| `letta/services/migration/enrich_pending_images.py` | Batch image enrichment |
| `letta/services/migration/historic_reembed.py` | Part 2 rolling re-embed |
| `letta/services/migration/tool_return_byte_strip.py` | Historic tool-return cleanup |
| `letta/services/migration/uplift_inventory.py` | Inventory + cost inputs |
| `alembic/versions/v060_unified_embedding_multimodal_recall.py` | Core schema migration |
| `alembic/versions/v061_file_archive_unified_embedding.py` | File archive embedding parity |
| `letta/embeddings/message_embed_text.py` | Message embed v2 + caption gists |
| `letta/services/image_ingest.py` | Ingest + enrichment + historic convert |
| `letta/services/image_fetch.py` | `image_fetch` tool |
| `letta/services/recall/hybrid_search.py` | Per-layer hybrid search |
| `letta/services/recall/recall_service.py` | Hit formatting + §6 post-processing |
| `letta/services/vision/render_policy.py` | Byte-budget render walk |
| `letta/services/vision/image_hydration.py` | LLM message hydration |
| `letta/services/vision/tool_return_storage.py` | Ref-only tool-return persistence |
| `letta/functions/function_sets/search_tools.py` | `search_all`, `image_fetch`, `image_search` |
| `letta/server/rest_api/routers/v1/images.py` | List w/ cursor, `POST /search`, content variants |
| `letta/schemas/image.py` | `ImageListResponse`, `ImageSearchRequest/Response` |
| `letta/prompts/system_prompts/letta_v1.py` | Hybrid search + MCP image instructions |
| `letta/constants.py` | `BASE_TOOLS`, `DEPRECATED_LETTA_TOOLS` |
| `letta/services/passage_manager.py` | Passage create; `embedding_space_id` on agent/source inserts |
| `letta/services/tool_executor/core_tool_executor.py` | Archival insert/search tool executors |
| `letta/llm_api/minimax_openai.py` | Duplicate thinking strip for MiniMax reasoner |
| `letta/interfaces/openai_streaming_interface.py` | Applies thinking strip when reasoning extracted |

### Client (`letta-vision-client`)

| Path | Role |
|------|------|
| `frontend/src/App.svelte` | Tab order; Chat default; mount-once |
| `frontend/src/routes/Images.svelte` | §12 Images inspector |
| `frontend/src/routes/Chat.svelte` | Windowed history; deferred memory |
| `frontend/src/lib/tools.js` | Base tool names |
| `frontend/src/lib/toolResultImages.js` | Tool-result image display + dedupe |
| `backend/routes/images.py` | Images API proxy + search |
| `backend/routes/messages.py` | Windowed history |
| `backend/routes/conversations.py` | Sidebar list (no N+1) |
| `backend/routes/folders.py` | Background recompile after file delete |
| `frontend/src/routes/Files.svelte` | Optimistic delete UX |

### Deploy (`letta-vision-deploy`)

| Path | Role |
|------|------|
| `docker-compose.yml` | MinIO, embedding handle, object store URI, caption model |
| `.env.example` | Documented unified embedding for all resource types |

---

## FR cross-check — r2 elements vs implementation

| FR r2 correction (Appendix A) | Implemented? |
|-------------------------------|--------------|
| (1) Passage dual-column 768 + legacy 4096 | **Yes** — `v060` |
| (2) Wire-byte cap convention | **Yes** — `wire_byte_size`, render policy |
| (3) Atomic monotonic version guard | **Yes** — `write_message_embedding_atomic` |
| (4) Current-turn on-demand 1MP | **Yes** — `generate_1mp_now` in hydration |
| (5) Tool embedding out of scope; lexical degrade | **Partial** — out of scope yes; lexical degrade **no** |
| (6) Pre-flag `supports_image_blocks_in_history` | **Yes** — aligned with `model_supports_vision()` |
| Caption VLM separate setting | **Yes** — `image_caption_model_handle` |
| `file_archives` embedding parity (§4.5b) | **Yes** — `v061` |
| Five-table recall vector leg (§12) | **Yes** |
| `pg_trgm` lexical leg (§12) | **Yes** |
| RRF fusion (§12) | **Yes** |
| Per-source diversity + dedup (§12) | **Yes** — `finalize_recall_hits` |
| Historic uplift (GA FR) | **Yes** — CLI + migration modules; **executed on sliver** |
| Images tab §12 inspector (GA FR) | **Yes** — list+detail, search, pagination |
| Client shell §13 (GA FR) | **Yes** — Chat default, mount-once, chat perf |
| Deployment-global embedding (Amendment A1) | **Yes** — documented here |
| Recall `file` + `filename=` output (A2) | **Yes** — documented here |
| Layer hybrid tools + `search_all` (A4) | **Yes** — documented here |

---

*End of report.*
