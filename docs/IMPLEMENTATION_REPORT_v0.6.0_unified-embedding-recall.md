# Implementation Report: Unified Embedding, Multimodal Image Memory & Unified Recall (v0.6.0-rc)

**To:** Ada  
**From:** Damon (letta-stack)  
**Date:** 2026-06-09  
**Status:** Release candidate — validated on **sliver** against *new* data; historic embedding uplift is the gate to GA  
**Baseline:** `v0.5.0` (three-tier filesystem memory)  
**Specification:** [FR: Unified Embedding Space, Multimodal Image Memory & Unified Recall](FR_letta-vision_Unified-Embedding-Multimodal-Recall_v0.6.0-rc.md) (revision r2)

---

## Executive summary

We implemented Ades's v0.6 FR through **Phases 1–9** of §14, with one deliberate architectural simplification beyond the FR text: **embedding is deployment-global** (`LETTA_DEFAULT_EMBEDDING_HANDLE`), not per-agent or per-folder. The resolver, native 768-dim pgvector storage, dual-column passage/archive migration, message and image vectors, object-store-backed image records, ingest/enrichment pipeline, tiered vision render policy, unified `recall` + `fetch_image` tools, and client **Images** tab are all in-tree and deployed on sliver.

**Historic data is intentionally excluded from vector recall** until the separate uplift FR runs: rows with `NULL` 768 `embedding` or `embedding_space_id = legacy-unknown` are filtered by the space guard. New writes (passages, file reading notes, messages, images) populate the 768 column under one shared `embedding_space_id`.

Live testing on sliver (Chat B, Files tab, Images tab) surfaced integration bugs that are fixed in-tree: `fetch_image` hydration, folder ingest padding regression, recall SQL parameter typing, and recall tool output clarity (`[file]` + `filename=`).

**Recommendation for Ada:** The stack is ready to **author and execute the historic embedding uplift FR** once you accept the documented partial gaps below (recall post-processing, TEXT-tier description injection, tool lexical fallback). None of those block uplift itself; they affect agent UX quality on mixed historic/new corpora.

---

## Scope delivered vs FR §14

| FR phase | Deliverable | Status |
|----------|-------------|--------|
| **Phase 1** — Embedding foundation | `EmbeddingConfig` extensions, resolver, provider client MRL/normalize/query override, padding removal | **Done** |
| **Phase 2** — Storage + guard | Alembic `v060`/`v061`, dual-column passages/archives, message vectors, HNSW + `pg_trgm`, atomic version guard | **Done** |
| **Phase 3** — Turbopuffer removal | Messages off tpuf; passage/file paths native when tpuf disabled | **Partial** — tpuf dual-write still optional when configured |
| **Phase 4** — Image records + object store | `images` table, `ImageManager`, S3/MinIO client, pixel embed path | **Done** |
| **Phase 5** — Ingest pipeline | Sync store, background 1MP + VLM captions + pixel embed, two-embed dance, retries | **Done** |
| **Phase 6** — Chat render policy | `LettaImage` refs, render walk, on-demand 1MP, serializer integration | **Partial** — TEXT tier uses placeholder, not image `description` |
| **Phase 7** — Recall tool | Vector + lexical + RRF; `fetch_image` multimodal return | **Partial** — diversity cap, dedup, filter params missing |
| **Phase 8** — Client Images tab | Server REST, client proxy, `Images.svelte` | **Done** |
| **Phase 9** — Base instructions | `recall` / `fetch_image` in `letta_v1.py` | **Done** |

### Acceptance criteria (§15)

| # | Criterion | Status | Notes |
|---|-----------|--------|-------|
| 1 | Fresh agent embeds all corpus types at 768 with shared `embedding_space_id` | **Pass** (new data) | Requires `LETTA_DEFAULT_EMBEDDING_HANDLE` + `LETTA_EMBED_ALL_MESSAGES=true` on deploy |
| 2 | No hardcoded embedding model; tpuf not required to boot/search | **Partial** | Message/recall paths tpuf-less; `TurbopufferClient.default_embedding_config` and optional dual-write remain |
| 3 | Vectors 768, unit-length, no `np.pad` on write/query | **Pass** | `prepare_vector_for_write`; passage schema padding validator removed (see Post-RC fixes) |
| 4 | Cross-space query returns zero vector hits + logs exclusion | **Partial** | `apply_embedding_space_guard` works; **no exclusion-count debug logging** |
| 5 | Image ingest: row, 1MP, captions, pixel embed, hash dedup | **Pass** | `image_ingest.py` + MinIO on sliver |
| 6 | Text query finds image via vector and/or trigram; `fetch_image` returns pixels | **Pass** | Empirical on sliver (Chat B + Images tab) |
| 7 | Long image-heavy chat: tiered render within wire-byte cap | **Partial** | Render walk + on-demand 1MP implemented; TEXT tier lacks description injection |
| 8 | `recall` fused deduped source-capped list | **Partial** | RRF fusion yes; per-source cap and image/message dedup **not implemented** |
| 9 | Enrichment failure: renderable image + text-only message embed | **Pass** | Failure path re-embeds message text |
| 10 | Client Images tab with metadata/actions | **Pass** | view-full, edit metadata, re-enrich, delete |
| 11 | HNSW + pg_trgm; dual-column passages/archives | **Pass** | `v060` + `v061` migrations |

Automated coverage includes: `test_render_policy.py`, `test_image_ingest.py`, `test_recall_service.py`, `test_fetch_image_and_recall_archives.py`, `test_openai_provider_embeddings.py`, embedding resolver tests.

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

### A3. `fetch_image` message hydration

**FR §12** covers multimodal tool return. Additional fix: tool-return `LettaImage` refs with `file_id` but `data: null` are **hydrated from MinIO** on message read (`image_hydration.py`, `message_manager.py`). `fetch_image` tool results are **not re-ingested** as new image records (`image_ingest.py` skip when `message.name == "fetch_image"`).

Empirical: Chat B shows fetched images in UI and passes pixels to the LLM after tool call.

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
| `letta/constants.py` | `DEPLOYMENT_EMBEDDING_DIM = 768`; `MAX_EMBEDDING_DIM` retained for legacy columns |
| `letta/llm_api/openai_client.py` | `dimensions`, `input_type`, query override, image embeddings |

**Deploy default:** `openrouter/google/gemini-embedding-2-preview` at 768-dim MRL, L2-normalized.

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
| `letta/services/image_ingest.py` | Sync hash-dedup store; background 1MP + VLM captions + pixel embed; two-embed dance (v1/v2) |
| `letta/services/object_store/client.py` | Content-addressed S3/MinIO; **wire-byte** size accounting |
| `letta/services/image_fetch.py` | `fetch_image` multimodal tool return |
| `letta/server/rest_api/routers/v1/images.py` | list/get/patch/delete/re-enrich/content/url |

**Caption VLM:** `settings.image_caption_model_handle` (sliver: `openrouter/minimax/minimax-m3` via compose).

### Vision render policy

| Module | Role |
|--------|------|
| `letta/services/vision/render_policy.py` | Byte-budget walk; `supports_image_blocks_in_history` pre-seeded for known models |
| `letta/services/vision/image_hydration.py` | `prepare_messages_for_vision_llm`; on-demand `generate_1mp_now`; tool-return hydration |
| `letta/agents/letta_agent_v2.py`, `letta_agent_v3.py` | Call hydration before `build_request_data` |
| `letta/schemas/message.py` | `image_render_decisions` threaded into `to_openai_dict` LettaImage branch |

**Wire bytes:** `file_size_full` / `file_size_1mp` stored as base64-encoded wire size (`object_store.wire_byte_size`). Cap: `settings.vision_context_byte_cap` (default 20 MiB).

**Gap:** TEXT tier renders `"[Image reference {handle} — use fetch_image…]"` without loading `images.description` from DB (FR §10).

### Unified recall

| Module | Role |
|--------|------|
| `letta/services/recall/recall_service.py` | Hybrid recall: 5 vector tables + trigram lexical + RRF (k=60) |
| `letta/functions/function_sets/recall_tools.py` | Agent tool wrapper |
| `letta/services/tool_executor/core_tool_executor.py` | Executor path |

**Vector leg:** `archival_passages`, `source_passages` (as layer `file`), `file_archives`, `messages`, `images` — all under space guard. Source passages and file archives scoped to agent's attached folders via `sources_agents` join.

**Lexical leg:** `pg_trgm` `similarity()` on passage text, message text, image caption/description/details, file archive title/content.

**Missing vs FR §12:** `layers` / `time_range` / `source` parameters; per-source diversity cap; image+message dedup; neighbor/offset handles for file chunks.

### Turbopuffer

| Path | Status |
|------|--------|
| Message embed/search | **PG-native only** |
| Passage/archive/file ingest | **Optional tpuf** when `use_tpuf()` — dual-write paths remain |
| Tool embed/search | **Still tpuf** when `embed_tools=True`; **no lexical fallback** when disabled (`tool_manager.search_tools_async` raises) |

Boot and recall work tpuf-less on sliver (default). FR §7 r2 lexical tool fallback is **not implemented**.

### Agent tools

`BASE_TOOLS` includes `recall` and `fetch_image` (`letta/constants.py`). Granular search tools remain registered and documented in `letta_v1.py` as follow-ups.

---

## Architecture (client — `letta-vision-client`)

| Surface | Purpose |
|---------|---------|
| **`Images.svelte`** | Grid/list, thumbnails, view full, edit caption/description/details, re-enrich, delete |
| **`backend/routes/images.py`** | Proxy to server `/v1/images/*` + content stream |
| **`App.svelte` / `stores.js`** | Images tab nav |
| **`Files.svelte`** | Folder upload; embedding picker **removed**; improved API error parsing |
| **`Agents.svelte`** | Embedding picker **removed** |
| **`api.js`** | `parseApiError()` for FastAPI nested errors |

---

## Architecture (deploy — `letta-vision-deploy`)

Sliver stack (`docker-compose.yml`):

| Variable | Purpose |
|----------|---------|
| `LETTA_DEFAULT_EMBEDDING_HANDLE` | `openrouter/google/gemini-embedding-2-preview` |
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
| Turbopuffer | §7 remove from path | **Partial** — messages off; optional dual-write when tpuf configured |
| Tool search tpuf-less | §7 lexical fallback | **Missing** — raises if `embed_tools=False` |
| Recall signature | §12 filters + post-steps | **Simplified** — `query` + `limit` only; no diversity/dedup |
| TEXT render tier | §10 description + handle | **Placeholder text only** |
| Space guard logging | §8 exclusion counts | **Not implemented** |
| Recall return shape | §12 layer names | **`file`** + **`filename=`** (Amendment A2) |
| Granular tools demoted | §12 internal only | Still in `BASE_TOOLS`; instructions list as follow-ups |

---

## Post-RC fixes (sliver validation, 2026-06-09)

| Issue | Root cause | Fix |
|-------|------------|-----|
| **`fetch_image` empty in UI/LLM** | Re-ingest stripped `data`; no hydration on read | Skip ingest for `fetch_image`; hydrate from MinIO; inline base64 in tool return |
| **Folder ingest 4096 vs 768 error** | `Passage` schema `pad_embeddings` validator zero-padded to 4096 before `prepare_vector_for_write` | Removed padding validator from `letta/schemas/passage.py` |
| **Per-agent/folder wrong embedding** | Stale UI pickers + resolver honored agent override | Deployment-global resolver; client embedding UI removed |
| **`recall` SQL crash** | `(:agent_id IS NULL OR sa.agent_id = :agent_id)` — asyncpg ambiguous type | Branch SQL on agent_id presence |
| **`recall` missed folder files** | Source passage vector/lexical not scoped to agent folders | Join `sources_agents` for `file` layer |
| **Agent misread recall hits** | Layer `source`, no filename | `file` layer + `filename=` in `format_recall_hit` |
| **File archive hits vague** | No filename in header | `filename=` on `file_archive` hits |

---

## Historic embedding uplift — readiness assessment

### What uplift can assume (already in place)

1. **Target schema:** 768 `embedding` column on `archival_passages`, `source_passages`, `file_archives`, plus `messages` and `images` born at 768.
2. **Source text:** `embedding_legacy_4096` rows retain original passage/archive text for re-embed.
3. **Space guard:** Queries only rank rows with matching `embedding_space_id`; historic NULL/legacy rows abstain from vector leg (lexical may still surface them).
4. **Indexes:** HNSW on 768 columns and GIN trigram indexes created forward-looking (empty/partial until uplift backfill).
5. **Single deployment model:** All backfill should use `LETTA_DEFAULT_EMBEDDING_HANDLE` (gemini-embedding-2-preview @ 768).
6. **Idempotent message writes:** Atomic `embedding_version` guard supports re-embed without races.

### What uplift must do (FR §16 — still the GA gate)

1. Re-embed historic `archival_passages` and `source_passages` from `text` → populate 768 `embedding` + stamp new `embedding_space_id`.
2. Re-embed historic `file_archives` from `title` + `content`.
3. Re-embed pre-v0.6.0 `messages` (and any images missing pixel vectors).
4. Validate HNSW performance and recall quality on full corpus.
5. Drop `embedding_legacy_4096` columns after validation.
6. Cost/time estimate and cutover strategy (shadow vs hard cutover).

### Pre-uplift quality gaps (non-blocking for uplift authoring)

| Gap | Impact on mixed corpus | Recommendation |
|-----|------------------------|----------------|
| Recall diversity/dedup | Noisy top-K until uplift fills vector leg | Implement before GA or accept during alpha |
| TEXT tier without description | Demoted images less useful in context | Load `images.description` in serializer |
| Tool lexical fallback | Tool picker search fails tpuf-less | Small follow-up PR |
| Space guard logging | Harder to observe mixed-space during uplift | Add debug log with exclusion counts |

---

## Empirical validation (sliver)

| Test | Result |
|------|--------|
| Folder file ingest (`villains.txt`, `heroes.txt`) | **Pass** after padding fix + deployment embedding |
| `recall("Victoria")` | **Pass** — finds file passage + file archive note |
| `recall` output clarity | **Pass** after `file` + `filename=` formatting |
| `write_file_archive` on `villains.txt` | **Pass** |
| `fetch_image` in Chat B | **Pass** — UI + LLM receive pixels |
| Images tab | **Pass** — list, thumbnails, metadata edit |
| Chat with vision model | **Pass** with render policy + hydration |

Agent `v060 test` on folder `v060-test`; deployment embedding `gemini-embedding-2-preview`.

---

## Open items (non-blocking for uplift FR)

| Item | Recommendation |
|------|----------------|
| Historic uplift FR | **Next** — gate to v0.6.0 GA |
| Recall diversity cap + dedup | Implement before GA or document as alpha limitation |
| TEXT tier description injection | §10 fidelity — load from `images` row |
| Tool lexical fallback | §7 r2 — `ILIKE` on name/description when tpuf off |
| Space guard exclusion logging | Observability for uplift validation |
| Remove tpuf dual-write paths | Cleanup after uplift if tpuf permanently retired |
| `content_hash` / `embedding_space_id` in Images UI | Cosmetic |
| Kimi-K2.6 `supports_image_blocks_in_history` | Confirm empirically on sliver |

---

## Recommendation to Ada

**Proceed with the historic embedding uplift FR.** The v0.6-rc implementation delivers Ades's core design property for *new* data:

- One embedding space (768-dim gemini-embedding-2-preview, deployment-wide)
- Native pgvector with space guard (no silent cross-model garbage)
- Images as first-class searchable records with object-store bytes and tiered chat render
- Unified `recall` over passages, files, file reading notes, messages, and images
- Reference-then-fetch via `fetch_image`

The uplift FR can target backfilling `embedding_legacy_4096` → 768 `embedding` across passages and file archives, re-embedding historic messages, building a full HNSW corpus, and dropping legacy columns. Documented partial gaps (recall post-processing, TEXT tier copy, tool fallback) are UX/quality items, not schema blockers.

---

## Key file index

### Server (`letta-vision`)

| Path | Role |
|------|------|
| `alembic/versions/v060_unified_embedding_multimodal_recall.py` | Core schema migration |
| `alembic/versions/v061_file_archive_unified_embedding.py` | File archive embedding parity |
| `letta/schemas/embedding_config.py` | Config + `compute_space_id` |
| `letta/embeddings/resolver.py` | Deployment-wide resolver |
| `letta/embeddings/util.py` | `prepare_vector_for_write` |
| `letta/embeddings/query.py` | Space guard + query embed |
| `letta/embeddings/write.py` | Atomic message embed write |
| `letta/orm/image.py` | Images table |
| `letta/orm/message.py` | Message vector columns |
| `letta/orm/passage.py` | Dual-column passages |
| `letta/services/image_ingest.py` | Ingest + enrichment |
| `letta/services/image_fetch.py` | `fetch_image` tool |
| `letta/services/image_manager.py` | Image CRUD |
| `letta/services/object_store/client.py` | MinIO/S3 client |
| `letta/services/recall/recall_service.py` | Unified recall + `format_recall_hit` |
| `letta/services/vision/render_policy.py` | Byte-budget render walk |
| `letta/services/vision/image_hydration.py` | LLM message hydration |
| `letta/services/message_manager.py` | PG message embed + search |
| `letta/services/passage_manager.py` | 768 passage writes |
| `letta/services/file_archive_embedding.py` | Archive 768 writes |
| `letta/server/rest_api/routers/v1/images.py` | Images REST |
| `letta/server/rest_api/routers/v1/folders.py` | Deployment embedding on ingest |
| `letta/server/server.py` | Agent create — deployment embedding only |
| `letta/prompts/system_prompts/letta_v1.py` | `recall` / `fetch_image` instructions |
| `letta/constants.py` | `DEPLOYMENT_EMBEDDING_DIM`, `BASE_TOOLS` |

### Client (`letta-vision-client`)

| Path | Role |
|------|------|
| `frontend/src/routes/Images.svelte` | Images tab |
| `backend/routes/images.py` | Images API proxy |
| `frontend/src/routes/Files.svelte` | Files (no embedding picker) |
| `frontend/src/routes/Agents.svelte` | Agents (no embedding picker) |
| `frontend/src/lib/api.js` | API error parsing |

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
| (6) Pre-flag `supports_image_blocks_in_history` | **Yes** — `render_policy.py` |
| Caption VLM separate setting | **Yes** — `image_caption_model_handle` |
| `file_archives` embedding parity (§4.5b) | **Yes** — `v061` |
| Five-table recall vector leg (§12) | **Yes** |
| `pg_trgm` lexical leg (§12) | **Yes** |
| RRF fusion (§12) | **Yes** |
| Per-source diversity + dedup (§12) | **No** |
| Deployment-global embedding (Amendment A1) | **Yes** — **not in FR text**; documented here |
| Recall `file` + `filename=` output (A2) | **Yes** — **not in FR text**; documented here |

---

*End of report.*
