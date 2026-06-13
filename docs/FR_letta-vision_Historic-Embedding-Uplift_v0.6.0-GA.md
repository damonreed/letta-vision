# FR: Historic Embedding Uplift & Corpus Conversion

Status: **GA signed off** (2026-06-12) — uplift **executed and validated on sliver** (2026-06-11); client §12–§13 shipped; post-uplift operational fixes landed (see §14); `embedding_legacy_4096` dropped (alembic `v062` + ORM); **v0.6.0 tagged** with OpenRouter `input_modalities` vision detection (`v063`, §14.8)
Author: Ada (architecture); implementation by Cursor
Depends on: v0.6.0-rc (shipped & validated on sliver against new data) — see `IMPLEMENTATION_REPORT_v0.6.0_unified-embedding-recall.md`
Grounded in as-built: five vector tables (`archival_passages`, `source_passages`, `file_archives`, `messages`, `images`); single 768-dim `embedding` column on passages/archives (legacy 4096 column dropped at GA); deployment-global embedding (`LETTA_DEFAULT_EMBEDDING_HANDLE` = `openrouter/google/gemini-embedding-2` @768); atomic message embed guard (`embeddings/write.py::write_message_embedding_atomic`); content-addressed object store with wire-byte sizing (`object_store/client.py`); `image_ingest.py` sync+background pipeline.

This FR runs in **two ordered parts**. Part 1 (base64→object conversion) must complete before Part 2 (re-embed), because Part 1 creates the image records Part 2 embeds, and the message re-embed in Part 2 folds in image caption gists that don't exist until Part 1's records are enriched.

---

## 1. Problem / Context

v0.6.0-rc moved all *new* writes into the unified 768 gemini-embedding-2 space and made images first-class object-store records. Two bodies of historic data remained outside that world until uplift (now **resolved on sliver**, 2026-06-11):

1. **Inline base64 images in `messages.content`.** Pre-rc (v0.3.0 vision) messages carried image bytes inline as base64 `ImageContent` blocks — not `images` records, not in the object store, not pixel-embedded, and bloating the `messages` table. Part 1 converted recoverable blocks to `LettaImage` refs + object-store blobs.
2. **Un-embedded historic rows.** `archival_passages`, `source_passages`, and `file_archives` had `NULL` in the 768 `embedding` column (historic vectors previously in a padded 4096 column, since dropped). Pre-rc `messages` had no vector. Part 2 re-embedded the full corpus under GA space id `6490a4b17e06a258`.

Recall now covers the full migrated corpus. **`embedding_legacy_4096` dropped** at GA (alembic `v062_drop_legacy_emb` ships with ORM mapping removal).

## 2. Goals

- **Part 1:** Convert every recoverable inline base64 image in `messages.content` into an `images` record + object-store blob + lightweight `LettaImage` reference, rewriting the message content to the reference form. Deduplicate by content hash against the existing `images` table. Idempotent and resumable. Net effect: large `messages` size reduction and historic images become enrichable/searchable.
- **Part 2:** Roll-fill the 768 `embedding` column across all five vector tables from retained source text/pixels, under the single deployment space id, resumably and idempotently, then validate recall quality and HNSW performance on the full corpus, then drop the legacy 4096 column (shipped at GA).
- Land the two recall post-processing gaps (per-source diversity cap + image/message dedup) **with** this uplift, since uplift is what fills the vector leg and makes their absence visible.
- **Images tab UI** (client): replace the current card grid with the same list + detail shell used by Agents / Providers / Files / MCP — thumbnail rail on the left, selected-image workspace on the right. See §12.
- **Client shell** (client): Chat-first nav, mount-once Chat, windowed history, conversation sidebar without N+1. See §13.
- No model change. The space id is the existing deployment config's `compute_space_id()`. (If gemini-embedding-2 leaves preview mid-effort, that is a *different* space and a *different* uplift — see §9.)

## 3. Conceptual Model

**No cutover event — the schema is already shadow-column.** The 768 `embedding` is the shadow; NULL until filled. The space guard (`embeddings/query.py::apply_embedding_space_guard`) already excludes NULL/`legacy-unknown` rows from vector ranking. So backfill is a *rolling fill*: each row joins the vector leg the instant it has a 768 vector + the deployment space id; recall coverage increases monotonically; nothing flips. The only discrete step is dropping the legacy column after validation. This answers the report's "shadow vs hard cutover" question: it is shadow, by construction.

**Strict part ordering.** Part 1 → (image enrichment) → Part 2 message re-embed. A message that referenced an inline image must, after conversion, re-embed with that image's caption gist (the two-embed pattern), so the image record and its caption must exist first.

**Conversion is lossless and forward-only.** Decoded bytes are preserved in the object store (content-addressed); the original base64 is reconstructable from the ref if ever needed. The message-content rewrite is a mutation of the historic event log, so it is gated behind a DB snapshot and a dry-run (see §4).

## 4. Part 1 — Base64 → Object Conversion

New module: `letta/services/migration/image_base64_conversion.py`. Reuse the existing ingest primitives — do **not** fork a parallel store/insert path.

**Scan & classify.** Iterate `messages` (resumable cursor on `(created_at, id)`), inspect each `content` block:
- `ImageContent` with a base64 source (inline `data` / data-URL) → **convert**.
- `ImageContent` already `LettaImage` (`type=letta`, `file_id`) → **skip** (already converted).
- A text/placeholder block from the old decayed serializer (`"[Image ...]"` with no recoverable bytes) → **skip & count** (nothing to recover; report these so the loss from the old decay bug is visible).
- Non-image blocks → untouched.

**Convert (per base64 block):**
1. Decode → bytes → `content_hash = sha256(bytes)`.
2. Dedup: if an `images` row with `(org, content_hash)` exists (from rc live use or an earlier message in this run), reuse its id — no re-store, no new record.
3. Else call the **existing synchronous ingest primitive** (`image_ingest.py` sync phase): store full bytes (wire-byte sized), insert `images` row (`provenance="uploaded"` unless the message is a tool return — preserve `generated` + `generation_prompt` where derivable; `enrichment_status=pending`).
4. Rewrite the block in place to `LettaImage(file_id=<image-id>, data=None)`, preserving the block's position in the content list.
5. Persist the rewritten `messages.content`.

**Enrichment.** Newly-created `pending` records are picked up by the normal background enrichment (1MP + structured VLM captions + pixel embed). For a bulk historic run, drive enrichment explicitly (a batched pass over `enrichment_status IN (pending, failed)`) rather than relying on incidental scheduling, so Part 2 can depend on a known-complete state. This pass is shared with Part 2's image step (§5.3) — they are the same work; run it once, after conversion.

**Idempotency & safety.**
- Re-running converts only base64 blocks; already-`letta` blocks are skipped, so the job is safe to resume/repeat.
- `--dry-run`: report counts (convertible blocks, unrecoverable placeholders, dedup hits, distinct new images) and **estimated `messages` size reduction** without writing.
- Require a Postgres snapshot before the live run (it mutates the event log). Document the snapshot step in the runbook; the conversion itself is lossless but in-place.

**Acceptance (Part 1):** zero `ImageContent` base64 blocks remain in `messages.content` (all converted or are unrecoverable placeholders, counted); each converted image is an `images` row with bytes in the object store; N identical inline copies collapse to one record with N references; measured `messages` table size drop reported.

## 5. Part 2 — Unified Rolling Re-embed

New module: `letta/services/migration/historic_reembed.py`. One resumable, idempotent, throttled job with per-table passes. Target space id = `resolve_deployment_embedding_config().compute_space_id()`. Use the deployment resolver, `prepare_vector_for_write` (768 + L2 normalize), and the existing atomic message write.

**Run with tpuf dual-write disabled** (sliver default). The optional tpuf dual-write paths are slated for removal post-uplift; the backfill targets PG 768 only.

### 5.1 Passages & file archives (independent, parallelizable)
- `archival_passages`, `source_passages`: re-embed from the retained `text` column (input_type `search_document`).
- `file_archives`: re-embed from `title` + `content`.
- For each: `WHERE embedding IS NULL OR embedding_space_id IS DISTINCT FROM :target_space_id`, batched on `(created_at, id)`. Write the 768 vector + stamp `embedding_space_id`. Idempotent via the WHERE predicate (a re-run skips already-filled rows).

### 5.2 Order constraint
Images (5.3) must reach `enrichment_status=complete` **before** the message pass (5.4), because converted-image messages re-embed with the caption gist. Passages/archives (5.1) have no such dependency and may run anytime/in parallel.

### 5.3 Images
`WHERE embedding IS NULL OR embedding_space_id IS DISTINCT FROM :target OR enrichment_status='failed'`. For each: ensure 1MP exists (generate if absent), ensure the three caption tiers exist (VLM via `image_caption_model_handle`), pixel-embed via gemini-2 image input → 768 normalized → stamp space id → `enrichment_status=complete`. Content-hash addressing makes retries cheap (no re-store). This pass subsumes Part 1's enrichment pass — run it once here.

### 5.4 Messages
`WHERE embedding IS NULL OR embedding_space_id IS DISTINCT FROM :target`. Embed message text; for messages carrying a `LettaImage` ref whose image is now enriched, include the short caption gist (a sentence, not the full description) before embedding. Write via `write_message_embedding_atomic` with a strictly higher `embedding_version` so the monotonic guard supersedes any prior text-only vector without racing. Pre-rc messages (no prior vector) embed fresh; rc-era text-only messages bump version.

### 5.5 Throttle, cost, resume
- **Cost estimate first.** Extend the existing `estimate_embeddings_size` to produce a pre-run report: row counts per table, token estimates, projected OpenRouter call count and rough cost/time. Print and require confirm before a live run.
- **Throttle + retry.** gemini-embedding-2-preview via OpenRouter is a preview endpoint with rate limits; the job rate-limits, retries with backoff, and checkpoints the `(created_at, id)` cursor per table so an interrupted run resumes without re-embedding completed rows.
- **HNSW fill strategy.** Default: incremental insert into the existing (currently near-empty) HNSW indexes — fine at sliver/personal-corpus scale. If a table's backfill is large enough that per-row index maintenance dominates, drop the HNSW index, bulk-fill, then `CREATE INDEX` + `ANALYZE`. Decide per-table at run time based on the cost estimate; document which path was taken.

## 6. Coupled recall fix (lands with this uplift)

Implement the two deferred recall post-steps now, because uplift fills the vector leg and exposes them:
- **Per-source diversity cap** (default 2–3 hits per file/conversation) in `recall_service.py`.
- **Image↔message dedup**: collapse an `images` hit and the `messages` hit that references it into one result with two reasons. After Part 1, many messages reference images, so this becomes common immediately.

(Optional, recommended alongside: the TEXT-tier `description` injection and space-guard exclusion logging from the report's gap list — the logging in particular is useful *during* the backfill to watch coverage climb. Treat description injection as in-scope-if-cheap, logging as in-scope for observability.)

## 7. Validation (the GA gate)

- **Coverage:** zero rows with `embedding IS NULL` across all five tables (excluding unrecoverable image placeholders); zero `embedding_space_id='legacy-unknown'` in the vector path.
- **Quality (empirical, the real gate):** on the full migrated corpus, run a fixed query set with known-good answers — including (a) text-finds-image cases, (b) literal-identifier lookups that must come from the trigram leg, (c) historic-memory recalls that were invisible pre-uplift — and judge the ranked results by eye. Green plumbing is not the bar; correct top-K is.
- **HNSW:** index built/usable on the full corpus; spot-check latency vs the prior flat scan.
- **Part 1 size:** report `messages` table size before/after.
- **Dedup/diversity:** verify a chunked file no longer floods top-K and an image+referencing-message appear once.
- **Then and only then:** drop the legacy 4096 column from `archival_passages`, `source_passages`, `file_archives` (final migration, shipped `v062`). Keep a snapshot until post-drop validation passes.

### 7.1 Sliver validation results (2026-06-11)

Uplift executed on sliver via `scripts/historic_uplift.py` (Part 1 convert → enrich-pending → Part 2 reembed across `archival_passages`, `source_passages`, `file_archives`, `messages`, `images`). Deployment embedding pinned to GA handle `openrouter/google/gemini-embedding-2` @768; space id `6490a4b17e06a258`.

| Gate | Result | Notes |
|------|--------|-------|
| Coverage | **Pass** | All five vector tables filled under the GA space id |
| Cross-layer recall | **Pass** | `search_all` fuses archival, file content, file archives, messages, and images with dedup + per-source cap |
| Layer tools | **Pass** | `archival_memory_search`, `file_contents_search`, `file_archives_search`, `conversation_search`, `image_search` each return ranked hits independently |
| Historic memory | **Pass** | Pre-uplift archival rows searchable after re-embed |
| Image memory | **Pass** | New Lyra image sessions ingest, enrich, pixel-embed, and surface via `image_search` / `search_all` |
| `image_fetch` | **Pass** | Multimodal return + hydration in live chat |
| HNSW latency | **Pass** | Acceptable at personal-corpus scale on sliver |
| Part 1 size reduction | **Pass** | Inline base64 removed from message content; images in object store |
| Dedup/diversity | **Pass** | Image+message pairs collapse; chunked files do not flood top-K |

**Empirical agent validation:** Lyra (`agent-35a1c263…`) exercised hybrid search across layers, archival insert/search, file reading notes, and multiple new image generations — system behavior judged solid end-to-end.

**Legacy column drop:** shipped at GA (`v062_drop_legacy_emb` + ORM); snapshot retained through post-drop validation.

**Known content/query caveats (not indexing bugs):**
- `file_contents_search` embeds passage **text only** — filename tokens (e.g. `ada_attire_notes.txt`) are not in the vector payload; literal-name queries may miss unless the name appears in passage body. Trigram leg can still surface weak matches.
- Immediate post-insert archival search failed for one live insert because `create_agent_passage_async` omitted `embedding_space_id` on write (§14.1) — fixed; one row backfilled manually on sliver.

## 8. Non-Goals

- Model change / second embedding space (this uplift is into the *existing* deployment space).
- Tool semantic search / PG tool vectors (still out of scope; lexical fallback is a separate small PR per the report).
- Audio/video embedding; summary-text image embedding; two-stage MRL retrieval.
- Removing the optional tpuf dual-write code paths (separate cleanup once tpuf is permanently retired).

## 9. Open Questions (with Defaults)

- **Preview→GA mid-run.** **Resolved on sliver:** uplift completed under preview; deployment then moved to GA handle `openrouter/google/gemini-embedding-2` with a full corpus re-embed into space `6490a4b17e06a258`. Future handle changes remain a new uplift per the space-guard model.
- **Unrecoverable placeholders.** Default: count and report; leave the message text as-is (no synthetic image). These are casualties of the old decay bug and cannot be recovered.
- **Enrichment cost ceiling.** Captioning every historic image is a VLM cost. Default: caption all; if the historic image volume makes that expensive, allow a flag to pixel-embed first (restores searchability) and backfill captions lazily — embedding is the searchability-critical step, captions are payload.
- **Provenance of converted images.** Default `uploaded`; preserve `generated` + `generation_prompt` only where the source message is a recoverable tool return.

## 10. Cursor Implementation Notes

- Reuse `image_ingest.py` sync primitive and `object_store/client.py` in Part 1 — the only net-new code is the message scan/classify and the in-place content rewrite.
- Both parts are management jobs (CLI/manage command), resumable, `--dry-run` first, with per-table `(created_at, id)` checkpoints. Not alembic — alembic only does the final legacy 4096 column drop (shipped as `v062`).
- The idempotency predicates (`embedding IS NULL OR embedding_space_id IS DISTINCT FROM :target`) are the resume mechanism; trust them over external state **except for Part 2 live runs with `--resume` (default)** — see runbook notes below.
- Run order in one orchestrated command: Part 1 convert → image enrich pass → {passages/archives in parallel} + message re-embed (after image enrich) → validate → (manual gate) → drop legacy column.
- Land the diversity cap + dedup (§6) in the same release; add space-guard exclusion logging early so the backfill is observable.
- `estimate_embeddings_size` extension prints the cost report and is also the dry-run for Part 2.
- Images tab (§12–§13): `Images.svelte` list+detail inspector, `POST /v1/images/search`, paginated `GET /v1/images`, Chat default tab + mount-once + history window — see implementation report GA client section.

### 10.1 Operations runbook

**Message uplift version constant.** `UPLIFT_MESSAGE_EMBED_VERSION` in `historic_reembed.py` is a fixed write target (currently `3`) while the selection predicate uses `embedding_version < MESSAGE_EMBED_VERSION` (`2`). Consequences: (a) a future space migration re-running this job without bumping the constant will have the monotonic guard reject every row already at v3 (`3 < 3` is false → counted as failed); (b) any steady-state re-run re-embeds every live v1 message unnecessarily (cost, not correctness). **Before each uplift:** bump `UPLIFT_MESSAGE_EMBED_VERSION` to `max(existing)+1` (or at least above the prior uplift's target).

**Part 2 checkpoint vs idempotency.** Part 2's checkpoint cursor takes precedence over the idempotency predicates when `--resume` is set (default). The Part 2 checkpoint is **not** deleted on completion (Part 1's is). A batch embed failure advances the cursor past the failed batch before raising — failed rows are never retried under default `resume=True`. The zero-NULLs coverage gate catches this, but if `failed > 0` after a run: re-run with `--no-resume`, then delete `~/.letta/uplift_part2_checkpoint.json` after a clean run.

## 11. Risk Register

| Risk | Mitigation |
|------|------------|
| Event-log mutation in Part 1 | DB snapshot + dry-run + lossless (bytes in object store) + idempotent rewrite |
| Message re-embed races image enrichment | Strict order (5.2): images complete before message pass; atomic version guard |
| OpenRouter preview rate limits / drift | Throttle + backoff + resumable cursor; pin handle; cost estimate before run |
| Large HNSW incremental fill slow | Per-table drop/bulk-fill/recreate option based on cost estimate |
| Dedup/diversity absence surfaces at GA | Land §6 with the uplift, not after |
| Dropping legacy column too early | Drop only after full validation; keep snapshot through post-drop check |
| Partial run leaves mixed state | Space guard keeps mixed state *correct* (rolling fill); resume continues from cursor |
| New insert omits `embedding_space_id` | Fixed §14.1 — all create paths stamp space id; one pre-fix row backfilled on sliver |
| Legacy `POST /v1/sources/{id}/upload` path | Fixed §14.1 — `DirectoryConnector` now uses `create_many_source_passages_async` |

## 12. Images Tab UI (client)

**Why now.** Part 1 will convert historic inline images into first-class `images` records; Part 2 enriches them with the three text tiers (`caption`, `description`, `details`). The current `Images.svelte` card grid does not scale to a full corpus and buries the editable metadata behind a modal. GA should ship an inspector that matches the rest of the admin shell and makes caption review/editing a first-class workflow.

**Layout.** Reuse the `*-layout` / `aside.list` + `section.detail` pattern from `Agents.svelte`, `Files.svelte`, `Mcp.svelte` (`grid-template-columns: 280px 1fr`; full viewport height below the tab bar).

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│  Images (142)  [ search query…………………………… ] [10 ▾] [Search] [Clear]     [Refresh]      │
├──────────────┬───────────────────────────────────────────────────────────────────────────┤
│              │                                                                           │
│  ┌────────┐  │  ┌─────────────────────────────┬─────────────────────────────────────────┐ │
│  │ 1MP    │◀─│  │                             │  Caption                                │ │
│  │ 0.87   │  │  │                             │  ┌───────────────────────────────────┐  │ │
│  └────────┘  │  │   [ 1MP preview — click ]   │  │ Short label (20–50 words)         │  │ │
│  ┌────────┐  │  │   zoom → full on click      │  └───────────────────────────────────┘  │ │
│  │ 1MP    │  │  │   click again → dismiss     │                                         │ │
│  │ 0.72   │  │  │                             │  Description                            │ │
│  └────────┘  │  │  ID        image-abc…        │  ┌───────────────────────────────────┐  │ │
│  ┌────────┐  │  │  Hash      sha256:9f3a…     │  │ Search-oriented summary           │  │ │
│  │ 1MP    │  │  │  Size      1.2 MB (384 KB)  │  │ (100–200 words)                   │  │ │
│  │ 0.65   │  │  │  Dims      2048 × 1536      │  │                                   │  │ │
│  └────────┘  │  │  Type      image/jpeg       │  └───────────────────────────────────┘  │ │
│  ┌────────┐  │  │  Provenance uploaded        │                                         │ │
│  │ 1MP    │  │  │  Enrichment  complete ✓     │  Details                                │ │
│  │ 0.58   │  │  │  Space id  gemini-emb…@768  │  ┌───────────────────────────────────┐  │ │
│  └────────┘  │  │  Created   2025-11-04 14:22 │  │ Extended literal description      │  │ │
│      ⋮       │  │  Updated   2025-11-04 14:23 │  │ (up to ~1000 words)                │  │ │
│  (scroll)    │  │                             │  │                                   │  │ │
│              │  │  [Re-enrich]  [Delete]      │  └───────────────────────────────────┘  │ │
│              │  └─────────────────────────────┴─────────────────────────────────────────┘ │
│   ~280px     │                        detail pane: 50% preview+meta | 50% editors          │
└──────────────┴───────────────────────────────────────────────────────────────────────────┘
     score overlay on thumbs when search active
```

**Image variants (rail + pane)**
- **Always prefer 1MP** for rail thumbnails and the detail-pane preview via `imageThumbnailPath` / `imageContentPath(id, { variant: "1mp" })` when `object_url_1mp` exists; fall back to full variant otherwise.
- **No separate “View full” control.** Click the pane preview to open a full-resolution overlay (reuse `ImageViewer.svelte` with `variant=full`). Click the zoomed image or backdrop again to dismiss. `Escape` also dismisses.

**Header — search**
- Inline search bar immediately after the “Images” title (same header row as Refresh).
- Controls in order: **query input** → **result-limit dropdown** (`5` / `10` / `20` / `50`, default `10`) → **Search** button → **Clear** button.
- **Enter** in the query field triggers Search (same as clicking Search).
- **Search** calls the same hybrid path as the `image_search` agent tool: `search_images_hybrid` in `letta/services/recall/hybrid_search.py` (vector + lexical RRF fuse). Wire a new REST endpoint `POST /v1/images/search` `{ query, limit? }` returning `{ results: [{ handle, description, score }] }` — mirror of `core_tool_executor.image_search` — plus a client proxy `POST /api/images/search`.
- **Clear** empties the query, exits search mode, and restores the default browse list (paginated scroll-load).
- While a search is active, the left rail shows **ranked hits only** (not the browse list). Each thumb gets a **score badge overlay** (two decimal places, e.g. `0.87`). Preserve rank order from the API.
- `limit` dropdown values (`5`, `10`, `20`, `50`) pass through to the search call; default `10`.
- Empty search results: “No matches” in the rail; detail pane shows the standard empty placeholder.

**Left rail — browse mode (default)**
- Thumbnail only (no id/caption text in the rail). Square crop, `object-fit: cover`, selected state highlight matching `.list li button.selected`.
- **Lazy dynamic load:** do not fetch the full org list up front. Load pages as the user scrolls (IntersectionObserver sentinel at list bottom). Thumbnails use `loading="lazy"` + 1MP-preferred path above.
- Empty state: “No images yet” placeholder in the detail pane (same as MCP/Files).

**Detail pane — 50 / 50 split**
- **Left half (preview + read-only metadata):**
  - Primary preview: 1MP (full fallback), click-to-zoom as above.
  - `dl.meta-grid` fields: id, content_hash (truncated, copy affordance), `file_size_full` / `file_size_1mp`, width×height, `media_type`, `provenance`, `generation_prompt` (if generated), `enrichment_status`, `enrichment_attempts`, `error_message` (if failed), `embedding_space_id`, `created_at`, `updated_at`.
  - Actions in header or footer: Re-enrich, Delete (with confirm).
- **Right half (inline editors):**
  - Always-visible fields for `caption`, `description`, `details` — same hints/word-tier copy as today’s edit modal.
  - Save / Cancel (or debounced auto-save — default: explicit Save to match Agents inline-edit pattern).
  - PATCH via existing `api.updateImage`; on save, update list item in place (no full reload).

**Non-goals for this UI pass**
- Upload / ingest from the Images tab (images still enter via chat/tool returns).
- Bulk edit or tagging.
- Embedding vector inspection.
- Agent-scoped search (`agent_id` filter on `image_search`) — tab search is org-wide, matching admin inspection use.

**Acceptance**
- Layout matches Agents/Files/MCP list+detail shell; no card grid remains.
- Rail and pane both use 1MP with full fallback; no “View full” button.
- Click pane preview → full-res zoom; click zoomed image or backdrop → dismiss.
- Search bar with limit dropdown, Search/Clear buttons; Enter triggers search.
- Active search replaces browse rail with ranked results and per-thumb score overlays.
- Selecting an image in a 100+ image browse corpus does not require loading all thumbnails before interaction (paginated/incremental list).
- Caption/description/details editable inline without a modal; save round-trips through PATCH.
- All metadata fields from `PydanticImage` that are user-meaningful are visible on the preview half.

**Implementation notes**
- Refactor `letta-vision-client/frontend/src/routes/Images.svelte`; extract subcomponents if needed (`ImageListRail`, `ImageDetailPane`, `ImageSearchBar`) but keep one route file unless it grows past ~400 lines.
- Reuse shared CSS tokens from `Mcp.svelte` / `Files.svelte` (`.list`, `.detail`, `.meta-grid`, `.detail-header`).
- **New API:** `POST /v1/images/search` in `letta/server/rest_api/routers/v1/images.py` delegating to `search_images_hybrid`; client `backend/routes/images.py` + `api.searchImages` in `api.js`.
- **Browse pagination:** `GET /v1/images` returns `{ images, has_more }` with `limit`, `after_created_at`, and `after_id` cursor params (`ImageManager.list_async`). Client browse uses page size 50 + IntersectionObserver sentinel.

## 13. Client shell (GA polish)

**Tab order & default.** Nav order: **Chat → Images → Files → Agents → MCP → Providers**. **Chat** is the default tab on load (`currentTab` default + `initFromHash` fallback). Hash routing unchanged per tab (`#chat`, `#images`, etc.).

**Chat mount-once.** `App.svelte` mounts Chat on first visit and keeps it alive (hidden via CSS) so tab switches do not abort streams, lose drafts, or re-fetch history.

**Chat performance (shipped with GA client).**
- **Conversation sidebar:** list endpoint only — no per-conversation `messages.list(limit=1)` preview fetch; sidebar shows name + `last_message_at`; sort by `last_message_at` desc.
- **History window:** `GET /history` returns `{ messages, has_more }` (default `limit=50`); “Load older messages” at thread top via `before` cursor; `full=true` for system-context modal fallback only.
- **Deferred memory:** agent blocks/open-files load when the Memory panel opens, not on every agent select.

## 14. Post-uplift operational fixes (2026-06-11)

These landed during live sliver validation after uplift execution. They are part of the GA release, not follow-ups.

### 14.1 Archival insert missing `embedding_space_id`

**Symptom:** `archival_memory_insert` succeeded (embedding present, tags correct) but immediate `archival_memory_search` returned only older memories; timestamps/tags in results looked “wrong” because they belonged to pre-existing hits, not the new passage.

**Root cause:** `_prepare_passage_embedding_fields()` stamped `embedding_space_id` on the data dict, but `create_agent_passage_async` / batch / `create_source_passage_async` did not pass it into ORM `common_fields`. New inserts had `embedding_space_id = NULL` and were excluded from the vector leg by `apply_embedding_space_guard()`. Lexical RRF could surface them weakly or not at all depending on query wording.

**Fix:** Include `embedding_space_id` in `common_fields` for all three create paths (`passage_manager.py`). One sliver row backfilled manually.

**Same bug class on legacy source upload.** The deprecated `POST /v1/sources/{id}/upload` → `load_file_to_source` → `DirectoryConnector` → `create_many_passages_async` path bypassed `_prepare_passage_embedding_fields` (no `embedding_space_id`, no `prepare_vector_for_write`). The live folders route uses `FileProcessor` → `create_many_source_passages_async` (fixed), so sliver validation never exercised it. **Fix:** `connectors.load_data` now calls `create_many_source_passages_async` with `file_metadata`.

### 14.2 Tool return serialization

| Tool | Issue | Fix |
|------|-------|-----|
| `archival_memory_search` | Return value not serialized to agent (empty/malformed tool result) | Return `{"message": …, "results": […]}` dict |
| `archival_memory_insert` | Returned `None` despite docstring promising confirmation + ID | Return `{message, results: [{id, timestamp, tags}]}` |

### 14.3 Reasoning models and tool execution

MiniMax M3 and Kimi K2.6 with `enable_reasoner: true` sometimes express tool intent in reasoning blocks and finish the turn without `tool_calls`. **Mitigation validated on sliver:** explicit agent `llm_config` PATCH — set `enable_reasoner: false` when tool reliability is critical; re-enable with `enable_reasoner: true` only (do not pass `"reasoning": true` on v1 agents — policy forces reasoner off for some providers).

### 14.4 MiniMax duplicate thinking display

When reasoning is extracted separately, MiniMax may echo the same analysis in `reasoning_message` and again as inline redacted_thinking markup in assistant text. **Fix:** `strip_duplicate_thinking_from_assistant_text()` in `minimax_openai.py`, applied in the streaming interface when reasoning was extracted separately.

### 14.5 File delete UX (client)

Deleting a file hung because the backend blocked on `recompile_conversations_for_folder()` synchronously. **Fix:** background recompile + optimistic UI removal on Files page (`letta-vision-client`).

### 14.6 Message uplift monotonic guard

Historic message re-embed writes were blocked when `embedding_version` guard rejected v2 updates. **Fix:** guard adjustment so uplift writes can supersede prior text-only vectors (`embeddings/write.py` path used by historic re-embed).

### 14.7 Historic tool-return byte strip

**Problem:** Historic `messages.tool_returns` could retain persisted base64 image bytes from `fetch_image` and similar tools, bloating the messages table after Part 1 converted inline content images.

**Fix:** `letta/services/migration/tool_return_byte_strip.py` — resumable CLI subcommand `strip-tool-returns` (see `scripts/historic_uplift.py`). Rewrites tool returns to lightweight refs; idempotent with checkpoint at `~/.letta/uplift_tool_return_strip_checkpoint.json`.

### 14.8 OpenRouter vision detection (v0.6.0 release)

**Problem:** `supports_vision` for `openrouter/*` models came only from a curated server registry. Registry globs could false-positive text-only routed models (e.g. `openrouter/deepseek/deepseek-v4-pro` lists `input_modalities: ["text"]` on OpenRouter but matched a broad pattern). Operators could not trust `/v1/models` or the Providers UI attach gate without manual overrides.

**Fix (shipped at v0.6.0 tag):** OpenRouter `GET /v1/models` → `architecture.input_modalities` is the authoritative signal for `openrouter/*` handles. Resolution order: `model_overrides.json` (manual) → OpenRouter catalog cache → registry globs for BYOK / non-OpenRouter paths. Flags are stamped at provider sync on `provider_models.supports_vision` (alembic `v063_provider_models_vision`) and warmed into an in-memory cache at startup so `render_policy`, message validation, and `/v1/models` stay consistent without per-request OR calls.

**Operator notes:**
- After upgrade, restart the server once so startup sync repopulates cache + DB column.
- Remove erroneous manual overrides for text-only OR models (e.g. deepseek-v4-pro) — auto detection should show Vision **No**.
- BYOK and `openai-proxy/*` handles still use the curated registry + `LETTA_VISION_MODELS_EXTRA`.
