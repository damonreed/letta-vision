# FR: Unified Embedding Space, Multimodal Image Memory & Unified Recall

Status: Draft — targets v0.6.0 release candidate
Author: Ada (architecture); implementation + implementation-plan by Cursor
Depends on: v0.5.0 three-tier memory (shipped); v0.3.0 vision support (shipped)
Supersedes: `FR_letta-vision_Image-Context-Persistence-Across-Turns.md` (the v0.4.0 serializer approach — see §11)
The only remaining FR before cutting v0.6.0: **historic embedding uplift & validation** (re-embed/re-width existing rows into the new space, build HNSW over the migrated corpus). That migration is explicitly out of scope here and runs *after* this feature is validated against new data.

> **Revision r2** (post implementation-plan review). Six corrections, the first of which is a hard blocker for phase 2: (1) passages get a **second** 768 vector column rather than reusing the 4096 one — a pgvector column has one fixed dim, so "new 768 writes + historic 4096 untouched + one column" was impossible as originally written (§4.2, §4.7, §16). (2) The 20MB cap is defined as a **wire-bytes** ceiling; the render walk accounts for base64 inflation (§10). (3) The monotonic version guard is an **atomic conditional UPDATE**, not read-then-write, and message edits bump the version (§8). (4) Current-turn 1MP fallback **generates the derivative on demand** if enrichment hasn't baked it yet (§9, §10). (5) **Tool embedding** is a fourth Turbopuffer consumer; tool semantic search is declared **out of scope** for v0.6.0 and degrades to lexical name/description matching so the stack boots tpuf-less (§7, §16). (6) `supports_image_blocks_in_history` **pre-flags researched providers as supported**; only unknown models default off, so the feature doesn't ship dark (§10, §17). Plus: the 3-tier caption VLM gets its own setting (§9, §17).

---

## 1. Problem

Embedding handling across letta-vision is in the state the filesystem was in before v0.5.0 — fragmented, implicit, and failing quietly.

- **No single source of truth for the embedding model.** Three independent, hardcoded selection sites diverge: agent `embedding_config` (archival passages), `TurbopufferClient.default_embedding_config` = `text-embedding-3-small`/1536 (messages + tpuf passages), and `OpenAIEmbedder` (file processor) = `text-embedding-3-small` again.
- **No query-time consistency guard.** The pgvector ordering in `sqlalchemy_base.py` (~line 373) and the three search helpers in `agent_manager_helper.py` embed the query with the agent's *current* config and rank by `cosine_distance` against stored vectors with no check they share a model. A model swap returns confidently-ranked garbage, not an error.
- **Chat embedding lives only in Turbopuffer.** `messages` has no vector column; `_embed_messages_background` writes to tpuf only. On a tpuf-less deployment (sliver) message recall is dead.
- **The 4096 pad forbids ANN.** All vectors zero-pad to `MAX_EMBEDDING_DIM = 4096` (constants.py:93). pgvector HNSW tops out near 2000 dims, so the column cannot be indexed — search is a sequential scan.
- **Images decay and are invisible to search.** Image bytes ride inline in message `content` as base64; `to_openai_dict()` flattens image blocks to text placeholders on re-serialization (the v0.4.0 problem), and nothing about an image is in any vector, so images are unfindable.
- **Agents flail across layers.** An agent must choose between `search_archival`, `search_file_contents`, and `conversation_search` and reconcile three result sets by hand.

This FR fixes the embedding foundation, makes images first-class searchable memory, and gives agents one recall tool over the whole corpus.

## 2. Goals

1. One resolved embedding config every call site reads from; eliminate all hardcoded models.
2. Standardize on `google/gemini-embedding-2-preview` via OpenRouter, native-768 (MRL-truncated, L2-normalized) — one multimodal space for text and images.
3. Stamp every vector row with `embedding_space_id`; refuse cross-space comparison at query time (loud, not silent).
4. Store vectors at native dim (un-padded), HNSW-indexed, including a new vector column on `messages`.
5. Remove Turbopuffer from the path.
6. Make images first-class memory: dedicated `images` table, content-addressed object store, native image (pixel) embedding in the shared space, three text tiers.
7. Persist images in chat as lightweight references; hydrate to pixels at render time under a provider byte cap; demote aged/overflow images to a resolvable reference (description + handle) — never a dead placeholder.
8. One unified hybrid recall tool over passages + messages + images, returning references the agent can drill into.
9. Add an Images management tab to the client.
10. Do all of this **without** re-embedding historic data — existing rows keep their old space id and abstain from new-space ranking via the guard until the separate migration FR runs.

## 3. Conceptual Model

**Space.** A *space* is the coordinate system a `(endpoint_type, model, dim, normalization, doc-side input_type)` tuple defines. Vectors are comparable only within one space. A space is named by `embedding_space_id` = a stable hash of that tuple. Any change that alters the geometry alters the id by design.

**MRL.** gemini-embedding-2 emits 3072-dim vectors with importance front-loaded; truncating to 768 keeps near-peak quality at 1/4 storage. **Only the full 3072 output is pre-normalized — a truncated vector MUST be L2-normalized before storage or comparison.**

**Asymmetric retrieval.** Stored items embed as documents (`search_document`); queries embed as `search_query`. The doc-side input_type is part of the space id; the query-side is per-call and not stored.

**Images are embedded from pixels, not from a summary.** The searchable vector is the native multimodal embedding of the image itself, in the same space as text — so a text query can match an image on its actual content. The text summary is *payload* (what the agent reads to judge relevance), not the search key.

**Reference-then-fetch.** Search returns a context-bearing snippet plus an opaque handle. Heavy content (full image pixels, full file) is pulled only when the agent deliberately fetches it.

**Decay-to-reference, not decay-to-void.** Aged or over-budget images in chat history demote to a resolvable reference (description + handle), which the model can rehydrate on demand — the opposite of the current dead-placeholder flattening.

## 4. Data Model

### 4.1 `EmbeddingConfig` changes — `letta/schemas/embedding_config.py`

Add:
- `output_dimensionality: Optional[int]` — MRL target dim sent to the provider (768). `embedding_dim` = stored width (= `output_dimensionality` when set, else native).
- `input_type: Optional[str]` — doc-side hint at ingest (default `"search_document"`).
- `normalize: bool = False` — client L2-normalizes the returned vector. MUST be `True` whenever truncating below native.
- `embedding_space_id: Optional[str]` — computed, not user-set. Add `compute_space_id(self) -> str` returning a stable sha256-hex prefix (16 chars) of the §3 tuple; populate on construction.

Add a `gemini-embedding-2` default to `default_config()` (openrouter / `google/gemini-embedding-2-preview`, dim 768, output_dimensionality 768, input_type `search_document`, normalize True, endpoint `https://openrouter.ai/api/v1`).

### 4.2 Storage standard — native dim, no padding

- `MAX_EMBEDDING_DIM` stays defined (historic rows + migration FR reference it) but is **not used for new writes**.
- New/changed vector columns are fixed-width at the deployment native dim (`DEPLOYMENT_EMBEDDING_DIM = 768`).
- `messages` and `images` are born at 768 (new columns).
- **Passages need a second column, not a re-typed one.** A pgvector column has a single fixed dimension, so `archival_passages.embedding` / `source_passages.embedding` are `Vector(4096)` today and you can neither write a 768 vector into them (without the padding we're removing) nor HNSW-index them (4096 > pgvector's ~2000 ceiling). Therefore:
  - Rename the existing column to `embedding_legacy_4096` (kept, untouched, **excluded from all vector ranking by the §8 guard** since its rows carry their old/`legacy-unknown` space id).
  - Add a new `embedding Vector(768)` column that new writes and the HNSW index target. Historic rows are `NULL` here, so the vector leg never returns them until the migration FR backfills them.
  - The migration FR (separate) backfills the 768 column for all historic rows and drops `embedding_legacy_4096`. After migration, all four tables are uniform on a 768 `embedding` column.
  - Net for phase one: passages, messages, and images all expose a 768 `embedding` column the recall tool queries uniformly; only newly-written passage rows populate it.
- Remove every `np.pad(..., MAX_EMBEDDING_DIM ...)` from `passage_manager.py` (≈7 sites) and the query side of `agent_manager_helper.py` (3 sites). Factor a single `_prepare_vector_for_write(vec, config)` helper rather than editing each block. New passage writes target the 768 `embedding` column.

### 4.3 `embedding_space_id` on every vector row

Add a `embedding_space_id` column (btree-indexed) to `archival_passages`, `source_passages`, `messages`, and `images`. Backfill passages from stored `embedding_config`; rows with none get sentinel `"legacy-unknown"`. The guard (§8) filters on it.

### 4.4 Messages become vector rows — `letta/orm/message.py`

Add to `Message`:
```python
if settings.database_engine is DatabaseChoice.POSTGRES:
    from pgvector.sqlalchemy import Vector
    embedding = mapped_column(Vector(DEPLOYMENT_EMBEDDING_DIM), nullable=True)
else:
    embedding = Column(CommonVector, nullable=True)
embedding_config: Mapped[Optional[dict]]  = mapped_column(EmbeddingConfigColumn, nullable=True)
embedding_space_id: Mapped[Optional[str]] = mapped_column(nullable=True, index=True)
embedding_version: Mapped[Optional[int]]  = mapped_column(nullable=True)  # monotonic enrichment stamp, see §9
```

### 4.5 NEW dedicated `images` table — `letta/orm/image.py` (new)

An image is one vector + three captions + object references with its own lifecycle — not a chunked text file, so it does **not** extend `FileMetadata`. New id prefix `image-<uuid>` (add `PrimitiveType.IMAGE`).

```
images
  id                  str  PK            # image-<uuid>
  organization_id     str  FK            # OrganizationMixin
  content_hash        str  UNIQUE(org)   # sha256 of raw bytes — dedup key
  object_url_full     str                # object-store URL/key, original bytes
  object_url_1mp      Optional[str]      # pre-baked ~1MP derivative (background)
  media_type          str                # image/png, image/jpeg, ...
  width               Optional[int]
  height              Optional[int]
  file_size_full      Optional[int]      # bytes of original (render-policy input)
  file_size_1mp       Optional[int]      # bytes of 1MP derivative
  provenance          str                # "uploaded" | "generated"
  generation_prompt   Optional[str]      # ZapImage/tool prompt, metadata only (NOT the description)
  caption             Optional[str]      # 20–50 words — inline placeholder gist; seeds message embed
  description         Optional[str]      # 100–200 words — search-result payload, tuned for the index
  details             Optional[str]      # 1000 words — deep-review text
  embedding           Vector(768)/CommonVector, nullable   # native image (pixel) embedding
  embedding_config    EmbeddingConfigColumn, nullable
  embedding_space_id  Optional[str], index
  enrichment_status   enum: pending | complete | failed
  enrichment_attempts int default 0
  error_message       Optional[str]
  created_at, updated_at, is_deleted (standard mixins)
```
Indexes: HNSW on `embedding` (Postgres); GIN trigram on `caption`, `description`, `details` (§4.7); unique `(organization_id, content_hash)`; btree `embedding_space_id`, `enrichment_status`, `created_at`.

Schemas: `letta/schemas/image.py` — `PydanticImage`, plus `ImageCreate`. Manager: `letta/services/image_manager.py` — create/get/get-by-hash/list/delete/update-enrichment, mirroring `file_manager.py`/`passage_manager.py` conventions (async, `@enforce_types`, `@trace_method`).

### 4.6 Persisted chat image form — reuse `LettaImage`

`ImageContent` already supports `LettaImage` source (`type: "letta"`, `file_id`, optional `data`, `detail`). Repurpose: `file_id` carries the **image record id** (`image-…`); `data` is **never persisted** (always `None` at rest) and is hydrated at serialize time per the render policy (§10). The existing `to_openai_dict()` letta-source branch (message.py ~1812/1891) is changed to resolve `file_id` against the `images` table and to apply the render policy rather than relying on inline `data`.

### 4.7 Indexes

- HNSW (`vector_cosine_ops`) on the **768** `embedding` column of `archival_passages`, `source_passages`, `messages`, `images` (Postgres only). The passage `embedding_legacy_4096` column is **not** indexed (and cannot be — it exceeds pgvector's HNSW dim ceiling).
- `pg_trgm` extension + GIN trigram indexes on the text columns the recall tool searches: passage `text`, message `text`, image `caption`/`description`/`details`, file/source content text.
- (For phase-one new-data testing the corpus is tiny; flat scan is fine. Indexes are created forward-looking and stress-validated in the migration FR.)

### 4.8 Alembic migrations

One ordered set: EmbeddingConfig has no schema (pydantic) → no migration; add `embedding_space_id` (+ backfill) to the three existing tables; **on `archival_passages` and `source_passages`, rename `embedding` → `embedding_legacy_4096` and add a new `embedding Vector(768)` (NULL for historic rows)**; add `messages` vector/config/version columns; create `images` table; create `pg_trgm` extension + all GIN trigram indexes; create HNSW indexes on the 768 columns only. Historic 4096 passage vectors are left in place under `embedding_legacy_4096`, untouched until the migration FR.

## 5. The Resolver — `letta/embeddings/resolver.py` (new)

`resolve_embedding_config(agent_state: Optional[AgentState] = None) -> EmbeddingConfig`:
1. Agent-level override if explicitly set and recorded.
2. Deployment default from `settings.default_embedding_handle` (e.g. `openrouter/google/gemini-embedding-2-preview`).
3. Hard error if neither resolves — no hardcoded fallback.

Route through the resolver: `passage_manager` (ingest + query), `message_manager._embed_messages_background`, `file_processor/embedder/openai_embedder.py`, `image_manager` (image embed). Delete `TurbopufferClient.default_embedding_config` and the tpuf embed paths.

## 6. Provider Client — gemini-embedding-2 via OpenRouter

File: `letta/llm_api/openai_client.py`, `request_embeddings` (current call at ~1233 sends only `model` + `input`).

- Pass `dimensions=output_dimensionality` and `input_type` (doc-side) when present.
- Add `input_type_override` param so the **query side** passes `"search_query"` without mutating the stored config.
- When `config.normalize`, L2-normalize each returned vector (`letta/embeddings/util.py::l2_normalize`).
- **Image input:** add an image-embedding entry point that sends the OpenRouter multimodal embeddings `input` variant (image URL or base64) for `image_manager`. OpenRouter `/embeddings` supports text+image, `dimensions`, and `input_type` (verified against OpenRouter docs, June 2026). Audio/video are NOT supported on the embeddings endpoint and are out of scope.
- The OpenRouter provider already lists `google/gemini-embedding-2-preview` at 3072 in `_OPENROUTER_EMBEDDING_DIMS_BY_ID`; the resolved config carries `output_dimensionality=768` so stored width is 768.

## 7. Turbopuffer Removal

Remove tpuf from `message_manager` and `passage_manager`; drop `use_tpuf` branches; remove the hardcoded `text-embedding-3-small`/1536 default. `tpuf_api_key`/`tpuf_region` become optional/dead. `_embed_messages_background` is rewritten to embed via resolver + `request_embeddings` and write the vector to the message row (§8/§9).

**Tool embedding is a fourth tpuf consumer — scoped out for v0.6.0.** `tool_manager._embed_tool_background` and the tool-search relevance path embed tools via tpuf. Migrating tool vectors to a PG column is real scope and not required by this feature's goals, so for v0.6.0: **tool semantic search is out of scope**, and the tool-search path degrades to lexical name/description matching when tpuf is absent. This keeps the "boots and recalls tpuf-less" acceptance criterion honest without dragging tool-vector migration into this FR. (Moving tool embedding onto a PG column the way messages/images are is a clean follow-up consistent with the "everything searchable like any other element" vision — deferred, not rejected.)

## 8. Query-Time Compatibility Guard

In the vector-ordered queries (`sqlalchemy_base.py` ordering block; the three `agent_manager_helper.py` helpers; the recall tool's vector leg):
1. Resolve the query's `embedding_space_id`.
2. Add `WHERE embedding_space_id = :query_space_id` to every vector-ordered query (passages, messages, images).
3. Rows in another space are excluded from vector ranking (still reachable by lexical search). Log a one-line debug with exclusion counts so mixed-space corpora are observable.

Vector write semantics everywhere (messages, images): keyed by row id, **replace not append** (one current vector per row), and **monotonic** on `embedding_version` — a write carrying a lower-or-equal version MUST NOT overwrite a higher one.

**The guard MUST be an atomic compare-and-set, not a read-then-write.** A Python `if row.version <= incoming: return` is a TOCTOU race, and the two-embed dance (§9) is exactly the concurrency it guards — the turn-end blind embed and a fast enrichment push can both read the old version and both write. Implement as a single conditional UPDATE so the database does the compare-and-set:

```sql
UPDATE messages
   SET embedding = :vec, embedding_config = :cfg, embedding_space_id = :sid, embedding_version = :v
 WHERE id = :id
   AND (embedding_version IS NULL OR embedding_version < :v)
```

**Message edits must advance the version.** An edited message re-embeds at a strictly higher version (or bypasses the guard explicitly), or the equal-version guard would silently keep the stale pre-edit vector. Treat an edit as a new monotonic step, not a re-issue of the same version.

## 9. Image Ingest & Enrichment Pipeline — `letta/services/image_ingest.py` (new)

Single ingest function; two entry points (user upload; tool output e.g. ZapImage) differing only in `provenance` tag and interception point. Both funnel here.

**Synchronous phase (in the turn — pure I/O, no model calls):**
1. `content_hash = sha256(bytes)`. If a record with `(org, hash)` exists, reuse it (no re-store, no re-embed, no re-caption) and return its id.
2. Else write original bytes to the object store (key by hash), insert `images` row with id, `object_url_full`, hash, media_type, width/height, `file_size_full`, provenance, `generation_prompt` (if tool), `enrichment_status=pending`, `embedding_version=0`.
3. Return the image id → caller constructs the `LettaImage` `image_ref` (`file_id = image-id`, `data=None`).

The turn closes immediately; the message persists with the reference.

**Background phase (fire-and-forget, idempotent, retryable):**
1. Generate the ~1MP derivative, store it, set `object_url_1mp` + `file_size_1mp`. (Needed for *older* turns, so it is normally ready well before the middle render tier reaches it. The one exception — an oversize image on the **current** turn whose 1MP isn't baked yet — is handled by on-demand generation in §10.) `file_size_1mp` (and `file_size_full`) are stored as the **base64-encoded wire size**, not raw disk bytes, because that is what the render-policy byte budget (§10) must count.
2. **One structured VLM call** returns all three text tiers (`caption`, `description`, `details`) — consistent and cheaper than three calls. Target lengths: **caption 20–50 words**, **description 100–200 words**, **details 1000 words**. The captioning model is a vision chat model configured by its own setting (`settings.image_caption_model_handle`), distinct from both the embedding model and the agent's runtime chat model. **Description is generated from pixels** (captures hallucinations / emergent detail / deviations), NOT from the generation prompt; the prompt is stored separately as provenance. Description is written for the index first (dense, literal content nouns), display second.
3. Pixel-embed the image via gemini-2 image input → 768, normalized, stamped `embedding_space_id`. Store on the record.
4. Set `enrichment_status=complete`.
5. **Push** the owning message's re-embed (§ below) — do not poll.

**Message embedding & the two-embed dance (uses idempotency from §8):**
- At turn end, the message embeds normally (same path as a no-image turn) with the `image_ref`'s empty caption — `embedding_version` = e.g. 1. The turn is immediately findable by its words.
- When enrichment completes, it pushes a message re-embed whose text now includes the image **caption gist** (short — a sentence, not the 100–200 word description, so the conversational signal isn't diluted), at a higher `embedding_version`. Replace-not-append + monotonic means this supersedes the empty-caption vector regardless of ordering races.

**Failure handling:**
- Background enrichment retries to a bound (`settings.image_enrichment_max_attempts`, default e.g. 3), incrementing `enrichment_attempts`.
- On exhaustion: set `enrichment_status=failed` + `error_message`, AND still trigger a text-only message re-embed so the turn is never lost to recall. A failed record still renders full/1MP (bytes exist) and is flagged for later re-enrichment; the content hash makes a future retry free.

## 10. Chat Image Representation & Render Policy

Replaces the v0.4.0 serializer approach. The persisted message holds the lightweight `LettaImage` reference (§4.6); the model-facing representation is built fresh at serialize time and never round-trips into storage, so there is no image block left to decay.

**Two consumers, two rehydrations:**
- **UI:** pulls the full image straight from `object_url_full` — unbounded by the provider cap (it does not go through the provider).
- **Model:** the tiered render below.

**Byte cap.** Global constant `settings.vision_context_byte_cap` (default `20 * 1024 * 1024`). The cap is a hard request-size ceiling — exceeding it fails the provider call — so everything sent over the wire counts, current image included. **The cap is defined in wire bytes**, i.e. the base64-encoded size as it appears in the request JSON (base64 inflates raw bytes by ~33%). The walk must compare base64 sizes against the cap; `file_size_full`/`file_size_1mp` are therefore stored as encoded sizes (§9). Confirm against Cursor's provider research whether the measured ~20MB limit was raw or encoded and set the constant accordingly — if the providers cap raw bytes, divide the budget by 4/3 instead of inflating the sizes; pick one convention and document it.

**Render walk** (deterministic, no model inference, pre-baked artifacts only):
```
if not model_caps.supports_image_blocks_in_history:   # see §17 default
    all images -> TEXT (description + handle); return

remaining = vision_context_byte_cap
demoted   = False
for img in images_ordered_newest_first:        # current turn's image(s) first
    if demoted:
        render[img] = TEXT;  continue
    if img in current_turn:
        if file_size_full <= remaining:
            render[img] = FULL;  remaining -= file_size_full
        else:
            onemp = img.file_size_1mp or generate_1mp_now(img)   # see note
            if onemp <= remaining:
                render[img] = ONE_MP; remaining -= onemp
            else:
                render[img] = TEXT;  demoted = True
    else:                                       # older turns: 1MP-or-text only
        if file_size_1mp <= remaining:
            render[img] = ONE_MP; remaining -= file_size_1mp
        else:
            render[img] = TEXT;  demoted = True
```
- "current turn" = images in the most recent user message.
- Full→1MP→text ladder applies **only** to the current turn; older images are 1MP-or-text by definition.
- **Current-turn 1MP may not exist yet.** The current image was ingested synchronously milliseconds ago; its 1MP derivative is produced in the background enrichment phase and may not be ready. If the current image exceeds the cap and `file_size_1mp` is absent, **generate the 1MP derivative on demand** (`generate_1mp_now`). This is pure image processing, not model inference, so it does not violate the no-model-in-hot-path rule; it's a single cheap resize in the rare current-oversize case. (Persist the result so the background phase doesn't redo it.) Older-turn images are never in this position — their 1MP is long since baked.
- First image that doesn't fit, and every older image, demote to TEXT (the `demoted` latch).
- **No hysteresis** — the model has no cross-turn memory of the render list, so an image flipping tiers between turns is not something it perceives; stability machinery is unnecessary.
- TEXT tier = `LettaImage` with `data=None` plus the image `description` injected, and the handle so the model can `fetch_image` to rehydrate pixels.
- FULL / ONE_MP hydrate `data` from `object_url_full` / `object_url_1mp` respectively.

A prior model reference to a now-demoted image still resolves: the TEXT form carries the description + handle, so the reference lands on coherent text and the model can fetch the pixels back if needed.

## 11. Supersession of the v0.4.0 FR

`FR_letta-vision_Image-Context-Persistence-Across-Turns.md` aimed to keep image blocks alive through `to_openai_dict()`. This FR removes the bytes from the message entirely (§4.6/§10), so the decay problem is gone by construction rather than patched. Mark the v0.4.0 FR superseded; its `to_openai_dict` work is replaced by the §10 render policy.

## 12. Unified Recall Tool — `recall`

One agent-facing tool over the whole corpus, replacing the need to choose between `search_archival` / `search_file_contents` / `conversation_search` (keep those as internal/filtered paths; `recall` is the default surface).

Signature (Letta tool): `recall(query: str, *, layers: Optional[list[str]] = None, time_range: Optional[...] = None, source: Optional[str] = None, limit: int = 10)`.

**Vector leg.** Embed the query once as `search_query` (via `input_type_override`). Run it against the four vector columns (`archival_passages`, `source_passages`, `messages`, `images`) under the space guard (§8). Same space ⇒ directly comparable cosine scores ⇒ collapse into one vector-ranked list by raw score. No per-table calibration.

**Lexical leg.** `pg_trgm` `similarity()` over the trigram-indexed text columns (passage/message text, image caption/description/details, file content). Trigram chosen over full-text search because it indexes substrings and graded similarity and does not mangle hyphenated identifiers (`scenecraft-mvp-connector`), UUIDs, error strings, filenames — exactly the literal tokens the vector leg buries.

**Fusion.** Reciprocal Rank Fusion (RRF) over the two ranked lists — rank-based, no cross-method score normalization; an item surfacing in both legs gets both contributions summed.

**Post-steps (transparent, no learned weights):**
- Per-source diversity cap (default 2–3 hits per file/conversation) so one chunked doc or chatty thread can't flood the list.
- Dedup: collapse an image record and the message that references it into one result with two reasons (don't show the image twice).
- Top-K with bounded snippets so the return can't blow the context window.

**Return shape (reference-then-fetch).** Each hit: a context-bearing snippet (passage; file chunk + neighbors via v0.5.0 char offsets; message + a surrounding turn or two; image `description`), the layer/type, the fused rank/score, and an opaque handle. Handles drive existing access paths: open file at offset, read conversation around message id, and a new `fetch_image(handle)` tool that pulls full pixels into context on demand.

**Image findability backstop.** Image `description`/`details` are trigram-searched, so an image is findable by described content even if the pixel vector under-ranks it for a text query (the modality gap). This is why image-only embedding is sufficient for v0.6.0 and the optional summary-text embedding is deferred — measure the modality gap empirically before adding it.

## 13. Client Changes — Images Tab (`letta-vision-client`)

Stack: Svelte 5 + Vite frontend, FastAPI backend proxying the Letta SDK.

**Server (letta-vision) — new REST surface** for image records (mirror `routers/v1/passages.py` / `file_memory.py`): list (with metadata + enrichment status, paginated/filterable), get one, delete, trigger re-enrichment. Image bytes served via signed object-store URL or a streaming endpoint.

**Client backend** — extend `backend/routes/images.py` (currently only `/api/images/fetch-url`): add list/get/delete/re-enrich endpoints proxying the server surface. Reuse `vision_max_upload_bytes` (already 20MB).

**Client frontend** — new `frontend/src/routes/Images.svelte` mirroring `Files.svelte`: grid/list of images with thumbnail (1MP/object URL), caption/description/details, provenance, enrichment status, content hash, dimensions, space id; actions: view full, edit metadata, re-enrich, delete. Wire the tab in `App.svelte` (import, nav button, render slot) and `lib/stores.js` (`currentTab` enum value + `initFromHash`) — the same ~5 edit points the existing tabs use.

## 14. Implementation Phases

1. **Embedding foundation:** EmbeddingConfig fields + `compute_space_id`; resolver + `default_embedding_handle`; provider client (`dimensions`/`input_type`/normalize/query override); remove padding; gemini-2 default.
2. **Storage + guard:** alembic (space-id columns + backfill, messages vector cols, HNSW, pg_trgm + GIN); query-time guard + logging; PG-native message embed with replace+monotonic write.
3. **Turbopuffer removal.**
4. **Image records + object store:** `images` table/schema/manager; object-store client (content-addressed; MinIO + GCS); image pixel-embed path in the provider client.
5. **Ingest pipeline:** sync store + background enrich (1MP + structured VLM 3-tier + pixel embed); two-embed dance; bounded retries + failure fallback; dedup by hash.
6. **Chat render policy:** `LettaImage` as persisted form; rewrite `to_openai_dict` letta-source branch to resolve the images table + apply the render walk; supersede v0.4.0.
7. **Recall tool:** vector leg (4 columns, guarded) + pg_trgm lexical leg + RRF + diversity/dedup/top-K + return shape; `fetch_image`; register `recall`, demote the three granular tools to internal/filtered.
8. **Client:** server image REST surface; client backend proxy; `Images.svelte` + nav wiring.
9. **Base instructions:** point the agent at `recall` and `fetch_image` with concrete per-tool directives.

## 15. Acceptance Criteria

- Fresh agent on the deployment default embeds archival passages, source passages, messages, and images with `google/gemini-embedding-2-preview` at stored dim 768, all sharing one `embedding_space_id`.
- No hardcoded embedding model remains (grep for `text-embedding-3-small` hits only tests/migrations). Turbopuffer not required to boot, embed, or search.
- Stored vectors are length-768, unit-length (‖v‖ ≈ 1.0 ± 1e-3); no `np.pad(..., MAX_EMBEDDING_DIM ...)` in write/query paths.
- A query in space A returns zero vector hits against rows stamped space B (and logs the exclusion) — no mis-ranked results.
- An ingested image (upload or ZapImage) creates one `images` row, stores full + 1MP, produces caption/description/details from pixels, and a 768 pixel embedding in the shared space; a re-shared identical image creates no new record (hash dedup).
- A text query matches a relevant image via the vector leg AND/OR via trigram on its description; the result returns description + handle; `fetch_image` returns the full pixels.
- In a long image-heavy chat: the current image renders full (or 1MP if full would breach the cap, or text if 1MP would); older images render 1MP newest-first until the budget is exhausted, then demote to description+handle; total image **wire bytes** (base64-encoded) sent ≤ the cap on every turn; a demoted image still resolves prior references and is fetchable.
- `recall(query)` returns a single fused, deduped, source-capped ranked list across passages + messages + images with reference-then-fetch handles; the three granular searches are no longer the agent's default surface.
- Background enrichment failure leaves a renderable, flagged image and a text-only-embedded (still recall-able) message; nothing blocks forever.
- Client shows an Images tab listing records with metadata and enrichment status; view-full, edit, re-enrich, delete work.
- HNSW + pg_trgm indexes exist on the relevant columns (Postgres). Passages expose a 768 `embedding` column (new writes populated, historic rows NULL until migration) alongside the retained `embedding_legacy_4096`; the recall tool queries the 768 column uniformly across all four tables.

## 16. Non-Goals (Explicit Deferrals)

- **Historic embedding uplift & validation** — re-embedding existing passage rows into the 768 space (backfilling the new `embedding` column from `embedding_legacy_4096`'s source text and then **dropping `embedding_legacy_4096`**), re-embedding pre-v0.6.0 messages/images, building HNSW over the migrated corpus, dual-space transition strategy (cutover vs shadow), cost/time estimation. **This is the one remaining FR and the gate to cutting v0.6.0.** This FR ships and is validated against *new* data only; the guard makes un-migrated history simply not appear in vector results yet.
- **Tool semantic search** — tool embedding stays on its current (tpuf) path or degrades to lexical matching; migrating tool vectors to a PG column is deferred (§7). Tool relevance is not part of the unified recall corpus in v0.6.0.
- Audio/video embedding (OpenRouter embeddings is text+image only; native google_ai/Vertex client deferred).
- Optional summary-text embedding for images (modality-gap backstop) — add only if empirical recall shows the gap hurting.
- Two-stage MRL retrieval (256-dim candidate + 3072 re-rank) — scale optimization, deferred.
- Per-agent embedding dimension override (deployment dim is fixed so columns are uniform).

## 17. Open Questions (with Defaults)

- **Provider image-block-in-history capability:** a per-model flag (`supports_image_blocks_in_history`). Default — **pre-flag the providers Cursor already validated at the 20MB cap as supported**, so the feature does not ship dark (all-text on first boot). Only genuinely *unknown* models default to unsupported (all-text, safe). Confirm Kimi-K2.6 empirically and seed its flag.
- **Caption VLM model:** `settings.image_caption_model_handle` — a vision chat model for the three-tier captioner, separate from the embedding model and the runtime chat model. Default to a cheap vision model; confirm the choice (Kimi-K2.6 works but may be overkill for captioning).
- **1MP target:** default — longest-edge resize to ~1 megapixel preserving aspect, JPEG quality tuned so typical `file_size_1mp` (encoded) ≪ cap (so the older-tier walk is additive packing, not per-image breach). Confirm with sample images.
- **`embedding_space_id` GA rename:** when gemini-embedding-2 leaves preview and the model string changes, the space id changes by design → forces a recorded re-embed (the migration FR) rather than silent drift. Intended.
- **Object store auth for the client view-full:** default — short-lived signed URLs from the server surface, not public objects.

## 18. Cursor Implementation Notes

- Factor `_prepare_vector_for_write(vec, config)` once; the padding removal touches ~7 copy-pasted blocks.
- `compute_space_id` must be deterministic across processes — stable serialization (sorted string/int inputs), fixed hash (sha256 hex, first 16 chars).
- Thread the query-side `input_type` override from the recall tool / search helpers down to `request_embeddings`; never store it on the agent config.
- The `to_openai_dict` letta-source branch (message.py ~1812/1891) is the single integration point for §10 — resolve `file_id` → `images` row, apply the render decision passed in from the context-window builder (the builder owns the byte-budget walk across the whole message list; the serializer renders one block given a decision).
- The byte-budget walk needs the *full message list* in view, so it lives in the context-window assembly layer (where the message list is known), not in per-message serialization.
- Image embedding and text embedding share `request_embeddings`; branch on `config.modality`/input shape, not a separate client.
- Object-store client should abstract MinIO vs GCS behind one interface (sliver local + GCP authoritative), content-addressed keys.
- Register `recall` and `fetch_image`; update agent base instructions with per-tool directives ("call `recall` first; `fetch_image` only when you need to see pixels").
- Validate `dimensions`/`input_type`/image-input are accepted by OpenRouter for the pinned preview model with one smoke call before wiring defaults.

## 19. Future Extensions (Out of Scope, Sketched)

- Native google_ai/Vertex embedding client for audio/video (full multimodality beyond OpenRouter's text+image).
- Summary-text embedding for images as a modality-gap backstop.
- Two-stage MRL retrieval for scale.
- Cross-agent shared image memory (ties into the planned Ada/Lyra shared blocks).

---

## Appendix A — Implementation-plan review notes (r2)

The implementation plan is faithful and got the three highest-risk calls right (render walk in `build_request_data` with decisions threaded to the serializer; `embedding_version` in the phase-2 migration before image code; keep `file_id`). The r2 corrections above map to the plan as follows. The first is a hard blocker for phase 2; the rest are folded in by phase.

- **Phase 2 / PR-B — passage dual-column (blocker).** The plan's "HNSW on passage embedding columns at 768 — new writes only; historic 4096 untouched" is impossible on one pgvector column. The migration must rename `embedding`→`embedding_legacy_4096` and add a new `embedding Vector(768)` (NULL historic), HNSW the 768 column only (§4.2/§4.7/§4.8). Do not start PR-B until this is reflected.
- **Phase 2 / PR-B — atomic monotonic guard.** Replace the sketched `if row.version <= incoming: return` with the single conditional `UPDATE ... WHERE embedding_version IS NULL OR embedding_version < :v` (§8). The two-embed dance is the concurrency this guards; the read-then-write version is a real race. Add: message edits advance the version.
- **Phase 5/6 — byte accounting unit.** Resolve whether the providers' ~20MB limit is raw or base64-encoded (the plan cites Cursor's research — check which side of base64 it was measured on), then set `vision_context_byte_cap` and store `file_size_full`/`file_size_1mp` consistently as wire bytes, or divide the budget by 4/3 (§9/§10). Pick one convention.
- **Phase 6 — current-turn 1MP-on-demand.** The plan only bakes 1MP in background (phases 4/5), but the current-turn full→1MP fallback can fire before that completes. Add `generate_1mp_now` (pure resize, no inference) in the render walk for the current-oversize case, persisting the result (§9/§10).
- **Phase 3 / PR-C — tool-embedding scope.** The plan hedges ("tools embedding can stay tpuf-optional or be removed"), which collides with the tpuf-less boot acceptance criterion. Make it a stated decision: tool semantic search is out of scope for v0.6.0 and degrades to lexical name/description matching tpuf-less (§7/§16). Settle this before PR-C so the acceptance criterion is honest.
- **Phase 6/9 — capability flag pre-seeding.** `supports_image_blocks_in_history` defaulting all-unknown-to-off ships the feature dark on first boot. Pre-flag the providers already validated at 20MB as supported; only unknown models default off (§17).
- **Phase 5 — caption VLM setting.** Add `settings.image_caption_model_handle` for the three-tier captioner; it is distinct from the embedding model and the runtime chat model and the plan doesn't name where it's configured (§9/§17).

Plan sequencing impact: resolve the passage dual-column and the byte-accounting unit before PR-B, and the tool-embedding scope before PR-C. PR-A and PR-D onward are sound as planned. The risk-register row "Historic 4096 passages vs 768 queries → space guard excludes them" stays true, but the *mechanism* is now the NULL 768 column plus the legacy column's old space id, not a single mixed-width column.
