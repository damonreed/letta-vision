# Release notes — v0.6.0 (letta-vision)

**Theme:** Unified 768-dim embedding space, multimodal image memory, hybrid recall, historic corpus uplift — GA.

## Highlights

- **Deployment-global embedding** — all ingest, search, and recall paths use `LETTA_DEFAULT_EMBEDDING_HANDLE` (`openrouter/google/gemini-embedding-2` @768). Per-agent embedding pickers removed from client flows.
- **Five-table vector leg** — `archival_passages`, `source_passages`, `file_archives`, `messages`, and `images` share one `embedding_space_id` with pgvector HNSW + `pg_trgm` lexical RRF.
- **Image records + object store** — content-addressed MinIO/S3 blobs, sync ingest, background 1MP + VLM captions + pixel embed; `LettaImage` refs replace inline base64 in chat.
- **Tiered vision render policy** — wire-byte budget walk with FULL / 1MP / TEXT tiers; tool-return and MCP-generated images participate in the byte cap.
- **Hybrid recall** — per-layer tools (`archival_memory_search`, `file_contents_search`, `file_archive_search`, `conversation_search`, `image_search`) plus fused `search_all` with image/message dedup and per-source diversity cap.
- **Historic uplift CLI** — `scripts/historic_uplift.py`: Part 1 base64→object conversion, batch enrichment, Part 2 rolling re-embed, inventory/cost estimates, tool-return byte strip.
- **Turbopuffer retired** — all memory search is PostgreSQL-native; dual-write paths removed.
- **OpenRouter vision detection** — `architecture.input_modalities` is authoritative for `openrouter/*` models; flags persisted on `provider_models.supports_vision` and warmed at startup (fixes registry false positives such as DeepSeek V4 Pro).

## Upgrade from v0.5.0

```bash
git checkout v0.6.0
docker build -t letta-vision:v0.6.0 -t letta-vision:latest .
```

Pair with **letta-vision-client v0.6.0** (or current `main` with GA client §12–§13) and **letta-vision-deploy** updated for MinIO + embedding defaults.

Set `LETTA_VERSION=0.6.0` in deploy `.env`.

### Migration steps

Alembic revisions (run on boot or manually via `docker compose run --rm letta-vision alembic upgrade head`):

| Revision | Purpose |
|----------|---------|
| `v060_unified_embedding_multimodal_recall` | Core unified embedding schema, message/image vectors, object store hooks |
| `v061_file_archive_unified_embedding` | `file_archives` embedding parity |
| `v062_drop_legacy_emb` | Drop `embedding_legacy_4096` from passage/archive tables (**requires** matching ORM — do not run SQL alone) |
| `v063_provider_models_vision` | `provider_models.supports_vision` for OpenRouter catalog flags |

### Deploy configuration (new / changed)

- `LETTA_DEFAULT_EMBEDDING_HANDLE` — deployment-wide embedding model (default: `openrouter/google/gemini-embedding-2`).
- `LETTA_OBJECT_STORE_URI` — S3-compatible object store for image bytes (e.g. MinIO in compose).
- `LETTA_IMAGE_CAPTION_MODEL_HANDLE` — VLM for three-tier captions (separate from agent chat model).
- `MODEL_OVERRIDES_PATH` — manual vision overrides still win over OpenRouter cache.

### Historic corpus uplift (existing deployments)

If upgrading from pre-v0.6 data with inline base64 images or NULL 768 embeddings:

```bash
docker exec letta-vision python scripts/historic_uplift.py inventory
docker exec letta-vision python scripts/historic_uplift.py convert --dry-run   # Part 1
docker exec letta-vision python scripts/historic_uplift.py convert             # after DB snapshot
docker exec letta-vision python scripts/historic_uplift.py enrich-pending
docker exec letta-vision python scripts/historic_uplift.py reembed             # Part 2
```

See [FR: Historic Embedding Uplift](FR_letta-vision_Historic-Embedding-Uplift_v0.6.0-GA.md) §10.1 runbook (checkpoint semantics, `UPLIFT_MESSAGE_EMBED_VERSION`).

### Post-upgrade verification

```bash
# Health
curl -sf http://localhost:8283/v1/health/

# OpenRouter vision flags (Bearer auth)
curl -sf -H "Authorization: Bearer $LETTA_SERVER_PASSWORD" http://localhost:8283/v1/models/ \
  | jq '.[] | select(.handle | test("deepseek-v4-pro|kimi-k2.6|gpt-4o")) | {handle, supports_vision}'

pytest tests/test_openrouter_vision_detection.py tests/test_vision_capability.py \
  tests/test_render_policy.py tests/test_recall_service.py
```

Expected after startup sync: `openrouter/deepseek/deepseek-v4-pro` → `supports_vision: false`; Kimi / gpt-4o OpenRouter entries → `true`.

## Operator notes

- **Reasoner + tools:** Kimi/MiniMax with `enable_reasoner: true` may finish turns without `tool_calls`. Set `enable_reasoner: false` on tool-critical agents (see uplift FR §14.3).
- **Legacy source upload:** `POST /v1/sources/{id}/upload` now stamps `embedding_space_id` via `create_many_source_passages_async`.
- **File delete:** client uses background folder recompile — no UI hang.
- **Recall filter params** (`layers`, `time_range`, `source`) deferred to v0.6.1.

## Vision detection change (v0.6.0)

Precedence for `supports_vision`:

1. `model_overrides.json` (letta-vision-client Providers UI)
2. OpenRouter catalog (`architecture.input_modalities`) for `openrouter/*` handles
3. Curated registry + `LETTA_VISION_MODELS_EXTRA` for BYOK / `openai-proxy/*`

Remove manual overrides that forced vision on text-only OpenRouter listings after upgrade.

## Documentation

- [IMPLEMENTATION_REPORT_v0.6.0_unified-embedding-recall.md](IMPLEMENTATION_REPORT_v0.6.0_unified-embedding-recall.md) — full GA report for Ada.
- [FR: Historic Embedding Uplift](FR_letta-vision_Historic-Embedding-Uplift_v0.6.0-GA.md) — uplift spec + §14 operational fixes (includes §14.8 vision detection).
- [FR: Unified Embedding & Recall (rc)](FR_letta-vision_Unified-Embedding-Multimodal-Recall_v0.6.0-rc.md) — original v0.6 FR.
- [IMPLEMENTATION_REPORT_v0.3.0_vision-support.md](IMPLEMENTATION_REPORT_v0.3.0_vision-support.md) — updated resolution order.
- [CHANGELOG](../CHANGELOG.md) — complete change list.
