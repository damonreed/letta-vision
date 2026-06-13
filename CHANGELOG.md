# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Fork releases use `v0.x.y` tags (diverged from upstream Letta `0.16.x` at v0.2.0).

## [Unreleased]

## [0.6.0] - 2026-06-13

### Added

- Unified 768-dim embedding space with deployment-global resolver (`LETTA_DEFAULT_EMBEDDING_HANDLE`).
- `images` table, object-store client, ingest/enrichment pipeline, pixel embed path.
- Message vectors with atomic version guard; caption-gist injection at embed v2+.
- Hybrid per-layer search (vector + lexical + RRF) and fused `search_all` recall tool.
- `image_fetch` / `image_search` tools; ref-only tool-return image persistence.
- Tiered vision render policy (FULL / 1MP / TEXT) with wire-byte cap.
- Historic uplift CLI (`scripts/historic_uplift.py`) — Part 1 conversion, enrichment, Part 2 re-embed, tool-return byte strip.
- Images REST API: paginated list, hybrid search, content variants, inline metadata PATCH.
- OpenRouter vision detection from `architecture.input_modalities`; `provider_models.supports_vision` (`v063`).
- Alembic `v060`–`v063` migrations (unified embedding, file archive parity, legacy column drop, vision flags).

### Changed

- Turbopuffer client and dual-write paths removed; all memory search is pgvector + `pg_trgm`.
- Deployment embedding moved to `openrouter/google/gemini-embedding-2` GA @768.
- `recall` renamed to `search_all`; `fetch_image` renamed to `image_fetch` (deprecated aliases retained).
- Archival/file recall tool output labels folder hits as `file` with `filename=`.
- `model_supports_vision()` precedence: manual override → OpenRouter cache → registry (non-OR only).

### Fixed

- Archival insert missing `embedding_space_id`; legacy source upload path stamping.
- Message uplift monotonic guard blocking v2 historic writes.
- MiniMax duplicate thinking in assistant text when reasoning extracted separately.
- Vision agents: `generate_image` / tool-return images in byte-budget walk and hydration.
- `list_llm_models` validation when `supports_vision` is null on DB rows.
- GitHub MCP `get_file_contents` EmbeddedResource text extraction.

## [0.5.0] - 2026-06-01

### Added

- Three-tier filesystem memory: `file_core_blocks`, `agent_open_files`, `file_archives` (migration `f1a2b3c4d5e6`).
- Read/nav tools: `file_read_page`, `file_read_next_page`, `file_read_prev_page`, `file_read_range`, `file_grep` via `CharPageReader`.
- Headline and archive tools: `update_file_headline`, `write_file_archive`, `search_file_archives`; `search_file_contents` rename.
- `add_text_file` — create text files in attached folders with optional headline and async ingestion.
- REST `/v1/file-memory/*` endpoints; conversation-scoped system recompile (`recompile_system_message_for_conversation`).
- Live system refresh after file-state tool mutations (`FILE_STATE_SYSTEM_REFRESH_TOOLS` in `LettaAgentV3`).
- E2B PIL image tool returns (`e2b_result_format.py`); MCP multimodal tool result formatter.
- LLM request log redaction (`log_redaction.py`); image-aware token estimates.
- Scripts: `backfill_file_core_blocks.py`, `refresh_letta_v1_system_prompts.py`.
- Tests: `test_three_tier_memory_compile.py`, `test_char_page_reader.py`, `test_add_text_file_tool.py`, and related coverage.

### Changed

- `letta_v1.py` rewrite: memory terminology, retrieval order, E2B sandbox documentation, final tool names.
- System prompt: `<directories>` shows file headlines for all attached files; `<open_files>` for active reading slots only.
- Context window lookup: model name normalization; `kimi-k2.6` support.

### Fixed

- `file_read_next_page` boundary skip; timezone on `update_file_headline`; archive search eager-load crash.
- Stale page-size limit during tool loop; stale system context after file mutations.
- `CONVERSATION_ID: default` in named conversations; fuzzy file ID resolution.
- Multimodal tool returns no longer overwritten by legacy text content on serialization.

## [0.4.0] - 2026-05-21

### Added

- Cross-turn image context: historical user images are preserved in LLM request serialization (OpenAI `image_url`, Anthropic `image`, Gemini `inline_data`).
- `user_content_to_openai_chat_content()` and order-based `fill_image_content_in_messages()` pairing (fixes tool-row expansion skipping images).
- Tests: extended `tests/test_message_serialization.py`; `tests/integration_test_image_persistence.py` (live OpenRouter when `OPENROUTER_API_KEY` is set).

### Changed

- README Vision section documents cross-turn behavior and super-linear token cost growth.
- Streamlined LLM failure notices in `LettaAgentV3` (single user-visible message, no injected JSON in the failure bubble).

## [0.3.0] - 2026-05-20

### Added

- Vision model registry (`letta/llm_api/model_registry.py`) with `LETTA_VISION_MODELS_EXTRA`.
- `supports_vision` on models API and `LLMConfig`; README registry table.
- Image validation (MIME, per-image and per-message size limits) with HTTP 422 for non-vision models.
- `LettaVisionCapabilityError` (422) and `LettaMessageTooLargeError` (413).
- Default `LETTA_LLM_REQUEST_TIMEOUT_SECONDS=300`; `LETTA_LLM_MAX_RETRIES`, `LETTA_LLM_RETRY_ON_TIMEOUT`.
- OpenRouter `provider_preferences` passthrough on OpenAI-compatible client.
- Tests: `tests/test_vision_capability.py`, `tests/test_llm_timeout_config.py`.
- Implementation report for Ada: `docs/IMPLEMENTATION_REPORT_v0.3.0_vision-support.md`.

### Changed

- Vision validation wired through REST `create_input_messages` and agent helpers.

## [0.2.0] - 2026-05-20

Pre-vision baseline: multimodal content blocks validated via K2.6 smoke test; partial timeout wiring.

[Unreleased]: https://github.com/damonreed/letta-vision/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/damonreed/letta-vision/releases/tag/v0.6.0
[0.5.0]: https://github.com/damonreed/letta-vision/releases/tag/v0.5.0
[0.4.0]: https://github.com/damonreed/letta-vision/releases/tag/v0.4.0
[0.3.0]: https://github.com/damonreed/letta-vision/releases/tag/v0.3.0
[0.2.0]: https://github.com/damonreed/letta-vision/releases/tag/v0.2.0
