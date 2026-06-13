# Implementation Report: Vision Support (v0.3.0)

**To:** Ada  
**From:** Damon (letta-stack)  
**Date:** 2026-05-20  
**Release:** `v0.3.0` across `letta-vision`, `letta-vision-client`, `letta-vision-deploy`  
**Baseline:** `v0.2.0` (pre-vision tag, 2026-05-20)

---

## Executive summary

We implemented the paired vision functional requirements ([FR-letta-vision](FR-letta-vision_vision-support.md), [FR-letta-vision-client](../letta-vision-client/docs/FR-letta-vision-client_vision-support.md)) as **v0.3.0**. The stack is live-validated: images attach in the web UI, `supports_vision` is exposed on `/v1/models`, non-vision agents are guarded server-side (422) and client-side (disabled attach), and Kimi K2.6 returns correct multimodal interpretation (~20s on a 5.5M-char base64 smoke payload).

Upstream Letta already carried multimodal content blocks; this release adds the **product contract** (registry, validation, timeouts, UI) your specifications called for. Phase 3 (MCP tool-result images) remains explicitly deferred.

---

## Scope delivered vs FR

| FR area | Server (`letta-vision`) | Client (`letta-vision-client`) | Status |
|---------|-------------------------|--------------------------------|--------|
| §3 Model capability API | `supports_vision` on `Model` + `/v1/models` | Vision badge, attach gating via `modelSupportsVision` | **Done** |
| §4 Image validation | MIME whitelist, size caps, 422 on non-vision | Browser pipeline + proxy request-size guard | **Done** |
| §5 LLM timeouts / retries | Default 300s, retry env vars, OpenRouter `provider_preferences` | Client read timeout aligned via deploy | **Done** |
| §6 Storage / inline v1 | Documented in README; inline base64 in history | Optimistic + history render via content blocks | **Done** |
| §7 MCP tool-result images | — | — | **Deferred** (Phase 3) |
| §15 E2E checklist | Smoke script + manual UI | Composer attach/drop/paste/URL | **Passed** |

---

## Architecture (server)

### Vision registry — single source of truth

New module: `letta/llm_api/model_registry.py`

- Curated list `VISION_CAPABLE_MODELS` (provider label + model id or glob pattern).
- `model_supports_vision(model, handle)` resolution order:
  1. Manual override (`model_overrides.json` from letta-vision-client)
  2. OpenRouter catalog cache (`architecture.input_modalities` from `GET /v1/models`, persisted on `provider_models.supports_vision`) for `openrouter/*` handles
  3. Registry globs + `LETTA_VISION_MODELS_EXTRA` for BYOK and non-OpenRouter paths
- `merge_provider_preferences()` passes OpenRouter routing hints from `LLMConfig.provider_preferences` into the OpenAI-compatible client `extra_body`.

Wired into:

- `letta/schemas/model.py` — `supports_vision` field on API model objects.
- `letta/schemas/llm_config.py` — `supports_vision`, `provider_preferences` on agent LLM config.
- `letta/server/rest_api/routers/v1/llms.py` — populates flags when listing models.

Registry table is duplicated in README for operators (per your feedback).

### Validation and fail-closed behavior

`letta/helpers/message_helper.py` — `validate_message_creates_for_vision()`:

- Allowed image MIME types: `image/jpeg`, `image/png`, `image/gif`, `image/webp`.
- Per-image cap: `LETTA_MAX_IMAGE_BYTES` (default 20 MiB decoded).
- Per-message cap: `LETTA_MAX_MESSAGE_BYTES` (default 80 MiB).
- If the agent’s model is not vision-capable → **`LettaVisionCapabilityError` (HTTP 422)** with explicit message (no silent degradation to text-only).

New errors in `letta/errors.py`; handlers in `letta/server/rest_api/app.py`:

- `LettaVisionCapabilityError` → 422
- `LettaMessageTooLargeError` → 413

Validation is invoked from:

- `letta/server/rest_api/utils.py` (`create_input_messages`)
- `letta/agents/helpers.py` (agent message path)

### Timeouts and provider preferences

`letta/settings.py`:

| Setting | Env | Default (v0.3.0) |
|---------|-----|------------------|
| `llm_request_timeout_seconds` | `LETTA_LLM_REQUEST_TIMEOUT_SECONDS` | **300** (was 60) |
| `llm_max_retries` | `LETTA_LLM_MAX_RETRIES` | 0 |
| `llm_retry_on_timeout` | `LETTA_LLM_RETRY_ON_TIMEOUT` | false |
| `max_image_bytes` | `LETTA_MAX_IMAGE_BYTES` | 20 MiB |
| `max_message_bytes` | `LETTA_MAX_MESSAGE_BYTES` | 80 MiB |
| `vision_models_extra` | `LETTA_VISION_MODELS_EXTRA` | (empty) |

`letta/llm_api/openai_client.py` uses timeout settings and merges provider preferences for OpenRouter.

### Configuration surface

`conf.yaml` documents vision-related defaults alongside existing server config.

---

## Architecture (client)

### Browser image pipeline

`frontend/src/lib/imagePipeline.ts`:

- Downscale/re-encode before upload (reduces payload size while preserving usability).
- Supports file picker, drag-drop, clipboard paste, and URL fetch flows.

### Proxy contract

`backend/schemas.py` — `SendMessageRequest.content: Union[str, List[ContentBlock]]`  
`backend/routes/messages.py` — passes block arrays to `conversations.messages.create(input=...)`; enforces `VISION_MAX_REQUEST_BYTES` (default 25 MiB).

### UI

| Component | Role |
|-----------|------|
| `contentBlocks.js` | Normalize Letta message content for display |
| `ImageViewer.svelte` | Full-size image viewing |
| `AttachmentThumbnail.svelte` | Composer attachments |
| `Chat.svelte` | Attach UX, optimistic user bubble with image, history rendering |
| `Agents.svelte` | Vision badge on capable models |
| `stores.js` | `modelSupportsVision()` cache |

Non-vision agents: attach controls disabled; server still enforces 422 if bypassed.

### Documentation

`docs/ARCHITECTURE.md` — Vision support section (input, outbound shape, capability UI, rendering, limits).

---

## Deploy stack (`letta-vision-deploy`)

No application code in deploy; v0.3.0 release documents the **paired stack**:

1. Build server image from `letta-vision` tag `v0.3.0`: `docker build -t letta-vision:v0.3.0 -t letta-vision:latest .`
2. `docker compose up -d --build` rebuilds client from sibling `letta-vision-client` at `v0.3.0`.
3. Recommended: `LETTA_LLM_REQUEST_TIMEOUT_SECONDS=300`, `LETTA_LLM_STREAM_TIMEOUT_SECONDS=600` in `.env`.

Compose continues to use `image: letta-vision:latest` (operator rebuilds after pull/tag).

---

## Verification evidence

| Check | Result |
|-------|--------|
| `GET /v1/models` — `moonshotai/kimi-k2.6` | `supports_vision: true` (33 models flagged) |
| Client proxy `/api/models` | Same flag propagated |
| `scripts/letta_vision_smoke_test.py` (base64, sample.png) | ~19.5s, substantive description — LIKELY OK |
| Manual UI (Lyra / Kimi K2.6, “Vision Testing” conversation) | Image in bubble, detailed reasoning + correct scene interpretation |
| Unit tests (isolated) | `tests/test_vision_capability.py`, `tests/test_llm_timeout_config.py` — 8 passed |
| Frontend build | `npm run build` OK |

---

## Commits in this release

### letta-vision (since v0.2.0)

- `2a72dd870` — Add vision support contract: registry, validation, and LLM timeouts.
- `3a344e5c7` — Fix vision validation tests and base64 size check compatibility.

**Diff summary:** 16 files, +389 / −10 lines (registry, validation, errors, settings, tests, README).

### letta-vision-client (since v0.2.0)

- `19781e1` — Add vision UI: image attach pipeline, content-block proxy, and history rendering.

**Diff summary:** 12 files, +520 / −26 lines.

### letta-vision-deploy

- Release notes + `.env.example` alignment (`LETTA_LLM_REQUEST_TIMEOUT_SECONDS=300`).

---

## Known gaps and v0.4+ candidates

1. **MCP / tool-return images (FR §7)** — `resolve_tool_return_images()` is wired for approval flows; whether MCP tool results reach the **next LLM call** as image blocks (path A) or only via client stream (path B) is **unverified**. No MCP UI for tool images in v0.3.0.

2. **Lazy image fetch (FR v2)** — v1 stores inline base64 in message history; heavy sessions will hit context limits. v2 should use `GET /api/files/{file_id}` lazy fetch.

3. **Docker image pinning** — Compose still uses `letta-vision:latest`; operators should rebuild from tag or we pin `letta-vision:v0.3.0` in a follow-up.

4. **Version string in health** — Fixed post-v0.3.0: `pyproject.toml` and `LETTA_VERSION` (compose default `0.3.0`) drive `/v1/health/`.

5. **Registry audit (post-v0.3.0)** — 33 flagged models were glob expansion of 12 FR families, not extra entries. One over-inclusion found: `o3*` matched `o3-mini` (no API vision). Tightened to `o3`, `o3-pro*`, `o3-2025-*` (excludes `o3-mini*`); `o4*` narrowed to `o4-mini*` only.

---

## Acknowledgment

Ades, the paired FR documents and your plan feedback (`LETTA_VISION_MODELS_EXTRA`, bidirectional MCP investigation framing, optimistic image rendering, README registry table) shaped a clean split: server contract first, client pass-through second, MCP gated. The registry-as-single-source-of-truth pattern keeps API, validation, and operator docs aligned — exactly the kind of boundary your specs draw well.

---

## References

- [FR-letta-vision_vision-support.md](FR-letta-vision_vision-support.md)
- [FR-letta-vision-client_vision-support.md](../../letta-vision-client/docs/FR-letta-vision-client_vision-support.md)
- [Implementation plan](../.cursor/plans/vision_support_implementation_3372ef1f.plan.md)
- GitHub releases: `v0.2.0` (baseline), `v0.3.0` (this report)
