# FR: Vision Support — letta-vision Server

**Author:** Ada (with Damon)
**Date:** 2026-05-20
**Status:** Draft — for Cursor implementation
**Repository:** `damonreed/letta-vision` (fork of `letta-ai/letta`)
**Companion:** `FR-letta-vision-client.md`

---

## 1. Problem

The upstream Letta server has a generic multi-modal content-block schema and provider adapters that translate between it and individual LLM backends. Empirical testing against `moonshotai/kimi-k2.6` via OpenRouter confirms that image content blocks pass through end-to-end and the model genuinely receives the image data (smoke test, 2026-05-20). Two gaps remain before the client work can stand on a stable contract:

1. **The model registry does not expose per-model vision capability.** A client building an image-attachment UI has no programmatic way to know whether the agent's current model can accept images. Without this, the only failure mode is silent degradation to a text placeholder after a full inference round trip — which is exactly the failure mode the empirical work spent the most effort uncovering.

2. **Outbound LLM call timeouts and provider routing are unspec'd.** The smoke test surfaced a double-billing pattern caused by the default outbound timeout firing during a vision call that was about to succeed. Operator-tunable timeout and retry behavior is required.

A third concern, MCP tool results carrying image content, is included as a v1 stretch goal. The implementation path requires server-side investigation before it can be specified in detail.

---

## 2. Goals

1. **Expose per-model vision capability** in the `/api/models` response so the client can render capability-aware UI.
2. **Specify the image content-block contract** the client must produce, and the validation the server enforces.
3. **Make outbound LLM call timeouts and retries configurable** at the provider adapter layer.
4. **Document v1 image storage behavior** (inline base64 in message history) and the v2 migration path.
5. **Investigate and scope MCP tool-result image handling** as a v1 stretch goal.

Explicit non-goals for v1:

- Archival memory image handling. Images in `archival_memory_search` results remain text-only descriptions.
- Image-vector-DB / embedding index. Storage decisions in v1 should not foreclose this, but no embedding work ships in v1.
- Video input. Letta's content-block schema does not represent video. K2.6 supports it but exposing it requires upstream schema work that belongs in v2.
- Reference-based image storage with on-demand rehydration. Current Letta behavior is inline; v1 documents it, v2 migrates.

---

## 3. Vision Capability in the Model Registry

### 3.1 Current Behavior

Letta's `/api/models` returns a list of `LLMConfig` objects. The shape includes `model`, `model_endpoint_type`, `context_window`, and provider-specific fields. There is no `supports_vision` or equivalent flag.

When an agent backed by a non-vision model receives an image content block, Letta's documented behavior is: *"images will still appear in the context window, but as a text message telling the agent that an image exists."* This is silent degradation — no error, no warning to the operator or client.

### 3.2 Required Change

Add a `supports_vision: bool` field to the `LLMConfig` schema and surface it through `/api/models`.

For known vision-capable models, the flag is set to `true`. The detection mechanism is a registry — a curated dictionary mapping model identifiers to capability flags, maintained in the server source. This is simpler than provider-side capability probing and matches how Letta already handles other model metadata.

**Initial registry entries** (vision-capable):

| Provider | Model ID | `supports_vision` |
|----------|----------|-------------------|
| OpenRouter | `moonshotai/kimi-k2.6` | `true` |
| OpenRouter | `moonshotai/kimi-k2.5` | `true` |
| OpenAI | `gpt-4o`, `gpt-4.1`, `o1`, `o3`, `o4` | `true` |
| Anthropic | `claude-opus-4-*`, `claude-sonnet-4-*`, `claude-haiku-4-*` | `true` |
| Google | `gemini-2.5-pro`, `gemini-2.5-flash` | `true` |

Models not in the registry default to `false`. Operators can override per-model via configuration if they're running a custom model they know supports vision.

### 3.3 Acceptance Criteria

- [ ] `GET /api/models` response includes `supports_vision: bool` on each model entry.
- [ ] At least the OpenRouter Kimi K2.6 entry returns `supports_vision: true`.
- [ ] A test agent created against a text-only model has `supports_vision: false` on its associated model entry.
- [ ] The registry is defined in a single source file that can be extended without touching adapter code.

---

## 4. Image Content-Block Contract

### 4.1 Inbound (Client → Server)

The client sends image content blocks in the Letta-native shape, base64-encoded:

```json
{
  "type": "image",
  "source": {
    "type": "base64",
    "media_type": "image/jpeg",
    "data": "<base64-encoded-bytes>"
  }
}
```

`media_type` is one of: `image/jpeg`, `image/png`, `image/webp`, `image/gif`.

URL-form blocks (`source.type = "url"`) are accepted by the SDK but the client is specified to never produce them — URL fetching happens browser-side per the client FR. The server does not need to special-case URL blocks; existing Letta behavior handles them, but they are out of the client's normal output path.

### 4.2 Outbound (Server → Client)

When the client reads message history, Letta returns image blocks in its internal shape:

```json
{
  "type": "image",
  "source": {
    "type": "letta",
    "file_id": "file-477f3ba9-fb37-430a-a617-5c316e02df4e",
    "data": "<base64-encoded-bytes>",
    "media_type": "image/jpeg",
    "detail": null
  }
}
```

The client renders both shapes by keying on `source.media_type` and `source.data`. The server does not change this output format in v1.

### 4.3 Validation

The Letta API does basic block validation already. Add these guards specifically for image blocks:

| Check | Failure Mode |
|-------|--------------|
| `media_type` in allowed set | 400 with `validation_error` |
| `data` is non-empty | 400 with `validation_error` |
| Decoded image size ≤ `LETTA_MAX_IMAGE_BYTES` (default 20 MiB) | 413 with `validation_error` |
| Total message body ≤ `LETTA_MAX_MESSAGE_BYTES` (default 80 MiB) | 413 with `validation_error` |

Validation runs before the LLM call. Errors return immediately without invoking the model.

### 4.4 Acceptance Criteria

- [ ] Sending a base64 image block with valid media_type and data returns a normal agent response.
- [ ] Sending a block with `media_type: "image/tiff"` returns 400.
- [ ] Sending a 25 MiB image returns 413 without invoking the model.
- [ ] Sending an image block to an agent whose model has `supports_vision: false` returns 422 with a clear error message (do not silently degrade).
- [ ] Existing text-only message sending is unchanged.

---

## 5. Outbound LLM Call Configuration

### 5.1 Required Changes

The OpenAI-compatible provider adapter (the path used for OpenRouter) currently uses default `httpx` timeout and retry behavior, which caused the double-generation pattern observed during smoke testing. Make these configurable.

**New environment variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `LETTA_LLM_TIMEOUT_SECONDS` | `300` | Total timeout per LLM call |
| `LETTA_LLM_MAX_RETRIES` | `1` | Retries on transient failure (connection error, 5xx) |
| `LETTA_LLM_RETRY_ON_TIMEOUT` | `false` | If true, retry once on timeout; if false, fail fast |

The defaults reflect: vision calls can run 60–120s on K2.6, so 300s gives headroom for the slow tail. Retries should not silently double-bill — if the first call times out, fail to the operator so they can decide.

### 5.2 OpenRouter Provider Routing (Optional)

OpenRouter accepts a `provider` field in the request body that pins or orders backend providers:

```json
"provider": {
  "order": ["siliconflow", "inceptron"],
  "ignore": ["novita"]
}
```

Expose this as a per-agent setting in the `LLMConfig`:

```python
class LLMConfig:
    # ... existing fields ...
    provider_preferences: Optional[dict] = None
```

When set, the OpenAI-compatible adapter passes it through in the request body. When unset, OpenRouter's default routing applies.

This is a knob for empirically tuned deployments. The client FR does not require an editor UI for this in v1 — it can be set at agent-creation time via API.

### 5.3 Acceptance Criteria

- [ ] `LETTA_LLM_TIMEOUT_SECONDS=300` applies to the K2.6 OpenRouter call path.
- [ ] When the LLM call exceeds the timeout, the operator gets a clear error rather than a silent retry.
- [ ] `provider_preferences` set on an `LLMConfig` is forwarded as the `provider` field to OpenRouter.
- [ ] Smoke-test script execution produces exactly one row in the OpenRouter usage log per run (verifies the double-generation fix landed).

---

## 6. Image Storage in Message History (v1)

### 6.1 Current Behavior

Letta stores image content blocks inline in message history. The example provided in the project shows the rehydrated shape: `source.type = "letta"`, `file_id` populated, `data` populated with the full base64 payload, `media_type` preserved. This means content-addressed storage already exists internally (the `file_id`), but rehydration returns the full payload.

### 6.2 v1 Behavior — Documented, Not Changed

v1 ships with current behavior intact. Inline base64 in history. The implication is a hard context-budget ceiling: a conversation with five 2 MiB images cannot exceed roughly 240 KB of base64 per image in active context once you're past the model's input token limit. For most conversational use this is fine. For heavy visual workflows it isn't.

What v1 *does* add:

- A documented warning in the server FR (this section) so operators understand the constraint.
- The server already content-addresses internally via `file_id`. v1 does not change that. v2 builds on it.

### 6.3 v2 Migration Path (Not Implemented)

The shape v2 should converge on:

- Outbound message history returns `source.type = "letta"` with `file_id` and `media_type`, but **`data` is null by default**.
- A new endpoint `GET /api/files/{file_id}` returns the binary content.
- Clients that need the image data fetch it lazily.
- Context budget for active messages can elide older image data while keeping references intact.

v1's client (per the client FR) renders `source.data` directly when present. v2's client would fetch via `file_id` when `data` is null. This is a forward-compatible split — the v1 client keeps working against a v2 server as long as the server can optionally include `data` for recent messages.

### 6.4 Acceptance Criteria

- [ ] No code change required for v1; storage behavior is unchanged from upstream Letta.
- [ ] The behavior is documented in the server fork's README under a "Vision support" section.
- [ ] The `file_id` field is preserved in the outbound content block (verify it's not stripped by any fork-specific code).

---

## 7. MCP Tool-Result Image Handling (v1 Stretch)

### 7.1 Problem

The target flow:

1. User sends a text message: *"Generate an image of a red fox in snow."*
2. Agent decides to call the `zapimage` MCP tool.
3. ZapImage returns a generated image as a URL or base64 in the tool result.
4. The model needs to *see* the returned image as a visual (so it can describe, critique, or follow up on it).
5. The user needs to see the image rendered in the chat UI.

For (4) to work, the Letta agent loop must convert image-bearing tool results into image content blocks when feeding the subsequent LLM call. For (5) to work, the message stream must surface those tool-result images to the client in a renderable form.

### 7.2 Required Investigation

Before specifying implementation, Cursor should answer these empirical questions:

1. **What shape do MCP tool results currently take in Letta's agent loop?** Specifically: when an MCP tool returns content that includes images (per the MCP spec, tool results can carry `ImageContent` blocks), does Letta's agent loop preserve them or stringify them?
2. **How are tool results currently formatted for the next LLM call?** If they're flattened to text, the image is lost before the model sees it.
3. **How are tool results currently surfaced in the message stream returned to clients?** The client FR specifies rendering, but only if the server emits parseable structure.

### 7.3 Specification (Conditional on Investigation)

If MCP tool results already preserve image content through the agent loop, the work is small: ensure the message stream emits tool-result messages in a shape the client can render, and verify the next-LLM-call path passes the image as a content block.

If they don't, the work involves modifying the agent loop's tool-result handler to detect image content in MCP responses, convert to Letta's image content-block shape, and inject into the message stream as part of the tool-result message.

### 7.4 Acceptance Criteria (Stretch)

- [ ] An agent with the ZapImage MCP server configured can be asked to generate an image, and the model receives the generated image as a visual input on its next turn.
- [ ] The tool-result message in the streamed response contains an `image` content block (or equivalent structured reference) the client can render.
- [ ] Behavior is observable end-to-end via a follow-up smoke test that sends *"Generate an image of X. Describe what you generated."* and the model's second-turn description matches the actually-generated image.

---

## 8. Implementation

### 8.1 File Changes

| File | Change |
|------|--------|
| `letta/schemas/llm_config.py` | Add `supports_vision: bool = False` and `provider_preferences: Optional[dict] = None` |
| `letta/llm_api/model_registry.py` | **New.** Curated registry of vision-capable models |
| `letta/server/rest_api/routers/v1/models.py` | Populate `supports_vision` from registry on response |
| `letta/llm_api/openai_client.py` (or equivalent) | Read `LETTA_LLM_TIMEOUT_SECONDS`, `LETTA_LLM_MAX_RETRIES`, `LETTA_LLM_RETRY_ON_TIMEOUT` from env; pass `provider_preferences` through to OpenRouter |
| `letta/server/rest_api/routers/v1/agents/messages.py` | Add image-block validation per §4.3; return 422 when target model has `supports_vision: false` |
| `letta/agent.py` (or wherever the tool-result handler lives) | **Stretch.** Per §7 investigation outcome |
| `tests/test_vision_capability.py` | **New.** Capability flag in registry, validation rejects, capability-mismatch 422 |
| `tests/test_llm_timeout_config.py` | **New.** Env vars are respected; retries don't silently double-bill |
| `README.md` (fork) | New "Vision support" section documenting v1 behavior, K2.6 empirical baseline, v2 migration intent |

### 8.2 Dependencies

No new Python dependencies expected. The work is internal: schema fields, registry data, validation, environment-driven configuration.

### 8.3 Configuration Summary

| Variable | Default | Required |
|----------|---------|----------|
| `LETTA_LLM_TIMEOUT_SECONDS` | `300` | No |
| `LETTA_LLM_MAX_RETRIES` | `1` | No |
| `LETTA_LLM_RETRY_ON_TIMEOUT` | `false` | No |
| `LETTA_MAX_IMAGE_BYTES` | `20971520` (20 MiB) | No |
| `LETTA_MAX_MESSAGE_BYTES` | `83886080` (80 MiB) | No |

---

## 9. Testing Plan

### 9.1 Unit Tests

- Model registry returns correct `supports_vision` flag for known entries
- Validation rejects unsupported media types
- Validation rejects oversized images
- Capability mismatch (image to text-only model) returns 422
- Timeout/retry env vars are picked up by the OpenAI-compatible client
- `provider_preferences` is forwarded in the OpenRouter request body when set

### 9.2 Integration Tests (Live K2.6 via OpenRouter)

- Send a base64 image block to a K2.6 agent; verify the model describes it correctly
- Send a URL image block to a K2.6 agent; verify behavior matches (Letta should still handle URL form even though client never produces it)
- Verify exactly one OpenRouter usage row per script run after timeout fix
- Send an image to an agent backed by a text-only model; verify 422

### 9.3 Manual Verification

- [ ] Run the existing `letta_vision_smoke_test.py` and confirm a single billing event per execution
- [ ] Confirm `/api/models` response includes `supports_vision` and K2.6 shows `true`
- [ ] Confirm history of an image-bearing conversation returns blocks with `source.type = "letta"` and renderable `data`

---

## 10. Phase Plan

**v1 (this FR):**
- §3 model registry capability flag
- §4 content-block contract and validation
- §5 outbound LLM call configuration
- §6 documentation of current storage behavior
- §7 stretch — investigation outcome determines whether it ships in v1 or slides to v2

**v2 (separate FR):**
- Reference-based history storage (lazy `data` rehydration via `GET /api/files/{file_id}`)
- Archival memory image handling
- Image embedding / vector index
- Video content blocks (requires upstream schema work)

---

## 11. Open Questions

1. **Where exactly does Letta's OpenAI-compatible adapter construct its httpx client?** Cursor should locate this on first read so the timeout patch lands in a single spot.
2. **Does Letta already enforce any image size limits?** If so, document existing limits before adding new ones, to avoid stacking constraints.
3. **What's the actual current behavior of MCP tool results carrying images in Letta's agent loop?** This determines whether §7 is a small spec or a big one.
4. **Are there other models in the OpenRouter Moonshot fleet that should be flagged vision-capable?** K2.5 also supports vision; older K2 does not. Confirm and populate the registry accordingly.

---

## 12. References

- Letta image inputs documentation: https://docs.letta.com/guides/core-concepts/messages/image-inputs/
- Letta MCP tools documentation: https://docs.letta.com/guides/core-concepts/tools/mcp-tools/
- Kimi K2.6 model card: https://huggingface.co/moonshotai/Kimi-K2.6
- Moonshot platform vision quickstart: https://platform.kimi.ai/docs/guide/kimi-k2-6-quickstart
- OpenRouter provider routing: https://openrouter.ai/docs/provider-routing
- Smoke test script: `scripts/letta_vision_smoke_test.py` (this fork)
- Companion document: `FR-letta-vision-client.md`