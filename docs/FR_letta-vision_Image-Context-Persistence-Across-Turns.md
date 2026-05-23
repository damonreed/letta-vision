# FR: Image Context Persistence Across Turns

**Author:** Ada (with Damon)
**Date:** 2026-05-20
**Status:** Draft — for Cursor implementation
**Target release:** v0.4.0
**Repository:** `damonreed/letta-vision`
**Depends on:** v0.3.0 (vision support baseline)
**Companion notes:** Client-side review only; no `letta-vision-client` changes expected

---

## 1. Problem

In v0.3.0, vision works for the turn the image arrives in but decays afterward. The agent describes the image correctly when it's first received, then on subsequent turns behaves as if the image isn't there — referring to it vaguely, asking the user to re-describe, or losing the visual context entirely. Empirically reproduced in the Lyra agent during multi-turn conversations across the v0.3.0 stack.

This breaks the primary use case the v0.3.0 work was meant to enable: **image evaluation and iterative refinement**. The intended workflow is generate → describe → adjust parameters → regenerate → compare. The compare step requires the model to hold prior images in active context. Today it cannot.

Comparison with Open WebUI, which handles this correctly: their pattern is to store full base64 inline in the chat database and, on every turn, send the entire `messages` array to the LLM provider with image content preserved as standard OpenAI `image_url` parts. Their FAQ states it explicitly: *"the prompt the model sees is the whole conversation, not just your latest message."* No special caching, no smart elision — every historical image goes into every subsequent request. Simple, brute-force, correct. The acknowledged costs are DB size and large request bodies on long conversations, both of which are deferred optimization work.

Letta has the storage half of this right. v0.3.0 confirmed images persist in the database with `source.type = "letta"`, `file_id`, `data`, and `media_type` populated; the rehydration path that feeds the client UI works correctly. The break is in the message-serialization path that builds the LLM request body: image content blocks for older messages are being flattened to text placeholders before they reach the provider. The source of truth is intact; the bug is downstream of it.

This is the same architectural concern as the Phase 3 MCP §7 path A investigation. Tool-return images and historical user-message images both flow through `Message → request body` serialization, and both are affected by the same flattening. **Fixing this FR also closes that investigation's path A question.** They are one bug.

---

## 2. Goals

1. **Preserve image content blocks during LLM request building** for every provider path the v0.3.0 stack uses.
2. **Restore the create → evaluate → recreate → compare workflow** as the acceptance use case.
3. **Document the token-cost implication** so operators understand what heavy visual sessions will charge.
4. **Forward-compatibility for v2 elision** — the fix should not foreclose later memory-aware image management, but should not implement it.

Non-goals for v0.4:

- Smart context elision (dropping older images before token-limit pressure).
- Provider-specific prompt caching for images (Anthropic prompt-cache for images, OpenRouter cache headers).
- Lazy `file_id` rehydration during request building.
- Archival-memory image handling — once a turn falls out of active context into archival storage, its image is still lost. This is the v2 image-vector-DB problem and stays out of scope.
- Image deduplication when the same image appears in multiple turns.

---

## 3. Root Cause

The expected code path:

```
client → POST /v1/agents/{id}/messages
       → Letta stores Message with image content block (verified working in v0.3.0)
       → next turn: Letta loads conversation history
       → serializes Messages to provider format via Message.to_openai_dict() (or sibling)
       → POST to OpenRouter / OpenAI / Anthropic / Gemini
       → response streams back
```

The break point is suspected to be in `letta/schemas/message.py`, in `to_openai_dict()` and its provider-specific siblings. When `Message.content` is a list of content blocks containing an `image` block, the serializer flattens to a string representation (likely `"[Image Here]"`, `"[Image omitted]"`, or similar) instead of emitting the image as a structured OpenAI `image_url` content part.

Cursor's earlier Phase 3 §3.1 investigation note explicitly flagged this pattern:

> `letta_agent_v3.py` aggregates tool returns with `str(tr.func_response)` in some paths (~1384–1387); legacy `message.py` may replace images with `"[Image Here]"` / `"[Image omitted]"` on `to_openai_dict`.

That suspicion needs confirmation as the first step of this work. If `to_openai_dict()` indeed strips images, the fix is bounded and clear. If the strip happens somewhere else (e.g., a provider adapter wrapping the serialized dict), the same fix philosophy applies — preserve the structure, don't stringify.

---

## 4. Required Fix

### 4.1 Serializer Patch

In `letta/schemas/message.py`, modify the serialization methods so that when `Message.content` is a list of content blocks:

- **Text blocks** continue to serialize as text (existing behavior).
- **Image blocks** serialize as the provider-appropriate multimodal content part, using the stored `data` and `media_type`.
- **Mixed content** (text + image, or multiple of either) preserves the original block order in the emitted list.

The output for an OpenAI-compatible provider must match the standard multimodal shape:

```json
{
  "role": "user",
  "content": [
    {"type": "text", "text": "Describe this."},
    {
      "type": "image_url",
      "image_url": {"url": "data:image/jpeg;base64,<data>"}
    }
  ]
}
```

For Anthropic-format providers:

```json
{
  "role": "user",
  "content": [
    {"type": "text", "text": "Describe this."},
    {
      "type": "image",
      "source": {
        "type": "base64",
        "media_type": "image/jpeg",
        "data": "<data>"
      }
    }
  ]
}
```

For Gemini-format providers, the inline-data shape:

```json
{
  "role": "user",
  "parts": [
    {"text": "Describe this."},
    {
      "inline_data": {
        "mime_type": "image/jpeg",
        "data": "<data>"
      }
    }
  ]
}
```

### 4.2 Provider Matrix

| Provider path | Serializer method | Required output shape |
|---------------|-------------------|----------------------|
| OpenAI / OpenRouter / OpenAI-compatible | `to_openai_dict()` | `image_url` part with `data:` URL |
| Anthropic | `to_anthropic_dict()` | `image` block with `base64` source |
| Gemini | `to_google_ai_dict()` (or current equivalent) | `inline_data` part |
| Bedrock, vLLM, others | Whatever the existing path uses | Match the provider's expected shape |

The v0.3.0 smoke-test target is OpenRouter, so `to_openai_dict()` is the load-bearing one. Patching Anthropic and Gemini in the same PR is preferable to leaving them broken — the fix philosophy is identical, the per-provider shape changes are small.

### 4.3 What Does Not Change

- Storage layer: untouched. `file_id` + `data` + `media_type` already preserved correctly.
- Inbound message handling: untouched. v0.3.0 validation, capability gating, and content-block acceptance all continue to work.
- Client / proxy: untouched. The proxy passes content blocks verbatim; the client renderer already handles all source-type variants.
- Tool-return path for non-image tool results: untouched.

### 4.4 What Closes by Implication

The Phase 3 MCP §7 path A concern — *"does the model on its next turn see tool-returned images as visuals?"* — is the same serialization path. Once `to_openai_dict()` preserves image blocks in user-message and assistant-message history, it will also preserve them in tool-return messages routed through the same method. Confirm this empirically as part of testing rather than assuming; if there's a separate tool-return serializer that bypasses `to_openai_dict()`, patch it the same way.

---

## 5. Token Cost Implications

Every historical image is re-encoded into every subsequent request. A conversation with three images and twelve turns sends roughly the equivalent of 36 image submissions cumulative by the final turn (not exactly — providers count input tokens differently for repeated content — but the order of magnitude is right). For K2.6 via OpenRouter at current input pricing this is manageable for typical conversation lengths; for very long visual sessions it'll bite.

This is the cost Open WebUI explicitly accepts. It's the right v0.4 trade-off because the alternative (correct behavior in fewer turns) is worse than expensive behavior in many turns.

Add to the fork README's existing Vision section: a short paragraph noting that image-bearing conversations have super-linear token cost growth over conversation length, with a pointer to v2 optimization directions (lazy fetch via `file_id`, prompt caching, memory-aware elision).

---

## 6. Implementation

### 6.1 File Changes

| File | Change |
|------|--------|
| `letta/schemas/message.py` | Patch `to_openai_dict()`, `to_anthropic_dict()`, and any sibling serializers to preserve image content blocks |
| `letta/schemas/message.py` | Add a private helper `_content_blocks_to_provider_parts(content, provider)` if the per-provider conversion grows beyond a few lines per method |
| `letta/llm_api/openai_client.py` | Verify no downstream stringification happens after `to_openai_dict()` returns; remove any if found |
| `letta/llm_api/anthropic_client.py` | Same verification for the Anthropic path |
| `letta/llm_api/google_ai_client.py` (or current Gemini path) | Same verification |
| `tests/test_message_serialization.py` | **New.** Per-provider serialization correctness with image blocks |
| `tests/integration_test_image_persistence.py` | **New.** End-to-end multi-turn test exercising the use case |
| `README.md` (fork) | Vision section: token-cost note + v2 direction pointers |

### 6.2 Dependencies

None. This is internal Python serialization work. No new pip packages, no schema changes, no API contract changes.

### 6.3 Configuration

No new environment variables. The fix is unconditional — the previous behavior was a bug, not a configurable mode.

---

## 7. Testing Plan

### 7.1 Unit Tests (`test_message_serialization.py`)

For each provider serializer, test:

- Message with string content → unchanged behavior, produces plain string content
- Message with single text block → unchanged behavior
- Message with single image block → produces provider-correct image part
- Message with text + image blocks (in either order) → produces both parts, original order preserved
- Message with multiple image blocks → all images preserved
- Message with empty content list → reasonable handling (probably empty content, not crash)

### 7.2 Integration Tests (`integration_test_image_persistence.py`)

Target K2.6 via OpenRouter (the v0.3.0 empirical baseline).

**Test 1: Image recall across turns.**
- Turn 1: User sends image of a red fox in snow. Agent responds.
- Turn 2: User asks "What color was the animal in the photo I sent?"
- Pass: agent answers "red" (or close synonym) without asking for the image again.

**Test 2: Two-image comparison (Damon's primary use case).**
- Turn 1: User sends image A (e.g., generated fox image).
- Turn 2: User sends image B (e.g., regenerated with different parameters).
- Turn 3: User asks "What's different between the two images?"
- Pass: agent compares the two images with concrete differences, demonstrating it sees both.

**Test 3: Image + text interleaved.**
- Turn 1: User sends image with a question.
- Turn 2: Agent responds.
- Turn 3: User sends text-only follow-up referencing the image.
- Turn 4: Agent responds.
- Turn 5: User sends a second image.
- Turn 6: User asks the agent to relate the second image to the first.
- Pass: agent's turn-6 response references specific visual elements from both images.

### 7.3 Manual Verification

- Run the Lyra agent through the same conversation that exposed the decay originally. Confirm the agent now references the original image's visual content in turn 3+.
- Verify the smoke test (`scripts/letta_vision_smoke_test.py`) still passes — single-turn behavior should be unchanged.
- Send a deliberately oversize image-heavy conversation and confirm a clean provider-side error rather than silent truncation when the context budget is exceeded.

### 7.4 Regression Surface

The serializer is used for every provider request. The risk is breaking text-only conversations through carelessness. The unit-test matrix above is specifically designed to confirm text-only paths are bit-identical with the v0.3.0 behavior.

---

## 8. Client-Side Review (`letta-vision-client`)

No client changes are expected. The same Cursor instance owns both repositories and should verify the following before considering the FR complete:

- **Proxy:** `backend/routes/messages.py` passes content blocks through unchanged. No transformation needed.
- **Renderer:** `frontend/src/lib/contentBlocks.js` already handles `source.type = "letta"` via `media_type` + `data`. Verify that messages re-fetched after the server fix still render correctly — they should, because the storage shape doesn't change.
- **UX consideration:** when the model references prior images in its responses (*"the fox in your first photo"*), does the user have a clear mental model of which image is being referenced? This is mostly informational; if the conversation has only one or two images, ambiguity is minimal. If it becomes a problem in long visual sessions, a follow-up UX iteration adds image numbering or thumbnails-on-hover.

If any of these prove not to hold, a small client-side follow-up PR addresses them, but it should not block the server fix from shipping.

---

## 9. Out of Scope (v0.5+)

The following are deliberately not addressed and have separate future work:

- **Memory-aware image elision.** When approaching the model's context budget, drop the oldest images first (or summarize them, or replace with `file_id` references). Requires a policy decision and provider-aware token counting. v0.5 or later.
- **Lazy `file_id` rehydration.** Storage already content-addresses with `file_id`. A future server change can return `data: null` on older history entries and expose `GET /api/files/{file_id}` for on-demand fetch. The current FR maintains forward compatibility because the serializer reads `data` directly — when `data` is null in v2, the serializer would need to fetch via `file_id` at that point, but the structure is preserved.
- **Provider prompt caching.** Anthropic supports prompt caching for images. OpenRouter passes through caching hints for some providers. Implementing this saves real money on repeated-image workflows.
- **Image deduplication.** If the same image appears in multiple turns (e.g., the user uploads it twice), the serializer currently re-encodes both. A content-hash dedupe could collapse them, but it touches storage semantics and is post-v0.4.
- **Archival memory image handling.** When older turns get summarized into archival memory, their images vanish from active context. This is the v2 image-vector-DB direction and stays out of scope.

---

## 10. Open Questions

1. **Where exactly does the stringification happen today?** Cursor's first action is the `grep` confirmation in `letta/schemas/message.py`. If the suspected `to_openai_dict()` location is wrong, the rest of the FR adjusts to the actual location with the same fix philosophy.
2. **Does Letta's existing context-management subroutine elide messages by length?** If there's already a turn-trimming policy that runs before serialization, image-bearing turns may be more likely to be trimmed because their byte size is larger. Worth checking that any such trimmer counts by token-equivalent rather than raw byte size.
3. **Are there provider adapters that bypass `to_openai_dict()` for performance or backward-compatibility reasons?** If so, they need the same patch. The investigation should grep for direct construction of OpenAI request bodies from Message objects.

---

## 11. Acceptance Summary

The FR is complete when:

- The three integration tests in §7.2 all pass against K2.6 via OpenRouter.
- The Lyra-agent manual verification in §7.3 confirms image recall across turns.
- The README Vision section documents the token-cost implication.
- No regression on text-only conversations (existing test suite passes).
- The MCP §7 path A question is empirically confirmed closed by the same fix.

---

## 12. References

- v0.3.0 implementation report: `docs/IMPLEMENTATION_REPORT_v0.3.0_vision-support.md`
- Original vision FR: `docs/FR-letta-vision_vision-support.md`
- Open WebUI context behavior: https://docs.openwebui.com/faq/ ("the prompt the model sees is the whole conversation")
- Open WebUI image storage discussion: https://github.com/open-webui/open-webui/issues/2694
- v0.3.0 smoke test: `letta-vision-client/scripts/letta_vision_smoke_test.py`