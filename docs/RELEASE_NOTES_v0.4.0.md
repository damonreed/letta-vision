# Release notes — v0.4.0 (letta-vision)

**Theme:** Image context survives across turns in the LLM request path.

## Highlights

- **Cross-turn vision** — Historical user images stay in provider payloads (OpenAI `image_url`, Anthropic `image`, Gemini `inline_data`) on later text-only turns.
- **Tool-row fix** — Order-based `fill_image_content_in_messages()` so expanded tool messages do not drop image blocks.
- **Clearer failures** — One user-visible notice when the LLM stream fails; diagnostics stay in logs.

## Upgrade from v0.3.0

```bash
git checkout v0.4.0
docker build -t letta-vision:v0.4.0 -t letta-vision:latest .
```

Pair with **letta-vision-client v0.4.0** and **letta-vision-deploy v0.4.0** for the full stack.

Set `LETTA_VERSION=0.4.0` in deploy `.env` (health endpoint).

## Verification

```bash
pytest tests/test_message_serialization.py
# Optional live OpenRouter persistence check:
# OPENROUTER_API_KEY=... pytest tests/integration_test_image_persistence.py
```

## Operator notes

- Every turn resends all in-context images to the provider; token use grows super-linearly on long visual threads.
- OpenRouter and other upstream providers may still return timeouts or 502s; use `LETTA_LLM_REQUEST_TIMEOUT_SECONDS` and retry env vars as needed.

## Documentation

- README — Vision / cross-turn section.
- [CHANGELOG](../CHANGELOG.md) — full change list.
