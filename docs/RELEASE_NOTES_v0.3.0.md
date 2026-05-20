# Release notes — v0.3.0 (letta-vision)

**Theme:** Vision support contract on the Letta Vision server.

## Highlights

- **`supports_vision`** on `GET /v1/models` driven by curated registry + `LETTA_VISION_MODELS_EXTRA`.
- **Fail-closed validation** — images on non-vision models return **422**, not silent degradation.
- **Image limits** — JPEG/PNG/GIF/WebP; defaults 20 MiB/image, 80 MiB/message.
- **LLM timeouts** — default request timeout **300s**; retry env vars documented.
- **OpenRouter** — `provider_preferences` from agent config passed through to the API client.

## Upgrade from v0.2.0

```bash
git checkout v0.3.0
docker build -t letta-vision:v0.3.0 -t letta-vision:latest .
```

Set `LETTA_LLM_REQUEST_TIMEOUT_SECONDS=300` (or higher) when using vision models.

## Verification

```bash
pytest tests/test_vision_capability.py tests/test_llm_timeout_config.py
# Against running server (from letta-vision-client repo):
python scripts/letta_vision_smoke_test.py --mode base64 --image scripts/sample.png
```

## Documentation

- [IMPLEMENTATION_REPORT_v0.3.0_vision-support.md](IMPLEMENTATION_REPORT_v0.3.0_vision-support.md) — full report for Ada.
- README — Vision support section and registry table.

## Not in this release

- MCP tool-result images in the next LLM turn (FR §7) — planned investigation for v0.4+.
