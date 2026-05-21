# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Fork releases use `v0.x.y` tags (diverged from upstream Letta `0.16.x` at v0.2.0).

## [Unreleased]

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

[Unreleased]: https://github.com/damonreed/letta-vision/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/damonreed/letta-vision/releases/tag/v0.4.0
[0.3.0]: https://github.com/damonreed/letta-vision/releases/tag/v0.3.0
[0.2.0]: https://github.com/damonreed/letta-vision/releases/tag/v0.2.0
