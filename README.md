# Letta (formerly MemGPT)

Build AI with advanced memory that can learn and self-improve over time.

* [Letta Code](https://docs.letta.com/letta-code): run agents locally in your terminal
* [Letta API](https://docs.letta.com/quickstart/): build agents into your applications

## Get started in the CLI

Requires [Node.js 18+](https://nodejs.org/en/download)

1. Install the [Letta Code](https://github.com/letta-ai/letta-code) CLI tool: `npm install -g @letta-ai/letta-code`
2. Run `letta` in your terminal to launch an agent with memory running on your local computer

When running the CLI tool, your agent help you code and do any task you can do on your computer.

Letta Code supports [skills](https://docs.letta.com/letta-code/skills) and [subagents](https://docs.letta.com/letta-code/subagents), and bundles pre-built skills/subagents for advanced memory and continual learning. Letta is fully model-agnostic, though we recommend Opus 4.5 and GPT-5.2 for best performance (see our [model leaderboard](https://leaderboard.letta.com/) for our rankings).

## Get started with the Letta API

Use the Letta API to integrate stateful agents into your own applications.
Letta has a full-featured agents API, and a Python and Typescript SDK (view our [API reference](https://docs.letta.com/api)).

### Installation

TypeScript / Node.js:
```bash
npm install @letta-ai/letta-client
```

Python:
```bash
pip install letta-client
```

### Hello World example

Below is a quick example of creating a stateful agent and sending it a message (requires a [Letta API key](https://app.letta.com)).
See the full [quickstart guide](https://docs.letta.com/quickstart) for complete documentation.

TypeScript:
```typescript
import Letta from "@letta-ai/letta-client";

const client = new Letta({ apiKey: process.env.LETTA_API_KEY });

// Create your agent
const agentState = await client.agents.create({
  model: "openai/gpt-5.2",
  memory_blocks: [
    {
      label: "human",
      value:
        "Name: Timber. Status: dog. Occupation: building Letta, infrastructure to democratize self-improving superintelligence",
    },
    {
      label: "persona",
      value:
        "I am a self-improving superintelligence. Timber is my best friend and collaborator.",
    },
  ],
  tools: ["web_search", "fetch_webpage"],
});

console.log("Agent created with ID:", agentState.id);

// Send your agent a message
const response = await client.agents.messages.create(agentState.id, {
  input: "What do you know about me?",
});

for (const message of response.messages) {
  console.log(message);
}
```

Python:
```python
from letta_client import Letta
import os

client = Letta(api_key=os.getenv("LETTA_API_KEY"))

# Create your agent
agent_state = client.agents.create(
    model="openai/gpt-5.2",
    memory_blocks=[
        {
          "label": "human",
          "value": "Name: Timber. Status: dog. Occupation: building Letta, infrastructure to democratize self-improving superintelligence"
        },
        {
          "label": "persona",
          "value": "I am a self-improving superintelligence. Timber is my best friend and collaborator."
        }
    ],
    tools=["web_search", "fetch_webpage"]
)

print(f"Agent created with ID: {agent_state.id}")

# Send your agent a message
response = client.agents.messages.create(
    agent_id=agent_state.id,
    input="What do you know about me?"
)

for message in response.messages:
    print(message)
```

## Contributing

Letta is an open source project built by over a hundred contributors from around the world. There are many ways to get involved in the Letta OSS project!

* [**Join the Discord**](https://discord.gg/letta): Chat with the Letta devs and other AI developers.
* [**Chat on our forum**](https://forum.letta.com/): If you're not into Discord, check out our developer forum.
* **Follow our socials**: [Twitter/X](https://twitter.com/Letta_AI), [LinkedIn](https://www.linkedin.com/in/letta), [YouTube](https://www.youtube.com/@letta-ai)

---

***Legal notices**: By using Letta and related Letta services (such as the Letta endpoint or hosted service), you are agreeing to our [privacy policy](https://www.letta.com/privacy-policy) and [terms of service](https://www.letta.com/terms-of-service).*


## Vision support

This fork adds vision contract hardening on top of upstream Letta multimodal content blocks.

### Behavior (v1)

- Image content blocks are stored **inline** in message history (base64 in `source.data` with `source.type = "letta"` and `file_id`).
- **Cross-turn context (v0.4.0):** On every agent turn, the full in-context history is serialized to the LLM provider with image blocks preserved as multimodal parts (`image_url` / Anthropic `image` / Gemini `inline_data`). Prior user images remain visible to the model on later text-only turns without re-uploading (Open WebUI-style: whole conversation in each request).
- **Token cost:** Image-bearing threads have **super-linear** input cost: each historical image is re-sent on every subsequent turn (roughly N images × M turns in cumulative payload size). Manageable for typical sessions; very long visual threads will hit context limits or large bills. v2 directions: lazy `file_id` fetch, provider prompt caching, memory-aware elision.
- Heavy image workflows hit context limits quickly; v2 will add lazy fetch via `GET /api/files/{file_id}`.

### Vision-capable models (registry)

| Provider | Model ID / pattern |
|----------|-------------------|
| OpenRouter | `moonshotai/kimi-k2.6` |
| OpenRouter | `moonshotai/kimi-k2.5` |
| OpenAI | `gpt-4o*`, `gpt-4.1*`, `o1*`, `o3*`, `o4*` |
| Anthropic | `claude-opus-4-*`, `claude-sonnet-4-*`, `claude-haiku-4-*` |
| Google | `gemini-2.5-pro*`, `gemini-2.5-flash*` |

Operators can add custom models: `LETTA_VISION_MODELS_EXTRA=my-custom-llava-v1,my-finetuned-qwen-vl` (comma-separated).

Glob patterns expand dated catalog entries (e.g. many `gpt-4o-*` snapshots). `o3-mini` is intentionally excluded (no API vision). Health reports fork version via `LETTA_VERSION` (default `0.4.0` in this repo).

Legacy non-v3 agent loops may still replace images with `"[Image Here]"` text placeholders; use `letta_v1_agent` / Letta Agent V3 for vision.

### Limits

| Variable | Default |
|----------|---------|
| `LETTA_MAX_IMAGE_BYTES` | 20 MiB |
| `LETTA_MAX_MESSAGE_BYTES` | 80 MiB |
| `LETTA_LLM_REQUEST_TIMEOUT_SECONDS` | 300 |
| `LETTA_LLM_RETRY_ON_TIMEOUT` | false |

Sending images to a non-vision model returns **422** (not silent text degradation).
