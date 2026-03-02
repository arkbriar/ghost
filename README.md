# ghost

A continuously running agent loop that:

1. Advances on a simulated uneven randomized interval (default range `1` to `10` seconds, no real per-tick sleep).
2. Appends a time line to context: `[T: HH:mm:ss] (+delta_from_start_with_ms)`.
3. Calls either the OpenAI Responses API or Anthropic Claude Messages API.
4. Prints either assistant text or detailed function-call output.

## Requirements

- Python 3.10+
- `uv`
- API credentials for your selected provider

## Setup

```bash
uv sync
```

## Run

Using environment variables:

```bash
export OPENAI_API_KEY="your-token"
uv run python main.py
```

Using explicit CLI overrides:

```bash
uv run python main.py --api-key "your-token" --base-url "https://api.openai.com/v1"
```

Claude configuration (supported, not experimentally validated in this repo):

```bash
export ANTHROPIC_API_KEY="your-token"
uv run python main.py --provider claude --model claude-sonnet-4-5-20250929
```

OpenAI experiment run (recommended model `gpt-5`):

```bash
uv run python main.py --model gpt-5 --api-key "your-token" --base-url "https://api.openai.com/v1"
```

## Key Options

- `--interval-min 1`
- `--interval-max 10`
- `--provider openai`
- `--model ...` (defaults by provider)
- `--time-role user`
- `--max-retries 3`
- `--retry-base-delay 1.0`
- `--timeout 60`
- `--max-tokens 256` (Claude only)
- `--api-key ...`
- `--base-url ...`
- `--stderr-log-file /path/to/stderr.log` (append stderr output to file and stderr)
- `--real-sleep` (sleep in real time between ticks)
- `--require-initial-tool-call` (ask the model to perform at least one initial `send_message` call)

## Notes

- The loop is autonomous; there is no human-input path.
- Provider-specific env vars:
  - OpenAI: `OPENAI_API_KEY`, optional `OPENAI_BASE_URL`
  - Claude: `ANTHROPIC_API_KEY`, optional `ANTHROPIC_BASE_URL`
- For Claude mode, `--time-role` must be `user` because Claude messages only support `user`/`assistant`.
- Loop cadence uses simulated random intervals by default (no real-time waiting between ticks).
- Use `--real-sleep` to wait in wall-clock time for each randomized interval.
- Use `--require-initial-tool-call` to include an extra system-prompt rule requiring at least one tool call at startup.
- OpenAI model note: `gpt-5.2` is currently not recommended for this loop profile because it may under-follow instructions or follow them too rigidly; use `gpt-5`.
- Experiment model (OpenAI): `gpt-5` (Codex).
- Trigger interval is uneven by default (`1.0` to `10.0` seconds, randomized each round).
- Tool `send_message(content)` is simulated as always successful.
- Function calls are printed in detail and appended back as tool results for normal agent-loop continuation.
