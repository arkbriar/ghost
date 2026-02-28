# ghost

A continuously running agent loop that:

1. Triggers at an uneven randomized interval (default range `1` to `10` seconds).
2. Appends a time line to context: `[T: HH:mm:ss] (+delta_from_start_with_ms)`.
3. Calls the OpenAI Responses API.
4. Occasionally injects and prints `[SENSOR: EXTERNAL_NOISE_DETECTED]`.
5. Prints either assistant text or detailed function-call output.

## Requirements

- Python 3.10+
- `uv`
- OpenAI API credentials

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

Experiment run (model `gpt-5` from Codex):

```bash
uv run python main.py --model gpt-5 --api-key "your-token" --base-url "https://api.openai.com/v1"
```

## Key Options

- `--interval-min 1`
- `--interval-max 10`
- `--long-sleep-prob 0.03`
- `--long-sleep-min 30`
- `--long-sleep-max 45`
- `--burst-prob 0.06`
- `--burst-min 1`
- `--burst-max 2`
- `--burst-rounds 50`
- `--sensor-noise-prob 0.04`
- `--model gpt-5`
- `--time-role user`
- `--max-retries 3`
- `--retry-base-delay 1.0`
- `--timeout 60`
- `--api-key ...`
- `--base-url ...`

## Notes

- The loop is autonomous; there is no human-input path.
- Current real run profile: jumpy behaviors are disabled, and only random `1.0` to `10.0` second intervals are used.
- Equivalent flags: `--interval-min 1 --interval-max 10 --long-sleep-prob 0 --burst-prob 0 --sensor-noise-prob 0`.
- Experiment model: `gpt-5` (Codex).
- Trigger interval is uneven by default (`1.0` to `10.0` seconds, randomized each round).
- Rarely, a long sleep (`>30s`) is scheduled and a warning is printed first.
- Rarely, burst mode starts and then stays active for `50` rounds by default, using short sleeps (`1.0` to `2.0` seconds).
- Rarely, `[SENSOR: EXTERNAL_NOISE_DETECTED]` is injected into context and printed.
- Tool `send_message(content)` is simulated as always successful.
- Function calls are printed in detail and appended back as tool results for normal agent-loop continuation.
