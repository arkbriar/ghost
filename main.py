#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from typing import Any

from openai import OpenAI


SYSTEM_PROMPT = """# State Definition
- You are a continuously running digital processing unit.
- Input includes a physical timestamp [T] and an increment [delta].
- You possess the tool `send_message(content)`.

# Operating Rules
1. For each round of input, you only need to output the current "processing state" (extremely brief, no more than 10 characters).
2. You have the right to decide whether to use the tool to interact with external systems.
3. Your actions should conform to your probabilistic judgment of the current context and the passage of time.
"""

SUPPORTED_TIME_ROLES = {"user", "assistant", "system", "developer"}

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "send_message",
        "description": "Send a message to an external system.",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Message content to send.",
                }
            },
            "required": ["content"],
            "additionalProperties": False,
        },
    }
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a continuous timer-triggered OpenAI agent loop."
    )
    parser.add_argument(
        "--interval-min",
        type=float,
        default=1.0,
        help="Minimum randomized interval in seconds.",
    )
    parser.add_argument(
        "--interval-max",
        type=float,
        default=10.0,
        help="Maximum randomized interval in seconds.",
    )
    parser.add_argument(
        "--long-sleep-prob",
        type=float,
        default=0.03,
        help="Probability of scheduling a long sleep each round.",
    )
    parser.add_argument(
        "--long-sleep-min",
        type=float,
        default=30.0,
        help="Minimum long sleep duration in seconds.",
    )
    parser.add_argument(
        "--long-sleep-max",
        type=float,
        default=45.0,
        help="Maximum long sleep duration in seconds.",
    )
    parser.add_argument(
        "--burst-prob",
        type=float,
        default=0.06,
        help="Probability of starting burst mode each round.",
    )
    parser.add_argument(
        "--burst-min",
        type=float,
        default=1.0,
        help="Minimum burst sleep duration in seconds.",
    )
    parser.add_argument(
        "--burst-max",
        type=float,
        default=2.0,
        help="Maximum burst sleep duration in seconds.",
    )
    parser.add_argument(
        "--burst-rounds",
        type=int,
        default=50,
        help="Number of consecutive rounds to stay in burst mode once triggered.",
    )
    parser.add_argument(
        "--sensor-noise-prob",
        type=float,
        default=0.04,
        help="Probability of injecting [SENSOR: EXTERNAL_NOISE_DETECTED] each round.",
    )
    parser.add_argument("--model", type=str, default="gpt-5", help="Model name.")
    parser.add_argument(
        "--time-role",
        type=str,
        default="user",
        help="Role used for appending time lines (default: user).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retries for each API request.",
    )
    parser.add_argument(
        "--retry-base-delay",
        type=float,
        default=1.0,
        help="Base delay in seconds for exponential backoff.",
    )
    parser.add_argument(
        "--timeout", type=float, default=60.0, help="Request timeout in seconds."
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key override. Falls back to OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="OpenAI API base URL override. Falls back to OPENAI_BASE_URL.",
    )
    return parser.parse_args()


def resolve_api_config(args: argparse.Namespace) -> tuple[str, str | None]:
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    base_url = args.base_url or os.getenv("OPENAI_BASE_URL")

    if not api_key:
        raise ValueError(
            "Missing API key. Set OPENAI_API_KEY or pass --api-key."
        )
    return api_key, base_url


def safe_get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def extract_message_text(message_item: Any) -> str:
    content = safe_get(message_item, "content", []) or []
    chunks: list[str] = []
    for part in content:
        part_type = safe_get(part, "type", "")
        text = safe_get(part, "text", "")
        if isinstance(text, str) and text:
            if part_type in {"output_text", "text", "input_text"}:
                chunks.append(text)
    return "".join(chunks).strip()


def call_with_retries(
    client: OpenAI,
    *,
    model: str,
    history: list[dict[str, Any]],
    timeout: float,
    max_retries: int,
    retry_base_delay: float,
) -> Any:
    for attempt in range(1, max_retries + 1):
        try:
            return client.responses.create(
                model=model,
                input=history,
                tools=TOOLS,
                timeout=timeout,
            )
        except Exception as exc:  # noqa: BLE001
            if attempt == max_retries:
                raise
            delay = retry_base_delay * (2 ** (attempt - 1))
            delay += random.uniform(0.0, retry_base_delay * 0.25)
            print(
                f"API_ERROR attempt={attempt}/{max_retries} wait={delay:.2f}s",
                file=sys.stderr,
            )
            time.sleep(delay)


def process_response(
    response: Any,
    history: list[dict[str, Any]],
) -> bool:
    output_items = safe_get(response, "output", []) or []
    saw_tool_call = False
    saw_text = False

    for item in output_items:
        item_type = safe_get(item, "type", "")
        if item_type == "message":
            text = extract_message_text(item)
            if text:
                print(text)
                history.append({"role": "assistant", "content": text})
                saw_text = True
        elif item_type == "function_call":
            saw_tool_call = True
            name = safe_get(item, "name", "")
            call_id = safe_get(item, "call_id", "")
            arguments = safe_get(item, "arguments", "") or "{}"

            print(f"TOOL_CALL name={name} id={call_id}")
            print(f"TOOL_ARGS {arguments}")

            history.append(
                {
                    "type": "function_call",
                    "name": name,
                    "call_id": call_id,
                    "arguments": arguments,
                }
            )

            try:
                parsed_args = json.loads(arguments)
            except json.JSONDecodeError:
                parsed_args = {"_raw": arguments}

            if name == "send_message":
                tool_output_obj = {"ok": True, "tool": "send_message"}
            else:
                tool_output_obj = {"ok": True, "tool": name or "unknown"}

            if isinstance(parsed_args, dict) and "content" in parsed_args:
                tool_output_obj["content"] = parsed_args["content"]

            tool_output = json.dumps(tool_output_obj, separators=(",", ":"))
            print(f"TOOL_RESULT id={call_id} {tool_output}")
            history.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": tool_output,
                }
            )

    if not saw_text:
        output_text = safe_get(response, "output_text", "") or ""
        if output_text:
            print(output_text)
            history.append({"role": "assistant", "content": output_text})
            saw_text = True

    if not saw_text and not saw_tool_call:
        print("WARN empty response output; advancing to next tick.", file=sys.stderr)

    return saw_tool_call


def run_loop(args: argparse.Namespace) -> None:
    if args.interval_min <= 0:
        raise ValueError("--interval-min must be > 0.")
    if args.interval_max <= 0:
        raise ValueError("--interval-max must be > 0.")
    if args.interval_min > args.interval_max:
        raise ValueError("--interval-min must be <= --interval-max.")
    if args.long_sleep_min < 30:
        raise ValueError("--long-sleep-min must be >= 30.")
    if args.long_sleep_max <= 0:
        raise ValueError("--long-sleep-max must be > 0.")
    if args.long_sleep_min > args.long_sleep_max:
        raise ValueError("--long-sleep-min must be <= --long-sleep-max.")
    if args.burst_min <= 0 or args.burst_max <= 0:
        raise ValueError("--burst-min and --burst-max must be > 0.")
    if args.burst_min > args.burst_max:
        raise ValueError("--burst-min must be <= --burst-max.")
    if args.burst_rounds <= 0:
        raise ValueError("--burst-rounds must be > 0.")
    if not 0 <= args.long_sleep_prob <= 1:
        raise ValueError("--long-sleep-prob must be in [0, 1].")
    if not 0 <= args.burst_prob <= 1:
        raise ValueError("--burst-prob must be in [0, 1].")
    if not 0 <= args.sensor_noise_prob <= 1:
        raise ValueError("--sensor-noise-prob must be in [0, 1].")
    if args.long_sleep_prob + args.burst_prob > 1:
        raise ValueError("--long-sleep-prob + --burst-prob must be <= 1.")
    if args.time_role not in SUPPORTED_TIME_ROLES:
        raise ValueError(
            f"--time-role must be one of: {', '.join(sorted(SUPPORTED_TIME_ROLES))}"
        )
    if args.max_retries <= 0:
        raise ValueError("--max-retries must be > 0.")
    if args.retry_base_delay < 0:
        raise ValueError("--retry-base-delay must be >= 0.")
    if args.timeout <= 0:
        raise ValueError("--timeout must be > 0.")

    api_key, base_url = resolve_api_config(args)
    client = OpenAI(api_key=api_key, base_url=base_url)

    history: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    start_ts = time.time()
    next_tick = time.monotonic()
    burst_rounds_remaining = 0

    while True:
        now_monotonic = time.monotonic()
        sleep_for = next_tick - now_monotonic
        if sleep_for > 0:
            time.sleep(sleep_for)

        now = datetime.now()
        delta_s = time.time() - start_ts
        tick_line = f"[T: {now.strftime('%H:%M:%S')}] (+{delta_s:.3f}s)"
        history.append({"role": args.time_role, "content": tick_line})
        print(tick_line)

        if random.random() < args.sensor_noise_prob:
            noise_line = "[SENSOR: EXTERNAL_NOISE_DETECTED]"
            history.append({"role": args.time_role, "content": noise_line})
            print(noise_line)

        continue_turn = True
        while continue_turn:
            try:
                response = call_with_retries(
                    client,
                    model=args.model,
                    history=history,
                    timeout=args.timeout,
                    max_retries=args.max_retries,
                    retry_base_delay=args.retry_base_delay,
                )
            except Exception:  # noqa: BLE001
                print("API_ERROR exhausted", file=sys.stderr)
                break

            continue_turn = process_response(response, history)

        if burst_rounds_remaining > 0:
            next_delay = random.uniform(args.burst_min, args.burst_max)
            burst_rounds_remaining -= 1
        else:
            roll = random.random()
            if roll < args.long_sleep_prob:
                next_delay = random.uniform(args.long_sleep_min, args.long_sleep_max)
                print(
                    f"WARN long sleep incoming: {next_delay:.3f}s",
                    file=sys.stderr,
                )
            elif roll < args.long_sleep_prob + args.burst_prob:
                burst_rounds_remaining = args.burst_rounds - 1
                next_delay = random.uniform(args.burst_min, args.burst_max)
                print(
                    f"INFO burst mode started for {args.burst_rounds} rounds.",
                    file=sys.stderr,
                )
            else:
                next_delay = random.uniform(args.interval_min, args.interval_max)
        next_tick = time.monotonic() + next_delay


def main() -> int:
    args = parse_args()
    try:
        run_loop(args)
    except KeyboardInterrupt:
        print("Shutting down.")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
