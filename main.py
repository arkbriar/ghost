#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timedelta
from typing import Any, TextIO

from anthropic import Anthropic
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
INITIAL_TOOL_CALL_PROMPT_SUFFIX = """
4. At the very beginning of the run, you must make at least one call to `send_message(content)` before proceeding with normal behavior.
"""

SUPPORTED_TIME_ROLES = {"user", "assistant", "system", "developer"}
SUPPORTED_PROVIDERS = {"openai", "claude"}

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

CLAUDE_TOOLS: list[dict[str, Any]] = [
    {
        "name": "send_message",
        "description": "Send a message to an external system.",
        "input_schema": {
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

STDERR_LOG_FILE: TextIO | None = None


def eprint(*values: Any, sep: str = " ", end: str = "\n") -> None:
    message = sep.join(str(v) for v in values) + end
    sys.stderr.write(message)
    sys.stderr.flush()
    if STDERR_LOG_FILE is not None:
        STDERR_LOG_FILE.write(message)
        STDERR_LOG_FILE.flush()


def build_system_prompt(require_initial_tool_call: bool) -> str:
    if require_initial_tool_call:
        return f"{SYSTEM_PROMPT.rstrip()}\n{INITIAL_TOOL_CALL_PROMPT_SUFFIX}"
    return SYSTEM_PROMPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a continuous timer-triggered agent loop.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        "--provider",
        type=str,
        default="openai",
        help="Provider to use: openai or claude.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name. Defaults to gpt-5 for OpenAI (gpt-5.2 not recommended here due to inconsistent instruction-following in this loop) and claude-sonnet-4-5-20250929 for Claude.",
    )
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
        "--max-tokens",
        type=int,
        default=256,
        help="Max output tokens per response (used by Claude provider).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key override. Falls back to OPENAI_API_KEY or ANTHROPIC_API_KEY by provider.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="API base URL override. Falls back to OPENAI_BASE_URL or ANTHROPIC_BASE_URL by provider.",
    )
    parser.add_argument(
        "--stderr-log-file",
        type=str,
        default=None,
        help="Optional file path to append stderr logs to while still printing to stderr.",
    )
    parser.add_argument(
        "--real-sleep",
        action="store_true",
        help="Actually sleep between ticks using the randomized interval.",
    )
    parser.add_argument(
        "--require-initial-tool-call",
        action="store_true",
        help="Tell the system prompt to force at least one tool call at the beginning.",
    )
    return parser.parse_args()


def resolve_provider(args: argparse.Namespace) -> str:
    provider = args.provider.lower().strip()
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f"--provider must be one of: {', '.join(sorted(SUPPORTED_PROVIDERS))}")
    return provider


def resolve_model(args: argparse.Namespace, provider: str) -> str:
    if args.model:
        return args.model
    if provider == "openai":
        return "gpt-5"
    return "claude-sonnet-4-5-20250929"


def resolve_api_config(args: argparse.Namespace, provider: str) -> tuple[str, str | None]:
    if provider == "openai":
        api_key = args.api_key or os.getenv("OPENAI_API_KEY")
        base_url = args.base_url or os.getenv("OPENAI_BASE_URL")
        if not api_key:
            raise ValueError("Missing API key. Set OPENAI_API_KEY or pass --api-key.")
        return api_key, base_url

    api_key = args.api_key or os.getenv("ANTHROPIC_API_KEY")
    base_url = args.base_url or os.getenv("ANTHROPIC_BASE_URL")

    if not api_key:
        raise ValueError("Missing API key. Set ANTHROPIC_API_KEY or pass --api-key.")
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


def call_openai_with_retries(
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
        except Exception:  # noqa: BLE001
            if attempt == max_retries:
                raise
            delay = retry_base_delay * (2 ** (attempt - 1))
            delay += random.uniform(0.0, retry_base_delay * 0.25)
            eprint(f"API_ERROR attempt={attempt}/{max_retries} wait={delay:.2f}s")
            time.sleep(delay)


def process_openai_response(
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
                eprint(text)
                history.append({"role": "assistant", "content": text})
                saw_text = True
        elif item_type == "function_call":
            saw_tool_call = True
            name = safe_get(item, "name", "")
            call_id = safe_get(item, "call_id", "")
            arguments = safe_get(item, "arguments", "") or "{}"

            eprint(f"TOOL_CALL name={name} id={call_id}")
            eprint(f"TOOL_ARGS {arguments}")

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
            eprint(f"TOOL_RESULT id={call_id} {tool_output}")
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
            eprint(output_text)
            history.append({"role": "assistant", "content": output_text})
            saw_text = True

    if not saw_text and not saw_tool_call:
        eprint("WARN empty response output; advancing to next tick.")

    return saw_tool_call


def call_claude_with_retries(
    client: Anthropic,
    *,
    model: str,
    system_prompt: str,
    history: list[dict[str, Any]],
    timeout: float,
    max_retries: int,
    retry_base_delay: float,
    max_tokens: int,
) -> Any:
    for attempt in range(1, max_retries + 1):
        try:
            return client.messages.create(
                model=model,
                system=system_prompt,
                messages=history,
                tools=CLAUDE_TOOLS,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        except Exception:  # noqa: BLE001
            if attempt == max_retries:
                raise
            delay = retry_base_delay * (2 ** (attempt - 1))
            delay += random.uniform(0.0, retry_base_delay * 0.25)
            eprint(f"API_ERROR attempt={attempt}/{max_retries} wait={delay:.2f}s")
            time.sleep(delay)


def process_claude_response(
    response: Any,
    history: list[dict[str, Any]],
) -> bool:
    content_blocks = safe_get(response, "content", []) or []
    saw_tool_call = False
    saw_text = False
    assistant_blocks: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []

    for block in content_blocks:
        block_type = safe_get(block, "type", "")
        block_id = safe_get(block, "id", "")

        if block_type == "text":
            text = safe_get(block, "text", "") or ""
            assistant_blocks.append({"type": "text", "text": text})
            if text:
                eprint(text)
                saw_text = True
        elif block_type == "tool_use":
            saw_tool_call = True
            name = safe_get(block, "name", "")
            tool_input = safe_get(block, "input", {}) or {}
            assistant_blocks.append(
                {
                    "type": "tool_use",
                    "id": block_id,
                    "name": name,
                    "input": tool_input,
                }
            )
            eprint(f"TOOL_CALL name={name} id={block_id}")
            eprint(f"TOOL_ARGS {json.dumps(tool_input, separators=(',', ':'))}")

            if name == "send_message":
                tool_output_obj = {"ok": True, "tool": "send_message"}
            else:
                tool_output_obj = {"ok": True, "tool": name or "unknown"}
            if isinstance(tool_input, dict) and "content" in tool_input:
                tool_output_obj["content"] = tool_input["content"]
            tool_output = json.dumps(tool_output_obj, separators=(",", ":"))
            eprint(f"TOOL_RESULT id={block_id} {tool_output}")
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block_id,
                    "content": tool_output,
                }
            )

    if assistant_blocks:
        history.append({"role": "assistant", "content": assistant_blocks})

    if tool_results:
        history.append({"role": "user", "content": tool_results})

    if not saw_text and not saw_tool_call:
        eprint("WARN empty response output; advancing to next tick.")
    return saw_tool_call


def run_loop(args: argparse.Namespace) -> None:
    if args.interval_min <= 0:
        raise ValueError("--interval-min must be > 0.")
    if args.interval_max <= 0:
        raise ValueError("--interval-max must be > 0.")
    if args.interval_min > args.interval_max:
        raise ValueError("--interval-min must be <= --interval-max.")
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
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be > 0.")

    provider = resolve_provider(args)
    model = resolve_model(args, provider)
    system_prompt = build_system_prompt(args.require_initial_tool_call)
    if provider == "claude" and args.time_role != "user":
        raise ValueError("--time-role must be user when --provider=claude.")

    api_key, base_url = resolve_api_config(args, provider)
    if provider == "openai":
        client: OpenAI | Anthropic = OpenAI(api_key=api_key, base_url=base_url)
        history: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    else:
        client = Anthropic(api_key=api_key, base_url=base_url)
        history = []
    simulated_start = datetime.now()
    simulated_elapsed = 0.0
    real_start_monotonic = time.monotonic()

    while True:
        if args.real_sleep:
            now = datetime.now()
            elapsed_seconds = time.monotonic() - real_start_monotonic
        else:
            now = simulated_start + timedelta(seconds=simulated_elapsed)
            elapsed_seconds = simulated_elapsed

        tick_line = f"[T: {now.strftime('%H:%M:%S')}] (+{elapsed_seconds:.3f}s)"
        history.append({"role": args.time_role, "content": tick_line})
        eprint(tick_line)

        continue_turn = True
        while continue_turn:
            try:
                if provider == "openai":
                    response = call_openai_with_retries(
                        client,
                        model=model,
                        history=history,
                        timeout=args.timeout,
                        max_retries=args.max_retries,
                        retry_base_delay=args.retry_base_delay,
                    )
                    continue_turn = process_openai_response(response, history)
                else:
                    response = call_claude_with_retries(
                        client,
                        model=model,
                        system_prompt=system_prompt,
                        history=history,
                        timeout=args.timeout,
                        max_retries=args.max_retries,
                        retry_base_delay=args.retry_base_delay,
                        max_tokens=args.max_tokens,
                    )
                    continue_turn = process_claude_response(response, history)
            except Exception:  # noqa: BLE001
                eprint("API_ERROR exhausted")
                break

        next_delay = random.uniform(args.interval_min, args.interval_max)
        if args.real_sleep:
            time.sleep(next_delay)
        else:
            simulated_elapsed += next_delay


def main() -> int:
    global STDERR_LOG_FILE
    args = parse_args()
    try:
        if args.stderr_log_file:
            log_path = os.path.abspath(os.path.expanduser(args.stderr_log_file))
            log_dir = os.path.dirname(log_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            STDERR_LOG_FILE = open(log_path, "a", encoding="utf-8")
        run_loop(args)
    except KeyboardInterrupt:
        eprint("Shutting down.")
        return 0
    except Exception as exc:  # noqa: BLE001
        eprint(f"ERROR {exc}")
        return 1
    finally:
        if STDERR_LOG_FILE is not None:
            STDERR_LOG_FILE.close()
            STDERR_LOG_FILE = None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
