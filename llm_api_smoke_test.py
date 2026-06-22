from __future__ import annotations

import argparse
import os
from pathlib import Path

from meetingscribe_core import call_openai_compatible_chat, parse_llm_json_response

DEFAULT_API_BASE = os.environ.get("DIZ_AI_BASE", "https://chat.ai.diz.uk-erlangen.de/api")
DEFAULT_MODEL = os.environ.get("DIZ_AI_MODEL", "mistral-large:123b")


def _get_default_api_key() -> str:
    if env_key := os.environ.get("DIZ_AI_KEY", ""):
        return env_key
    try:
        apikey_file = Path("apikey.txt")
        if apikey_file.exists():
            return apikey_file.read_text(encoding="utf-8").strip()
    except Exception:
        pass
    return ""


DEFAULT_API_KEY = _get_default_api_key()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the hosted LLM chat endpoint.")
    parser.add_argument("--llm-api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--llm-model", default=DEFAULT_MODEL)
    parser.add_argument("--llm-api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--no-json", action="store_true", help="Skip response_format=json_object on the first call")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    messages = [
        {"role": "system", "content": "Return valid JSON with one key named ok and the value 'yes'."},
        {"role": "user", "content": "Say hello and only return the JSON object."},
    ]

    content = call_openai_compatible_chat(
        api_base=args.llm_api_base,
        api_key=args.llm_api_key,
        model=args.llm_model,
        messages=messages,
        timeout_seconds=120,
        use_json_response_format=not args.no_json,
    )
    print("RAW_RESPONSE_START")
    print(content)
    print("RAW_RESPONSE_END")

    try:
        decoded = parse_llm_json_response(content)
        print("PARSED_JSON_START")
        print(decoded)
        print("PARSED_JSON_END")
    except Exception as exc:  # noqa: BLE001
        print(f"PARSE_FAILED: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
