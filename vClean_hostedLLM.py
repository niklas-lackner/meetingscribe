from __future__ import annotations

import os
import sys
from pathlib import Path

from record_and_transcribe_vClean import main

DEFAULT_API_BASE = os.environ.get("DIZ_AI_BASE", "https://chat.ai.diz.uk-erlangen.de/api")
DEFAULT_MODEL = os.environ.get("DIZ_AI_MODEL", "gemma4:31b")

# Try to load API key from environment, then from apikey.txt
_api_key = os.environ.get("DIZ_AI_KEY", "")
if not _api_key:
    api_key_file = Path(__file__).parent / "apikey.txt"
    if api_key_file.exists():
        _api_key = api_key_file.read_text().strip()
DEFAULT_API_KEY = _api_key


def _has_cli_flag(argv: list[str], flag: str) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in argv)


def _inject_default_args(argv: list[str]) -> list[str]:
    injected = list(argv)

    if not _has_cli_flag(injected, "--llm-finalize") and not _has_cli_flag(injected, "--no-llm-finalize"):
        injected.append("--llm-finalize")
    if not _has_cli_flag(injected, "--llm-api-base"):
        injected.extend(["--llm-api-base", DEFAULT_API_BASE])
    if not _has_cli_flag(injected, "--llm-model"):
        injected.extend(["--llm-model", DEFAULT_MODEL])
    if DEFAULT_API_KEY and not _has_cli_flag(injected, "--llm-api-key"):
        injected.extend(["--llm-api-key", DEFAULT_API_KEY])

    return injected


def main_hosted() -> int:
    os.environ.setdefault("OPENAI_API_KEY", DEFAULT_API_KEY)
    sys.argv = [sys.argv[0], *_inject_default_args(sys.argv[1:])]
    return main()


if __name__ == "__main__":
    raise SystemExit(main_hosted())
