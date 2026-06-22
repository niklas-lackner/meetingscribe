from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import SimpleNamespace

from record_and_transcribe_vClean import regenerate_reports_from_raw_json

DEFAULT_API_BASE = os.environ.get("DIZ_AI_BASE", "https://chat.ai.diz.uk-erlangen.de/api")
DEFAULT_MODEL = os.environ.get("DIZ_AI_MODEL", "gemma4:31b")


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


def build_args(raw_json_path: Path, api_base: str, model: str, api_key: str) -> SimpleNamespace:
    return SimpleNamespace(
        raw_json=raw_json_path,
        meeting_folder=raw_json_path.parent,
        txt=Path("meeting.txt"),
        log=Path("meeting_raw_segments.jsonl"),
        speaker_log=Path("meeting_speakers.jsonl"),
        dialog=Path("meeting_dialog.md"),
        full_report=Path("meeting_report_full.md"),
        long_report=Path("meeting_report_long.md"),
        short_report=Path("meeting_protocol_short.md"),
        llm_finalize=True,
        llm_provider="openai-compatible",
        llm_api_base=api_base,
        llm_model=model,
        llm_api_key=api_key,
        llm_timeout_seconds=10000,
        llm_max_input_chars=220000,
        llm_chunk_input_chars=80000,
        llm_system_prompt=None,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply hosted LLM postprocessing retroactively to existing vClean meeting_raw.json files."
    )
    parser.add_argument("--meetings-root", type=Path, default=Path("meetings"), help="Root folder with meeting sessions")
    parser.add_argument("--pattern", default="meeting_raw.json", help="Filename to search for under the meetings root")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on how many meetings to process")
    parser.add_argument("--dry-run", action="store_true", help="List matches without processing them")
    parser.add_argument("--llm-api-base", default=DEFAULT_API_BASE, help="OpenAI-compatible API base URL")
    parser.add_argument("--llm-model", default=DEFAULT_MODEL, help="Model name for retroactive finalization")
    parser.add_argument("--llm-api-key", default=DEFAULT_API_KEY, help="API key for the hosted endpoint")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    meetings_root = args.meetings_root
    if not meetings_root.exists():
        raise RuntimeError(f"Meetings root not found: {meetings_root}")

    raw_files = sorted(meetings_root.rglob(args.pattern))
    if args.limit > 0:
        raw_files = raw_files[: args.limit]

    if args.dry_run:
        for raw_file in raw_files:
            print(raw_file)
        return 0

    if not raw_files:
        print(f"No matching raw files found under {meetings_root}")
        return 0

    success = 0
    failures = 0
    for raw_file in raw_files:
        print(f"Processing {raw_file}")
        try:
            regen_args = build_args(raw_file, args.llm_api_base, args.llm_model, args.llm_api_key)
            regenerate_reports_from_raw_json(regen_args)
            success += 1
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"Failed for {raw_file}: {exc}")

    print(f"Finished. Success: {success}, Failures: {failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
