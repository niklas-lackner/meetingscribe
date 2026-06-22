"""One-click demo: live-transcribe whatever is playing on your system audio.

Typical demo flow:
  1. Open a YouTube video (or any video) in your browser, pause it.
  2. Run this script:  python demo.py --device-index <N>
       (find <N> once via:  python demo.py --list-devices)
  3. Hit play on the video. The TV-style window shows the transcript live.
  4. When the duration ends (default 180s) the LLM report is written to the
     meetings/ session folder next to the recording.

This is a thin wrapper around record_and_transcribe_vClean.py that turns on
sensible demo defaults (system-audio loopback + live TV window + short run).
Everything can be overridden on the command line.
"""

from __future__ import annotations

import sys

from record_and_transcribe_vClean import main


def _has_flag(argv: list[str], flag: str) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in argv)


def _inject_demo_defaults(argv: list[str]) -> list[str]:
    injected = list(argv)

    # Capture system audio (the played video) via WASAPI loopback.
    if not _has_flag(injected, "--mic") and not _has_flag(injected, "--output-loopback"):
        injected.append("--output-loopback")

    # Show the big live "TV screen" next to the video.
    if not _has_flag(injected, "--tv-window"):
        injected.append("--tv-window")
    if not _has_flag(injected, "--tv-title"):
        injected.extend(["--tv-title", "Live Transcription Demo"])

    # Keep the demo short by default.
    if not _has_flag(injected, "--duration"):
        injected.extend(["--duration", "180"])

    return injected


def main_demo() -> int:
    argv = sys.argv[1:]
    # Pass --list-devices straight through so users can find their device index.
    if not _has_flag(argv, "--list-devices"):
        argv = _inject_demo_defaults(argv)
    sys.argv = [sys.argv[0], *argv]
    return main()


if __name__ == "__main__":
    raise SystemExit(main_demo())
