# Demo media

Drop the demo recording here as **`demo.gif`** (or `demo.mp4`) and it will show
up in the main README.

## How to record the demo

1. Open a YouTube video in your browser and pause it.
2. Start a screen recorder (Xbox Game Bar `Win+G`, OBS, or ScreenToGif for a GIF).
   Capture both the browser video **and** the Live Transcription window.
3. Run:
   ```powershell
   python demo.py --device-index <N>
   ```
4. Hit play on the video. Let it run ~20–40s so the transcript visibly fills in.
5. Stop the recording, export as `docs/demo.gif`, commit it.

Tip: for a GIF keep it short (<15 MB). For longer clips use `demo.mp4` and link it
in the README instead of embedding.
