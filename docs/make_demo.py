"""Generate docs/demo.gif — a stylized, animated illustration of MeetingScribe.

This is NOT a real screen capture; it's a hand-rendered mockup (same colors/fonts
as the live TV window) showing the idea: a video call on the left, the live
transcript streaming in on the right, and the AI report appearing at the end.

Replace it with a real screen recording whenever you like. Regenerate with:
    python docs/make_demo.py
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# --- Canvas / theme (matches LiveChunkWindow) ---------------------------------
W, H = 1100, 620
BG = (10, 13, 20)
PANEL = (15, 21, 38)
PANEL_EDGE = (32, 42, 70)
HEADER_FG = (154, 167, 194)
TEXT_FG = (244, 247, 255)
DIM_FG = (120, 132, 160)
ACCENT = (96, 132, 255)
GREEN = (90, 210, 150)
RED = (240, 70, 70)

OUT = Path(__file__).parent / "demo.gif"


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    for path in (f"C:/Windows/Fonts/{name}", f"/usr/share/fonts/truetype/dejavu/{name}"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


F_BIG = font("seguisb.ttf", 30)
F_H = font("seguisb.ttf", 20)
F_TXT = font("segoeui.ttf", 23)
F_SMALL = font("segoeui.ttf", 16)
F_BADGE = font("seguisb.ttf", 15)

# --- Layout -------------------------------------------------------------------
M = 22
VID_X0, VID_Y0, VID_X1, VID_Y1 = M, 52, 600, 340         # video call area
REP_X0, REP_Y0, REP_X1, REP_Y1 = M, 360, 600, H - M      # AI report area
TR_X0, TR_Y0, TR_X1, TR_Y1 = 622, 52, W - M, H - M       # transcript area

TILES = [
    ("Alex", (52, 70, 120)),
    ("Sam", (104, 60, 96)),
    ("Mia", (56, 96, 92)),
    ("You", (88, 80, 52)),
]

TRANSCRIPT = [
    "Welcome everyone, thanks for joining today's sync.",
    "Let's start with a quick status on the Q3 roadmap.",
    "The transcription module now runs fully on-device.",
    "Latency dropped to under two seconds per segment.",
    "Speaker separation holds up well even in noisy calls.",
    "Next step: ship the live report export by Friday.",
]

REPORT = [
    "On-device transcription, <2s latency",
    "Speaker separation stable in noisy calls",
    "Action: ship live report export by Fri",
]


def rrect(d: ImageDraw.ImageDraw, box, radius, fill=None, outline=None, width=1):
    d.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def wrap(text: str, fnt, max_w: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if fnt.getlength(trial) <= max_w:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def draw_video(d: ImageDraw.ImageDraw, t: float):
    rrect(d, (VID_X0, VID_Y0, VID_X1, VID_Y1), 14, fill=(7, 10, 16), outline=PANEL_EDGE, width=2)
    # 2x2 call tiles
    gx0, gy0, gx1, gy1 = VID_X0 + 14, VID_Y0 + 14, VID_X1 - 14, VID_Y1 - 44
    gap = 10
    tw = (gx1 - gx0 - gap) / 2
    th = (gy1 - gy0 - gap) / 2
    for i, (name, col) in enumerate(TILES):
        cx = gx0 + (i % 2) * (tw + gap)
        cy = gy0 + (i // 2) * (th + gap)
        rrect(d, (cx, cy, cx + tw, cy + th), 10, fill=col)
        # avatar head
        hr = th * 0.26
        hx, hy = cx + tw / 2, cy + th * 0.42
        d.ellipse((hx - hr, hy - hr, hx + hr, hy + hr), fill=(230, 232, 240))
        d.pieslice((hx - hr * 1.5, hy + hr * 0.3, hx + hr * 1.5, hy + hr * 2.6), 180, 360, fill=(230, 232, 240))
        d.rectangle((cx + 8, cy + th - 22, cx + 8 + F_SMALL.getlength(name) + 12, cy + th - 4), fill=(0, 0, 0, 120))
        d.text((cx + 14, cy + th - 21), name, font=F_SMALL, fill=TEXT_FG)
    # speaking highlight cycles through tiles
    spk = int(t * 3) % 4
    sx = gx0 + (spk % 2) * (tw + gap)
    sy = gy0 + (spk // 2) * (th + gap)
    rrect(d, (sx, sy, sx + tw, sy + th), 10, outline=GREEN, width=3)
    # waveform + progress bar
    by = VID_Y1 - 24
    n = 60
    for k in range(n):
        bx = VID_X0 + 16 + k * ((VID_X1 - VID_X0 - 32) / n)
        amp = abs(math.sin(k * 0.5 + t * 6)) * (10 if k % 3 else 6) + 2
        d.line((bx, by - amp, bx, by + amp), fill=ACCENT, width=2)
    # red "playing" progress
    frac = min(1.0, 0.1 + t * 0.12)
    d.line((VID_X0 + 16, VID_Y1 - 6, VID_X0 + 16 + (VID_X1 - VID_X0 - 32) * frac, VID_Y1 - 6), fill=RED, width=3)
    # REC dot
    d.ellipse((VID_X0 + 16, VID_Y0 + 14, VID_X0 + 30, VID_Y0 + 28), fill=RED)
    d.text((VID_X0 + 36, VID_Y0 + 13), "LIVE", font=F_BADGE, fill=TEXT_FG)


def draw_transcript(d: ImageDraw.ImageDraw, shown: list[str], partial: str, cursor: bool):
    rrect(d, (TR_X0, TR_Y0, TR_X1, TR_Y1), 14, fill=PANEL, outline=PANEL_EDGE, width=2)
    d.text((TR_X0 + 18, TR_Y0 + 14), "LIVE TRANSCRIPTION", font=F_H, fill=HEADER_FG)
    d.line((TR_X0 + 18, TR_Y0 + 44, TR_X1 - 18, TR_Y0 + 44), fill=PANEL_EDGE, width=1)
    y = TR_Y0 + 58
    max_w = TR_X1 - TR_X0 - 60
    entries = [f"[{i + 1}] {s}" for i, s in enumerate(shown)]
    if partial is not None:
        entries.append(f"[{len(shown) + 1}] {partial}" + ("|" if cursor else ""))
    for e in entries:
        for ln in wrap(e, F_TXT, max_w):
            d.text((TR_X0 + 22, y), ln, font=F_TXT, fill=TEXT_FG)
            y += 30
        y += 8
    if not shown and partial is None:
        d.text((TR_X0 + 22, y), "listening…" + ("|" if cursor else ""), font=F_TXT, fill=DIM_FG)


def draw_report(d: ImageDraw.ImageDraw, reveal: float):
    rrect(d, (REP_X0, REP_Y0, REP_X1, REP_Y1), 14, fill=PANEL, outline=PANEL_EDGE, width=2)
    if reveal <= 0:
        d.text((REP_X0 + 18, REP_Y0 + 16), "AI report generates after the call…",
               font=F_SMALL, fill=DIM_FG)
        return
    d.text((REP_X0 + 18, REP_Y0 + 14), "AI MEETING REPORT", font=F_H, fill=GREEN)
    d.line((REP_X0 + 18, REP_Y0 + 42, REP_X1 - 18, REP_Y0 + 42), fill=PANEL_EDGE, width=1)
    y = REP_Y0 + 54
    shown = REPORT[: max(0, int(reveal * len(REPORT) + 0.001))]
    for item in shown:
        d.ellipse((REP_X0 + 22, y + 9, REP_X0 + 30, y + 17), fill=ACCENT)
        for j, ln in enumerate(wrap(item, F_SMALL, REP_X1 - REP_X0 - 60)):
            d.text((REP_X0 + 40, y), ln, font=F_SMALL, fill=TEXT_FG)
            y += 22
        y += 6


def badge(d: ImageDraw.ImageDraw):
    txt = "DEMO · illustration"
    w = F_BADGE.getlength(txt) + 18
    rrect(d, (W - M - w, H - M - 26, W - M, H - M - 2), 8, fill=(40, 40, 52))
    d.text((W - M - w + 9, H - M - 25), txt, font=F_BADGE, fill=DIM_FG)


def frame(t: float, shown, partial, cursor, report_reveal) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((M, 8), "MeetingScribe", font=F_BIG, fill=TEXT_FG)
    tw = F_BIG.getlength("MeetingScribe")
    d.text((M + tw + 16, 18), "live meeting transcription, on your machine",
           font=F_SMALL, fill=DIM_FG)
    draw_video(d, t)
    draw_transcript(d, shown, partial, cursor)
    draw_report(d, report_reveal)
    badge(d)
    return img.convert("P", palette=Image.ADAPTIVE, colors=128)


def build():
    frames, durations = [], []
    t = 0.0

    def add(img, ms):
        nonlocal t
        frames.append(img)
        durations.append(ms)
        t += ms / 1000.0

    # intro: listening
    for k in range(6):
        add(frame(t, [], None, k % 2 == 0, 0), 180)
    # typing, word by word
    shown: list[str] = []
    for line in TRANSCRIPT:
        words = line.split()
        for w in range(1, len(words) + 1):
            partial = " ".join(words[:w])
            add(frame(t, shown, partial, True, 0), 90)
        shown.append(line)
        add(frame(t, shown, None, False, 0), 220)
    # hold full transcript
    for _ in range(3):
        add(frame(t, shown, None, False, 0), 260)
    # report reveal
    for r in (0.34, 0.67, 1.0):
        add(frame(t, shown, None, False, r), 360)
    for _ in range(8):
        add(frame(t, shown, None, False, 1.0), 320)

    frames[0].save(
        OUT, save_all=True, append_images=frames[1:], duration=durations,
        loop=0, optimize=True, disposal=2,
    )
    kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT}  ({len(frames)} frames, {kb:.0f} KB)")


if __name__ == "__main__":
    build()
