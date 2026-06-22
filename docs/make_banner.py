"""Generate docs/banner.png — the hero banner shown at the top of the README.

Regenerate with:  python docs/make_banner.py
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

W, H = 1280, 360
OUT = Path(__file__).parent / "banner.png"

BG_TOP = (16, 21, 36)
BG_BOT = (7, 9, 15)
TEXT = (244, 247, 255)
DIM = (150, 162, 190)
ACCENT = (96, 132, 255)
GREEN = (90, 210, 150)
PANEL = (16, 22, 40)
EDGE = (44, 56, 92)


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    for path in (f"C:/Windows/Fonts/{name}", f"/usr/share/fonts/truetype/dejavu/{name}"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


F_TITLE = font("seguisb.ttf", 78)
F_TAG = font("segoeui.ttf", 30)
F_SMALL = font("seguisb.ttf", 19)
F_TINY = font("segoeui.ttf", 18)


def gradient() -> Image.Image:
    img = Image.new("RGB", (W, H))
    px = img.load()
    for y in range(H):
        f = y / H
        col = tuple(int(BG_TOP[i] + (BG_BOT[i] - BG_TOP[i]) * f) for i in range(3))
        for x in range(W):
            px[x, y] = col
    # soft accent glow bottom-left
    glow = Image.new("RGB", (W, H), (0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.ellipse((-160, 120, 560, 640), fill=(26, 40, 96))
    gd.ellipse((820, -220, 1500, 360), fill=(40, 26, 70))
    glow = glow.filter(ImageFilter.GaussianBlur(130))
    return ImageChops.screen(img, glow)


def rrect(d, box, r, **kw):
    d.rounded_rectangle(box, radius=r, **kw)


def build():
    img = gradient()
    d = ImageDraw.Draw(img)

    # headphone-ish mark (drawn, since emoji fonts don't render in PIL)
    mx, my = 70, 150
    d.arc((mx, my, mx + 70, my + 70), 180, 360, fill=ACCENT, width=8)
    d.rounded_rectangle((mx - 2, my + 32, mx + 16, my + 74), 7, fill=ACCENT)
    d.rounded_rectangle((mx + 54, my + 32, mx + 72, my + 74), 7, fill=ACCENT)

    # title + tagline
    tx = 170
    d.text((tx, 96), "MeetingScribe", font=F_TITLE, fill=TEXT)
    tw = F_TITLE.getlength("MeetingScribe")
    d.line((tx + 4, 188, tx + tw, 188), fill=ACCENT, width=5)
    d.text((tx + 4, 200), "Never miss a word — even in the long ones.",
           font=F_TAG, fill=DIM)
    d.text((tx + 4, 244), "100% local  ·  faster-whisper + CUDA  ·  live transcript + AI report",
           font=F_TINY, fill=(110, 122, 150))

    # mini transcript card (right)
    cx0, cy0, cx1, cy1 = 905, 70, 1230, 250
    rrect(d, (cx0, cy0, cx1, cy1), 16, fill=PANEL, outline=EDGE, width=2)
    d.ellipse((cx0 + 18, cy0 + 20, cx0 + 30, cy0 + 32), fill=GREEN)
    d.text((cx0 + 40, cy0 + 16), "LIVE TRANSCRIPTION", font=F_SMALL, fill=DIM)
    d.line((cx0 + 18, cy0 + 48, cx1 - 18, cy0 + 48), fill=EDGE, width=1)
    lines = ["[1] Welcome everyone,", "    thanks for joining.", "[2] Let's start with the", "    Q3 roadmap|"]
    y = cy0 + 60
    for ln in lines:
        d.text((cx0 + 20, y), ln, font=F_TINY, fill=TEXT)
        y += 26

    # waveform strip along the bottom
    by = H - 34
    n = 150
    for k in range(n):
        bx = 24 + k * ((W - 48) / n)
        amp = abs(math.sin(k * 0.4)) * (16 if k % 4 else 9) + 3
        col = GREEN if (cx0 - 24) < bx < (cx1 - 24) and False else ACCENT
        d.line((bx, by - amp, bx, by + amp), fill=col, width=2)

    img.save(OUT)
    print(f"wrote {OUT}  ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    build()
