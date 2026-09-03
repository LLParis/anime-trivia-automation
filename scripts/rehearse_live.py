"""Paint a real saved card inside the calibrated capture region: red, then green.

Run the app with ``--rehearse`` first (it types the answer into the real
Discord composer at green but withholds Enter), then run this script. The
window is borderless and topmost and covers only the capture region, so the
Discord composer below it stays visible and focusable; the app sees the card
exactly as it would during a quiz.

    .venv\\Scripts\\python.exe scripts\\rehearse_live.py --config config.json \\
        --card "C:\\path\\to\\Screenshot 2026-08-31 120050.png" --red 7 --green 25
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import tkinter as tk
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageTk

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anime_trivia_automation.config import load_config

DISCORD_CHAT_BACKGROUND = (49, 51, 56)


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            exit_code = wintypes.DWORD()
            return bool(
                kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
                and exit_code.value == 259  # STILL_ACTIVE
            )
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def require_rehearsal_worker(
    status_path: Path,
    *,
    expected_run_id: str | None = None,
    max_age_seconds: float = 5.0,
) -> str:
    """Refuse to paint unless a fresh, live rehearsal worker owns the status."""

    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "No readable rehearsal worker status; start anime-trivia --rehearse first"
        ) from exc
    if not isinstance(payload, dict) or payload.get("mode") != "REHEARSAL":
        raise RuntimeError("Refusing to paint cards: the active worker is not in REHEARSAL mode")
    run_id = str(payload.get("run_id") or "")
    if not run_id or (expected_run_id is not None and run_id != expected_run_id):
        raise RuntimeError("Refusing to paint cards: the rehearsal worker changed")
    phase = str(payload.get("phase") or "")
    if phase in {"", "STARTING", "LOADING", "STOPPING", "STOPPED", "ERROR"}:
        raise RuntimeError(f"Refusing to paint cards: rehearsal worker phase is {phase or 'missing'}")
    try:
        updated_at = datetime.fromisoformat(str(payload["updated_at"]))
        if updated_at.tzinfo is None:
            raise ValueError("updated_at has no timezone")
        age = (datetime.now(UTC) - updated_at.astimezone(UTC)).total_seconds()
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Refusing to paint cards: rehearsal status timestamp is invalid") from exc
    if age < -2.0 or age > max(1.0, float(max_age_seconds)):
        raise RuntimeError(f"Refusing to paint cards: rehearsal status is stale ({age:.1f}s)")
    try:
        pid = int(payload.get("pid"))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Refusing to paint cards: rehearsal worker PID is invalid") from exc
    if not _pid_is_running(pid):
        raise RuntimeError("Refusing to paint cards: rehearsal worker is not running")
    return run_id


# Sampled from a real Anime Soul card (Screenshot 2026-08-31 070005.png): the
# embed sits at x=47 behind a 6 px accent strip, on a #1A1827 body.
CARD_SIZE = (967, 288)
CARD_OUTER = (13, 10, 31)
CARD_BODY = (26, 24, 39)
ACCENT_X = 47
ACCENT_WIDTH = 6
ACCENT_RED = (237, 66, 69)
ACCENT_GREEN = (59, 165, 93)
CARD_FONT = "C:/Windows/Fonts/segoeui.ttf"


def render_card(
    clue: str,
    *,
    question_label: str = "1/10",
    answer_type: str = "anime_title",
    ready: bool = False,
) -> Image.Image:
    """Draw a card carrying an arbitrary clue, for rehearsing a specific round.

    Painting a saved screenshot cannot rehearse a chosen text clue: the app
    reads the clue out of the pixels, so the screenshot's own clue wins and the
    injected one is never consulted. This draws the real card layout with the
    wanted text instead, including every marker the OCR keys on -- the header,
    the "Answer with the ..." line, the question label, the status words, and
    the coloured accent strip that is the authoritative red/green signal.

    Deliberately free of emoji glyphs: the markers are plain text, and a font
    fallback box would only risk confusing the OCR.
    """

    from PIL import ImageDraw, ImageFont

    width, height = CARD_SIZE
    image = Image.new("RGB", (width, height), CARD_OUTER)
    draw = ImageDraw.Draw(image)
    draw.rectangle([ACCENT_X + ACCENT_WIDTH, 8, width - 8, height - 8], fill=CARD_BODY)
    draw.rectangle(
        [ACCENT_X, 8, ACCENT_X + ACCENT_WIDTH - 1, height - 8],
        fill=ACCENT_GREEN if ready else ACCENT_RED,
    )

    def font(size: int):
        try:
            return ImageFont.truetype(CARD_FONT, size)
        except OSError:
            return ImageFont.load_default()

    left = ACCENT_X + ACCENT_WIDTH + 22
    status = "Answer Now!" if ready else "Get Ready..."
    draw.text((left, 18), f"Anime Guessing Game - {status}", font=font(21), fill=(220, 221, 222))

    # Wrap the clue by measured width rather than a character count, so a long
    # quotation lays out the way a real card does.
    body = font(29)
    words, lines, current = clue.split(), [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textlength(candidate, font=body) > width - left - 30 and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    for index, line in enumerate(lines[:4]):
        draw.text((left, 62 + index * 38), line, font=body, fill=(255, 255, 255))

    kind = "anime title" if answer_type == "anime_title" else "character name"
    draw.text(
        (left, 214),
        f"Answer with the {kind} - first correct guess in chat wins!",
        font=font(20),
        fill=(220, 221, 222),
    )
    footer = (
        "answers OPEN you have 60s"
        if ready
        else "Reading time - answers open in 5s"
    )
    draw.text(
        (left, 250),
        f"Question {question_label} - {footer}",
        font=font(18),
        fill=(148, 155, 164),
    )
    return image


def recolor_accent_to_green(card: Image.Image, accent_width: int = 60) -> Image.Image:
    """Turn the card's red left accent (and the red status dot) green."""

    array = np.array(card.convert("RGB")).astype(np.int32)
    red, green, blue = array[..., 0], array[..., 1], array[..., 2]
    mask = (red >= 140) & (red >= green * 1.25) & (red >= blue * 1.25)
    out = array.copy()
    out[..., 0][mask] = np.minimum(green[mask] + 20, 80)
    out[..., 1][mask] = red[mask]
    return Image.fromarray(out.astype(np.uint8))


def compose(region_size: tuple[int, int], card: Image.Image, offset: tuple[int, int]) -> Image.Image:
    canvas = Image.new("RGB", region_size, DISCORD_CHAT_BACKGROUND)
    canvas.paste(card.convert("RGB"), offset)
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--card", required=True, help="saved screenshot of a RED card crop")
    parser.add_argument("--red", type=float, default=7.0, help="seconds to show the red card")
    parser.add_argument("--green", type=float, default=25.0, help="seconds to show the green card")
    parser.add_argument("--x", type=int, default=24, help="card x offset inside the region")
    parser.add_argument("--y", type=int, default=None, help="card y offset inside the region")
    args = parser.parse_args()

    config = load_config(args.config)
    rehearsal_run_id = require_rehearsal_worker(
        config.runtime.status_path,
        max_age_seconds=config.status.stale_after_seconds,
    )
    left, top, right, bottom = config.capture.region
    width, height = right - left, bottom - top
    card = Image.open(args.card)
    y = args.y if args.y is not None else max(0, height - card.height - 240)
    red_frame = compose((width, height), card, (args.x, y))
    green_frame = compose((width, height), recolor_accent_to_green(card), (args.x, y))

    # The capture region is in physical pixels; without DPI awareness Tk would
    # scale the window and paint the card outside the region.
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        pass
    root = tk.Tk()
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    root.geometry(f"{width}x{height}+{left}+{top}")
    label = tk.Label(root, borderwidth=0, highlightthickness=0)
    label.pack()
    red_photo = ImageTk.PhotoImage(red_frame)
    green_photo = ImageTk.PhotoImage(green_frame)
    label.configure(image=red_photo)
    started = time.monotonic()
    print(f"RED card shown at {time.strftime('%H:%M:%S')} inside region {config.capture.region}", flush=True)

    def go_green() -> None:
        try:
            require_rehearsal_worker(
                config.runtime.status_path,
                expected_run_id=rehearsal_run_id,
                max_age_seconds=config.status.stale_after_seconds,
            )
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr, flush=True)
            root.destroy()
            return
        label.configure(image=green_photo)
        print(f"GREEN card shown at {time.strftime('%H:%M:%S')} (+{time.monotonic()-started:.1f}s)", flush=True)
        root.after(int(args.green * 1000), root.destroy)

    root.after(int(args.red * 1000), go_green)
    root.mainloop()
    print(f"card window closed at {time.strftime('%H:%M:%S')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
