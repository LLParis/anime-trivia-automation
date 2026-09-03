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
import sys
import time
import tkinter as tk
from pathlib import Path

import numpy as np
from PIL import Image, ImageTk

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anime_trivia_automation.config import load_config  # noqa: E402

DISCORD_CHAT_BACKGROUND = (49, 51, 56)


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
        label.configure(image=green_photo)
        print(f"GREEN card shown at {time.strftime('%H:%M:%S')} (+{time.monotonic()-started:.1f}s)", flush=True)
        root.after(int(args.green * 1000), root.destroy)

    root.after(int(args.red * 1000), go_green)
    root.mainloop()
    print(f"card window closed at {time.strftime('%H:%M:%S')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
