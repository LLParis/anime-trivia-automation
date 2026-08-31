from __future__ import annotations

import argparse
import ctypes
import json
import os
import tempfile
import tkinter as tk
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def enable_physical_pixels() -> None:
    if os.name == "nt":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except (AttributeError, OSError):
            ctypes.windll.user32.SetProcessDPIAware()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select the physical-pixel feed band and update config.json."
    )
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config.json")
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    enable_physical_pixels()
    root = tk.Tk()
    root.attributes("-fullscreen", True)
    root.attributes("-topmost", True)
    root.attributes("-alpha", 0.22)
    root.configure(bg="black")
    canvas = tk.Canvas(root, bg="black", highlightthickness=0, cursor="crosshair")
    canvas.pack(fill="both", expand=True)
    canvas.create_text(
        root.winfo_screenwidth() // 2,
        32,
        text="Drag a stable feed band large enough for the tallest Anime Soul card. Esc cancels.",
        fill="white",
        font=("Segoe UI", 18, "bold"),
    )
    start: tuple[int, int] | None = None
    rectangle: int | None = None

    def mouse_down(event: tk.Event) -> None:
        nonlocal start, rectangle
        start = (event.x_root, event.y_root)
        rectangle = canvas.create_rectangle(
            event.x, event.y, event.x, event.y, outline="#ff334f", width=4
        )

    def mouse_move(event: tk.Event) -> None:
        if start is not None and rectangle is not None:
            canvas.coords(rectangle, start[0], start[1], event.x_root, event.y_root)

    def mouse_up(event: tk.Event) -> None:
        if start is None:
            return
        left, right = sorted((start[0], event.x_root))
        top, bottom = sorted((start[1], event.y_root))
        if right - left < 20 or bottom - top < 20:
            return
        region = [left, top, right, bottom]
        root.clipboard_clear()
        root.clipboard_append(json.dumps(region))
        root.update()

        source_path = config_path
        if not source_path.exists():
            source_path = REPO_ROOT / "config.example.json"
        with source_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        config.setdefault("capture", {})["region"] = region
        config["capture"]["calibrated"] = True
        config_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                prefix=f".{config_path.name}.",
                suffix=".tmp",
                dir=config_path.parent,
                delete=False,
            ) as handle:
                temporary_path = handle.name
                json.dump(config, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, config_path)
        finally:
            if temporary_path and os.path.exists(temporary_path):
                os.unlink(temporary_path)

        print(json.dumps({"region": region}, indent=2))
        print(f"Updated {config_path} and copied the bare coordinate array.")
        root.destroy()

    canvas.bind("<ButtonPress-1>", mouse_down)
    canvas.bind("<B1-Motion>", mouse_move)
    canvas.bind("<ButtonRelease-1>", mouse_up)
    root.bind("<Escape>", lambda _event: root.destroy())
    root.mainloop()


if __name__ == "__main__":
    main()
