"""Practice every saved card through the real pipeline, scoring each round.

Run the app with ``--rehearse`` first. This paints each red card screenshot
inside the capture region, flips it green, waits for the app to type its
answer into the real Discord composer (Enter withheld), and records what was
typed and how long after green. Between cards a blank frame is shown so the
next card starts a fresh round.

    .venv\\Scripts\\python.exe scripts\\rehearse_batch.py --config config.json \\
        --manifest runtime\\card_manifest.json --screens "C:\\...\\Screenshots"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import tkinter as tk
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageTk

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from anime_trivia_automation.config import load_config  # noqa: E402
from anime_trivia_automation.utils import normalize_question  # noqa: E402
from rehearse_live import compose, recolor_accent_to_green  # noqa: E402

REHEARSAL_LINE = re.compile(r"^(\d\d:\d\d:\d\d\.\d\d\d) WARNING .*REHEARSAL: '(.+?)' is typed and verified")


def newest_log(log_dir: Path) -> Path:
    return max(log_dir.glob("anime-trivia-*.log"), key=lambda p: p.stat().st_mtime)


def typed_answers(log_path: Path) -> list[tuple[str, str]]:
    out = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = REHEARSAL_LINE.match(line)
        if m:
            out.append((m.group(1), m.group(2)))
    return out


def order_without_label_repeats(cards: list[dict]) -> list[dict]:
    by_label: dict[str, list[dict]] = defaultdict(list)
    for card in cards:
        by_label[card.get("question") or "?"].append(card)
    ordered: list[dict] = []
    while any(by_label.values()):
        for label in list(by_label):
            if by_label[label]:
                ordered.append(by_label[label].pop(0))
    return ordered


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--manifest", default="runtime/card_manifest.json")
    parser.add_argument("--screens", required=True)
    parser.add_argument("--kinds", default="text", help="comma list: text,visual")
    parser.add_argument("--red", type=float, default=5.0)
    parser.add_argument("--green-max", type=float, default=12.0)
    parser.add_argument("--blank", type=float, default=2.0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    left, top, right, bottom = config.capture.region
    width, height = right - left, bottom - top
    kinds = set(args.kinds.split(","))
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    cards = [
        c for c in manifest
        if c.get("card") and c.get("readiness") == "locked" and c.get("answer") and c.get("kind") in kinds
    ]
    cards = order_without_label_repeats(cards)
    if args.limit:
        cards = cards[: args.limit]
    log_path = newest_log(config.runtime.log_dir or Path("runtime/logs"))
    print(f"{len(cards)} cards | app log {log_path.name}", flush=True)

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
    blank = ImageTk.PhotoImage(Image.new("RGB", (width, height), (49, 51, 56)))
    label.configure(image=blank)
    root.update()

    results = []
    for index, card in enumerate(cards, 1):
        image = Image.open(Path(args.screens) / card["file"])
        y = max(0, height - image.height - 240)
        red = ImageTk.PhotoImage(compose((width, height), image, (24, y)))
        green = ImageTk.PhotoImage(compose((width, height), recolor_accent_to_green(image), (24, y)))
        seen_before = len(typed_answers(log_path))
        label.configure(image=red)
        root.update()
        time.sleep(args.red)
        label.configure(image=green)
        root.update()
        green_at = time.perf_counter()
        typed = None
        latency = None
        deadline = green_at + args.green_max
        while time.perf_counter() < deadline:
            root.update()
            entries = typed_answers(log_path)
            if len(entries) > seen_before:
                typed = entries[-1][1]
                latency = time.perf_counter() - green_at
                break
            time.sleep(0.05)
        expected = card["answer"]
        ok = typed is not None and normalize_question(typed) == normalize_question(expected)
        results.append({"file": card["file"], "question": card.get("question"), "kind": card.get("kind"),
                        "expected": expected, "typed": typed, "latency_s": round(latency, 2) if latency else None, "ok": ok})
        mark = "OK " if ok else ("TYPO" if typed else "NONE")
        print(f"[{index:2}/{len(cards)}] {mark} q={card.get('question'):5} {expected!r:34} typed={typed!r} "
              f"{'%.2fs' % latency if latency else ''} | {card.get('hint','')[:50]}", flush=True)
        label.configure(image=blank)
        root.update()
        time.sleep(args.blank)
    root.destroy()
    ok = sum(1 for r in results if r["ok"])
    typed_n = sum(1 for r in results if r["typed"])
    lat = sorted(r["latency_s"] for r in results if r["latency_s"] is not None)
    median = lat[len(lat) // 2] if lat else None
    print(f"\nRESULT: {ok}/{len(results)} typed the known answer | typed anything {typed_n} | median green->typed {median} s", flush=True)
    out = Path("runtime") / f"rehearsal-batch-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"report: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
