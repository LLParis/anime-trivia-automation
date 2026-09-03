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

from rehearse_live import (
    compose,
    recolor_accent_to_green,
    require_rehearsal_worker,
)

from anime_trivia_automation.config import load_config
from anime_trivia_automation.utils import normalize_question

REHEARSAL_LINE = re.compile(
    r"^(\d\d:\d\d:\d\d\.\d\d\d) WARNING .*REHEARSAL: (?:'(.+?)'|\"(.+?)\") is typed and verified"
)


def newest_log(log_dir: Path) -> Path:
    return max(log_dir.glob("anime-trivia-*.log"), key=lambda p: p.stat().st_mtime)


def typed_answers(log_path: Path) -> list[tuple[str, str]]:
    out = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = REHEARSAL_LINE.match(line)
        if m:
            out.append((m.group(1), m.group(2) or m.group(3) or ""))
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
    parser.add_argument(
        "--cases",
        default=None,
        help="JSON list of {clue, answer, type, kind} to rehearse instead of the manifest",
    )
    parser.add_argument("--red", type=float, default=5.0)
    parser.add_argument("--green-max", type=float, default=12.0)
    parser.add_argument("--blank", type=float, default=2.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--force-novel",
        action="store_true",
        help="skip the history lookup so the live solver is exercised on every card",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    rehearsal_run_id = require_rehearsal_worker(
        config.runtime.status_path,
        max_age_seconds=config.status.stale_after_seconds,
    )
    left, top, right, bottom = config.capture.region
    width, height = right - left, bottom - top
    kinds = set(args.kinds.split(","))
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    cards = [
        c for c in manifest
        if c.get("card") and c.get("readiness") == "locked" and c.get("answer") and c.get("kind") in kinds
    ]
    if args.cases:
        # Rehearse specific clues rather than the saved manifest. This is what
        # you want after a round fails live: replay that exact clue through the
        # real pipeline. Any manifest card of the right kind supplies the
        # painted pixels; the clue itself is injected, so the card art is only
        # there to drive red/green detection.
        cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
        # Draw art from the whole manifest: --kinds filters the manifest run,
        # but each case declares its own kind.
        # Rotate through distinct art per kind: two cases painted on the same
        # card produce the same round signature, and the app answers a round
        # once, so the second would look like a silent failure.
        art = {
            kind: [
                c
                for c in manifest
                if c.get("card")
                and c.get("readiness") == "locked"
                and c.get("kind") == kind
            ]
            for kind in ("text", "visual")
        }
        used: dict[str, int] = {"text": 0, "visual": 0}
        cards = []
        for case in cases:
            kind = case.get("kind") or "text"
            pool = art.get(kind) or []
            if not pool:
                raise SystemExit(f"no {kind} card art in the manifest to paint")
            source = pool[used[kind] % len(pool)]
            used[kind] += 1
            cards.append(
                {
                    "file": source["file"],
                    "kind": kind,
                    "answer": case["answer"],
                    "hint": case["clue"],
                    "type": case.get("type") or "anime_title",
                    # The label must be the one the app will OCR off the painted
                    # pixels, or the injected clue is rejected as belonging to a
                    # different round and the case never reaches the solver.
                    "question": source.get("question") or "1/10",
                }
            )
    else:
        cards = order_without_label_repeats(cards)
    if args.limit:
        cards = cards[: args.limit]
    log_path = newest_log(config.runtime.log_dir or Path("runtime/logs"))
    print(f"{len(cards)} cards | app log {log_path.name}", flush=True)
    # Emoji clues live in Discord's accessibility tree, which a painted card
    # lacks; recover each card's clue from the reviewed history by its answer.
    history = json.loads(config.runtime.history_path.read_text(encoding="utf-8"))["pairs"]
    clue_by_answer: dict[tuple[str, str], str] = {}
    for pair in history:
        key = (normalize_question(pair["answer"]), pair["type"])
        prose = sum(ch.isalpha() for ch in pair["clue"]) >= 3
        if key not in clue_by_answer or not prose:
            clue_by_answer[key] = pair["clue"]
    clue_file = config.runtime.status_path.parent / "rehearsal_clue.json"

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
        require_rehearsal_worker(
            config.runtime.status_path,
            expected_run_id=rehearsal_run_id,
            max_age_seconds=config.status.stale_after_seconds,
        )
        image = Image.open(Path(args.screens) / card["file"])
        y = max(0, height - image.height - 240)
        red = ImageTk.PhotoImage(compose((width, height), image, (24, y)))
        green = ImageTk.PhotoImage(compose((width, height), recolor_accent_to_green(image), (24, y)))
        seen_before = len(typed_answers(log_path))
        if args.cases:
            clue = card.get("hint")  # explicit case clue, whatever the kind
        else:
            clue = card.get("hint") if card.get("kind") == "text" else clue_by_answer.get(
                (normalize_question(card["answer"]), card.get("type") or "")
            )
        if clue:
            clue_file.write_text(
                json.dumps(
                    {
                        "question_label": card.get("question"),
                        "expected_answer_type": card.get("type"),
                        "clue": clue,
                        "force_novel": bool(args.force_novel),
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        elif clue_file.exists():
            clue_file.unlink()
        label.configure(image=red)
        root.update()
        time.sleep(args.red)
        require_rehearsal_worker(
            config.runtime.status_path,
            expected_run_id=rehearsal_run_id,
            max_age_seconds=config.status.stale_after_seconds,
        )
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
        if clue_file.exists():
            clue_file.unlink()
        time.sleep(args.blank)
    root.destroy()
    # Leave the box the way we found it: erase the last typed answer if it is
    # still there untouched.
    last = next((r["typed"] for r in reversed(results) if r["typed"]), None)
    if last:
        try:
            from pynput.keyboard import Controller, Key

            from anime_trivia_automation.discord import DiscordComposerLocator
            from anime_trivia_automation.typing import ForegroundWindowGuard

            guard = ForegroundWindowGuard(config.typing)
            target = guard.expected_window()
            composer = (
                DiscordComposerLocator(
                    config.typing.composer_name_prefix, config.typing.composer_class_fragment
                ).find(target.hwnd, target.process_id)
                if target
                else None
            )
            if composer is not None and composer.value() == last and guard.activate(target.hwnd):
                if not composer.focused():
                    composer.set_focus()
                    time.sleep(0.1)
                controller = Controller()
                with controller.pressed(Key.ctrl):
                    controller.press("a")
                    controller.release("a")
                controller.press(Key.backspace)
                controller.release(Key.backspace)
                deadline = time.monotonic() + max(
                    1.0, float(config.typing.composer_settle_timeout_seconds)
                )
                while composer.value() != "" and time.monotonic() < deadline:
                    time.sleep(0.01)
                if composer.value() == "":
                    print(
                        f"cleared and verified the last practice answer {last!r}",
                        flush=True,
                    )
                    owned = config.runtime.status_path.parent / "owned_draft.json"
                    if owned.exists():
                        owned.unlink()
                else:
                    print(
                        "could not verify final practice cleanup; ownership marker retained",
                        flush=True,
                    )
            else:
                print(
                    "final practice cleanup skipped: exact owned composer was unavailable",
                    flush=True,
                )
        except Exception as exc:
            print(f"could not clear the last answer: {type(exc).__name__}", flush=True)
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
