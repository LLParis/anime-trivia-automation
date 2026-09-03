"""Harvest past Anime Soul rounds from the Discord channel scrollback.

A history hit resolves in about 0.5 s; the live solver takes about 6 s and only
returns after answers have already opened. So every clue that lands in
``data/trivia_history.seed.json`` converts a future repeat into a race won on
write speed alone -- and an emoji rebus the model cannot solve becomes a plain
lookup. All ten clues on 2026-09-03 were novel; the channel holds months more.

This walks the message list through the SAME accessibility tree the live app
reads, reusing ``DiscordQuestionLocator``'s card and reveal parsers, so a clue
harvested here is byte-identical to one captured live (emoji included).

READ ONLY by construction: it scrolls the message list via UI Automation and
reads accessible names. It never touches the composer, never types, and never
clicks, so it cannot send anything to the channel. It restores the channel to
the newest message when it finishes.

    .venv\\Scripts\\python.exe scripts\\harvest_channel_history.py --dry-run

Keep Discord shown on the trivia channel and don't scroll it while this runs.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anime_trivia_automation.config import load_config  # noqa: E402
from anime_trivia_automation.discord import DiscordQuestionLocator  # noqa: E402
from anime_trivia_automation.typing import ForegroundWindowGuard  # noqa: E402
from anime_trivia_automation.utils import normalize_question  # noqa: E402

# Every Discord message row's accessible name carries its own timestamp. That
# is what separates a real message from a sidebar entry, and it is how the walk
# knows scrolling is still making progress through ordinary conversation.
_MESSAGE_STAMP = re.compile(r"\d{1,2}:\d{2}\s*(?:AM|PM)", re.IGNORECASE)


@dataclass
class Row:
    """One trivia row: either a bot card or a bot reveal."""

    name: str
    kind: str  # "card" | "reveal"
    clue: str = ""
    answer_type: str = ""
    question_label: str = ""
    answer: str = ""


@dataclass
class Scan:
    """One pass over the materialized message list."""

    message_names: list[str]
    rows: list[Row]
    total_list_items: int
    top_element: Any = None
    bottom_element: Any = None


def scan(locator: DiscordQuestionLocator, target: Any) -> Scan:
    """Read the virtualized list once: messages, trivia rows, and the extremes.

    Offscreen rows are kept: Discord materializes a band above and below the
    viewport and those rows carry identical accessible names. Trivia rows are
    ordered by vertical position so a reveal that follows a card stays after it.
    """

    automation, uia_types = locator._client()
    root = automation.ElementFromHandle(target.hwnd)
    condition = automation.CreatePropertyCondition(
        uia_types.UIA_ControlTypePropertyId,
        uia_types.UIA_ListItemControlTypeId,
    )
    elements = root.FindAll(uia_types.TreeScope_Descendants, condition)
    messages: list[str] = []
    placed: list[tuple[int, int, Row]] = []
    top_element = bottom_element = None
    top_at = bottom_at = None
    for index in range(elements.Length):
        element = elements.GetElement(index)
        try:
            if int(element.CurrentProcessId) != target.process_id:
                continue
            name = str(element.CurrentName or "")
            top = int(element.CurrentBoundingRectangle.top)
        except Exception:
            continue
        if not name:
            continue
        is_message = bool(_MESSAGE_STAMP.search(name))
        if is_message:
            messages.append(name)
            if top_at is None or top < top_at:
                top_at, top_element = top, element
            if bottom_at is None or top > bottom_at:
                bottom_at, bottom_element = top, element
        if "Anime Guessing Game" in name:
            try:
                full = (
                    locator._message_group_name(element, automation, uia_types) or name
                )
            except Exception:
                full = name
            card = locator.parse_card_name(full)
            if card is None:
                continue
            clue, answer_type, label = card
            placed.append(
                (top, index, Row(full, "card", clue, answer_type, label))
            )
        elif locator.is_official_reveal_name(name):
            answer = locator.parse_reveal_answer(name)
            if answer:
                placed.append((top, index, Row(name, "reveal", answer=answer)))
    placed.sort(key=lambda item: (item[0], item[1]))
    return Scan(
        message_names=messages,
        rows=[row for _, _, row in placed],
        total_list_items=int(elements.Length),
        top_element=top_element,
        bottom_element=bottom_element,
    )


def scroll_to(locator: DiscordQuestionLocator, element: Any) -> bool:
    """Bring one already-located row into view. No mouse, no keys, no clicks."""

    if element is None:
        return False
    _automation, uia_types = locator._client()
    try:
        element.GetCurrentPattern(uia_types.UIA_ScrollItemPatternId).QueryInterface(
            uia_types.IUIAutomationScrollItemPattern
        ).ScrollIntoView()
        return True
    except Exception:
        return False


def pair_rounds(rows: list[Row]) -> list[dict]:
    """Pair each card with the first reveal posted after it."""

    paired: list[dict] = []
    for position, row in enumerate(rows):
        if row.kind != "card":
            continue
        for later in rows[position + 1:]:
            if later.kind == "card":
                break  # a new card started; this one's reveal is not in view
            paired.append(
                {
                    "clue": row.clue,
                    "type": row.answer_type,
                    "answer": later.answer,
                    "question_label": row.question_label,
                    "reveal_name": later.name,
                }
            )
            break
    return paired


def _live_app_running() -> bool:
    """True when the live automation is up, so harvesting must not scroll."""

    try:
        output = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq anime-trivia.exe", "/NH"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return "anime-trivia.exe" in output


def walk(
    locator: DiscordQuestionLocator,
    target: Any,
    *,
    oldest: bool,
    steps: int,
    settle: float,
    known: set[tuple[str, str]] | None = None,
    harvested: dict[tuple[str, str], dict] | None = None,
    want: int = 0,
    label: str = "",
) -> int:
    """Scroll one direction, collecting clue/answer pairs on the way.

    Progress is judged by NEW MESSAGE rows, not by trivia rows: most of the
    channel is ordinary conversation, and a card-only test would call a
    perfectly good scroll barren and stop early.
    """

    seen_messages: set[str] = set()
    barren = 0
    for step in range(1, steps + 1):
        current = scan(locator, target)
        fresh = [n for n in current.message_names if n not in seen_messages]
        seen_messages.update(current.message_names)
        added = 0
        pairs = 0
        cards = sum(1 for r in current.rows if r.kind == "card")
        if known is not None and harvested is not None:
            for pair in pair_rounds(current.rows):
                pairs += 1
                key = (normalize_question(pair["clue"]), pair["type"])
                if key in known or key in harvested:
                    continue
                harvested[key] = pair
                added += 1
        if label:
            print(
                f"[{label} {step:3}/{steps}] messages={len(current.message_names):3} "
                f"new={len(fresh):3} cards={cards:2} reveals={len(current.rows) - cards:2} "
                f"paired={pairs:2} new_clues={added:2} total_new={len(harvested or {})}",
                flush=True,
            )
        if want and harvested is not None and len(harvested) >= want:
            print("reached the requested round count", flush=True)
            return len(harvested)
        barren = barren + 1 if not fresh else 0
        if barren >= 4:
            print(f"{label or 'walk'}: no new messages after four scrolls", flush=True)
            break
        element = current.top_element if oldest else current.bottom_element
        if not scroll_to(locator, element):
            print(f"{label or 'walk'}: could not scroll further", flush=True)
            break
        time.sleep(settle)
    return len(harvested or {})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--rounds", type=int, default=400, help="stop after this many new pairs")
    parser.add_argument("--max-scrolls", type=int, default=600, help="hard ceiling; always terminates")
    parser.add_argument("--settle", type=float, default=0.45, help="seconds to let Discord render")
    parser.add_argument("--dry-run", action="store_true", help="report without writing history")
    parser.add_argument(
        "--include-known",
        action="store_true",
        help="also list pairs already in history, to prove the pairing works",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    target = ForegroundWindowGuard(config.typing).expected_window()
    if target is None:
        print("Discord was not found. Open it on the trivia channel.", file=sys.stderr)
        return 2
    locator = DiscordQuestionLocator()

    # Harvesting scrolls the channel into history, which would blind the live
    # app's newest-card read mid-quiz.
    if _live_app_running():
        print(
            "anime-trivia is running. Stop it (F12) first: scrolling the channel "
            "into history would blind the live card read.",
            file=sys.stderr,
        )
        return 2
    if ctypes.windll.user32.IsIconic(target.hwnd):
        print(
            "Discord is minimized, so Electron is not rendering the message list. "
            "Bring it up on the trivia channel and run this again.",
            file=sys.stderr,
        )
        return 2

    probe = scan(locator, target)
    if not probe.message_names:
        print(
            f"Discord's message list is not readable ({probe.total_list_items} list "
            "rows, none of them messages). Show the Discord window on the trivia "
            "channel and run this again.",
            file=sys.stderr,
        )
        return 2
    print(
        f"message list readable: {len(probe.message_names)} messages, "
        f"{len(probe.rows)} trivia rows in view",
        flush=True,
    )

    history_path = config.runtime.history_path
    history = json.loads(history_path.read_text(encoding="utf-8"))
    known = (
        set()
        if args.include_known
        else {
            (normalize_question(p["clue"]), p.get("type") or "")
            for p in history["pairs"]
        }
    )
    before = len(history["pairs"])
    print(f"history starts with {before} clues", flush=True)

    harvested: dict[tuple[str, str], dict] = {}
    walk(
        locator,
        target,
        oldest=True,
        steps=args.max_scrolls,
        settle=args.settle,
        known=known,
        harvested=harvested,
        want=args.rounds,
        label="up",
    )
    # Leave the channel where the operator had it: the live app reads the newest
    # visible card, so a channel parked in history would blind it.
    walk(
        locator,
        target,
        oldest=False,
        steps=args.max_scrolls,
        settle=args.settle,
        label="back",
    )

    raw_path = Path("runtime") / "harvest_raw.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(
        json.dumps(list(harvested.values()), indent=1, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n{len(harvested)} new clue/answer pairs -> {raw_path}", flush=True)
    for pair in list(harvested.values())[:20]:
        print(f"  {pair['type']:12} {pair['clue'][:52]!r:56} -> {pair['answer']}")
    if len(harvested) > 20:
        print(f"  ... and {len(harvested) - 20} more")

    if args.include_known:
        # This mode zeroes the known-set to demonstrate pairing, so everything
        # it reports is already in history. Writing would duplicate all of it.
        print(
            "\ninspection run (--include-known): history not written",
            flush=True,
        )
        return 0
    if args.dry_run:
        print("\ndry run: history not written", flush=True)
        return 0
    if not harvested:
        print("nothing new to add", flush=True)
        return 0

    # History entries keep their three-field shape; provenance stays in the raw
    # dump so the file the app loads is exactly the shape it expects.
    history["pairs"].extend(
        {"clue": p["clue"], "type": p["type"], "answer": p["answer"]}
        for p in harvested.values()
    )
    history_path.write_text(
        json.dumps(history, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    print(f"history {before} -> {len(history['pairs'])} clues", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
