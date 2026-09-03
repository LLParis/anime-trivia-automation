"""Harvest every past Anime Soul round from the Discord channel scrollback.

A history hit resolves in about 0.5 s; the live solver takes about 6 s and only
returns after answers have already opened. So every clue that lands in
``data/trivia_history.seed.json`` converts a future repeat into a race we win on
write speed alone -- and an emoji rebus we could never solve becomes a plain
lookup. On 2026-09-03 all ten clues were novel; the channel holds months more.

This walks the message list through the SAME accessibility tree the live app
reads, reusing ``DiscordQuestionLocator``'s card and reveal parsers so a clue
harvested here is byte-identical to one captured live (emoji included).

READ ONLY, by construction: it scrolls the message list via UI Automation and
reads names. It never touches the composer, never types, never posts, and never
clicks -- so it cannot send anything to the channel.

    .venv\\Scripts\\python.exe scripts\\harvest_channel_history.py --rounds 400

Leave Discord on the trivia channel and don't scroll it while this runs.
"""

from __future__ import annotations

import argparse
import ctypes
import json
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


@dataclass
class Row:
    """One message row: either a card or a bot reveal, in document order."""

    order: int
    name: str
    kind: str  # "card" | "reveal"
    clue: str = ""
    answer_type: str = ""
    question_label: str = ""
    answer: str = ""


def _rows(locator: DiscordQuestionLocator, hwnd: int, process_id: int) -> list[Row]:
    """Every card and reveal currently materialized in the virtualized list.

    Offscreen rows are kept: Discord renders a band above and below the
    viewport, and those rows carry exactly the same accessible names. Ordering
    is by vertical position so a reveal that follows a card stays after it.
    """

    automation, uia_types = locator._client()
    root = automation.ElementFromHandle(hwnd)
    condition = automation.CreatePropertyCondition(
        uia_types.UIA_ControlTypePropertyId,
        uia_types.UIA_ListItemControlTypeId,
    )
    elements = root.FindAll(uia_types.TreeScope_Descendants, condition)
    found: list[tuple[int, int, Row]] = []
    for index in range(elements.Length):
        element = elements.GetElement(index)
        try:
            if int(element.CurrentProcessId) != process_id:
                continue
            name = str(element.CurrentName or "")
        except Exception:
            continue
        if not name:
            continue
        try:
            top = int(element.CurrentBoundingRectangle.top)
        except Exception:
            top = index
        if "Anime Guessing Game" in name:
            try:
                full = locator._message_group_name(element, automation, uia_types) or name
            except Exception:
                full = name
            card = locator.parse_card_name(full)
            if card is None:
                continue
            clue, answer_type, label = card
            found.append(
                (top, index, Row(index, full, "card", clue, answer_type, label))
            )
            continue
        if locator.is_official_reveal_name(name):
            answer = locator.parse_reveal_answer(name)
            if answer:
                found.append((top, index, Row(index, name, "reveal", answer=answer)))
    found.sort(key=lambda item: (item[0], item[1]))
    return [row for _, _, row in found]


def _scroll_extreme(
    locator: DiscordQuestionLocator, hwnd: int, process_id: int, *, oldest: bool
) -> bool:
    """Scroll the top-most (oldest) or bottom-most (newest) row into view."""

    automation, uia_types = locator._client()
    root = automation.ElementFromHandle(hwnd)
    condition = automation.CreatePropertyCondition(
        uia_types.UIA_ControlTypePropertyId,
        uia_types.UIA_ListItemControlTypeId,
    )
    elements = root.FindAll(uia_types.TreeScope_Descendants, condition)
    best = None
    best_top = None
    for index in range(elements.Length):
        element = elements.GetElement(index)
        try:
            if int(element.CurrentProcessId) != process_id:
                continue
            top = int(element.CurrentBoundingRectangle.top)
        except Exception:
            continue
        if best_top is None or (top < best_top if oldest else top > best_top):
            best_top, best = top, element
    if best is None:
        return False
    try:
        best.GetCurrentPattern(uia_types.UIA_ScrollItemPatternId).QueryInterface(
            uia_types.IUIAutomationScrollItemPattern
        ).ScrollIntoView()
        return True
    except Exception:
        return False


def restore_to_newest(
    locator: DiscordQuestionLocator,
    hwnd: int,
    process_id: int,
    *,
    settle: float,
    limit: int = 400,
) -> None:
    """Leave the channel where the operator had it: at the newest message.

    The live app reads the newest visible card, so a channel left scrolled into
    history would blind it. Mirror of the harvest walk, downward.
    """

    seen: set[str] = set()
    barren = 0
    for _ in range(limit):
        rows = _rows(locator, hwnd, process_id)
        fresh = [r.name for r in rows if r.name not in seen]
        seen.update(r.name for r in rows)
        barren = barren + 1 if not fresh else 0
        if barren >= 3:
            break
        if not _scroll_extreme(locator, hwnd, process_id, oldest=False):
            break
        time.sleep(settle)
    print("channel restored to the newest messages", flush=True)


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


def _list_item_count(locator: DiscordQuestionLocator, target: Any) -> int:
    """How many list rows exist at all, to tell 'suspended' from 'empty'."""

    try:
        automation, uia_types = locator._client()
        root = automation.ElementFromHandle(target.hwnd)
        condition = automation.CreatePropertyCondition(
            uia_types.UIA_ControlTypePropertyId,
            uia_types.UIA_ListItemControlTypeId,
        )
        return int(root.FindAll(uia_types.TreeScope_Descendants, condition).Length)
    except Exception:
        return 0


def pair_rounds(rows: list[Row]) -> list[dict]:
    """Pair each card with the first reveal posted after it."""

    paired: list[dict] = []
    for position, row in enumerate(rows):
        if row.kind != "card":
            continue
        for later in rows[position + 1:]:
            if later.kind == "card":
                break  # a new card started; this one was never revealed here
            if later.kind == "reveal":
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    parser.add_argument(
        "--rounds",
        type=int,
        default=400,
        help="stop after this many newly harvested clue/answer pairs",
    )
    parser.add_argument(
        "--max-scrolls",
        type=int,
        default=600,
        help="hard ceiling on scroll steps so this always terminates",
    )
    parser.add_argument(
        "--settle",
        type=float,
        default=0.45,
        help="seconds to let Discord render after each scroll step",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be added without writing history",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    guard = ForegroundWindowGuard(config.typing)
    target = guard.expected_window()
    if target is None:
        print(
            "Discord was not found. Open Discord on the trivia channel and retry.",
            file=sys.stderr,
        )
        return 2
    locator = DiscordQuestionLocator()

    # Refuse to run alongside the live app: harvesting scrolls the channel into
    # history, which would blind the app's newest-card read mid-quiz.
    if _live_app_running():
        print(
            "anime-trivia is running. Stop it (F12) before harvesting: scrolling "
            "the channel into history would blind the live card read.",
            file=sys.stderr,
        )
        return 2

    # Electron suspends the virtualized message list when the window is
    # minimized or hidden: the accessibility tree then holds only the channel
    # sidebar. Say so plainly instead of reporting "nothing to harvest".
    if ctypes.windll.user32.IsIconic(target.hwnd):
        print(
            "Discord is minimized, so it is not rendering the message list. "
            "Bring Discord up on the trivia channel and run this again.",
            file=sys.stderr,
        )
        return 2
    probe = _rows(locator, target.hwnd, target.process_id)
    if not probe:
        print(
            f"Discord's message list is not readable ({_list_item_count(locator, target)} "
            "rows found, none of them messages). Bring the Discord window to the "
            "front on the trivia channel and run this again; Electron only renders "
            "the message list while the window is shown.",
            file=sys.stderr,
        )
        return 2
    print(f"message list is readable: {len(probe)} card/reveal rows visible", flush=True)

    history_path = config.runtime.history_path
    history = json.loads(history_path.read_text(encoding="utf-8"))
    known = {
        (normalize_question(p["clue"]), p.get("type") or "")
        for p in history["pairs"]
    }
    before = len(history["pairs"])
    print(f"history starts with {before} clues", flush=True)

    harvested: dict[tuple[str, str], dict] = {}
    seen_names: set[str] = set()
    barren = 0

    for step in range(1, args.max_scrolls + 1):
        rows = _rows(locator, target.hwnd, target.process_id)
        fresh_names = [r.name for r in rows if r.name not in seen_names]
        seen_names.update(r.name for r in rows)
        added_this_step = 0
        for pair in pair_rounds(rows):
            key = (normalize_question(pair["clue"]), pair["type"])
            if key in known or key in harvested:
                continue
            harvested[key] = pair
            added_this_step += 1
        print(
            f"[{step:3}/{args.max_scrolls}] rows={len(rows):3} new_messages={len(fresh_names):3} "
            f"new_clues={added_this_step:2} total_new={len(harvested)}",
            flush=True,
        )
        if len(harvested) >= args.rounds:
            print("reached the requested round count", flush=True)
            break
        # Two consecutive steps with no new message at all means the scroll is
        # not moving (top of channel, or Discord stopped virtualizing).
        barren = barren + 1 if not fresh_names else 0
        if barren >= 3:
            print("no new messages after three scrolls; stopping", flush=True)
            break
        if not _scroll_extreme(
            locator, target.hwnd, target.process_id, oldest=True
        ):
            print("could not scroll any further; stopping", flush=True)
            break
        time.sleep(args.settle)

    restore_to_newest(
        locator, target.hwnd, target.process_id, settle=args.settle
    )

    raw_path = Path("runtime") / "harvest_raw.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(
        json.dumps(list(harvested.values()), indent=1, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n{len(harvested)} new clue/answer pairs -> {raw_path}", flush=True)
    for pair in list(harvested.values())[:15]:
        print(f"  {pair['type']:12} {pair['clue'][:56]!r:60} -> {pair['answer']}")
    if len(harvested) > 15:
        print(f"  ... and {len(harvested) - 15} more")

    if args.dry_run:
        print("\ndry run: history not written", flush=True)
        return 0
    if not harvested:
        print("nothing new to add", flush=True)
        return 0

    # History entries keep their three-field shape; provenance lives in the raw
    # dump so the file the app loads stays exactly the shape it expects.
    history["pairs"].extend(
        {"clue": p["clue"], "type": p["type"], "answer": p["answer"]}
        for p in harvested.values()
    )
    history_path.write_text(
        json.dumps(history, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"history {before} -> {len(history['pairs'])} clues ({history_path})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
