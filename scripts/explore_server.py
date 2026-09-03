"""Read Discord channels through the accessibility tree the app already uses.

Maps the server so opportunities can be judged from what the bot actually posts
rather than from guesswork: the AS Points economy, the store, the leaderboards,
and where the quiz does and does not run.

Read-only by construction. It opens a channel by clicking the sidebar row whose
rectangle the accessibility tree reported, scrolls the message list, and reads
names. It never touches the composer, never types, and never clicks a control
whose label could commit a purchase.

    .venv\\Scripts\\python.exe scripts\\explore_server.py --list
    .venv\\Scripts\\python.exe scripts\\explore_server.py --channel as-store
    .venv\\Scripts\\python.exe scripts\\explore_server.py --channel bots \\
        --scrolls 40 --grep "AS Points|streak|rep"
    .venv\\Scripts\\python.exe scripts\\explore_server.py --channel bots --controls
"""

from __future__ import annotations

import argparse
import ctypes
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anime_trivia_automation.config import load_config  # noqa: E402
from anime_trivia_automation.discord import DiscordQuestionLocator  # noqa: E402
from anime_trivia_automation.typing import ForegroundWindowGuard  # noqa: E402

# Every Discord message row's accessible name carries its own timestamp, which
# is how a chat message is told apart from a sidebar entry.
STAMP = re.compile(r"\d{1,2}:\d{2}\s*(AM|PM)", re.IGNORECASE)

# Anything that might spend points or commit a change is never clicked.
FORBIDDEN = re.compile(
    r"buy|purchase|confirm|redeem|checkout|equip|spend|apply|delete|leave",
    re.IGNORECASE,
)


def _client(locator, target):
    automation, uia = locator._client()
    return automation, uia, automation.ElementFromHandle(target.hwnd)


def _find(locator, target, control_type):
    automation, uia, root = _client(locator, target)
    return automation, uia, root.FindAll(
        uia.TreeScope_Descendants,
        automation.CreatePropertyCondition(
            uia.UIA_ControlTypePropertyId, control_type
        ),
    )


def channels(locator, target) -> list[tuple[str, object]]:
    """Sidebar channel rows with their real names.

    A row's own accessible name is only "TextInvite to Channel"; the channel
    name lives on a descendant, which is why this reaches one level in.
    """

    automation, uia, els = _find(locator, target, None) if False else _find(
        locator, target, locator._client()[1].UIA_ListItemControlTypeId
    )
    found = []
    for index in range(els.Length):
        element = els.GetElement(index)
        name = str(element.CurrentName or "")
        if STAMP.search(name) or "Invite to Channel" not in name:
            continue
        kids = element.FindAll(
            uia.TreeScope_Descendants, automation.CreateTrueCondition()
        )
        for k in range(min(kids.Length, 8)):
            text = str(kids.GetElement(k).CurrentName or "").strip()
            if "(text channel)" in text or "(forum channel)" in text:
                found.append(
                    (text.split(" (")[0].replace("unread, ", "").strip(), element)
                )
                break
    return found


def _click(element) -> bool:
    """Click an element's centre, using the rectangle the tree reported.

    Taking the rectangle from the accessibility tree rather than guessing pixels
    means the click lands on the intended control or nowhere.
    """

    try:
        rect = element.CurrentBoundingRectangle
    except Exception:
        return False
    if rect.right <= rect.left or rect.bottom <= rect.top:
        return False
    user32 = ctypes.windll.user32
    user32.SetCursorPos(
        int((rect.left + rect.right) / 2), int((rect.top + rect.bottom) / 2)
    )
    time.sleep(0.05)
    user32.mouse_event(0x0002, 0, 0, 0, 0)  # left down
    time.sleep(0.03)
    user32.mouse_event(0x0004, 0, 0, 0, 0)  # left up
    return True


def open_channel(locator, target, wanted: str) -> str | None:
    for label, element in channels(locator, target):
        if wanted in label:
            if _click(element):
                time.sleep(2.4)
                return label
    return None


def messages(locator, target) -> tuple[list[str], object]:
    _automation, _uia, els = _find(
        locator, target, locator._client()[1].UIA_ListItemControlTypeId
    )
    out, top, top_at = [], None, None
    for index in range(els.Length):
        element = els.GetElement(index)
        try:
            name = str(element.CurrentName or "")
            y = int(element.CurrentBoundingRectangle.top)
        except Exception:
            continue
        if not STAMP.search(name):
            continue
        out.append(" ".join(name.split()))
        if top_at is None or y < top_at:
            top_at, top = y, element
    return out, top


def scroll_up(locator, element) -> bool:
    _automation, uia = locator._client()
    if element is None:
        return False
    try:
        element.GetCurrentPattern(uia.UIA_ScrollItemPatternId).QueryInterface(
            uia.IUIAutomationScrollItemPattern
        ).ScrollIntoView()
        return True
    except Exception:
        return False


def controls(locator, target) -> None:
    """List interactive controls, flagging any that could commit a purchase."""

    automation, uia, _root = _client(locator, target)
    for control, kind in (
        (uia.UIA_ButtonControlTypeId, "BUTTON"),
        (uia.UIA_ComboBoxControlTypeId, "DROPDOWN"),
    ):
        _a, _u, els = _find(locator, target, control)
        print(f"--- {kind}: {els.Length} ---")
        for index in range(els.Length):
            name = " ".join(str(els.GetElement(index).CurrentName or "").split())
            if not name:
                continue
            flag = "   <-- NEVER CLICKED" if FORBIDDEN.search(name) else ""
            print(f"   {name[:70]}{flag}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--list", action="store_true", help="list sidebar channels")
    parser.add_argument("--channel", default=None, help="channel name fragment to open")
    parser.add_argument("--scrolls", type=int, default=1, help="scroll-up steps")
    parser.add_argument("--grep", default=None, help="only print matching lines")
    parser.add_argument("--controls", action="store_true", help="list interactive controls")
    parser.add_argument(
        "--back-to",
        default="\U0001f49canime-chat",
        help="channel to return to when finished",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    target = ForegroundWindowGuard(config.typing).expected_window()
    if target is None:
        print("Discord was not found. Open it and retry.", file=sys.stderr)
        return 2
    locator = DiscordQuestionLocator()

    if args.list or not args.channel:
        print("channels in the sidebar:")
        for label, _ in channels(locator, target):
            print(f"   {label}")
        if not args.channel:
            return 0
        print()

    opened = open_channel(locator, target, args.channel)
    if opened is None:
        print(f"channel {args.channel!r} not found", file=sys.stderr)
        return 2
    print(f"opened {opened}\n")

    if args.controls:
        controls(locator, target)
    else:
        seen: list[str] = []
        known: set[str] = set()
        barren = 0
        for _ in range(max(1, args.scrolls)):
            batch, top = messages(locator, target)
            fresh = [m for m in batch if m not in known]
            known.update(batch)
            seen = fresh + seen
            barren = barren + 1 if not fresh else 0
            if barren >= 3 or not scroll_up(locator, top):
                break
            time.sleep(0.4)
        needle = re.compile(args.grep, re.IGNORECASE) if args.grep else None
        shown = 0
        for line in seen:
            if needle is None or needle.search(line):
                print(f"  {line[:400]}")
                shown += 1
        print(f"\n({shown} of {len(known)} messages shown)")

    if args.back_to:
        back = open_channel(locator, target, args.back_to)
        if back:
            print(f"returned to {back}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
