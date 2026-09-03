"""Claim the Anime Soul daily streak, once, with confirmation.

The daily reward is the single biggest points lever we have found. It ramps from
+250 on day 1 to +500 at day 5 and stays there, and it resets at 5:00 PM. Missing
one day drops the streak back to +250, which is why an 8-day run was lost. At the
top rate it is worth ten trivia wins for one message.

This POSTS ONE MESSAGE as the operator: exactly "!asdaily" in the bots channel.
Nothing else is ever sent. It carries the same guards as the live answer path:

  * refuses while the quiz automation is running, so it cannot collide with a round
  * refuses unless the composer is empty, so it never touches text you are writing
  * types through SendInput, reads the composer back, and only then presses Enter
  * confirms the composer cleared, then reads the bot's reply and reports it
  * returns you to the channel you were on

    .venv\\Scripts\\python.exe scripts\\claim_daily.py            # claim
    .venv\\Scripts\\python.exe scripts\\claim_daily.py --dry-run  # check only
"""

from __future__ import annotations

import argparse
import ctypes
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anime_trivia_automation.config import load_config  # noqa: E402
from anime_trivia_automation.discord import (  # noqa: E402
    DiscordComposerLocator,
    DiscordQuestionLocator,
)
from anime_trivia_automation.typing import ForegroundWindowGuard  # noqa: E402
from anime_trivia_automation.windows_input import BatchedWindowsInput  # noqa: E402

COMMAND = "!asdaily"
# The daily reward resets at this hour, local time.
RESET_HOUR = 17
BOTS_CHANNEL = "bots"
STAMP = re.compile(r"\d{1,2}:\d{2}\s*(AM|PM)", re.IGNORECASE)
# What the bot says back, so a claim is confirmed rather than assumed.
POINTS = re.compile(r"\+(\d[\d,]*)\s*AS Points", re.IGNORECASE)
STREAK = re.compile(r"Streak:\s*Day\s*(\d+)\s*(?:·|\|)?\s*Best:\s*(\d+)", re.IGNORECASE)


def _client(locator, target):
    automation, uia = locator._client()
    return automation, uia, automation.ElementFromHandle(target.hwnd)


def channels(locator, target):
    automation, uia, root = _client(locator, target)
    els = root.FindAll(
        uia.TreeScope_Descendants,
        automation.CreatePropertyCondition(
            uia.UIA_ControlTypePropertyId, uia.UIA_ListItemControlTypeId
        ),
    )
    found = []
    for index in range(els.Length):
        element = els.GetElement(index)
        name = str(element.CurrentName or "")
        if STAMP.search(name) or "Invite to Channel" not in name:
            continue
        kids = element.FindAll(uia.TreeScope_Descendants, automation.CreateTrueCondition())
        for k in range(min(kids.Length, 8)):
            text = str(kids.GetElement(k).CurrentName or "").strip()
            if "(text channel)" in text:
                found.append(
                    (text.split(" (")[0].replace("unread, ", "").strip(), element)
                )
                break
    return found


def click(element) -> bool:
    try:
        rect = element.CurrentBoundingRectangle
    except Exception:
        return False
    if rect.right <= rect.left:
        return False
    user32 = ctypes.windll.user32
    user32.SetCursorPos(
        int((rect.left + rect.right) / 2), int((rect.top + rect.bottom) / 2)
    )
    time.sleep(0.05)
    user32.mouse_event(0x0002, 0, 0, 0, 0)
    time.sleep(0.03)
    user32.mouse_event(0x0004, 0, 0, 0, 0)
    return True


def newest_messages(locator, target, limit: int = 12) -> list[str]:
    _automation, _uia, _root = _client(locator, target)
    automation, uia, root = _client(locator, target)
    els = root.FindAll(
        uia.TreeScope_Descendants,
        automation.CreatePropertyCondition(
            uia.UIA_ControlTypePropertyId, uia.UIA_ListItemControlTypeId
        ),
    )
    out = []
    for index in range(els.Length):
        name = str(els.GetElement(index).CurrentName or "")
        if STAMP.search(name):
            out.append(" ".join(name.split()))
    return out[-limit:]


def quiz_running() -> bool:
    try:
        output = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq anime-trivia.exe", "/NH"],
            capture_output=True, text=True, timeout=15, check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return "anime-trivia.exe" in output


def describe(messages: list[str], operator: str) -> str | None:
    """Pull the bot's answer to our claim out of the newest messages."""

    for message in reversed(messages):
        folded = message.casefold()
        # "You've already claimed your daily today" carries neither "AS Points"
        # nor "streak", and missing it made a successful run look like a failure.
        if not any(
            word in folded
            for word in ("as points", "streak", "claimed", "daily", "come back at")
        ):
            continue
        if operator and operator not in message:
            continue
        # Classify rather than slice: the reply contains times like "5:00 PM",
        # so any character-class slice trips over the colon.
        parts = []
        if "already claimed" in folded:
            parts.append("already claimed today, next reset 5:00 PM")
        elif "maxed the streak" in folded:
            parts.append("streak maxed, top daily reward")
        elif "started their daily streak" in folded:
            parts.append("streak started")
        points = POINTS.search(message)
        streak = STREAK.search(message)
        if points:
            parts.append(f"+{points.group(1)} AS Points")
        if streak:
            parts.append(f"streak day {streak.group(1)}, best {streak.group(2)}")
        if parts:
            return " | ".join(parts)
    return None


def wait_until(after: str, jitter_minutes: int) -> None:
    """Sleep until a randomly chosen minute past the reset.

    Claiming at 17:00:00 every single day is a signature in itself, so the exact
    moment moves within a window. The reward does not decay inside the day, so
    the delay costs nothing.
    """

    import datetime as dt
    import random

    hour, _, minute = after.partition(":")
    target = dt.datetime.now().replace(
        hour=int(hour), minute=int(minute or 0), second=0, microsecond=0
    )
    if target < dt.datetime.now():
        target += dt.timedelta(days=1)
    target += dt.timedelta(seconds=random.randint(0, max(0, jitter_minutes) * 60))
    seconds = (target - dt.datetime.now()).total_seconds()
    print(f"waiting until {target:%H:%M:%S} ({seconds / 60:.0f} min)", flush=True)
    while seconds > 0:
        time.sleep(min(seconds, 30))
        seconds = (target - dt.datetime.now()).total_seconds()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run every check and stop before sending anything",
    )
    parser.add_argument(
        "--after",
        default=None,
        help="wait until this local time before claiming, e.g. 17:00 (the reset)",
    )
    parser.add_argument(
        "--jitter",
        type=int,
        default=40,
        help="spread the claim over this many minutes past --after",
    )
    parser.add_argument(
        "--before-reset",
        action="store_true",
        help="allow sending before the 17:00 reset (normally refused)",
    )
    args = parser.parse_args()

    # The reward resets at 17:00 and the operator's instruction is to claim at
    # or after that. Sending earlier wastes the run and posts a message at an
    # hour that was never sanctioned, which is exactly what happened at 14:59 on
    # 2026-09-03. Refuse by default rather than rely on remembering.
    import datetime as _dt

    if not args.dry_run and not args.after and not args.before_reset:
        now = _dt.datetime.now()
        if now.hour < RESET_HOUR:
            print(
                f"It is {now:%H:%M} and the daily resets at {RESET_HOUR}:00. "
                f"Run with --after {RESET_HOUR}:00 to wait, --dry-run to check, "
                "or --before-reset if you really mean now.",
                file=sys.stderr,
            )
            return 2

    if args.after:
        wait_until(args.after, args.jitter)

    config = load_config(args.config)
    operator = config.runtime.operator_display_name
    guard = ForegroundWindowGuard(config.typing)
    target = guard.expected_window()
    if target is None:
        print("Discord was not found. Open it and retry.", file=sys.stderr)
        return 2
    if quiz_running():
        print(
            "anime-trivia is running. Claim before or after a quiz so this cannot "
            "collide with a round.",
            file=sys.stderr,
        )
        return 2
    if ctypes.windll.user32.IsIconic(target.hwnd):
        print("Discord is minimized; restore it and retry.", file=sys.stderr)
        return 2

    locator = DiscordQuestionLocator()
    started_on = target.title or ""
    print(f"on {started_on}")

    opened = None
    for label, element in channels(locator, target):
        if BOTS_CHANNEL in label:
            if click(element):
                time.sleep(2.2)
                opened = label
            break
    if opened is None:
        print("could not open the bots channel", file=sys.stderr)
        return 2
    print(f"opened {opened}")

    # The configured prefix names the trivia channel's composer ("Message
    # #anime-chat"); in #bots the box is named for that channel instead, so
    # match the generic prefix here.
    composer = DiscordComposerLocator(
        "Message #", config.typing.composer_class_fragment
    ).find(target.hwnd, target.process_id)
    if composer is None:
        print("composer not found", file=sys.stderr)
        return 2
    existing = composer.value()
    if existing:
        print(
            f"composer already holds {existing[:40]!r}; leaving it alone",
            file=sys.stderr,
        )
        return 2

    before = newest_messages(locator, target)
    if args.dry_run:
        print(f"dry run: would send {COMMAND!r}. Nothing sent.")
        return 0

    if not guard.activate(target.hwnd):
        print("could not bring Discord forward", file=sys.stderr)
        return 2
    time.sleep(0.25)
    if not composer.focused():
        composer.set_focus()
        time.sleep(0.15)

    BatchedWindowsInput().send_text(COMMAND)
    deadline = time.monotonic() + max(1.0, config.typing.composer_settle_timeout_seconds)
    while composer.value() != COMMAND and time.monotonic() < deadline:
        time.sleep(0.02)
    if composer.value() != COMMAND:
        print(
            f"composer holds {composer.value()[:40]!r}, not the command; nothing sent",
            file=sys.stderr,
        )
        return 2

    from pynput.keyboard import Controller, Key

    controller = Controller()
    controller.press(Key.enter)
    controller.release(Key.enter)
    print(f"sent {COMMAND}")

    deadline = time.monotonic() + 3.0
    while composer.value() != "" and time.monotonic() < deadline:
        time.sleep(0.05)
    if composer.value() != "":
        print("warning: the composer did not clear; check Discord", file=sys.stderr)

    for _ in range(12):
        time.sleep(0.6)
        result = describe(newest_messages(locator, target), operator)
        if result and describe(before, operator) != result:
            print(f"bot replied: {result}")
            break
    else:
        print("no reply matched; check #bots yourself")

    # Put the channel back, so the quiz automation is never left looking at the
    # wrong one.
    for label, element in channels(locator, target):
        if label != opened and started_on and label.strip("#") in started_on:
            if click(element):
                time.sleep(1.2)
                print(f"returned to {label}")
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
