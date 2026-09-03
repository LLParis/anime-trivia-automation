from __future__ import annotations

import logging
import re
import threading
import unicodedata
from dataclasses import dataclass
from typing import Any

LOGGER = logging.getLogger(__name__)


def normalize_composer_value(value: str) -> str:
    """Remove Discord's invisible empty-editor sentinels, not user text."""

    return (
        value.replace("\ufeff", "")
        .replace("\u200b", "")
        .replace("\r\n", "\n")
        .rstrip("\n")
    )


@dataclass
class DiscordComposer:
    automation: Any
    element: Any
    value_pattern: Any
    name: str

    def value(self) -> str:
        return normalize_composer_value(str(self.value_pattern.CurrentValue))

    def focused(self) -> bool:
        focused = self.automation.GetFocusedElement()
        return bool(self.automation.CompareElements(self.element, focused))

    def set_focus(self) -> None:
        self.element.SetFocus()

    def set_owned_value(self, value: str) -> None:
        """Replace the empty editor with one complete automation-owned value."""

        self.value_pattern.SetValue(value)

    def clear_owned_value(self) -> None:
        self.value_pattern.SetValue("")


class DiscordComposerLocator:
    """Find the one Discord Slate message editor through Windows UI Automation."""

    def __init__(self, name_prefix: str, class_fragment: str) -> None:
        self._name_prefix = name_prefix.casefold()
        self._class_fragment = class_fragment.casefold()
        self._thread_local = threading.local()

    def _client(self) -> tuple[Any, Any]:
        cached = getattr(self._thread_local, "client", None)
        if cached is not None:
            return cached

        try:
            import comtypes
            from comtypes import client

            comtypes.CoInitialize()
            client.GetModule("UIAutomationCore.dll")
            from comtypes.gen import UIAutomationClient as uia_types
        except ImportError as exc:
            raise RuntimeError(
                "comtypes is required for Discord composer verification"
            ) from exc

        automation = client.CreateObject(
            uia_types.CUIAutomation,
            interface=uia_types.IUIAutomation,
        )
        cached = (automation, uia_types)
        self._thread_local.client = cached
        return cached

    def find(self, hwnd: int, process_id: int) -> DiscordComposer | None:
        automation, uia_types = self._client()
        root = automation.ElementFromHandle(hwnd)
        edit_condition = automation.CreatePropertyCondition(
            uia_types.UIA_ControlTypePropertyId,
            uia_types.UIA_EditControlTypeId,
        )
        elements = root.FindAll(uia_types.TreeScope_Descendants, edit_condition)
        candidates: list[Any] = []
        for index in range(elements.Length):
            element = elements.GetElement(index)
            name = str(element.CurrentName or "")
            class_name = str(element.CurrentClassName or "")
            if int(element.CurrentProcessId) != process_id:
                continue
            if not name.casefold().startswith(self._name_prefix):
                continue
            if self._class_fragment not in class_name.casefold():
                continue
            if not bool(element.CurrentIsEnabled):
                continue
            if bool(element.CurrentIsOffscreen):
                continue
            if not bool(element.CurrentIsKeyboardFocusable):
                continue
            candidates.append(element)

        if len(candidates) != 1:
            LOGGER.warning(
                "Expected exactly one Discord composer, found %d", len(candidates)
            )
            return None

        element = candidates[0]
        pattern = element.GetCurrentPattern(
            uia_types.UIA_ValuePatternId
        ).QueryInterface(uia_types.IUIAutomationValuePattern)
        return DiscordComposer(
            automation=automation,
            element=element,
            value_pattern=pattern,
            name=str(element.CurrentName or ""),
        )


@dataclass(frozen=True)
class AccessibleQuestion:
    clue: str
    expected_answer_type: str
    question_label: str
    screen_bottom: int


@dataclass(frozen=True)
class AccessibleReveal:
    identity: str
    answer: str
    screen_top: int


class DiscordQuestionLocator(DiscordComposerLocator):
    """Read the newest visible Anime Soul card's semantic accessibility name."""

    _CARD_PATTERN = re.compile(
        r"Anime Guessing Game\s+—\s+[^\s]+\s+"
        r"(?:Get Ready(?:(?:\.\.\.)|…)?|Answer Now!?|Round Over)\s+"
        r"(.+?)\s*,?\s*🎯\s*Answer with the "
        r"(anime title|character name).*?Question\s*(\d+)\s*/\s*(\d+)",
        flags=re.IGNORECASE,
    )
    _REVEAL_PATTERN = re.compile(
        r"\bthe answer was\s+(.+?)(?:\.\s*(?:\+\d+\s+AS\s+Points|$)|$)",
        flags=re.IGNORECASE,
    )

    def __init__(self) -> None:
        super().__init__("Message #", "slateTextArea")

    def find_question(
        self, hwnd: int, process_id: int
    ) -> AccessibleQuestion | None:
        automation, uia_types = self._client()
        root = automation.ElementFromHandle(hwnd)
        condition = automation.CreatePropertyCondition(
            uia_types.UIA_ControlTypePropertyId,
            uia_types.UIA_ListItemControlTypeId,
        )
        elements = root.FindAll(uia_types.TreeScope_Descendants, condition)
        parsed: list[tuple[int, int, AccessibleQuestion]] = []
        for index in range(elements.Length):
            element = elements.GetElement(index)
            if int(element.CurrentProcessId) != process_id:
                continue
            if bool(element.CurrentIsOffscreen):
                continue
            name = str(element.CurrentName or "")
            if "Anime Guessing Game" not in name:
                continue
            card = self.parse_card_name(name)
            if card is None:
                LOGGER.warning(
                    "Anime Soul card did not parse; accessible name was %r",
                    name[:400],
                )
                continue
            clue, expected_type, question_label = card
            try:
                rectangle = element.CurrentBoundingRectangle
                screen_top = int(rectangle.top)
                screen_bottom = int(rectangle.bottom)
            except Exception:
                screen_top = index
                screen_bottom = index
            parsed.append(
                (
                    screen_top,
                    index,
                    AccessibleQuestion(
                        clue=clue,
                        expected_answer_type=expected_type,
                        question_label=question_label,
                        screen_bottom=screen_bottom,
                    ),
                )
            )
        return max(parsed, key=lambda item: (item[0], item[1]))[2] if parsed else None

    _QUESTION_LABEL = re.compile(r"Question\s*(\d+)\s*/\s*(\d+)", re.IGNORECASE)
    _ANSWER_TYPE = re.compile(
        r"Answer with the\s+(anime title|character name)", re.IGNORECASE
    )
    _STATUS_WORDS = re.compile(
        r"(?:Get Ready(?:\.\.\.|…)?|Answer Now!?|Round Over)", re.IGNORECASE
    )
    # Card chrome that must never be mistaken for the clue itself.
    _CHROME_EMOJI = frozenset("🎮🔴🟢⚪🎯🔒🔓⏳⌛🕐🕑🕒⏱️⏰✅❌")

    @classmethod
    def parse_card_name(cls, name: str) -> tuple[str, str, str] | None:
        """Extract (clue, answer type, question label) from a card's accessible name.

        The strict regex is tried first. If Discord renders the card slightly
        differently (missing 🎯, extra separators, emoji exposed as separate
        runs), fall back to structural parsing: the clue is what sits between
        the status word and the "Answer with the ..." instruction, and for an
        emoji rebus it is the emoji sequence itself.
        """

        match = cls._CARD_PATTERN.search(name)
        if match is not None:
            return (
                match.group(1).strip(),
                "anime_title" if match.group(2).casefold() == "anime title" else "character",
                f"{match.group(3)}/{match.group(4)}",
            )
        type_match = cls._ANSWER_TYPE.search(name)
        label_match = cls._QUESTION_LABEL.search(name)
        if type_match is None or label_match is None:
            return None
        expected_type = (
            "anime_title" if type_match.group(1).casefold() == "anime title" else "character"
        )
        question_label = f"{label_match.group(1)}/{label_match.group(2)}"
        head = name[: type_match.start()]
        status = None
        for status in cls._STATUS_WORDS.finditer(head):
            pass
        clue = head[status.end():] if status is not None else head
        clue = clue.replace("🎯", " ").strip(" ,\u2014-\t\r\n")
        if not clue:
            clue = cls.emoji_sequence(head)
        if not clue:
            return None
        return clue, expected_type, question_label

    @staticmethod
    def emoji_sequence(text: str) -> str:
        """The emoji glyphs of a clue in order, without Discord's card chrome."""

        symbols: list[str] = []
        for character in text:
            if character in ("️", "‍"):
                if symbols:
                    symbols[-1] += character
                continue
            category = unicodedata.category(character)
            if category.startswith("S") or ord(character) > 0xFFFF:
                if character in DiscordQuestionLocator._CHROME_EMOJI:
                    continue
                symbols.append(character)
        return " ".join(symbols)

    @classmethod
    def parse_reveal_answer(cls, accessible_name: str) -> str | None:
        """Extract one official Anime Soul answer from a semantic list-item name."""

        match = cls._REVEAL_PATTERN.search(accessible_name)
        if match is None:
            return None
        answer = " ".join(match.group(1).split()).strip()
        return answer or None

    @classmethod
    def is_official_reveal_name(cls, accessible_name: str) -> bool:
        folded = accessible_name.casefold()
        return (
            re.match(
                r"^Anime Soul\d{1,2}:\d{2}\s*(?:AM|PM)",
                accessible_name,
            )
            is not None
            and ("correct!" in folded or "time's up" in folded)
            and cls.parse_reveal_answer(accessible_name) is not None
        )

    def find_reveal_records(
        self, hwnd: int, process_id: int
    ) -> tuple[AccessibleReveal, ...]:
        """Read visible official bot reveals with stable identity and position."""

        automation, uia_types = self._client()
        root = automation.ElementFromHandle(hwnd)
        condition = automation.CreatePropertyCondition(
            uia_types.UIA_ControlTypePropertyId,
            uia_types.UIA_ListItemControlTypeId,
        )
        elements = root.FindAll(uia_types.TreeScope_Descendants, condition)
        records: list[AccessibleReveal] = []
        for index in range(elements.Length):
            element = elements.GetElement(index)
            if int(element.CurrentProcessId) != process_id:
                continue
            if bool(element.CurrentIsOffscreen):
                continue
            name = str(element.CurrentName or "")
            if not self.is_official_reveal_name(name):
                continue
            answer = self.parse_reveal_answer(name)
            if answer is not None:
                try:
                    screen_top = int(element.CurrentBoundingRectangle.top)
                except Exception:
                    screen_top = index
                records.append(
                    AccessibleReveal(
                        identity=name,
                        answer=answer,
                        screen_top=screen_top,
                    )
                )
        return tuple(records)

    def find_reveal_answers(self, hwnd: int, process_id: int) -> tuple[str, ...]:
        """Compatibility projection of visible official reveal answers."""

        return tuple(
            record.answer
            for record in self.find_reveal_records(hwnd, process_id)
        )
