from __future__ import annotations

import logging
import re
import threading
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
            import comtypes.client as client

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


class DiscordQuestionLocator(DiscordComposerLocator):
    """Read the newest visible Anime Soul card's semantic accessibility name."""

    _CARD_PATTERN = re.compile(
        r"Anime Guessing Game\s+—\s+[^\s]+\s+"
        r"(?:Get Ready(?:(?:\.\.\.)|…)?|Answer Now!?|Round Over)\s+"
        r"(.+?)\s*,?\s*🎯\s*Answer with the "
        r"(anime title|character name).*?Question\s*(\d+)\s*/\s*(\d+)",
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
            match = self._CARD_PATTERN.search(name)
            if match is None:
                continue
            expected_type = (
                "anime_title"
                if match.group(2).casefold() == "anime title"
                else "character"
            )
            try:
                screen_top = int(element.CurrentBoundingRectangle.top)
            except Exception:
                screen_top = index
            parsed.append(
                (
                    screen_top,
                    index,
                    AccessibleQuestion(
                    clue=match.group(1).strip(),
                    expected_answer_type=expected_type,
                    question_label=f"{match.group(3)}/{match.group(4)}",
                    ),
                )
            )
        return max(parsed, key=lambda item: (item[0], item[1]))[2] if parsed else None
