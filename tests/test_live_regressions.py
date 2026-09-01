from __future__ import annotations

import threading
import time
import unittest
from contextlib import contextmanager

from anime_trivia_automation.config import ReadinessConfig, TypingConfig
from anime_trivia_automation.discord import (
    DiscordQuestionLocator,
    normalize_composer_value,
)
from anime_trivia_automation.models import AnswerTask
from anime_trivia_automation.typing import (
    ActivePromptState,
    ForegroundWindow,
    SafeKeyboardExecutor,
)


class FakeComposer:
    def __init__(self) -> None:
        self.text = ""
        self.has_focus = True
        self.name = "Message #💜anime-chat"

    def value(self) -> str:
        return self.text

    def focused(self) -> bool:
        return self.has_focus

    def set_focus(self) -> None:
        self.has_focus = True

    def clear_owned_value(self) -> None:
        self.text = ""


class FakeGuard:
    window = ForegroundWindow(1, 2, "Discord.exe", "Anime Soul - Discord")

    def allowed(self) -> tuple[bool, str]:
        return True, "Discord"

    def validate(self, _window: ForegroundWindow | None) -> tuple[bool, str]:
        return True, "Discord"

    def current(self) -> ForegroundWindow:
        return self.window


class FakeLocator:
    def __init__(self, composer: FakeComposer) -> None:
        self.composer = composer

    def find(self, _hwnd: int, _process_id: int) -> FakeComposer:
        return self.composer


class FakeController:
    def __init__(self, composer: FakeComposer, enter_key: object) -> None:
        self.composer = composer
        self.enter_key = enter_key
        self.entered = False

    def type(self, character: str) -> None:
        self.composer.text += character

    def press(self, key: object) -> None:
        if key == self.enter_key:
            self.entered = True
            self.composer.text = ""

    def release(self, _key: object) -> None:
        return

    @contextmanager
    def pressed(self, _key: object):
        yield


class AmbiguousEnterController(FakeController):
    def press(self, key: object) -> None:
        if key == self.enter_key:
            self.entered = True
            self.composer.text = ""
            raise RuntimeError("simulated post-dispatch failure")
        super().press(key)


def make_executor(
    active: ActivePromptState, composer: FakeComposer
) -> SafeKeyboardExecutor:
    executor = SafeKeyboardExecutor.__new__(SafeKeyboardExecutor)
    executor._config = TypingConfig(
        enabled=True,
        expected_process_names=("Discord.exe",),
        expected_window_title_contains="Anime Soul - Discord",
        pre_delay_seconds=(0.0, 0.0),
        key_delay_seconds=(0.005, 0.005),
        draft_while_locked=True,
        verify_composer=True,
        auto_focus_composer=True,
        composer_name_prefix="Message #💜anime-chat",
        composer_class_fragment="slateTextArea",
        enter_after_open_slack_seconds=0.0,
    )
    executor._readiness_config = ReadinessConfig(
        require_green_outline=True,
        ready_wait_timeout_seconds=1.0,
    )
    executor._active_prompt = active
    executor._stop_event = threading.Event()
    executor._guard = FakeGuard()
    executor._composer_locator = FakeLocator(composer)
    executor._enter_key = object()
    executor._orphaned_draft = None
    executor._controller = FakeController(composer, executor._enter_key)
    return executor


class LiveRegressionTests(unittest.TestCase):
    def test_composer_empty_sentinel_is_normalized(self) -> None:
        self.assertEqual(normalize_composer_value("\ufeff\n"), "")

    def test_accessibility_card_pattern_preserves_emoji(self) -> None:
        name = (
            "Anime Soul App 🎮 Anime Guessing Game — 🔴 Get Ready... "
            "🚀 🌙 👨‍🚀 👬 , 🎯 Answer with the anime title — first correct "
            "guess in chat wins! Question 2/10 · Reading time"
        )
        match = DiscordQuestionLocator._CARD_PATTERN.search(name)
        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.group(1), "🚀 🌙 👨‍🚀 👬")
        self.assertEqual(match.group(3), "2")

    def test_draft_is_complete_before_green_and_enter_waits(self) -> None:
        active = ActivePromptState()
        signature = "round:anime_title:1/10"
        active.update(signature, "locked", 1)
        composer = FakeComposer()
        executor = make_executor(active, composer)
        task = AnswerTask(
            answer="Fruits Basket",
            prompt_signature=signature,
            expected_answer_type="anime_title",
            question_label="1/10",
            detected_at=time.monotonic(),
            countdown_seconds=5.0,
            source="history-cache",
        )
        result: list[bool] = []
        thread = threading.Thread(target=lambda: result.append(executor.execute(task)))
        thread.start()
        deadline = time.monotonic() + 1.0
        while composer.text != task.answer and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertEqual(composer.text, task.answer)
        self.assertFalse(executor._controller.entered)
        active.update(signature, "ready", 2)
        thread.join(1.0)
        self.assertEqual(result, [True])
        self.assertTrue(executor._controller.entered)
        self.assertEqual(composer.text, "")

    def test_human_interference_aborts_without_erasing_user_text(self) -> None:
        active = ActivePromptState()
        signature = "round:character:1/10"
        active.update(signature, "locked", 1)
        composer = FakeComposer()
        executor = make_executor(active, composer)
        task = AnswerTask(
            answer="Rei Kiriyama",
            prompt_signature=signature,
            expected_answer_type="character",
            question_label="1/10",
            detected_at=time.monotonic(),
            countdown_seconds=5.0,
            source="history-cache",
        )
        result: list[bool] = []
        thread = threading.Thread(target=lambda: result.append(executor.execute(task)))
        thread.start()
        deadline = time.monotonic() + 1.0
        while len(composer.text) < 2 and time.monotonic() < deadline:
            time.sleep(0.002)
        composer.text += "USER"
        thread.join(1.0)
        self.assertEqual(result, [False])
        self.assertIn("USER", composer.text)
        self.assertFalse(executor._controller.entered)

    def test_closed_race_after_ready_never_dispatches_enter(self) -> None:
        active = ActivePromptState()
        signature = "round:anime_title:1/10"
        active.update(signature, "locked", 1)
        original_wait_ready = active.wait_ready

        def wait_ready_then_close(*args, **kwargs):
            result = original_wait_ready(*args, **kwargs)
            if result:
                active.update(signature, "closed", 3)
            return result

        active.wait_ready = wait_ready_then_close  # type: ignore[method-assign]
        composer = FakeComposer()
        executor = make_executor(active, composer)
        task = AnswerTask(
            answer="Fruits Basket",
            prompt_signature=signature,
            expected_answer_type="anime_title",
            question_label="1/10",
            detected_at=time.monotonic(),
            countdown_seconds=5.0,
            source="history-cache",
        )
        result: list[bool] = []
        thread = threading.Thread(target=lambda: result.append(executor.execute(task)))
        thread.start()
        deadline = time.monotonic() + 1.0
        while composer.text != task.answer and time.monotonic() < deadline:
            time.sleep(0.005)
        active.update(signature, "ready", 2)
        thread.join(1.0)
        self.assertEqual(result, [False])
        self.assertFalse(executor._controller.entered)
        self.assertEqual(composer.text, "")

    def test_enter_keydown_exception_is_consumed_without_duplicate(self) -> None:
        active = ActivePromptState()
        signature = "round:anime_title:1/10"
        active.update(signature, "ready", 1)
        composer = FakeComposer()
        executor = make_executor(active, composer)
        executor._controller = AmbiguousEnterController(composer, executor._enter_key)
        task = AnswerTask(
            answer="Fruits Basket",
            prompt_signature=signature,
            expected_answer_type="anime_title",
            question_label="1/10",
            detected_at=time.monotonic(),
            countdown_seconds=0.0,
            source="history-cache",
        )
        self.assertTrue(executor.execute(task))
        self.assertTrue(executor._controller.entered)

    def test_orphan_cleanup_targets_owned_editor_without_keyboard_shortcuts(self) -> None:
        active = ActivePromptState()
        composer = FakeComposer()
        composer.text = "Owned Draft"
        executor = make_executor(active, composer)
        executor._orphaned_draft = "Owned Draft"
        executor.service_orphan()
        self.assertEqual(composer.text, "")
        self.assertIsNone(executor._orphaned_draft)


if __name__ == "__main__":
    unittest.main()
