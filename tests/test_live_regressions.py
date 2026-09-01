from __future__ import annotations

import threading
import time
import unittest
from contextlib import contextmanager
from types import SimpleNamespace

from anime_trivia_automation.config import ReadinessConfig, TypingConfig
from anime_trivia_automation.app import AnimeTriviaAutomation
from anime_trivia_automation.discord import (
    DiscordQuestionLocator,
    normalize_composer_value,
)
from anime_trivia_automation.models import AnswerTask
from anime_trivia_automation.status import NullStatus
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
        self.focus_calls = 0

    def value(self) -> str:
        return self.text

    def focused(self) -> bool:
        return self.has_focus

    def set_focus(self) -> None:
        self.focus_calls += 1
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


class AmbiguousCharacterController(FakeController):
    def type(self, character: str) -> None:
        self.composer.text += character
        raise RuntimeError("simulated post-character failure")


class RecordingStatus:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def emit(self, phase: str, **fields) -> None:
        self.events.append((phase, fields))


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
    executor._status = NullStatus()
    executor._controller = FakeController(composer, executor._enter_key)
    return executor


class LiveRegressionTests(unittest.TestCase):
    def test_quiz_complete_marker_only_closes_the_final_card(self) -> None:
        self.assertFalse(AnimeTriviaAutomation._is_final_question(None))
        self.assertFalse(AnimeTriviaAutomation._is_final_question("Question 4/10"))
        self.assertTrue(AnimeTriviaAutomation._is_final_question("Question 10/10"))
        self.assertTrue(AnimeTriviaAutomation._is_final_question("3/3"))

    def test_new_session_requires_a_locked_first_question(self) -> None:
        self.assertTrue(AnimeTriviaAutomation._is_new_quiz_start("1/10", "locked"))
        self.assertFalse(AnimeTriviaAutomation._is_new_quiz_start("1/10", "ready"))
        self.assertFalse(AnimeTriviaAutomation._is_new_quiz_start("5/10", "locked"))

    def test_short_quote_is_treated_as_text_for_reveal_learning(self) -> None:
        self.assertEqual(
            AnimeTriviaAutomation._effective_prompt_kind("visual", '"Sit, boy!"'),
            "text",
        )
        self.assertEqual(
            AnimeTriviaAutomation._effective_prompt_kind("visual", "🏍️ ❄️ 🏚️ 🥫"),
            "visual",
        )

    def test_clue_change_blocks_a_stale_same_round_enter(self) -> None:
        active = ActivePromptState()
        signature = "round:character:1/10"
        active.update(signature, "locked", 1, "first-clue")
        active.update(signature, "ready", 2, "corrected-clue")
        called = False

        def dispatch() -> bool:
            nonlocal called
            called = True
            return True

        self.assertFalse(
            active.execute_if_ready(
                signature,
                threading.Event(),
                dispatch,
                clue_fingerprint="first-clue",
            )
        )
        self.assertFalse(called)

    def test_visual_fingerprint_uses_hash_without_semantic_emoji(self) -> None:
        app = AnimeTriviaAutomation.__new__(AnimeTriviaAutomation)
        app._accessible_round = None
        observation = SimpleNamespace(
            signature="round:anime_title:4/10",
            prompt_kind="visual",
            perceptual_hash="abc123",
            hint_text="Visual / emoji clue",
        )
        self.assertEqual(app._clue_fingerprint(observation), "visual:abc123")
        app._accessible_round = (observation.signature, "🥽 🦖 💻 🌐")
        self.assertTrue(app._clue_fingerprint(observation).startswith("semantic:"))

    def test_status_round_identity_survives_footer_ocr_variation(self) -> None:
        app = AnimeTriviaAutomation.__new__(AnimeTriviaAutomation)
        app._status_session_id = 1
        app._status_round_id = 0
        app._active_status_signature = None
        app._active_status_question_label = None
        app._active_status_clue_key = None
        app._active_status_token = None
        app._active_status_closed = False

        fallback = SimpleNamespace(
            signature="clue:fallback",
            question_label=None,
            hint_text="A stable clue",
            perceptual_hash=None,
        )
        labeled = SimpleNamespace(
            signature="round:character:4/10",
            question_label="4/10",
            hint_text="A stable clue",
            perceptual_hash=None,
        )
        changed_type = SimpleNamespace(
            signature="round:anime_title:4/10",
            question_label="4/10",
            hint_text="A stable clue",
            perceptual_hash=None,
        )

        first_token, first_is_new = app._round_token(fallback, live=True)
        labeled_token, labeled_is_new = app._round_token(labeled, live=True)
        changed_token, changed_is_new = app._round_token(changed_type, live=True)

        self.assertTrue(first_is_new)
        self.assertFalse(labeled_is_new)
        self.assertFalse(changed_is_new)
        self.assertEqual(first_token, labeled_token)
        self.assertEqual(first_token, changed_token)
        self.assertEqual(app._status_round_id, 1)

    def test_footer_upgrade_rejects_a_different_physical_card(self) -> None:
        app = AnimeTriviaAutomation.__new__(AnimeTriviaAutomation)
        app._status_session_id = 1
        app._status_round_id = 0
        app._active_status_signature = None
        app._active_status_question_label = None
        app._active_status_clue_key = None
        app._active_status_token = None
        app._active_status_closed = False
        first = SimpleNamespace(
            signature="clue:first",
            question_label=None,
            hint_text="First clue",
            perceptual_hash=None,
        )
        second = SimpleNamespace(
            signature="round:character:2/10",
            question_label="2/10",
            hint_text="Second clue",
            perceptual_hash=None,
        )

        first_token, _ = app._round_token(first, live=True)
        second_token, second_is_new = app._round_token(second, live=True)

        self.assertTrue(second_is_new)
        self.assertNotEqual(first_token, second_token)
        self.assertEqual(app._status_round_id, 2)

    def test_viewport_quiz_complete_requires_a_closed_final_round(self) -> None:
        app = AnimeTriviaAutomation.__new__(AnimeTriviaAutomation)
        app._active_status_question_label = "10/10"
        app._active_status_closed = False
        self.assertFalse(app._should_finish_without_card(True))
        app._active_status_closed = True
        self.assertTrue(app._should_finish_without_card(True))

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
        recording = RecordingStatus()
        executor._status = recording
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
        phases = [phase for phase, _fields in recording.events]
        self.assertIn("DRAFTING", phases)
        self.assertIn("WAITING_GREEN", phases)
        self.assertIn("SUBMITTED", phases)

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
        with self.assertLogs(
            "anime_trivia_automation.typing", level="WARNING"
        ):
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
        self.assertEqual(composer.focus_calls, 0)

    def test_ambiguous_character_injection_retains_exact_prefix_ownership(self) -> None:
        active = ActivePromptState()
        signature = "round:character:1/10"
        active.update(signature, "locked", 1)
        composer = FakeComposer()
        executor = make_executor(active, composer)
        executor._controller = AmbiguousCharacterController(
            composer, executor._enter_key
        )
        task = AnswerTask(
            answer="Rei Kiriyama",
            prompt_signature=signature,
            expected_answer_type="character",
            question_label="1/10",
            detected_at=time.monotonic(),
            countdown_seconds=5.0,
            source="history-cache",
        )
        with self.assertLogs(
            "anime_trivia_automation.typing", level="WARNING"
        ):
            self.assertFalse(executor.execute(task))
        self.assertEqual(composer.text, "R")
        self.assertEqual(executor._orphaned_draft, "R")
        self.assertFalse(executor._controller.entered)


if __name__ == "__main__":
    unittest.main()
