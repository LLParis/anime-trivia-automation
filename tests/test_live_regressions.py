from __future__ import annotations

import threading
import time
import unittest
from contextlib import contextmanager
from types import SimpleNamespace

from anime_trivia_automation.app import AnimeTriviaAutomation
from anime_trivia_automation.config import (
    AppConfig,
    CaptureConfig,
    ReadinessConfig,
    TypingConfig,
    validate_config,
)
from anime_trivia_automation.discord import (
    DiscordQuestionLocator,
    normalize_composer_value,
)
from anime_trivia_automation.models import AnswerTask
from anime_trivia_automation.status import NullStatus
from anime_trivia_automation.typing import (
    ActivePromptState,
    AnswerDispatcher,
    ForegroundWindow,
    ForegroundWindowGuard,
    SafeKeyboardExecutor,
)


class FakeComposer:
    def __init__(self) -> None:
        self.text = ""
        self.has_focus = True
        self.name = "Message #💜anime-chat"
        self.focus_calls = 0
        self.set_values: list[str] = []

    def value(self) -> str:
        return self.text

    def focused(self) -> bool:
        return self.has_focus

    def set_focus(self) -> None:
        self.focus_calls += 1
        self.has_focus = True

    def set_owned_value(self, value: str) -> None:
        self.set_values.append(value)
        self.text = value

    def clear_owned_value(self) -> None:
        self.text = ""


class AmbiguousAtomicComposer(FakeComposer):
    def set_owned_value(self, value: str) -> None:
        super().set_owned_value(value)
        raise RuntimeError("simulated exception after complete atomic write")


class DelayedAtomicComposer(FakeComposer):
    """Expose UIA writes only after a few CurrentValue reads, like Discord Slate."""

    def __init__(self, lag_reads: int = 3) -> None:
        super().__init__()
        self._lag_reads = lag_reads
        self._pending: str | None = None
        self._remaining_reads = 0

    def value(self) -> str:
        if self._pending is not None:
            if self._remaining_reads > 0:
                self._remaining_reads -= 1
            else:
                self.text = self._pending
                self._pending = None
        return self.text

    def set_owned_value(self, value: str) -> None:
        self.set_values.append(value)
        self._pending = value
        self._remaining_reads = self._lag_reads

    def clear_owned_value(self) -> None:
        self._pending = ""
        self._remaining_reads = self._lag_reads


class FakeGuard:
    window = ForegroundWindow(1, 2, "Discord.exe", "Anime Soul - Discord")

    def allowed(self) -> tuple[bool, str]:
        return True, "Discord"

    def validate(self, _window: ForegroundWindow | None) -> tuple[bool, str]:
        return True, "Discord"

    def current(self) -> ForegroundWindow:
        return self.window

    def expected_window(self) -> ForegroundWindow:
        return self.window

    @staticmethod
    def idle_milliseconds() -> int:
        return 10_000

    def activate(self, _hwnd: int) -> bool:
        return True


class SwitchingGuard:
    discord = ForegroundWindow(1, 2, "Discord.exe", "Anime Soul - Discord")
    chrome = ForegroundWindow(3, 4, "chrome.exe", "Gemini")

    def __init__(self, idle_ms: int = 0) -> None:
        self.window = self.chrome
        self.idle_ms = idle_ms
        self.activations: list[int] = []

    def idle_milliseconds(self) -> int:
        return self.idle_ms

    def activate(self, hwnd: int) -> bool:
        self.activations.append(hwnd)
        if hwnd == self.discord.hwnd:
            self.window = self.discord
            return True
        if hwnd == self.chrome.hwnd:
            self.window = self.chrome
            return True
        return False

    def allowed(self) -> tuple[bool, str]:
        return self.validate(self.window)

    def validate(self, window: ForegroundWindow | None) -> tuple[bool, str]:
        if window == self.discord:
            return True, "Discord"
        return False, "foreground process is 'chrome.exe'"

    def current(self) -> ForegroundWindow:
        return self.window

    def expected_window(self) -> ForegroundWindow:
        return self.discord


class FakeLocator:
    def __init__(self, composer: FakeComposer) -> None:
        self.composer = composer
        self.find_calls = 0

    def find(self, _hwnd: int, _process_id: int) -> FakeComposer:
        self.find_calls += 1
        return self.composer


class SwitchingLocator(FakeLocator):
    def __init__(self, composer: FakeComposer) -> None:
        super().__init__(composer)
        self.available = False

    def find(self, _hwnd: int, _process_id: int) -> FakeComposer | None:
        self.find_calls += 1
        return self.composer if self.available else None


class FlakyLocator(FakeLocator):
    def __init__(self, composer: FakeComposer, failures: int = 2) -> None:
        super().__init__(composer)
        self.failures = failures

    def find(self, _hwnd: int, _process_id: int) -> FakeComposer:
        self.find_calls += 1
        if self.find_calls <= self.failures:
            raise RuntimeError("simulated Discord UIA rerender")
        return self.composer


class BlockingExecutor:
    def __init__(self, first_result: bool) -> None:
        self.first_result = first_result
        self.calls: list[AnswerTask] = []
        self.first_started = threading.Event()
        self.release_first = threading.Event()
        self.second_finished = threading.Event()
        self.suppressed: set[str] = set()

    def observe_prompt(self, _signature: str | None, _fingerprint: str | None) -> None:
        return

    def is_suppressed(self, task: AnswerTask) -> bool:
        return (task.round_token or task.prompt_signature) in self.suppressed

    def service_orphan(self) -> None:
        return

    def execute(self, task: AnswerTask) -> bool:
        self.calls.append(task)
        if len(self.calls) == 1:
            self.first_started.set()
            self.release_first.wait(1.0)
            return self.first_result
        self.second_finished.set()
        return True


class FakeController:
    def __init__(self, composer: FakeComposer, enter_key: object) -> None:
        self.composer = composer
        self.enter_key = enter_key
        self.entered = False
        self.type_calls: list[str] = []
        self.selected_all = False
        self._modifier_held = False

    def type(self, character: str) -> None:
        self.type_calls.append(character)
        self.composer.text += character

    def press(self, key: object) -> None:
        if key == self.enter_key:
            self.entered = True
            self.composer.text = ""
            return
        if key == "a" and self._modifier_held:
            self.selected_all = True
            return
        if getattr(key, "name", None) == "backspace" and self.selected_all:
            # Select-all + Backspace erases the whole (focused) editor.
            self.composer.text = ""
            self.selected_all = False

    def release(self, _key: object) -> None:
        return

    @contextmanager
    def pressed(self, _key: object):
        self._modifier_held = True
        try:
            yield
        finally:
            self._modifier_held = False


class FakeTextInput:
    """Stands in for the SendInput batch writer: text lands in the composer."""

    def __init__(self, composer: FakeComposer) -> None:
        self.composer = composer
        self.sent: list[str] = []

    def send_text(self, text: str) -> int:
        self.sent.append(text)
        self.composer.text += text
        return len(text) * 2


class AmbiguousEnterController(FakeController):
    def press(self, key: object) -> None:
        if key == self.enter_key:
            self.entered = True
            self.composer.text = ""
            raise RuntimeError("simulated post-dispatch failure")
        super().press(key)


class StickyEnterController(FakeController):
    """Simulate Discord rejecting Enter and leaving the exact draft behind."""

    def press(self, key: object) -> None:
        if key == self.enter_key:
            self.entered = True
            return
        super().press(key)


class RecordingStatus:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def emit(self, phase: str, **fields) -> None:
        self.events.append((phase, fields))


def make_executor(
    active: ActivePromptState,
    composer: FakeComposer,
    *,
    write_mode: str = "type",
    auto_activate: bool = True,
    humanize: bool = False,
) -> SafeKeyboardExecutor:
    executor = SafeKeyboardExecutor.__new__(SafeKeyboardExecutor)
    executor._config = TypingConfig(
        enabled=True,
        expected_process_names=("Discord.exe",),
        expected_window_title_contains="Anime Soul - Discord",
        pre_delay_seconds=(0.0, 0.0),
        key_delay_seconds=(0.001, 0.001),
        draft_while_locked=True,
        verify_composer=True,
        auto_focus_composer=True,
        foreground_wait_timeout_seconds=0.75,
        composer_name_prefix="Message #💜anime-chat",
        composer_class_fragment="slateTextArea",
        enter_after_open_slack_seconds=0.0,
        composer_write_mode=write_mode,
        auto_activate_discord=auto_activate,
        activation_idle_ms=350,
        humanize_answers=humanize,
    )
    executor._activated_from = None
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
    executor._suppression_lock = threading.Lock()
    executor._suppressed_rounds = set()
    executor._status = NullStatus()
    executor._controller = FakeController(composer, executor._enter_key)
    executor._text_input = FakeTextInput(composer)
    executor._last_owned_answer = None
    return executor


class LiveRegressionTests(unittest.TestCase):
    def test_quiz_complete_marker_only_closes_the_final_card(self) -> None:
        self.assertFalse(AnimeTriviaAutomation._is_final_question(None))
        self.assertFalse(AnimeTriviaAutomation._is_final_question("Question 4/10"))
        self.assertTrue(AnimeTriviaAutomation._is_final_question("Question 10/10"))
        self.assertTrue(AnimeTriviaAutomation._is_final_question("3/3"))

    def test_live_card_after_quiz_completion_starts_a_new_session(self) -> None:
        # The old locked-Q1 latch ignored every live card of a quiz joined
        # mid-way (2026-09-02 7 AM: Q7-Q10 dropped). Any red/green card now
        # re-arms the worker; only grey cards are inert.
        self.assertFalse(hasattr(AnimeTriviaAutomation, "_is_new_quiz_start"))
        self.assertFalse(hasattr(AnimeTriviaAutomation, "_awaiting_quiz_start"))

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

    def test_red_clue_correction_retires_stale_task_without_suppressing_round(self) -> None:
        active = ActivePromptState()
        signature = "round:anime_title:1/10"
        active.update(signature, "locked", 1, "text:ocr-a")
        composer = FakeComposer()
        executor = make_executor(active, composer)
        first = AnswerTask(
            answer="Wrong Candidate",
            prompt_signature=signature,
            expected_answer_type="anime_title",
            question_label="1/10",
            detected_at=time.monotonic(),
            countdown_seconds=5.0,
            source="test",
            round_token="session-1:round-1",
            clue_fingerprint="text:ocr-a",
        )
        first_result: list[bool] = []
        thread = threading.Thread(
            target=lambda: first_result.append(executor.execute(first))
        )
        thread.start()
        time.sleep(0.05)
        active.update(signature, "locked", 2, "text:ocr-b")
        thread.join(1.0)
        self.assertEqual(first_result, [False])

        corrected = AnswerTask(
            **{
                **first.__dict__,
                "answer": "Correct Candidate",
                "clue_fingerprint": "text:ocr-b",
            }
        )
        self.assertFalse(executor.is_suppressed(corrected))
        active.update(signature, "ready", 3, "text:ocr-b")
        self.assertTrue(executor.execute(corrected))
        self.assertTrue(executor._controller.entered)

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
        observation.hint_text = "界人细"
        self.assertEqual(app._clue_fingerprint(observation), "visual:abc123")
        app._accessible_round = (observation.signature, "🥽 🦖 💻 🌐")
        self.assertTrue(app._clue_fingerprint(observation).startswith("semantic:"))

    def test_reveal_transaction_survives_noon_ocr_suffix_variation(self) -> None:
        app = AnimeTriviaAutomation.__new__(AnimeTriviaAutomation)
        app._accessible_round = None
        app._pending_round = None
        app._ocr = SimpleNamespace(recognize=lambda _frame: ())
        scene = SimpleNamespace(frame=None)
        base = SimpleNamespace(
            signature="round:anime_title:9/10",
            question_label="9/10",
            expected_answer_type="anime_title",
            prompt_kind="text",
            perceptual_hash=None,
            readiness="locked",
            hint_text=(
                '"Those who break the rules are scum, but those who abandon '
                'their friends are worse than scum." T'
            ),
        )
        app._arm_pending_round(base, scene)
        first = app._pending_round
        assert first is not None
        first.saw_ready = True
        corrected = SimpleNamespace(**{**base.__dict__, "hint_text": base.hint_text[:-1] + "1"})
        app._arm_pending_round(corrected, scene)
        self.assertIs(app._pending_round, first)
        self.assertTrue(app._pending_round.saw_ready)
        self.assertFalse(app._pending_round.clue.endswith((" T", " 1")))

    def test_text_fingerprint_ignores_noon_trailing_ocr_artifact(self) -> None:
        app = AnimeTriviaAutomation.__new__(AnimeTriviaAutomation)
        app._accessible_round = None
        base = SimpleNamespace(
            signature="round:anime_title:9/10",
            prompt_kind="text",
            perceptual_hash=None,
            hint_text=(
                '"Those who break the rules are scum, but those who abandon '
                'their friends are worse than scum."'
            ),
        )
        noisy_t = SimpleNamespace(**{**base.__dict__, "hint_text": base.hint_text + " T"})
        noisy_one = SimpleNamespace(
            **{**base.__dict__, "hint_text": base.hint_text + " 1"}
        )
        self.assertEqual(app._clue_fingerprint(base), app._clue_fingerprint(noisy_t))
        self.assertEqual(
            app._clue_fingerprint(base), app._clue_fingerprint(noisy_one)
        )

    def test_text_fingerprint_preserves_meaningful_terminal_tokens(self) -> None:
        app = AnimeTriviaAutomation.__new__(AnimeTriviaAutomation)
        app._accessible_round = None
        team_7 = SimpleNamespace(
            signature="round:character:2/10",
            prompt_kind="text",
            perceptual_hash=None,
            hint_text="A relaxed ninja became the teacher of Team 7.",
        )
        team_8 = SimpleNamespace(
            **{**team_7.__dict__, "hint_text": "A relaxed ninja became the teacher of Team 8."}
        )
        self.assertNotEqual(
            app._clue_fingerprint(team_7), app._clue_fingerprint(team_8)
        )

    def test_background_expected_window_is_unambiguous_and_read_only(self) -> None:
        guard = ForegroundWindowGuard(
            TypingConfig(
                expected_process_names=("Discord.exe",),
                expected_window_title_contains="Anime Soul - Discord",
            )
        )
        chrome = ForegroundWindow(3, 4, "chrome.exe", "Gemini")
        discord = ForegroundWindow(1, 2, "Discord.exe", "Anime Soul - Discord")
        guard.current = lambda: chrome  # type: ignore[method-assign]
        guard.visible_windows = lambda: (chrome, discord)  # type: ignore[method-assign]
        self.assertEqual(guard.expected_window(), discord)

        second = ForegroundWindow(5, 6, "Discord.exe", "Anime Soul - Discord")
        guard.visible_windows = lambda: (discord, second)  # type: ignore[method-assign]
        self.assertIsNone(guard.expected_window())

    def test_live_typing_requires_exact_composer_verification(self) -> None:
        config = AppConfig(
            capture=CaptureConfig(region=(0, 0, 1920, 1080), calibrated=True),
            typing=TypingConfig(enabled=True, verify_composer=False)
        )
        with self.assertRaisesRegex(ValueError, "verify_composer"):
            validate_config(config)

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

    def test_card_parser_handles_the_2026_09_02_discord_row_and_group_names(self) -> None:
        # Exact strings read from the live #anime-chat window on 2026-09-02.
        row_name = (
            "Anime Soul6:03 PMWednesday, September 2, 2026 6:03 PM Anime Guessing Game "
            "—  Round Over,Answer with the anime title — first correct guess in chat wins!"
            "Question 10/10 · round over(edited)Wednesday, September 2, 2026 6:04 PM"
        )
        group_name = (
            "Anime Soul App , 🎮 Anime Guessing Game — ⚪ Round Over 👦 📜 👺 🐱 🍶 , "
            "🎯 Answer with the anime title — first correct guess in chat wins! "
            "Question 10/10 · round over"
        )
        text_row = (
            "Anime Soul6:03 PMWednesday, September 2, 2026 6:03 PM Anime Guessing Game "
            '—  Round Over"You believe in aliens, but not ghosts?",Answer with the anime '
            "title — first correct guess in chat wins!Question 9/10 · round over(edited)"
        )
        # The row name has no emoji at all: an emoji card cannot be parsed from it.
        self.assertIsNone(DiscordQuestionLocator.parse_card_name(row_name))
        self.assertEqual(
            DiscordQuestionLocator.parse_card_name(group_name),
            ("👦 📜 👺 🐱 🍶", "anime_title", "10/10"),
        )
        # A text card still parses from the emoji-less row name via the fallback.
        self.assertEqual(
            DiscordQuestionLocator.parse_card_name(text_row),
            ('"You believe in aliens, but not ghosts?"', "anime_title", "9/10"),
        )

    def test_accessibility_reveal_parser_extracts_official_answer(self) -> None:
        accessible_name = (
            "Anime Soul6:04 PMTuesday, September 1, 2026 6:04 PM "
            "Correct! @player got it in 10.6s — "
            "the answer was Fate Zero. +50 AS Points (balance: 139,943)"
        )
        self.assertEqual(
            DiscordQuestionLocator.parse_reveal_answer(accessible_name),
            "Fate Zero",
        )
        self.assertTrue(
            DiscordQuestionLocator.is_official_reveal_name(accessible_name)
        )
        self.assertFalse(
            DiscordQuestionLocator.is_official_reveal_name(
                "random user6:04 PM Anime Soul Correct! the answer was Naruto."
            )
        )

    def test_composer_is_untouched_until_green_then_answer_is_atomic(self) -> None:
        active = ActivePromptState()
        signature = "round:anime_title:1/10"
        active.update(signature, "locked", 1)
        composer = FakeComposer()
        executor = make_executor(active, composer, write_mode="uia")
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
        while (
            not any(phase == "WAITING_GREEN" for phase, _ in recording.events)
            and time.monotonic() < deadline
        ):
            time.sleep(0.005)
        self.assertEqual(composer.text, "")
        self.assertEqual(composer.set_values, [])
        self.assertEqual(executor._controller.type_calls, [])
        self.assertEqual(executor._composer_locator.find_calls, 0)
        self.assertFalse(executor._controller.entered)
        active.update(signature, "ready", 2)
        thread.join(1.0)
        self.assertEqual(result, [True])
        self.assertTrue(executor._controller.entered)
        self.assertEqual(composer.set_values, [task.answer])
        self.assertEqual(executor._controller.type_calls, [])
        self.assertEqual(composer.text, "")
        phases = [phase for phase, _fields in recording.events]
        self.assertIn("DRAFTING", phases)
        self.assertIn("WAITING_GREEN", phases)
        self.assertIn("SUBMITTED", phases)

    def test_solved_answer_waits_for_manual_return_from_gemini(self) -> None:
        active = ActivePromptState()
        signature = "round:anime_title:1/10"
        active.update(signature, "ready", 1, "text:known")
        composer = FakeComposer()
        executor = make_executor(active, composer)
        guard = SwitchingGuard()
        executor._guard = guard
        recording = RecordingStatus()
        executor._status = recording
        task = AnswerTask(
            answer="Fruits Basket",
            prompt_signature=signature,
            expected_answer_type="anime_title",
            question_label="1/10",
            detected_at=time.monotonic(),
            countdown_seconds=0.0,
            source="history-cache",
            clue_fingerprint="text:known",
        )
        result: list[bool] = []
        thread = threading.Thread(target=lambda: result.append(executor.execute(task)))
        thread.start()
        deadline = time.monotonic() + 0.5
        while (
            not any(phase == "WAITING_DISCORD" for phase, _ in recording.events)
            and time.monotonic() < deadline
        ):
            time.sleep(0.005)
        self.assertEqual(executor._composer_locator.find_calls, 0)
        self.assertFalse(executor._controller.entered)
        guard.window = guard.discord
        thread.join(1.0)
        self.assertEqual(result, [True])
        self.assertEqual(executor._composer_locator.find_calls, 1)
        self.assertTrue(executor._controller.entered)

    def test_stale_foreground_task_never_touches_the_composer(self) -> None:
        active = ActivePromptState()
        signature = "round:anime_title:1/10"
        active.update(signature, "ready", 2, "text:new")
        composer = FakeComposer()
        executor = make_executor(active, composer)
        task = AnswerTask(
            answer="Fruits Basket",
            prompt_signature=signature,
            expected_answer_type="anime_title",
            question_label="1/10",
            detected_at=time.monotonic(),
            countdown_seconds=0.0,
            source="history-cache",
            clue_fingerprint="text:old",
        )
        self.assertFalse(executor.execute(task))
        self.assertEqual(executor._composer_locator.find_calls, 0)
        self.assertEqual(composer.focus_calls, 0)

    def test_wrong_channel_waits_until_exact_composer_appears(self) -> None:
        active = ActivePromptState()
        signature = "round:anime_title:1/10"
        active.update(signature, "ready", 1, "text:known")
        composer = FakeComposer()
        executor = make_executor(active, composer)
        locator = SwitchingLocator(composer)
        executor._composer_locator = locator
        task = AnswerTask(
            answer="Fruits Basket",
            prompt_signature=signature,
            expected_answer_type="anime_title",
            question_label="1/10",
            detected_at=time.monotonic(),
            countdown_seconds=0.0,
            source="history-cache",
            clue_fingerprint="text:known",
        )
        result: list[bool] = []
        thread = threading.Thread(target=lambda: result.append(executor.execute(task)))
        thread.start()
        deadline = time.monotonic() + 0.4
        while locator.find_calls == 0 and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertFalse(executor._controller.entered)
        locator.available = True
        thread.join(1.0)
        self.assertEqual(result, [True])
        self.assertTrue(executor._controller.entered)

    def test_existing_orphan_waits_for_discord_then_cleans_and_continues(self) -> None:
        active = ActivePromptState()
        signature = "round:anime_title:1/10"
        active.update(signature, "ready", 1, "text:known")
        composer = FakeComposer()
        composer.text = "Old Owned Draft"
        executor = make_executor(active, composer)
        executor._orphaned_draft = composer.text
        guard = SwitchingGuard()
        executor._guard = guard
        task = AnswerTask(
            answer="Fruits Basket",
            prompt_signature=signature,
            expected_answer_type="anime_title",
            question_label="1/10",
            detected_at=time.monotonic(),
            countdown_seconds=0.0,
            source="history-cache",
            clue_fingerprint="text:known",
        )
        result: list[bool] = []
        thread = threading.Thread(target=lambda: result.append(executor.execute(task)))
        thread.start()
        time.sleep(0.05)
        self.assertEqual(executor._composer_locator.find_calls, 0)
        guard.window = guard.discord
        thread.join(1.0)
        self.assertEqual(result, [True])
        self.assertIsNone(executor._orphaned_draft)
        self.assertTrue(executor._controller.entered)

    def test_foreground_timeout_is_terminal_for_the_round(self) -> None:
        active = ActivePromptState()
        signature = "round:anime_title:1/10"
        active.update(signature, "ready", 1, "text:known")
        composer = FakeComposer()
        executor = make_executor(active, composer)
        executor._config = TypingConfig(
            **{
                **executor._config.__dict__,
                "foreground_wait_timeout_seconds": 0.05,
            }
        )
        executor._guard = SwitchingGuard()
        task = AnswerTask(
            answer="Fruits Basket",
            prompt_signature=signature,
            expected_answer_type="anime_title",
            question_label="1/10",
            detected_at=time.monotonic(),
            countdown_seconds=0.0,
            source="history-cache",
            round_token="session-1:round-1",
            clue_fingerprint="text:known",
        )
        self.assertFalse(executor.execute(task))
        self.assertTrue(executor.is_suppressed(task))
        started = time.monotonic()
        self.assertFalse(executor.execute(task))
        self.assertLess(time.monotonic() - started, 0.02)

    def test_uncertain_green_wait_does_not_suppress_corrected_round(self) -> None:
        active = ActivePromptState()
        signature = "round:anime_title:1/10"
        active.update(signature, "ready", 1, "text:known")
        active.mark_uncertain(2)
        composer = FakeComposer()
        executor = make_executor(active, composer)
        executor._config = TypingConfig(
            **{
                **executor._config.__dict__,
                "foreground_wait_timeout_seconds": 0.05,
            }
        )
        executor._guard = SwitchingGuard()
        task = AnswerTask(
            answer="Fruits Basket",
            prompt_signature=signature,
            expected_answer_type="anime_title",
            question_label="1/10",
            detected_at=time.monotonic(),
            countdown_seconds=0.0,
            source="history-cache",
            round_token="session-1:round-1",
            clue_fingerprint="text:known",
        )
        self.assertFalse(executor.execute(task))
        self.assertFalse(executor.is_suppressed(task))

    def test_transient_uia_rerender_keeps_solved_task_pending(self) -> None:
        active = ActivePromptState()
        signature = "round:anime_title:1/10"
        active.update(signature, "ready", 1, "text:known")
        composer = FakeComposer()
        executor = make_executor(active, composer)
        locator = FlakyLocator(composer)
        executor._composer_locator = locator
        task = AnswerTask(
            answer="Fruits Basket",
            prompt_signature=signature,
            expected_answer_type="anime_title",
            question_label="1/10",
            detected_at=time.monotonic(),
            countdown_seconds=0.0,
            source="history-cache",
            clue_fingerprint="text:known",
        )
        self.assertTrue(executor.execute(task))
        self.assertGreaterEqual(locator.find_calls, 3)
        self.assertTrue(executor._controller.entered)

    def test_manual_text_during_pre_delay_suppresses_the_round(self) -> None:
        active = ActivePromptState()
        signature = "round:anime_title:1/10"
        active.update(signature, "ready", 1, "text:known")
        composer = FakeComposer()
        executor = make_executor(active, composer)
        executor._config = TypingConfig(
            **{
                **executor._config.__dict__,
                "pre_delay_seconds": (0.2, 0.2),
            }
        )
        task = AnswerTask(
            answer="Fruits Basket",
            prompt_signature=signature,
            expected_answer_type="anime_title",
            question_label="1/10",
            detected_at=time.monotonic(),
            countdown_seconds=0.0,
            source="history-cache",
            round_token="session-1:round-1",
            clue_fingerprint="text:known",
        )
        result: list[bool] = []
        thread = threading.Thread(target=lambda: result.append(executor.execute(task)))
        thread.start()
        time.sleep(0.05)
        composer.text = "my manual answer"
        thread.join(1.0)
        self.assertEqual(result, [False])
        self.assertEqual(composer.text, "my manual answer")
        self.assertTrue(executor.is_suppressed(task))
        self.assertFalse(executor._controller.entered)

    def test_manual_text_at_atomic_commit_boundary_suppresses_round(self) -> None:
        active = ActivePromptState()
        signature = "round:anime_title:1/10"
        active.update(signature, "ready", 1, "text:known")
        composer = FakeComposer()
        executor = make_executor(active, composer, write_mode="uia")
        task = AnswerTask(
            answer="Fruits Basket",
            prompt_signature=signature,
            expected_answer_type="anime_title",
            question_label="1/10",
            detected_at=time.monotonic(),
            countdown_seconds=0.0,
            source="history-cache",
            round_token="session-1:round-1",
            clue_fingerprint="text:known",
        )

        def inject_at_boundary(*_args, **_kwargs) -> str:
            composer.text = "my manual answer"
            return "ready"

        executor._wait_pre_delay = inject_at_boundary  # type: ignore[method-assign]
        self.assertFalse(executor.execute(task))
        self.assertEqual(composer.text, "my manual answer")
        self.assertTrue(executor.is_suppressed(task))
        self.assertFalse(executor._controller.entered)

    def test_round_close_cancels_foreground_wait_without_ui_access(self) -> None:
        active = ActivePromptState()
        signature = "round:character:1/10"
        active.update(signature, "ready", 1, "text:known")
        composer = FakeComposer()
        executor = make_executor(active, composer)
        executor._guard = SwitchingGuard()
        task = AnswerTask(
            answer="Sakura Kinomoto",
            prompt_signature=signature,
            expected_answer_type="character",
            question_label="1/10",
            detected_at=time.monotonic(),
            countdown_seconds=0.0,
            source="history-cache",
            clue_fingerprint="text:known",
        )
        result: list[bool] = []
        thread = threading.Thread(target=lambda: result.append(executor.execute(task)))
        thread.start()
        time.sleep(0.05)
        active.update(signature, "closed", 2, "text:known")
        thread.join(1.0)
        self.assertEqual(result, [False])
        self.assertEqual(executor._composer_locator.find_calls, 0)
        self.assertFalse(executor._controller.entered)

    def test_manual_composer_text_wins_when_discord_returns(self) -> None:
        active = ActivePromptState()
        signature = "round:anime_title:1/10"
        active.update(signature, "ready", 1, "text:known")
        composer = FakeComposer()
        executor = make_executor(active, composer)
        guard = SwitchingGuard()
        executor._guard = guard
        task = AnswerTask(
            answer="Fruits Basket",
            prompt_signature=signature,
            expected_answer_type="anime_title",
            question_label="1/10",
            detected_at=time.monotonic(),
            countdown_seconds=0.0,
            source="history-cache",
            round_token="session-1:round-1",
            clue_fingerprint="text:known",
        )
        result: list[bool] = []
        thread = threading.Thread(target=lambda: result.append(executor.execute(task)))
        thread.start()
        time.sleep(0.05)
        composer.text = "my manual answer"
        guard.window = guard.discord
        thread.join(1.0)
        self.assertEqual(result, [False])
        self.assertEqual(composer.text, "my manual answer")
        self.assertFalse(executor._controller.entered)
        self.assertTrue(executor.is_suppressed(task))
        corrected = AnswerTask(
            **{**task.__dict__, "clue_fingerprint": "text:corrected"}
        )
        executor.observe_prompt(signature, corrected.clue_fingerprint)
        self.assertTrue(executor.is_suppressed(corrected))

    def test_corrected_answer_is_sent_as_a_follow_up_guess_after_the_gap(self) -> None:
        active = ActivePromptState()
        stop_event = threading.Event()
        executor = BlockingExecutor(first_result=True)
        dispatcher = AnswerDispatcher(
            executor, active, stop_event, guess_gap_seconds=0.05  # type: ignore[arg-type]
        )
        dispatcher.start()
        signature = "round:anime_title:1/10"
        dispatcher.observe_prompt(signature, "ready", 1, "text:a")
        first = AnswerTask(
            answer="First",
            prompt_signature=signature,
            expected_answer_type="anime_title",
            question_label="1/10",
            detected_at=time.monotonic(),
            countdown_seconds=0.0,
            source="test",
            round_token="session-1:round-1",
            clue_fingerprint="text:a",
        )
        self.assertTrue(dispatcher.submit(first))
        self.assertTrue(executor.first_started.wait(0.5))
        dispatcher.observe_prompt(signature, "ready", 2, "text:b")
        second = AnswerTask(**{**first.__dict__, "answer": "Second", "clue_fingerprint": "text:b"})
        self.assertTrue(dispatcher.submit(second))
        executor.release_first.set()
        self.assertTrue(executor.second_finished.wait(1.0))
        stop_event.set()
        dispatcher.join(1.0)
        # Wrong guesses are free: a different answer for the same open round is
        # typed as guess 2 once the gap has elapsed.
        self.assertEqual([task.answer for task in executor.calls], ["First", "Second"])

    def test_same_answer_is_never_resent_and_ladder_is_capped(self) -> None:
        active = ActivePromptState()
        stop_event = threading.Event()
        executor = BlockingExecutor(first_result=True)
        dispatcher = AnswerDispatcher(
            executor,  # type: ignore[arg-type]
            active,
            stop_event,
            max_guesses_per_round=2,
            guess_gap_seconds=0.0,
        )
        dispatcher.start()
        signature = "round:character:3/10"
        dispatcher.observe_prompt(signature, "ready", 1, "text:a")
        first = AnswerTask(
            answer="Fuu Kasumi",
            prompt_signature=signature,
            expected_answer_type="character",
            question_label="3/10",
            detected_at=time.monotonic(),
            countdown_seconds=0.0,
            source="test",
            round_token="session-1:round-3",
            clue_fingerprint="text:a",
        )
        self.assertTrue(dispatcher.submit(first))
        self.assertTrue(executor.first_started.wait(0.5))
        duplicate = AnswerTask(**{**first.__dict__, "answer": "fuu kasumi"})
        self.assertFalse(dispatcher.submit(duplicate))
        second = AnswerTask(**{**first.__dict__, "answer": "Mugen", "guess_index": 2})
        self.assertTrue(dispatcher.submit(second))
        third = AnswerTask(**{**first.__dict__, "answer": "Jin", "guess_index": 3})
        self.assertFalse(dispatcher.submit(third))
        executor.release_first.set()
        self.assertTrue(executor.second_finished.wait(1.0))
        stop_event.set()
        dispatcher.join(1.0)
        self.assertEqual(
            [task.answer for task in executor.calls], ["Fuu Kasumi", "Mugen"]
        )

    def test_follow_up_guess_is_dropped_when_the_round_closes(self) -> None:
        active = ActivePromptState()
        stop_event = threading.Event()
        executor = BlockingExecutor(first_result=True)
        dispatcher = AnswerDispatcher(
            executor, active, stop_event, guess_gap_seconds=0.3  # type: ignore[arg-type]
        )
        dispatcher.start()
        signature = "round:anime_title:4/10"
        dispatcher.observe_prompt(signature, "ready", 1, "text:a")
        first = AnswerTask(
            answer="Bleach",
            prompt_signature=signature,
            expected_answer_type="anime_title",
            question_label="4/10",
            detected_at=time.monotonic(),
            countdown_seconds=0.0,
            source="test",
            round_token="session-1:round-4",
            clue_fingerprint="text:a",
        )
        self.assertTrue(dispatcher.submit(first))
        self.assertTrue(executor.first_started.wait(0.5))
        second = AnswerTask(**{**first.__dict__, "answer": "Naruto", "guess_index": 2})
        self.assertTrue(dispatcher.submit(second))
        executor.release_first.set()
        time.sleep(0.05)
        dispatcher.observe_prompt(signature, "closed", 2, "text:a")
        time.sleep(0.4)
        stop_event.set()
        dispatcher.join(1.0)
        self.assertEqual([task.answer for task in executor.calls], ["Bleach"])

    def test_idle_operator_gets_discord_raised_and_focus_returned(self) -> None:
        active = ActivePromptState()
        signature = "round:anime_title:1/10"
        active.update(signature, "ready", 1, "text:known")
        composer = FakeComposer()
        executor = make_executor(active, composer)
        guard = SwitchingGuard(idle_ms=2000)
        executor._guard = guard
        task = AnswerTask(
            answer="Fruits Basket",
            prompt_signature=signature,
            expected_answer_type="anime_title",
            question_label="1/10",
            detected_at=time.monotonic(),
            countdown_seconds=0.0,
            source="history-cache",
            clue_fingerprint="text:known",
        )
        self.assertTrue(executor.execute(task))
        self.assertTrue(executor._controller.entered)
        # Discord was raised for the send, then Chrome got focus back.
        self.assertEqual(guard.activations, [guard.discord.hwnd, guard.chrome.hwnd])
        self.assertEqual(guard.window, guard.chrome)

    def test_active_operator_is_never_interrupted_by_activation(self) -> None:
        active = ActivePromptState()
        signature = "round:anime_title:1/10"
        active.update(signature, "ready", 1, "text:known")
        composer = FakeComposer()
        executor = make_executor(active, composer)
        executor._config = TypingConfig(
            **{**executor._config.__dict__, "foreground_wait_timeout_seconds": 0.2}
        )
        guard = SwitchingGuard(idle_ms=0)
        executor._guard = guard
        task = AnswerTask(
            answer="Fruits Basket",
            prompt_signature=signature,
            expected_answer_type="anime_title",
            question_label="1/10",
            detected_at=time.monotonic(),
            countdown_seconds=0.0,
            source="history-cache",
            clue_fingerprint="text:known",
        )
        self.assertFalse(executor.execute(task))
        self.assertEqual(guard.activations, [])
        self.assertFalse(executor._controller.entered)

    def test_atomic_commit_stages_full_answer_after_green_then_enters(self) -> None:
        active = ActivePromptState()
        signature = "round:character:2/10"
        active.update(signature, "locked", 1)
        composer = FakeComposer()
        executor = make_executor(active, composer, write_mode="uia")
        task = AnswerTask(
            answer="Fuu Kasumi",
            prompt_signature=signature,
            expected_answer_type="character",
            question_label="2/10",
            detected_at=time.monotonic(),
            countdown_seconds=5.0,
            source="gemini-3.8-structured",
        )
        result: list[bool] = []
        thread = threading.Thread(target=lambda: result.append(executor.execute(task)))
        thread.start()
        time.sleep(0.05)
        self.assertEqual(executor._controller.type_calls, [])
        active.update(signature, "ready", 2)
        thread.join(1.0)
        self.assertEqual(result, [True])
        self.assertEqual(executor._controller.type_calls, [])
        self.assertEqual(composer.set_values, [task.answer])
        self.assertTrue(executor._controller.entered)
        self.assertEqual(composer.text, "")

    def test_atomic_commit_waits_for_discord_uia_value_acknowledgement(self) -> None:
        active = ActivePromptState()
        signature = "round:character:2/10"
        active.update(signature, "ready", 1)
        composer = DelayedAtomicComposer()
        executor = make_executor(active, composer, write_mode="uia")
        task = AnswerTask(
            answer="Benedict Blue",
            prompt_signature=signature,
            expected_answer_type="character",
            question_label="2/10",
            detected_at=time.monotonic(),
            countdown_seconds=0.0,
            source="history-cache",
        )

        self.assertTrue(executor.execute(task))
        self.assertEqual(composer.set_values, [task.answer])
        self.assertEqual(executor._controller.type_calls, [])
        self.assertTrue(executor._controller.entered)
        self.assertEqual(composer.text, "")

    def test_only_the_two_known_composer_writers_are_accepted(self) -> None:
        for mode in ("type", "uia"):
            validate_config(
                AppConfig(
                    capture=CaptureConfig(region=(0, 0, 1920, 1080), calibrated=True),
                    typing=TypingConfig(composer_write_mode=mode),
                )
            )
        with self.assertRaisesRegex(ValueError, "composer_write_mode"):
            validate_config(
                AppConfig(
                    capture=CaptureConfig(region=(0, 0, 1920, 1080), calibrated=True),
                    typing=TypingConfig(composer_write_mode="clipboard"),
                )
            )

    def test_answers_are_typed_in_human_form(self) -> None:
        from anime_trivia_automation.utils import humanize_answer

        self.assertEqual(humanize_answer("Girls' Last Tour"), "girls last tour")
        self.assertEqual(humanize_answer("Steins;Gate"), "steins gate")
        self.assertEqual(humanize_answer("One-Punch Man"), "one punch man")
        self.assertEqual(humanize_answer("86 Eighty-Six"), "86 eighty six")
        self.assertEqual(humanize_answer("Natsume's Book of Friends"), "natsumes book of friends")
        self.assertEqual(humanize_answer("...", strip_punctuation=True), "...")

        active = ActivePromptState()
        signature = "round:anime_title:8/10"
        active.update(signature, "ready", 1, "text:girls")
        composer = FakeComposer()
        executor = make_executor(active, composer, humanize=True)
        task = AnswerTask(
            answer="Girls' Last Tour",
            prompt_signature=signature,
            expected_answer_type="anime_title",
            question_label="8/10",
            detected_at=time.monotonic(),
            countdown_seconds=0.0,
            source="antigravity-account",
            clue_fingerprint="text:girls",
        )
        self.assertTrue(executor.execute(task))
        self.assertEqual(executor._text_input.sent, ["girls last tour"])

    def test_stuck_owned_answer_is_cleared_before_the_next_round(self) -> None:
        # 2026-09-02 18:00: Q3's text stayed in the box after Enter, so Q4, Q5,
        # and Q9 were refused as "manual text". Our own leftover is ours to clear.
        active = ActivePromptState()
        signature = "round:character:4/10"
        active.update(signature, "ready", 1, "text:bleach")
        composer = FakeComposer()
        composer.text = "Hyoma Chigiri"
        executor = make_executor(active, composer)
        executor._last_owned_answer = "Hyoma Chigiri"
        task = AnswerTask(
            answer="Bleach",
            prompt_signature=signature,
            expected_answer_type="anime_title",
            question_label="4/10",
            detected_at=time.monotonic(),
            countdown_seconds=0.0,
            source="antigravity-account",
            clue_fingerprint="text:bleach",
        )
        self.assertTrue(executor.execute(task))
        self.assertTrue(executor._controller.entered)
        self.assertEqual(executor._text_input.sent, ["Bleach"])
        self.assertEqual(composer.text, "")

    def test_human_text_in_the_box_is_never_cleared(self) -> None:
        active = ActivePromptState()
        signature = "round:character:4/10"
        active.update(signature, "ready", 1, "text:bleach")
        composer = FakeComposer()
        composer.text = "my own guess"
        executor = make_executor(active, composer)
        executor._last_owned_answer = "Hyoma Chigiri"
        executor._config = TypingConfig(
            **{**executor._config.__dict__, "foreground_wait_timeout_seconds": 0.2}
        )
        task = AnswerTask(
            answer="Bleach",
            prompt_signature=signature,
            expected_answer_type="anime_title",
            question_label="4/10",
            detected_at=time.monotonic(),
            countdown_seconds=0.0,
            source="antigravity-account",
            clue_fingerprint="text:bleach",
        )
        self.assertFalse(executor.execute(task))
        self.assertFalse(executor._controller.entered)
        self.assertEqual(composer.text, "my own guess")

    def test_second_enter_and_cleanup_when_discord_ignores_the_first(self) -> None:
        active = ActivePromptState()
        signature = "round:anime_title:7/10"
        active.update(signature, "ready", 1, "text:steins")
        composer = FakeComposer()
        executor = make_executor(active, composer)

        class DeafController(FakeController):
            def __init__(self, composer, enter_key):
                super().__init__(composer, enter_key)
                self.enter_presses = 0

            def press(self, key):
                if key == self.enter_key:
                    self.enter_presses += 1
                    self.entered = True
                    return  # Discord ignores Enter; text stays
                super().press(key)

        executor._controller = DeafController(composer, executor._enter_key)
        task = AnswerTask(
            answer="Steins;Gate",
            prompt_signature=signature,
            expected_answer_type="anime_title",
            question_label="7/10",
            detected_at=time.monotonic(),
            countdown_seconds=0.0,
            source="antigravity-account",
            clue_fingerprint="text:steins",
        )
        self.assertTrue(executor.execute(task))
        self.assertEqual(executor._controller.enter_presses, 2)
        # Nothing is left behind to block the next round.
        self.assertEqual(composer.text, "")

    def test_latest_only_dispatch_replaces_stale_fingerprint_variants(self) -> None:
        active = ActivePromptState()
        stop_event = threading.Event()
        executor = BlockingExecutor(first_result=False)
        dispatcher = AnswerDispatcher(executor, active, stop_event)  # type: ignore[arg-type]
        dispatcher.start()
        signature = "round:anime_title:1/10"
        base = AnswerTask(
            answer="A",
            prompt_signature=signature,
            expected_answer_type="anime_title",
            question_label="1/10",
            detected_at=time.monotonic(),
            countdown_seconds=0.0,
            source="test",
            round_token="session-1:round-1",
            clue_fingerprint="text:a",
        )
        dispatcher.observe_prompt(signature, "ready", 1, "text:a")
        self.assertTrue(dispatcher.submit(base))
        self.assertTrue(executor.first_started.wait(0.5))
        for generation, letter in enumerate("bcdef", start=2):
            fingerprint = f"text:{letter}"
            dispatcher.observe_prompt(signature, "ready", generation, fingerprint)
            task = AnswerTask(
                **{**base.__dict__, "answer": letter.upper(), "clue_fingerprint": fingerprint}
            )
            self.assertTrue(dispatcher.submit(task))
        executor.release_first.set()
        self.assertTrue(executor.second_finished.wait(0.5))
        stop_event.set()
        dispatcher.join(1.0)
        self.assertEqual([task.answer for task in executor.calls], ["A", "F"])

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
        time.sleep(0.05)
        self.assertEqual(composer.text, "")
        composer.text = "USER"
        active.update(signature, "ready", 2)
        thread.join(1.0)
        self.assertEqual(result, [False])
        self.assertEqual(composer.text, "USER")
        self.assertEqual(composer.set_values, [])
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
        time.sleep(0.05)
        self.assertEqual(composer.text, "")
        active.update(signature, "ready", 2)
        thread.join(1.0)
        self.assertEqual(result, [False])
        self.assertFalse(executor._controller.entered)
        self.assertEqual(composer.text, "")
        self.assertEqual(composer.set_values, [])

    def test_round_close_after_atomic_commit_clears_complete_value(self) -> None:
        active = ActivePromptState()
        signature = "round:anime_title:1/10"
        active.update(signature, "ready", 1, "text:known")
        composer = FakeComposer()
        executor = make_executor(active, composer, write_mode="uia")
        executor._config = TypingConfig(
            **{
                **executor._config.__dict__,
                "enter_after_open_slack_seconds": 0.15,
            }
        )
        task = AnswerTask(
            answer="Fruits Basket",
            prompt_signature=signature,
            expected_answer_type="anime_title",
            question_label="1/10",
            detected_at=time.monotonic(),
            countdown_seconds=0.0,
            source="history-cache",
            clue_fingerprint="text:known",
        )
        result: list[bool] = []
        thread = threading.Thread(target=lambda: result.append(executor.execute(task)))
        thread.start()
        deadline = time.monotonic() + 1.0
        while not composer.set_values and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertEqual(composer.text, task.answer)
        active.update(signature, "closed", 2, "text:known")
        thread.join(1.0)
        self.assertEqual(result, [False])
        self.assertFalse(executor._controller.entered)
        self.assertEqual(composer.text, "")
        self.assertEqual(composer.set_values, [task.answer])

    def test_manual_edit_after_atomic_commit_blocks_enter_without_erasure(self) -> None:
        active = ActivePromptState()
        signature = "round:anime_title:1/10"
        active.update(signature, "ready", 1, "text:known")
        composer = FakeComposer()
        executor = make_executor(active, composer, write_mode="uia")
        executor._config = TypingConfig(
            **{
                **executor._config.__dict__,
                "enter_after_open_slack_seconds": 0.15,
            }
        )
        task = AnswerTask(
            answer="Fruits Basket",
            prompt_signature=signature,
            expected_answer_type="anime_title",
            question_label="1/10",
            detected_at=time.monotonic(),
            countdown_seconds=0.0,
            source="history-cache",
            round_token="session-1:round-1",
            clue_fingerprint="text:known",
        )
        result: list[bool] = []
        thread = threading.Thread(target=lambda: result.append(executor.execute(task)))
        thread.start()
        deadline = time.monotonic() + 1.0
        while not composer.set_values and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertEqual(composer.text, task.answer)
        composer.text += " USER"
        thread.join(1.0)
        self.assertEqual(result, [False])
        self.assertEqual(composer.text, task.answer + " USER")
        self.assertFalse(executor._controller.entered)
        self.assertTrue(executor.is_suppressed(task))

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

    def test_rejected_enter_clears_owned_draft_before_next_round(self) -> None:
        active = ActivePromptState()
        signature = "round:anime_title:1/10"
        active.update(signature, "ready", 1, "text:known")
        composer = FakeComposer()
        executor = make_executor(active, composer)
        executor._controller = StickyEnterController(composer, executor._enter_key)
        task = AnswerTask(
            answer="Fruits Basket",
            prompt_signature=signature,
            expected_answer_type="anime_title",
            question_label="1/10",
            detected_at=time.monotonic(),
            countdown_seconds=0.0,
            source="history-cache",
            round_token="session-1:round-1",
            clue_fingerprint="text:known",
        )

        with self.assertLogs(
            "anime_trivia_automation.typing", level="WARNING"
        ):
            self.assertTrue(executor.execute(task))

        self.assertTrue(executor._controller.entered)
        self.assertEqual(composer.text, "")
        self.assertIsNone(executor._orphaned_draft)

    def test_transient_commit_uncertainty_retries_same_green_task(self) -> None:
        active = ActivePromptState()
        signature = "round:anime_title:1/10"
        active.update(signature, "ready", 1, "text:future-rules")
        composer = FakeComposer()
        executor = make_executor(active, composer)
        task = AnswerTask(
            answer="Chainsaw Man",
            prompt_signature=signature,
            expected_answer_type="anime_title",
            question_label="1/10",
            detected_at=time.monotonic(),
            countdown_seconds=0.0,
            source="antigravity-account-3.8-low",
            round_token="session-1:round-1",
            clue_fingerprint="text:future-rules",
        )
        attempts: list[str] = []

        def commit(_task, answer, _window, _composer):
            attempts.append(answer)
            if len(attempts) == 1:
                return "stale"
            composer.text = answer
            return "committed"

        executor._commit_complete_answer = commit

        self.assertTrue(executor.execute(task))
        self.assertEqual(attempts, ["Chainsaw Man", "Chainsaw Man"])
        self.assertTrue(executor._controller.entered)
        self.assertEqual(composer.text, "")

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

    def test_atomic_write_exception_after_full_value_can_be_verified(self) -> None:
        active = ActivePromptState()
        signature = "round:character:1/10"
        active.update(signature, "ready", 1)
        composer = AmbiguousAtomicComposer()
        executor = make_executor(active, composer, write_mode="uia")
        task = AnswerTask(
            answer="Rei Kiriyama",
            prompt_signature=signature,
            expected_answer_type="character",
            question_label="1/10",
            detected_at=time.monotonic(),
            countdown_seconds=0.0,
            source="history-cache",
        )
        with self.assertLogs(
            "anime_trivia_automation.typing", level="WARNING"
        ):
            self.assertTrue(executor.execute(task))
        self.assertEqual(composer.set_values, [task.answer])
        self.assertEqual(executor._controller.type_calls, [])
        self.assertIsNone(executor._orphaned_draft)
        self.assertTrue(executor._controller.entered)


if __name__ == "__main__":
    unittest.main()
