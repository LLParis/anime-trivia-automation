from __future__ import annotations

import asyncio
import threading
import time
import unittest
from types import SimpleNamespace

from anime_trivia_automation.app import (
    AnimeTriviaAutomation,
    AsyncResolutionRound,
    PendingRound,
    ProviderResolution,
    ResolutionRequest,
)
from anime_trivia_automation.status import NullStatus
from anime_trivia_automation.typing import ActivePromptState


class RecordingDispatcher:
    def __init__(self) -> None:
        self.tasks = []

    def submit(self, task) -> bool:
        self.tasks.append(task)
        return True


def make_async_app(signature: str, fingerprint: str, round_token: str):
    app = AnimeTriviaAutomation.__new__(AnimeTriviaAutomation)
    app._active_prompt = ActivePromptState()
    app._active_prompt.update(signature, "locked", 1, fingerprint)
    app._active_status_token = round_token
    app._active_status_closed = False
    app._active_resolution_key = (round_token, fingerprint)
    app._quiz_ended = False
    app._ephemeral_answer = None
    app._status_resolution = {}
    app._status = NullStatus()
    app._dispatcher = RecordingDispatcher()
    scene = SimpleNamespace(detected_at=time.monotonic())
    observation = SimpleNamespace(
        signature=signature,
        expected_answer_type="anime_title",
        question_label="1/10",
        countdown_seconds=5.0,
        readiness="locked",
        scene=scene,
    )
    request = ResolutionRequest(
        key=(round_token, fingerprint),
        round_token=round_token,
        signature=signature,
        clue_fingerprint=fingerprint,
        clue='"Who decides limits? And based on what?"',
        observation=observation,
        started_at=time.perf_counter(),
    )
    state = AsyncResolutionRound(
        request=request,
        providers={"qwen"},
        pending={"qwen"},
        ocr_ms=70.0,
        extract_ms=0.0,
        lookup_ms=1.0,
    )
    app._resolution_rounds = {request.key: state}
    app._stop_event = threading.Event()
    return app, state


class AsyncResolutionTests(unittest.TestCase):
    def test_stop_cancels_long_gemini_coroutine_without_visual_timeout_delay(self) -> None:
        app = AnimeTriviaAutomation.__new__(AnimeTriviaAutomation)
        app._stop_event = threading.Event()
        app._gemini_loop = None
        app._gemini_loop_thread = None
        app._gemini_loop_ready = threading.Event()
        app._start_gemini_event_loop()
        errors: list[str] = []

        def run_long_call() -> None:
            try:
                app._run_gemini_coroutine(asyncio.sleep(30.0), timeout=31.0)
            except Exception as exc:
                errors.append(type(exc).__name__)

        worker = threading.Thread(target=run_long_call)
        started = time.monotonic()
        worker.start()
        time.sleep(0.05)
        app._stop_event.set()
        worker.join(1.0)
        elapsed = time.monotonic() - started
        self.assertFalse(worker.is_alive())
        self.assertLess(elapsed, 1.0)
        self.assertEqual(errors, ["RuntimeError"])

        close_ran: list[bool] = []

        async def close_marker() -> None:
            close_ran.append(True)

        app._run_gemini_coroutine(
            close_marker(),
            timeout=1.0,
            cancel_on_stop=False,
        )
        self.assertEqual(close_ran, [True])

        assert app._gemini_loop is not None
        app._gemini_loop.call_soon_threadsafe(app._gemini_loop.stop)
        assert app._gemini_loop_thread is not None
        app._gemini_loop_thread.join(1.0)

    def test_old_reveal_newly_visible_above_card_cannot_be_promoted(self) -> None:
        pending = PendingRound(
            signature="round:anime_title:10/10",
            question_label="10/10",
            expected_answer_type="anime_title",
            prompt_kind="visual",
            clue="🏆 ⚔️ 🪄 👑",
            clue_fingerprint="semantic:🏆 ⚔️ 🪄 👑",
            baseline_reveal_ids={"baseline-visible"},
            saw_ready=True,
        )
        records = [
            ("naruto", "Naruto", "baseline-visible", 1300),
            ("bleach", "Bleach", "old-item-newly-virtualized", 900),
            ("fate zero", "Fate Zero", "new-result", 1350),
        ]
        self.assertEqual(
            AnimeTriviaAutomation._select_new_semantic_answers(
                pending, records, card_screen_bottom=1200
            ),
            {"fate zero": "Fate Zero"},
        )

    def test_result_survives_uncertain_red_to_green_transition(self) -> None:
        signature = "round:anime_title:1/10"
        fingerprint = "text:who decides limits and based on what"
        round_token = "session-1:round-1:1/10"
        app, state = make_async_app(signature, fingerprint, round_token)

        # This is the exact 6 PM race: capture sees the red-to-green change
        # while the provider completes. The answer must be retained, not
        # synchronously wait for the scene thread to revalidate itself.
        app._active_prompt.mark_uncertain(2)
        app._accept_resolution_result(
            ProviderResolution(
                key=state.request.key,
                provider="qwen",
                source="qwen38-retrieval-consensus",
                answer="One-Punch Man",
                confidence=0.99,
                elapsed_ms=3825.0,
            )
        )
        self.assertEqual(state.candidate.answer, "One-Punch Man")
        self.assertFalse(state.queued)
        self.assertEqual(app._dispatcher.tasks, [])

        app._active_prompt.update(signature, "ready", 3, fingerprint)
        self.assertTrue(app._queue_resolution_candidate(state))
        self.assertTrue(state.queued)
        self.assertEqual(len(app._dispatcher.tasks), 1)
        self.assertEqual(app._dispatcher.tasks[0].answer, "One-Punch Man")

    def test_late_result_from_an_old_round_is_discarded(self) -> None:
        signature = "round:anime_title:1/10"
        fingerprint = "text:old clue"
        round_token = "session-1:round-1:1/10"
        app, state = make_async_app(signature, fingerprint, round_token)
        app._active_status_token = "session-1:round-2:2/10"
        app._active_resolution_key = (app._active_status_token, "text:new clue")

        app._accept_resolution_result(
            ProviderResolution(
                key=state.request.key,
                provider="qwen",
                source="qwen38-retrieval-consensus",
                answer="Wrong Round",
                confidence=1.0,
                elapsed_ms=100.0,
            )
        )
        self.assertTrue(state.retired)
        self.assertIsNone(state.candidate)
        self.assertEqual(app._dispatcher.tasks, [])

    def test_same_round_fingerprint_can_return_and_use_retained_result(self) -> None:
        signature = "round:anime_title:1/10"
        fingerprint_a = "text:stable clue a"
        fingerprint_b = "text:temporary ocr b"
        round_token = "session-1:round-1:1/10"
        app, state = make_async_app(signature, fingerprint_a, round_token)

        app._active_prompt.update(signature, "locked", 2, fingerprint_b)
        app._active_resolution_key = (round_token, fingerprint_b)
        app._accept_resolution_result(
            ProviderResolution(
                key=state.request.key,
                provider="qwen",
                source="qwen38-retrieval-consensus",
                answer="One-Punch Man",
                confidence=0.99,
                elapsed_ms=3800.0,
            )
        )
        self.assertFalse(state.retired)
        self.assertEqual(state.candidate.answer, "One-Punch Man")
        self.assertEqual(app._dispatcher.tasks, [])

        app._active_prompt.update(signature, "ready", 3, fingerprint_a)
        app._active_resolution_key = state.request.key
        self.assertTrue(app._queue_resolution_candidate(state))
        self.assertEqual(app._dispatcher.tasks[0].answer, "One-Punch Man")

    def test_provider_disagreement_never_queues_first_answer(self) -> None:
        signature = "round:anime_title:1/10"
        fingerprint = "text:ambiguous clue"
        round_token = "session-1:round-1:1/10"
        app, state = make_async_app(signature, fingerprint, round_token)
        state.providers = {"gemini", "qwen"}
        state.pending = {"gemini", "qwen"}

        app._accept_resolution_result(
            ProviderResolution(
                key=state.request.key,
                provider="gemini",
                source="gemini-3.7-structured",
                answer="Wrong Answer",
                confidence=0.99,
                elapsed_ms=1500.0,
            )
        )
        self.assertIsNone(state.candidate)
        self.assertEqual(app._dispatcher.tasks, [])

        app._accept_resolution_result(
            ProviderResolution(
                key=state.request.key,
                provider="qwen",
                source="qwen38-retrieval-consensus",
                answer="Correct Answer",
                confidence=0.99,
                elapsed_ms=3800.0,
            )
        )
        self.assertTrue(state.conflicted)
        self.assertIsNone(state.candidate)
        self.assertEqual(app._dispatcher.tasks, [])


if __name__ == "__main__":
    unittest.main()
