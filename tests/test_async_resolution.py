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
    app._config = SimpleNamespace(
        antigravity=SimpleNamespace(enabled=False),
        typing=SimpleNamespace(max_guesses_per_round=3, max_answer_characters=96),
    )
    app._antigravity = SimpleNamespace(
        availability=SimpleNamespace(available=False),
    )
    scene = SimpleNamespace(detected_at=time.monotonic())
    observation = SimpleNamespace(
        signature=signature,
        prompt_kind="text",
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
        semantic_clue=True,
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
    def test_semantic_emoji_uses_account_provider_without_double_voting(self) -> None:
        signature = "round:anime_title:3/10"
        app, state = make_async_app(
            signature,
            "semantic:dragon hot springs ghost pig",
            "session-1:round-3:3/10",
        )
        state.request.observation.prompt_kind = "visual"
        app._accessible_round = (signature, "🐉 ♨️ 👻 🐖")
        app._config.novel = SimpleNamespace(enabled=True)
        app._config.gemini = SimpleNamespace(enabled=True)
        app._config.vlm = SimpleNamespace(
            allow_unverified_submission=False,
            allow_novel_visual_submission=False,
        )
        app._novel = SimpleNamespace(ready_for_resolve=True)
        app._gemini = SimpleNamespace(
            availability=SimpleNamespace(available=True),
            rate_limited=False,
        )
        app._config.antigravity.enabled = True
        app._antigravity.availability.available = True

        providers = app._enabled_resolution_providers(
            state.request.observation, "🐉 ♨️ 👻 🐖"
        )

        self.assertEqual(providers, {"antigravity"})

    def test_raw_visual_uses_only_image_gemini(self) -> None:
        signature = "round:anime_title:3/10"
        app, state = make_async_app(
            signature,
            "visual:abc",
            "session-1:round-3:3/10",
        )
        state.request.observation.prompt_kind = "visual"
        app._accessible_round = None
        app._config.novel = SimpleNamespace(enabled=True)
        app._config.gemini = SimpleNamespace(enabled=True)
        app._config.vlm = SimpleNamespace(
            allow_unverified_submission=False,
            allow_novel_visual_submission=False,
        )
        app._novel = SimpleNamespace(ready_for_resolve=True)
        app._gemini = SimpleNamespace(
            availability=SimpleNamespace(available=True),
            rate_limited=False,
        )
        app._config.antigravity.enabled = True
        app._antigravity.availability.available = True

        providers = app._enabled_resolution_providers(
            state.request.observation, "Visual / emoji clue"
        )

        self.assertEqual(providers, {"gemini"})

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

    def test_first_answer_queues_immediately_and_disagreement_ladders(self) -> None:
        signature = "round:anime_title:1/10"
        fingerprint = "text:ambiguous clue"
        round_token = "session-1:round-1:1/10"
        app, state = make_async_app(signature, fingerprint, round_token)
        state.providers = {"gemini", "qwen"}
        state.pending = {"gemini", "qwen"}

        # Gemini answers first: it is queued right away, without waiting for
        # the slower local provider. Follow-ups remain slowmode-spaced.
        app._accept_resolution_result(
            ProviderResolution(
                key=state.request.key,
                provider="gemini",
                source="gemini-3.7-structured",
                answer="First Answer",
                confidence=0.80,
                elapsed_ms=1500.0,
                alternatives=("Alt One",),
            )
        )
        self.assertEqual(state.candidate.answer, "First Answer")
        # Its own alternatives wait until every provider has reported, so a
        # fast provider cannot fill the guess cap before a slower one answers.
        self.assertEqual(
            [(task.answer, task.guess_index) for task in app._dispatcher.tasks],
            [("First Answer", 1)],
        )

        # A disagreeing provider becomes the next rung of the ladder, then the
        # held-back alternatives follow (deduplicated against the ladder).
        app._accept_resolution_result(
            ProviderResolution(
                key=state.request.key,
                provider="qwen",
                source="qwen38-retrieval-consensus",
                answer="Other Answer",
                confidence=0.99,
                elapsed_ms=3800.0,
                alternatives=("First Answer",),
            )
        )
        self.assertEqual(
            [(task.answer, task.guess_index) for task in app._dispatcher.tasks],
            [("First Answer", 1), ("Other Answer", 2), ("Alt One", 3)],
        )
        self.assertEqual(state.pending, set())
        self.assertFalse(state.unknown_emitted)

    def test_antigravity_fallback_answers_after_primary_abstention(self) -> None:
        signature = "round:character:7/10"
        fingerprint = "text:airi clue"
        round_token = "session-1:round-7:7/10"
        app, state = make_async_app(signature, fingerprint, round_token)
        state.request.observation.prompt_kind = "text"
        state.providers = {"gemini", "qwen"}
        state.pending = {"gemini", "qwen"}
        app._config.antigravity.enabled = True
        app._antigravity.availability.available = True
        submitted_providers: list[str] = []
        app._submit_resolution_provider = (  # type: ignore[method-assign]
            lambda provider, _request: submitted_providers.append(provider)
        )

        for provider in ("gemini", "qwen"):
            app._accept_resolution_result(
                ProviderResolution(
                    key=state.request.key,
                    provider=provider,
                    source=f"{provider}-resolver",
                    answer=None,
                    confidence=0.0,
                    elapsed_ms=100.0,
                )
            )
        self.assertTrue(state.fallback_started)
        self.assertEqual(state.pending, {"antigravity"})
        self.assertEqual(submitted_providers, ["antigravity"])

        app._accept_resolution_result(
            ProviderResolution(
                key=state.request.key,
                provider="antigravity",
                source="antigravity-account-3.7-low",
                answer="Airi Katagiri",
                confidence=0.95,
                elapsed_ms=3700.0,
            )
        )
        self.assertEqual(state.candidate.answer, "Airi Katagiri")
        self.assertEqual(app._dispatcher.tasks[0].answer, "Airi Katagiri")

    def test_antigravity_can_run_when_no_primary_provider_is_available(self) -> None:
        signature = "round:character:7/10"
        fingerprint = "text:airi clue"
        round_token = "session-1:round-7:7/10"
        app, original_state = make_async_app(signature, fingerprint, round_token)
        app._config.antigravity.enabled = True
        app._antigravity.availability.available = True
        app._enabled_resolution_providers = lambda *_args: set()  # type: ignore[method-assign]
        submitted: list[tuple[str, ResolutionRequest]] = []
        app._submit_resolution_provider = (  # type: ignore[method-assign]
            lambda provider, request: submitted.append((provider, request))
        )

        state = app._start_async_resolution(
            key=(round_token, fingerprint),
            round_token=round_token,
            clue_fingerprint=fingerprint,
            clue="A hardworking student who encourages an older manga artist.",
            observation=original_state.request.observation,
            ocr_ms=1.0,
            extract_ms=1.0,
            lookup_ms=1.0,
        )

        self.assertTrue(state.fallback_started)
        self.assertEqual(state.pending, {"antigravity"})
        self.assertEqual(submitted[0][0], "antigravity")
        self.assertGreaterEqual(submitted[0][1].started_at, state.request.started_at)

    def test_inactive_same_round_fingerprint_does_not_spend_fallback_quota(self) -> None:
        signature = "round:anime_title:1/10"
        fingerprint_a = "text:stable clue a"
        fingerprint_b = "text:corrected clue b"
        round_token = "session-1:round-1:1/10"
        app, state = make_async_app(signature, fingerprint_a, round_token)
        state.providers = {"qwen"}
        state.pending = {"qwen"}
        app._config.antigravity.enabled = True
        app._antigravity.availability.available = True
        app._active_resolution_key = (round_token, fingerprint_b)
        submitted: list[str] = []
        app._submit_resolution_provider = (  # type: ignore[method-assign]
            lambda provider, _request: submitted.append(provider)
        )

        app._accept_resolution_result(
            ProviderResolution(
                key=state.request.key,
                provider="qwen",
                source="qwen38-retrieval-consensus",
                answer=None,
                confidence=0.0,
                elapsed_ms=3800.0,
            )
        )

        self.assertFalse(state.fallback_started)
        self.assertEqual(submitted, [])

        app._active_resolution_key = state.request.key
        app._active_prompt.update(signature, "locked", 3, fingerprint_a)
        app._emit_unknown_if_complete(state)
        self.assertTrue(state.fallback_started)
        self.assertEqual(submitted, ["antigravity"])

    def test_queued_antigravity_revalidates_fingerprint_before_cli_launch(self) -> None:
        signature = "round:anime_title:1/10"
        fingerprint_a = "text:stable clue a"
        fingerprint_b = "text:corrected clue b"
        round_token = "session-1:round-1:1/10"
        app, state = make_async_app(signature, fingerprint_a, round_token)
        app._active_resolution_key = (round_token, fingerprint_b)
        app._provider_locks = {"antigravity": threading.Lock()}
        app._config.antigravity.total_timeout_seconds = 6.0
        calls: list[str] = []

        async def resolve_never_called(_request):
            calls.append("called")
            raise AssertionError("stale fallback launched the CLI")

        app._antigravity.resolve = resolve_never_called
        result = app._run_resolution_provider("antigravity", state.request)

        self.assertEqual(result.error, "stale")
        self.assertEqual(calls, [])

    def test_queued_qwen_revalidates_fingerprint_before_model_launch(self) -> None:
        signature = "round:anime_title:1/10"
        fingerprint_a = "text:stable clue a"
        fingerprint_b = "text:corrected clue b"
        round_token = "session-1:round-1:1/10"
        app, state = make_async_app(signature, fingerprint_a, round_token)
        app._active_resolution_key = (round_token, fingerprint_b)
        app._provider_locks = {"qwen": threading.Lock()}
        app._config.novel = SimpleNamespace(total_timeout_seconds=8.0)
        calls: list[str] = []
        app._novel = SimpleNamespace(
            resolve_ranked=lambda *_args: calls.append("called"),
            last_confidence=0.0,
            last_detail="",
        )

        result = app._run_resolution_provider("qwen", state.request)

        self.assertEqual(result.error, "stale")
        self.assertEqual(calls, [])

    def test_returning_fingerprint_advances_from_antigravity_to_qwen(self) -> None:
        signature = "round:character:5/10"
        fingerprint_a = "text:elias clue"
        fingerprint_b = "text:temporary ocr"
        round_token = "session-1:round-5:5/10"
        app, state = make_async_app(signature, fingerprint_a, round_token)
        state.providers = {"antigravity"}
        state.pending = {"antigravity"}
        state.fallback_started = True
        app._config.novel = SimpleNamespace(enabled=True)
        app._active_resolution_key = (round_token, fingerprint_b)
        submitted: list[str] = []
        app._submit_resolution_provider = (  # type: ignore[method-assign]
            lambda provider, _request: submitted.append(provider)
        )

        app._accept_resolution_result(
            ProviderResolution(
                key=state.request.key,
                provider="antigravity",
                source="antigravity-account-3.7-low",
                answer=None,
                confidence=0.0,
                elapsed_ms=8000.0,
            )
        )
        self.assertEqual(submitted, [])

        app._active_resolution_key = state.request.key
        app._active_prompt.update(signature, "locked", 3, fingerprint_a)
        app._emit_unknown_if_complete(state)

        self.assertEqual(submitted, ["qwen"])
        self.assertEqual(state.pending, {"qwen"})

    def test_antigravity_fallback_is_not_spent_when_primaries_answered(self) -> None:
        signature = "round:anime_title:1/10"
        fingerprint = "text:ambiguous"
        round_token = "session-1:round-1:1/10"
        app, state = make_async_app(signature, fingerprint, round_token)
        state.request.observation.prompt_kind = "text"
        state.providers = {"gemini", "qwen"}
        state.pending = {"gemini", "qwen"}
        app._config.antigravity.enabled = True
        app._antigravity.availability.available = True
        submitted: list[str] = []
        app._submit_resolution_provider = (  # type: ignore[method-assign]
            lambda provider, _request: submitted.append(provider)
        )

        app._accept_resolution_result(
            ProviderResolution(
                key=state.request.key,
                provider="gemini",
                source="gemini-3.7-structured",
                answer="One-Punch Man",
                confidence=0.99,
                elapsed_ms=1500.0,
            )
        )
        app._accept_resolution_result(
            ProviderResolution(
                key=state.request.key,
                provider="qwen",
                source="qwen38-retrieval-consensus",
                answer="Dragon Ball Z",
                confidence=0.99,
                elapsed_ms=3800.0,
            )
        )
        self.assertEqual(submitted, [])
        self.assertFalse(state.fallback_started)
        self.assertEqual(state.candidate.answer, "One-Punch Man")
        self.assertEqual(
            [task.answer for task in app._dispatcher.tasks],
            ["One-Punch Man", "Dragon Ball Z"],
        )

    def test_duplicate_answers_across_providers_are_queued_once(self) -> None:
        signature = "round:character:5/10"
        fingerprint = "text:same"
        round_token = "session-1:round-5:5/10"
        app, state = make_async_app(signature, fingerprint, round_token)
        state.providers = {"gemini", "qwen"}
        state.pending = {"gemini", "qwen"}
        for provider, source in (
            ("gemini", "gemini-3.7-structured"),
            ("qwen", "qwen38-retrieval-consensus"),
        ):
            app._accept_resolution_result(
                ProviderResolution(
                    key=state.request.key,
                    provider=provider,
                    source=source,
                    answer="Elias Ainsworth",
                    confidence=0.9,
                    elapsed_ms=1000.0,
                )
            )
        self.assertEqual([task.answer for task in app._dispatcher.tasks], ["Elias Ainsworth"])
        self.assertEqual(len(state.guesses), 1)


if __name__ == "__main__":
    unittest.main()
