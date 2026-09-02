from __future__ import annotations

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
from anime_trivia_automation.models import AnswerTask
from anime_trivia_automation.status import NullStatus
from anime_trivia_automation.typing import ActivePromptState, AnswerDispatcher


class RecordingDispatcher:
    def __init__(self) -> None:
        self.tasks = []

    def submit(self, task) -> bool:
        self.tasks.append(task)
        return True


class PassiveExecutor:
    def observe_prompt(self, _signature, _fingerprint) -> None:
        pass

    @staticmethod
    def is_suppressed(_task) -> bool:
        return False


class RoundTokenTypingIdentityTests(unittest.TestCase):
    @staticmethod
    def _round_token_app() -> AnimeTriviaAutomation:
        app = AnimeTriviaAutomation.__new__(AnimeTriviaAutomation)
        app._status_session_id = 1
        app._status_round_id = 0
        app._active_status_signature = None
        app._active_status_question_label = None
        app._active_status_clue_key = None
        app._active_status_token = None
        app._active_status_closed = False
        app._pending_round = None
        app._accessible_round = None
        return app

    def test_typing_identity_is_unique_while_raw_card_signature_is_preserved(self) -> None:
        raw_signature = "round:anime_title:1/10"
        fingerprint = "text:repeated clue"
        first_round = "session-1:round-1:1/10"
        second_round = "session-2:round-1:1/10"
        scene = SimpleNamespace(detected_at=time.monotonic())
        observation = SimpleNamespace(
            signature=raw_signature,
            prompt_kind="text",
            expected_answer_type="anime_title",
            question_label="1/10",
            countdown_seconds=0.0,
            readiness="ready",
            scene=scene,
        )

        app = AnimeTriviaAutomation.__new__(AnimeTriviaAutomation)
        app._accessible_round = None
        app._resolution_rounds = {}
        app._enabled_resolution_providers = lambda *_args: {"qwen"}
        app._submit_resolution_provider = lambda *_args: None
        app._status = NullStatus()
        app._status_resolution = {}
        app._ephemeral_answer = None
        app._active_resolution_key = (first_round, fingerprint)
        app._active_status_token = first_round
        app._active_status_closed = False
        app._active_prompt = ActivePromptState()
        app._active_prompt.update(first_round, "ready", 1, fingerprint)
        app._dispatcher = RecordingDispatcher()

        state = app._start_async_resolution(
            key=(first_round, fingerprint),
            round_token=first_round,
            clue_fingerprint=fingerprint,
            clue="Repeated clue",
            observation=observation,
            ocr_ms=1.0,
            extract_ms=1.0,
            lookup_ms=1.0,
        )
        self.assertEqual(state.request.signature, first_round)
        self.assertEqual(state.request.observation.signature, raw_signature)

        guess = ProviderResolution(
            key=state.request.key,
            provider="qwen",
            source="qwen38-retrieval-consensus",
            answer="Repeated Anime",
            confidence=1.0,
            elapsed_ms=1.0,
        )
        state.guesses.append(guess)
        state.candidate = guess
        app._announce_resolution_guesses(state, [guess])
        self.assertEqual(app._ephemeral_answer[0], raw_signature)

        self.assertTrue(
            app._queue_answer_if_current(
                answer="Repeated Anime",
                source="history-cache",
                observation=observation,
                clue_fingerprint=fingerprint,
                round_token=first_round,
                detected_at=scene.detected_at,
                ocr_ms=1.0,
                extract_ms=1.0,
                lookup_ms=1.0,
                provider_ms=0.0,
            )
        )
        stale_task = app._dispatcher.tasks[0]
        self.assertEqual(stale_task.prompt_signature, first_round)
        self.assertEqual(stale_task.round_token, first_round)

        # A later physical round may have the exact same card signature and
        # clue. Its unique round token must still invalidate the old task.
        app._active_prompt.update(second_round, "ready", 2, fingerprint)
        self.assertFalse(
            app._active_prompt.is_open(
                stale_task.prompt_signature,
                stale_task.clue_fingerprint,
            )
        )

    def test_dispatcher_retires_other_rounds_and_old_current_fingerprints(self) -> None:
        active = ActivePromptState()
        dispatcher = AnswerDispatcher(
            PassiveExecutor(),  # type: ignore[arg-type]
            active,
            threading.Event(),
        )
        first_round = "session-1:round-1:1/10"
        second_round = "session-2:round-1:1/10"

        def task(round_token: str, fingerprint: str, answer: str) -> AnswerTask:
            return AnswerTask(
                answer=answer,
                prompt_signature=round_token,
                expected_answer_type="anime_title",
                question_label="1/10",
                detected_at=time.monotonic(),
                countdown_seconds=0.0,
                source="test",
                round_token=round_token,
                clue_fingerprint=fingerprint,
            )

        self.assertTrue(
            dispatcher.observe_prompt(first_round, "ready", 1, "text:same")
        )
        self.assertTrue(dispatcher.submit(task(first_round, "text:same", "Old")))

        # The next physical round intentionally has the same card/clue shape.
        self.assertTrue(
            dispatcher.observe_prompt(second_round, "ready", 2, "text:same")
        )
        self.assertEqual(list(dispatcher._queue), [])

        self.assertTrue(dispatcher.submit(task(second_round, "text:same", "Current")))
        self.assertTrue(
            dispatcher.observe_prompt(second_round, "ready", 3, "text:corrected")
        )
        self.assertEqual(list(dispatcher._queue), [])

        current = task(second_round, "text:corrected", "Corrected")
        self.assertTrue(dispatcher.submit(current))
        self.assertEqual(list(dispatcher._queue), [current])

    def test_noon_visual_footer_miss_retains_the_labeled_round_token(self) -> None:
        app = self._round_token_app()
        labeled_hash = (
            "b5276c9b4ae993242e10b655fd3749eb95ec2daaf568fd9402d14b2a7a2b0644"
        )
        footerless_hash = (
            "b5646db24ecb936d083fa6d5e0205baab7c92daaf54ba1351e964a6a422a16dd"
        )
        labeled = SimpleNamespace(
            signature="round:anime_title:4/10",
            question_label="4/10",
            hint_text="",
            perceptual_hash=labeled_hash,
            prompt_kind="visual",
            expected_answer_type="anime_title",
            readiness="locked",
        )
        footerless = SimpleNamespace(
            signature=f"visual:{footerless_hash}",
            question_label=None,
            hint_text="",
            perceptual_hash=footerless_hash,
            prompt_kind="visual",
            expected_answer_type="anime_title",
            readiness="locked",
        )

        token, new_round = app._round_token(labeled, live=True)
        self.assertTrue(new_round)
        app._pending_round = PendingRound(
            signature=labeled.signature,
            question_label="4/10",
            expected_answer_type="anime_title",
            prompt_kind="visual",
            clue="Visual / emoji clue",
            clue_fingerprint=f"visual:{labeled_hash}",
            hashes={labeled_hash},
        )

        # These exact noon hashes are 68 bits apart—far above the cache's
        # distance-10 threshold—because losing the footer shifts the crop.
        self.assertEqual(
            (int(labeled_hash, 16) ^ int(footerless_hash, 16)).bit_count(),
            68,
        )
        footerless_token, footerless_is_new = app._round_token(
            footerless, live=True
        )
        self.assertFalse(footerless_is_new)
        self.assertEqual(footerless_token, token)
        self.assertEqual(app._active_status_signature, labeled.signature)
        self.assertEqual(
            app._stable_live_clue_fingerprint(footerless),
            f"visual:{labeled_hash}",
        )

        # Multiple consecutive missed-footer frames remain the same card.
        another_footerless = SimpleNamespace(
            **{
                **footerless.__dict__,
                "signature": "visual:" + "f" * 64,
                "perceptual_hash": "f" * 64,
            }
        )
        repeated_token, repeated_is_new = app._round_token(
            another_footerless, live=True
        )
        self.assertFalse(repeated_is_new)
        self.assertEqual(repeated_token, token)

        restored_token, restored_is_new = app._round_token(labeled, live=True)
        self.assertFalse(restored_is_new)
        self.assertEqual(restored_token, token)

    def test_ready_to_locked_footerless_visual_is_a_new_round(self) -> None:
        app = self._round_token_app()
        labeled = SimpleNamespace(
            signature="round:anime_title:4/10",
            question_label="4/10",
            hint_text="",
            perceptual_hash="0" * 64,
            prompt_kind="visual",
            expected_answer_type="anime_title",
            readiness="ready",
        )
        old_token, _ = app._round_token(labeled, live=True)
        app._pending_round = PendingRound(
            signature=labeled.signature,
            question_label="4/10",
            expected_answer_type="anime_title",
            prompt_kind="visual",
            clue="Visual / emoji clue",
            clue_fingerprint="visual:" + "0" * 64,
            hashes={"0" * 64},
            saw_ready=True,
        )
        next_locked = SimpleNamespace(
            **{
                **labeled.__dict__,
                "signature": "visual:" + "1" * 64,
                "question_label": None,
                "perceptual_hash": "1" * 64,
                "readiness": "locked",
            }
        )

        new_token, is_new = app._round_token(next_locked, live=True)
        self.assertTrue(is_new)
        self.assertNotEqual(new_token, old_token)

    def test_retired_fingerprint_releases_app_queue_claim_for_a_b_a_return(self) -> None:
        app = AnimeTriviaAutomation.__new__(AnimeTriviaAutomation)
        token = "session-1:round-4:4/10"
        observation = SimpleNamespace(
            signature="round:anime_title:4/10",
            prompt_kind="visual",
            expected_answer_type="anime_title",
            question_label="4/10",
            countdown_seconds=0.0,
            readiness="ready",
            scene=SimpleNamespace(detected_at=time.monotonic()),
        )

        def state(fingerprint: str) -> AsyncResolutionRound:
            request = ResolutionRequest(
                key=(token, fingerprint),
                round_token=token,
                signature=token,
                clue_fingerprint=fingerprint,
                clue="Visual / emoji clue",
                observation=observation,
                started_at=time.perf_counter(),
            )
            guess = ProviderResolution(
                key=request.key,
                provider="gemini",
                source="gemini-3.8-structured",
                answer="Rurouni Kenshin",
                confidence=0.99,
                elapsed_ms=1.0,
            )
            return AsyncResolutionRound(
                request=request,
                providers={"gemini"},
                pending=set(),
                ocr_ms=1.0,
                extract_ms=1.0,
                lookup_ms=1.0,
                guesses=[guess],
                candidate=guess,
            )

        state_a = state("visual:a")
        state_b = state("visual:b")
        state_a.queued_answers.add("rurouni kenshin")
        state_a.queued = True
        app._resolution_rounds = {
            state_a.request.key: state_a,
            state_b.request.key: state_b,
        }
        app._reconcile_retired_dispatch_tasks(token, "visual:b")
        self.assertFalse(state_a.queued)
        self.assertEqual(state_a.queued_answers, set())

        app._active_resolution_key = state_a.request.key
        app._active_status_token = token
        app._active_status_closed = False
        app._active_prompt = ActivePromptState()
        app._active_prompt.update(token, "ready", 3, "visual:a")
        app._dispatcher = RecordingDispatcher()
        app._config = SimpleNamespace(
            typing=SimpleNamespace(max_guesses_per_round=3)
        )
        self.assertTrue(app._queue_resolution_candidate(state_a))
        self.assertEqual(
            [task.answer for task in app._dispatcher.tasks],
            ["Rurouni Kenshin"],
        )


if __name__ == "__main__":
    unittest.main()
