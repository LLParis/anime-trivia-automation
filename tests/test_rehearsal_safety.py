from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from anime_trivia_automation.config import StatusConfig
from anime_trivia_automation.models import AnswerTask
from anime_trivia_automation.status import OperatorStatus
from anime_trivia_automation.typing import ActivePromptState, SafeKeyboardExecutor
from scripts.rehearse_live import require_rehearsal_worker
from tests.test_live_regressions import FakeComposer, RecordingStatus, make_executor


class RehearsalSafetyTests(unittest.TestCase):
    def test_operator_status_identifies_rehearsal_without_counting_a_submit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            status = OperatorStatus(
                StatusConfig(enabled=False),
                root / "status.json",
                dry_run=False,
                rehearsal=True,
                ledger_path=root / "ledger.jsonl",
            )
            try:
                status.emit(
                    "REHEARSAL",
                    title="Enter withheld",
                    detail="Typed and verified only",
                    question="1/10",
                    answer="answer",
                )
                snapshot = status.snapshot
                self.assertEqual(snapshot["mode"], "REHEARSAL")
                self.assertEqual(snapshot["counters"]["submitted"], 0)
            finally:
                status.close()

    @staticmethod
    def _write_status(
        path: Path,
        *,
        mode: str = "REHEARSAL",
        phase: str = "ARMED",
        run_id: str = "rehearsal-run",
        age_seconds: float = 0.0,
    ) -> None:
        path.write_text(
            json.dumps(
                {
                    "mode": mode,
                    "phase": phase,
                    "run_id": run_id,
                    "pid": 1234,
                    "updated_at": (
                        datetime.now(UTC) - timedelta(seconds=age_seconds)
                    ).isoformat(),
                }
            ),
            encoding="utf-8",
        )

    def test_rehearsal_handshake_requires_fresh_matching_live_worker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "status.json"
            self._write_status(path)
            with patch("scripts.rehearse_live._pid_is_running", return_value=True):
                self.assertEqual(require_rehearsal_worker(path), "rehearsal-run")
                self.assertEqual(
                    require_rehearsal_worker(
                        path, expected_run_id="rehearsal-run"
                    ),
                    "rehearsal-run",
                )
                with self.assertRaisesRegex(RuntimeError, "worker changed"):
                    require_rehearsal_worker(path, expected_run_id="other-run")

            self._write_status(path, mode="LIVE")
            with self.assertRaisesRegex(RuntimeError, "not in REHEARSAL mode"):
                require_rehearsal_worker(path)

            self._write_status(path, age_seconds=30.0)
            with (
                patch("scripts.rehearse_live._pid_is_running", return_value=True),
                self.assertRaisesRegex(RuntimeError, "status is stale"),
            ):
                require_rehearsal_worker(path, max_age_seconds=5.0)

            self._write_status(path)
            with (
                patch("scripts.rehearse_live._pid_is_running", return_value=False),
                self.assertRaisesRegex(RuntimeError, "is not running"),
            ):
                require_rehearsal_worker(path)

    def test_suffix_repair_rechecks_the_exact_live_round_before_input(self) -> None:
        first_round = "session-1:round-1:1/10"
        second_round = "session-1:round-2:2/10"
        fingerprint = "text:first"
        active = ActivePromptState()
        active.update(first_round, "ready", 1, fingerprint)

        class Composer:
            text = ""

            def value(self) -> str:
                return self.text

            @staticmethod
            def focused() -> bool:
                return True

        class PartialInput:
            def __init__(self, composer) -> None:
                self.composer = composer
                self.calls = []
                self.first_call = threading.Event()

            def send_text(self, text: str) -> None:
                self.calls.append(text)
                if len(self.calls) == 1:
                    self.composer.text = text[:2]
                    self.first_call.set()
                else:
                    self.composer.text += text

        window = type("Window", (), {"hwnd": 42})()
        composer = Composer()
        text_input = PartialInput(composer)
        executor = SafeKeyboardExecutor.__new__(SafeKeyboardExecutor)
        executor._active_prompt = active
        executor._readiness_config = type(
            "Readiness", (), {"require_green_outline": True}
        )()
        executor._config = type(
            "Typing", (), {"composer_settle_timeout_seconds": 0.0}
        )()
        executor._stop_event = threading.Event()
        executor._guard = type(
            "Guard",
            (),
            {
                "current": lambda _self: window,
                "validate": lambda _self, _window: (True, "ok"),
            },
        )()
        executor._text_input = text_input
        cleared = []
        executor._clear_or_remember = lambda _composer, value: cleared.append(value)
        executor._remember_orphan = lambda value: None
        task = AnswerTask(
            answer="first answer",
            prompt_signature=first_round,
            expected_answer_type="anime_title",
            question_label="1/10",
            detected_at=time.monotonic(),
            countdown_seconds=0.0,
            source="test",
            round_token=first_round,
            clue_fingerprint=fingerprint,
        )
        result = []
        worker = threading.Thread(
            target=lambda: result.append(
                executor._type_complete_answer(
                    task, task.answer, window, composer  # type: ignore[arg-type]
                )
            )
        )
        worker.start()
        self.assertTrue(text_input.first_call.wait(0.5))
        active.update(second_round, "ready", 2, "text:second")
        worker.join(1.5)

        self.assertEqual(result, ["stale"])
        self.assertEqual(text_input.calls, [task.answer])
        self.assertEqual(cleared, [task.answer[:2]])

    def test_rehearsal_withholds_enter_and_never_reports_a_submission(self) -> None:
        round_token = "session-1:round-1:1/10"
        fingerprint = "text:known"
        active = ActivePromptState()
        active.update(round_token, "ready", 1, fingerprint)
        composer = FakeComposer()
        executor = make_executor(active, composer)
        executor._config = replace(executor._config, press_enter=False)
        status = RecordingStatus()
        executor._status = status
        task = AnswerTask(
            answer="Taiga Aisaka",
            prompt_signature=round_token,
            expected_answer_type="character",
            question_label="1/10",
            detected_at=time.monotonic(),
            countdown_seconds=0.0,
            source="history-cache",
            round_token=round_token,
            clue_fingerprint=fingerprint,
        )

        self.assertTrue(executor.execute(task))

        self.assertFalse(executor._controller.entered)
        phases = [phase for phase, _fields in status.events]
        self.assertIn("DRAFTING", phases)
        self.assertIn("REHEARSAL", phases)
        self.assertNotIn("SUBMITTED", phases)
        drafting = next(fields for phase, fields in status.events if phase == "DRAFTING")
        rehearsal = next(fields for phase, fields in status.events if phase == "REHEARSAL")
        self.assertIn("withheld", drafting["detail"])
        self.assertNotEqual(rehearsal.get("increment"), "submitted")


if __name__ == "__main__":
    unittest.main()
