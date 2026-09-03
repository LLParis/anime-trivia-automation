from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path

from anime_trivia_automation.report import load_rounds, render_report
from anime_trivia_automation.typing import ActivePromptState

try:
    from test_live_regressions import FakeComposer, SwitchingGuard, make_executor
except ModuleNotFoundError:  # invoked as tests.test_launch_gates
    from tests.test_live_regressions import FakeComposer, SwitchingGuard, make_executor


def _row(
    phase,
    question,
    *,
    answer=None,
    detail="",
    clue=None,
    run="run-1",
    mono=0.0,
    event_id=None,
):
    return {
        "ts": "2026-09-03T01:00:00",
        "monotonic": mono,
        "run_id": run,
        "phase": phase,
        "question": question,
        "answer": answer,
        "detail": detail,
        "clue": clue,
        "source": "antigravity-account" if phase == "NOVEL" else None,
        "event_id": event_id,
    }


class QuizReportTests(unittest.TestCase):
    def test_report_names_the_layer_where_each_round_ended(self) -> None:
        rows = [
            _row("RED", "3/10", clue="A red-haired footballer", mono=100.0),
            _row("NOVEL", "3/10", answer="Hyoma Chigiri", mono=104.0),
            _row("SUBMITTED", "3/10", answer="Hyoma Chigiri", detail="The app will not retry this round"),
            _row("LEARNED", "3/10", answer="Hyoma Chigiri"),
            _row("RED", "4/10", clue="Admiration", mono=120.0),
            _row("NOVEL", "4/10", answer="Bleach", mono=124.5),
            _row("MANUAL", "4/10", answer="Bleach", detail="Your composer text is untouched"),
            _row("LEARNED", "4/10", answer="Bleach"),
            _row("RED", "6/10", clue="A confident blond esper", mono=140.0),
            _row("NOVEL", "6/10", answer="Teruki Hanazawa", mono=143.0),
            _row("SUBMITTED", "6/10", answer="Teruki Hanazawa", detail="Composer cleared"),
            _row("LEARNED", "6/10", answer="Teruki Hanazawa"),
            _row("RED", "8/10", clue="Visual / emoji clue", mono=160.0),
            _row("CLOSED", "8/10", detail="Submission gate is closed"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / "ledger.jsonl"
            ledger.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            rounds = load_rounds(ledger)
        outcomes = {r.question: r.outcome for r in rounds}
        self.assertTrue(outcomes["3/10"].startswith("UNCONFIRMED"))
        self.assertTrue(outcomes["4/10"].startswith("HAD IT, not sent: text already in composer"))
        self.assertEqual(outcomes["6/10"], "CORRECT (sent)")
        self.assertEqual(outcomes["8/10"], "not resolved")
        self.assertAlmostEqual(rounds[0].resolve_ms, 4000.0)
        text = render_report(rounds)
        self.assertIn("had the answer but did not send 1", text)
        self.assertIn("| 6/10 |", text)

    def test_report_keeps_repeated_question_labels_as_distinct_rounds(self) -> None:
        rows = [
            _row(
                "RED",
                "2/10",
                clue="first painted card",
                event_id="session-1:round-1:2/10",
            ),
            _row(
                "REHEARSAL",
                "2/10",
                answer="First",
                event_id="session-1:round-1:2/10:rehearsal",
            ),
            _row(
                "RED",
                "2/10",
                clue="second painted card",
                event_id="session-1:round-7:2/10",
            ),
            _row(
                "REHEARSAL",
                "2/10",
                answer="Second",
                event_id="session-1:round-7:2/10:rehearsal",
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / "ledger.jsonl"
            ledger.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            rounds = load_rounds(ledger)

        self.assertEqual(len(rounds), 2)
        self.assertEqual([item.round_id for item in rounds], [
            "session-1:round-1",
            "session-1:round-7",
        ])
        self.assertEqual([item.clue for item in rounds], [
            "first painted card",
            "second painted card",
        ])

    def test_unconfirmed_enter_attention_is_not_reported_as_not_sent(self) -> None:
        rows = [
            _row(
                "RED",
                "3/10",
                clue="A red-haired footballer",
                event_id="session-1:round-3:3/10",
            ),
            _row(
                "NOVEL",
                "3/10",
                answer="Hyoma Chigiri",
                event_id="session-1:round-3:3/10",
            ),
            {
                **_row(
                    "ATTENTION",
                    "3/10",
                    answer="Hyoma Chigiri",
                    detail="Discord re-rendered before the composer could confirm delivery",
                    event_id="session-1:round-3:3/10:submission-unconfirmed",
                ),
                "title": "Submission confirmation unavailable",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / "ledger.jsonl"
            ledger.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            rounds = load_rounds(ledger)

        self.assertEqual(len(rounds), 1)
        self.assertEqual(
            rounds[0].outcome,
            "UNCONFIRMED (Enter sent, delivery not acknowledged)",
        )
        report = render_report(rounds)
        self.assertIn("unconfirmed 1", report)
        self.assertIn("sent 0", report)


class LiveProbeTests(unittest.TestCase):
    def test_probe_types_verifies_clears_and_restores_focus(self) -> None:
        active = ActivePromptState()
        composer = FakeComposer()
        executor = make_executor(active, composer)
        guard = SwitchingGuard(idle_ms=5000)
        executor._guard = guard
        result = executor.live_probe(wait_seconds=1.0, idle_ms=2000)
        self.assertTrue(result.ok, result.detail)
        self.assertEqual(executor._text_input.sent, ["ok"])
        self.assertEqual(composer.text, "")
        self.assertEqual(guard.activations, [guard.discord.hwnd, guard.chrome.hwnd])
        self.assertGreaterEqual(result.value_lag_ms, 0.0)

    def test_probe_refuses_while_the_operator_is_typing(self) -> None:
        active = ActivePromptState()
        composer = FakeComposer()
        executor = make_executor(active, composer)
        executor._guard = SwitchingGuard(idle_ms=0)
        result = executor.live_probe(wait_seconds=0.3, idle_ms=2000)
        self.assertFalse(result.ok)
        self.assertEqual(executor._text_input.sent, [])

    def test_probe_fails_closed_when_the_composer_holds_text(self) -> None:
        active = ActivePromptState()
        composer = FakeComposer()
        composer.text = "someone's draft"
        executor = make_executor(active, composer)
        executor._guard = SwitchingGuard(idle_ms=5000)
        result = executor.live_probe(wait_seconds=1.0, idle_ms=2000)
        self.assertFalse(result.ok)
        self.assertEqual(composer.text, "someone's draft")

    def test_measured_lag_widens_the_settle_window(self) -> None:
        active = ActivePromptState()
        executor = make_executor(active, FakeComposer())
        executor.adopt_measured_lag(300.0)
        self.assertAlmostEqual(executor._config.composer_settle_timeout_seconds, 1.5)
        executor.adopt_measured_lag(10.0)
        self.assertAlmostEqual(executor._config.composer_settle_timeout_seconds, 1.5)


if __name__ == "__main__":
    unittest.main()
