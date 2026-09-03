from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from anime_trivia_automation.antigravity import (
    AntigravityProvider,
    AntigravityRequest,
    _ProcessOutcome,
)
from anime_trivia_automation.config import (
    AntigravityConfig,
    load_config,
    validate_config,
)


class FakeRunner:
    def __init__(self, outcomes: list[_ProcessOutcome]) -> None:
        self.outcomes = outcomes
        self.calls = []

    def __call__(self, argv, cwd, env, timeout):
        self.calls.append((tuple(argv), Path(cwd), dict(env), timeout))
        return self.outcomes.pop(0)


def success_document(answer: str = "One-Punch Man") -> bytes:
    return json.dumps(
        {
            "status": "SUCCESS",
            "duration_seconds": 1.7,
            "structured_output": {
                "answer": answer,
                "answer_type": "anime_title",
                "confidence": 0.95,
                "confidence_label": "high",
                "abstain": False,
            },
        }
    ).encode()


class AntigravityProviderTests(unittest.TestCase):
    def make_provider(self, runner: FakeRunner, root: Path) -> AntigravityProvider:
        return AntigravityProvider(
            AntigravityConfig(
                enabled=True,
                executable=Path("C:/Users/sirlo/AppData/Local/agy/bin/agy.exe"),
                working_root=root,
            ),
            environ={
                "PATH": "safe-path",
                "GEMINI_API_KEY": "must-not-leak",
                "GOOGLE_API_KEY": "must-not-leak-either",
                "OPENAI_API_KEY": "must-not-leak-too",
            },
            process_runner=runner,
        )

    def test_account_auth_preflight_and_structured_text_answer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runner = FakeRunner(
                [
                    _ProcessOutcome(
                        0,
                        b"gemini-3.8-flash-low\tGemini 3.8 Flash (Low)\n",
                        0.2,
                    ),
                    _ProcessOutcome(0, success_document(), 1.7),
                ]
            )
            provider = self.make_provider(runner, Path(directory))
            availability = asyncio.run(provider.preflight())
            self.assertTrue(availability.available)
            result = asyncio.run(
                provider.resolve(
                    AntigravityRequest(
                        clue='"Who decides limits? And based on what?"',
                        expected_answer_type="anime_title",
                    )
                )
            )
            self.assertTrue(result.accepted)
            self.assertEqual(result.answer, "One-Punch Man")
            resolve_argv, resolve_cwd, child_env, _timeout = runner.calls[1]
            self.assertIn("gemini-3.8-flash-low", resolve_argv)
            self.assertIn("--output-format", resolve_argv)
            self.assertIn("--json-schema", resolve_argv)
            self.assertNotIn("GEMINI_API_KEY", child_env)
            self.assertNotIn("GOOGLE_API_KEY", child_env)
            self.assertNotIn("OPENAI_API_KEY", child_env)
            self.assertEqual(child_env, {"PATH": "safe-path"})
            self.assertFalse(resolve_cwd.exists())

    def test_visual_request_abstains_without_spawning_cli(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runner = FakeRunner(
                [
                    _ProcessOutcome(
                        0,
                        b"gemini-3.8-flash-low\tGemini 3.8 Flash (Low)\n",
                        0.1,
                    )
                ]
            )
            provider = self.make_provider(runner, Path(directory))
            asyncio.run(provider.preflight())
            result = asyncio.run(
                provider.resolve(
                    AntigravityRequest(
                        clue="Visual / emoji clue",
                        expected_answer_type="anime_title",
                        prompt_kind="visual",
                    )
                )
            )
            self.assertEqual(result.status, "abstained")
            self.assertEqual(len(runner.calls), 1)

    def test_timeout_and_unsafe_answer_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runner = FakeRunner(
                [
                    _ProcessOutcome(
                        0,
                        b"gemini-3.8-flash-low\tGemini 3.8 Flash (Low)\n",
                        0.1,
                    ),
                    _ProcessOutcome(-1, b"", 6.0, timed_out=True),
                    _ProcessOutcome(0, success_document("@everyone"), 1.0),
                ]
            )
            provider = self.make_provider(runner, Path(directory))
            asyncio.run(provider.preflight())
            request = AntigravityRequest(
                clue='"Who decides limits?"',
                expected_answer_type="anime_title",
            )
            self.assertEqual(asyncio.run(provider.resolve(request)).status, "timeout")
            self.assertEqual(
                asyncio.run(provider.resolve(request)).status,
                "invalid_response",
            )

    def test_close_kills_an_exact_inflight_cli_process(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            provider = AntigravityProvider(
                AntigravityConfig(
                    enabled=True,
                    executable=Path(sys.executable),
                    working_root=Path(directory),
                ),
                environ=os.environ,
            )

            async def exercise() -> _ProcessOutcome:
                worker = asyncio.create_task(
                    asyncio.to_thread(
                        provider._run_process,
                        (sys.executable, "-c", "import time; time.sleep(30)"),
                        Path(directory),
                        os.environ,
                        30.0,
                    )
                )
                deadline = time.monotonic() + 2.0
                while not provider._active_processes and time.monotonic() < deadline:
                    await asyncio.sleep(0.01)
                self.assertTrue(provider._active_processes)
                await provider.close()
                return await asyncio.wait_for(worker, timeout=2.0)

            started = time.monotonic()
            outcome = asyncio.run(exercise())
            self.assertLess(time.monotonic() - started, 2.0)
            self.assertLess(outcome.duration_seconds, 2.0)
            self.assertEqual(provider._active_processes, {})

    def test_request_directory_cleanup_retries_a_windows_handle_race(self) -> None:
        with patch(
            "anime_trivia_automation.antigravity.shutil.rmtree",
            side_effect=[PermissionError(), None],
        ) as remove, patch("anime_trivia_automation.antigravity.time.sleep") as sleep:
            AntigravityProvider._cleanup_request_directory(Path("request-race"))

        self.assertEqual(remove.call_count, 2)
        sleep.assert_called_once_with(0.05)

    def test_verified_cli_can_live_outside_virtualized_local_app_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "agy.exe"
            executable.write_bytes(b"versioned signed CLI fixture")
            config = load_config("config.example.json")
            installed = replace(
                config.antigravity,
                enabled=True,
                executable=executable,
            )

            with patch(
                "anime_trivia_automation.config._is_valid_google_signed_executable",
                return_value=True,
            ):
                validate_config(replace(config, antigravity=installed))
            with patch(
                "anime_trivia_automation.config._is_valid_google_signed_executable",
                return_value=False,
            ), self.assertRaisesRegex(ValueError, "valid Google LLC signature"):
                validate_config(replace(config, antigravity=installed))



class AnswerBudgetTests(unittest.TestCase):
    """The solver budget must cover the real Anime Soul answer window.

    On 2026-09-03 07:00 the three unanswered rounds (Princess Mononoke, Blue
    Period, Solo Leveling) each abstained exactly 12.0 s after the card
    appeared: they were the old budget expiring, not the model declining. The
    same clues answered in 5-22 s when given room, and the card stays green for
    about 60 s, so a 12 s budget silently discarded winnable rounds.
    """

    def test_default_budget_covers_a_slow_clue(self) -> None:
        self.assertGreaterEqual(AntigravityConfig().total_timeout_seconds, 25.0)

    def test_budget_stays_inside_the_answer_window(self) -> None:
        config = AntigravityConfig()
        self.assertLessEqual(config.total_timeout_seconds, 45.0)

    def test_a_budget_past_the_answer_window_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "agy.exe"
            executable.write_bytes(b"versioned signed CLI fixture")
            config = load_config("config.example.json")
            # The budget rule only applies to an enabled, installed provider.
            too_long = replace(
                config.antigravity,
                enabled=True,
                executable=executable,
                total_timeout_seconds=46.0,
            )
            with patch(
                "anime_trivia_automation.config._is_valid_google_signed_executable",
                return_value=True,
            ), self.assertRaisesRegex(ValueError, "cannot exceed 45 seconds"):
                validate_config(replace(config, antigravity=too_long))

if __name__ == "__main__":
    unittest.main()
