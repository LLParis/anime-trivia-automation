from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from anime_trivia_automation.config import StatusConfig
from anime_trivia_automation.status import OperatorStatus


class OperatorStatusTests(unittest.TestCase):
    def test_atomic_snapshot_and_event_deduplication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "status.json"
            status = OperatorStatus(StatusConfig(), path, dry_run=False)
            status.emit(
                "RED",
                title="Q1 — Get Ready",
                question="1/10",
                clue="Example clue",
                readiness="locked",
                event_id="round-1",
                increment="rounds_seen",
                new_round=True,
            )
            status.emit(
                "RED",
                event_id="round-1",
                increment="rounds_seen",
            )
            status.emit(
                "KNOWN",
                answer="Example Answer",
                source="history-cache",
                event_id="round-1",
                increment="known",
            )
            status.emit(
                "KNOWN",
                event_id="round-1",
                increment="known",
            )
            self.assertTrue(status.flush())
            with path.open("r", encoding="utf-8") as handle:
                snapshot = json.load(handle)
            self.assertEqual(snapshot["phase"], "KNOWN")
            self.assertEqual(snapshot["answer"], "Example Answer")
            self.assertEqual(snapshot["counters"]["rounds_seen"], 1)
            self.assertEqual(snapshot["counters"]["known"], 1)
            self.assertGreater(snapshot["sequence"], 1)
            status.close()

    def test_error_is_not_overwritten_by_close(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "status.json"
            status = OperatorStatus(StatusConfig(), path, dry_run=True)
            status.emit(
                "ERROR",
                title="Capture failure",
                increment="fatal_errors",
            )
            status.emit("STOPPING", title="Stopping")
            status.close()
            self.assertEqual(status.snapshot["phase"], "ERROR")
            self.assertEqual(status.snapshot["counters"]["fatal_errors"], 1)

    def test_atomic_replace_retries_transient_windows_reader_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "status.json"
            status = OperatorStatus(StatusConfig(), path, dry_run=False)
            real_replace = os.replace
            attempts = 0

            def flaky_replace(source, destination):
                nonlocal attempts
                attempts += 1
                if attempts < 3:
                    raise PermissionError("simulated reader sharing lock")
                return real_replace(source, destination)

            with mock.patch(
                "anime_trivia_automation.status.os.replace",
                side_effect=flaky_replace,
            ):
                status.emit("ARMED", title="Armed")
                self.assertTrue(status.flush())
            self.assertEqual(attempts, 3)
            with path.open("r", encoding="utf-8") as handle:
                self.assertEqual(json.load(handle)["phase"], "ARMED")
            status.close()

    def test_status_io_failure_is_fail_soft(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "status.json"
            status = OperatorStatus(StatusConfig(), path, dry_run=False)
            with self.assertLogs(
                "anime_trivia_automation.status", level="ERROR"
            ):
                with mock.patch(
                    "anime_trivia_automation.status.os.replace",
                    side_effect=PermissionError("persistent status lock"),
                ):
                    status.emit("ARMED", title="Armed")
                    status.flush()
            self.assertFalse(status.enabled)
            status.heartbeat()
            status.close()

    def test_unknown_to_known_correction_keeps_exclusive_counters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "status.json"
            status = OperatorStatus(StatusConfig(), path, dry_run=False)
            status.emit(
                "UNKNOWN",
                event_id="round-1",
                increment="unknown",
            )
            status.emit(
                "KNOWN",
                event_id="round-1",
                increment="known",
                decrement="unknown",
            )
            self.assertEqual(status.snapshot["counters"]["known"], 1)
            self.assertEqual(status.snapshot["counters"]["unknown"], 0)
            status.close()

    def test_unknown_counter_name_cannot_break_automation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "status.json"
            status = OperatorStatus(StatusConfig(), path, dry_run=False)
            with self.assertLogs(
                "anime_trivia_automation.status", level="ERROR"
            ):
                status.emit("ARMED", increment="does_not_exist")
            self.assertEqual(status.snapshot["phase"], "ARMED")
            status.close()


if __name__ == "__main__":
    unittest.main()
