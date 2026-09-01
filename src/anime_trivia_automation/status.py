from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import StatusConfig

LOGGER = logging.getLogger(__name__)


class NullStatus:
    def launch(self) -> None:
        return

    def emit(self, _phase: str, **_fields: Any) -> None:
        return

    def close(self, _detail: str = "Stopped") -> None:
        return

    def heartbeat(self) -> None:
        return


class OperatorStatus:
    """Thread-safe structured status state consumed by the no-focus overlay."""

    def __init__(
        self,
        config: StatusConfig,
        path: Path,
        *,
        dry_run: bool,
        avoid_region: tuple[int, int, int, int] | None = None,
    ) -> None:
        self._config = config
        self._path = path
        self._avoid_region = avoid_region
        self._lock = threading.RLock()
        self._emitted: set[tuple[str, str]] = set()
        self._sequence = 0
        self._last_heartbeat = 0.0
        self._failed = False
        self._run_id = uuid.uuid4().hex
        self._process: subprocess.Popen[bytes] | None = None
        self._write_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
        self._writer_stop = threading.Event()
        self._writer_condition = threading.Condition()
        self._last_written_sequence = 0
        self._writer_thread: threading.Thread | None = None
        self._state: dict[str, Any] = {
            "schema_version": 1,
            "pid": os.getpid(),
            "run_id": self._run_id,
            "mode": "DRY RUN" if dry_run else "LIVE",
            "phase": "STARTING",
            "title": "Starting Anime Trivia",
            "detail": "Opening the operator panel",
            "question": "—",
            "clue": "Waiting for startup",
            "answer": "—",
            "source": "—",
            "readiness": "unknown",
            "history_entries": 0,
            "counters": {
                "rounds_seen": 0,
                "known": 0,
                "unknown": 0,
                "drafts_started": 0,
                "submitted": 0,
                "learned": 0,
                "closed": 0,
                "fatal_errors": 0,
            },
        }
        try:
            initial = self._prepare_snapshot_locked()
            self._write_snapshot(initial)
            self._last_written_sequence = int(initial["sequence"])
            self._writer_thread = threading.Thread(
                target=self._writer_loop,
                name="operator-status-writer",
                daemon=True,
            )
            self._writer_thread.start()
        except Exception:
            self._failed = True
            LOGGER.exception(
                "Operator status initialization failed; automation will continue without the panel"
            )

    @property
    def enabled(self) -> bool:
        return self._config.enabled and not self._failed

    @property
    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self._state)

    def launch(self) -> None:
        if not self.enabled or self._process is not None:
            return
        python_path = Path(sys.executable)
        pythonw = python_path.with_name("pythonw.exe")
        executable = pythonw if pythonw.exists() else python_path
        command = [
            str(executable),
            "-m",
            "anime_trivia_automation.status_window",
            "--status-path",
            str(self._path),
            "--run-id",
            self._run_id,
            "--worker-pid",
            str(os.getpid()),
            "--width",
            str(self._config.width),
            "--height",
            str(self._config.height),
            "--margin-x",
            str(self._config.margin_x),
            "--margin-y",
            str(self._config.margin_y),
            "--opacity",
            str(self._config.opacity),
            "--poll-ms",
            str(self._config.poll_ms),
            "--stale-after",
            str(self._config.stale_after_seconds),
            "--auto-close",
            str(self._config.auto_close_seconds),
            "--error-close",
            str(self._config.error_close_seconds),
        ]
        if self._avoid_region is not None:
            command.extend(
                ["--avoid-region", *(str(value) for value in self._avoid_region)]
            )
        if self._config.topmost:
            command.append("--topmost")
        if self._config.click_through:
            command.append("--click-through")
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
                creationflags=creationflags,
            )
        except OSError:
            LOGGER.exception("Could not launch the operator status panel")

    def emit(
        self,
        phase: str,
        *,
        title: str | None = None,
        detail: str | None = None,
        question: str | None = None,
        clue: str | None = None,
        answer: str | None = None,
        source: str | None = None,
        readiness: str | None = None,
        history_entries: int | None = None,
        event_id: str | None = None,
        increment: str | None = None,
        decrement: str | None = None,
        new_round: bool = False,
    ) -> None:
        if not self.enabled:
            return
        with self._lock:
            if self._state.get("phase") == "ERROR" and phase != "ERROR":
                return
            dedupe_key = (event_id, phase) if event_id else None
            if dedupe_key is not None and dedupe_key in self._emitted:
                return
            if dedupe_key is not None:
                self._emitted.add(dedupe_key)
            if new_round:
                self._state.update(
                    {
                        "answer": "—",
                        "source": "—",
                        "detail": "",
                    }
                )
            self._state["phase"] = phase
            for key, value in (
                ("title", title),
                ("detail", detail),
                ("question", question),
                ("clue", clue),
                ("answer", answer),
                ("source", source),
                ("readiness", readiness),
                ("history_entries", history_entries),
            ):
                if value is not None:
                    self._state[key] = value
            if increment is not None:
                counters = self._state["counters"]
                if increment not in counters:
                    LOGGER.error("Unknown operator counter: %s", increment)
                else:
                    counters[increment] += 1
            if decrement is not None:
                counters = self._state["counters"]
                if decrement not in counters:
                    LOGGER.error("Unknown operator counter: %s", decrement)
                else:
                    counters[decrement] = max(0, counters[decrement] - 1)
            self._enqueue_locked()

    def close(self, detail: str = "Automation stopped") -> None:
        with self._lock:
            is_error = self._state.get("phase") == "ERROR"
        if not is_error:
            self.emit("STOPPED", title="Stopped", detail=detail, readiness="closed")
        self._writer_stop.set()
        thread = self._writer_thread
        if thread is not None:
            thread.join(timeout=2.0)

    def heartbeat(self) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if now - self._last_heartbeat < 1.0:
            return
        with self._lock:
            self._last_heartbeat = now
            self._enqueue_locked()

    def flush(self, timeout: float = 1.0) -> bool:
        deadline = time.monotonic() + timeout
        with self._lock:
            target = self._sequence
        while time.monotonic() < deadline:
            with self._writer_condition:
                if self._last_written_sequence >= target:
                    return True
                self._writer_condition.wait(timeout=0.02)
        return False

    def _prepare_snapshot_locked(self) -> dict[str, Any]:
        self._sequence += 1
        self._state["sequence"] = self._sequence
        self._state["updated_at"] = datetime.now(UTC).isoformat()
        return deepcopy(self._state)

    def _enqueue_locked(self) -> None:
        if self._failed:
            return
        snapshot = self._prepare_snapshot_locked()
        try:
            self._write_queue.put_nowait(snapshot)
            return
        except queue.Full:
            pass
        try:
            self._write_queue.get_nowait()
        except queue.Empty:
            pass
        self._write_queue.put_nowait(snapshot)

    def _writer_loop(self) -> None:
        while not self._writer_stop.is_set() or not self._write_queue.empty():
            try:
                snapshot = self._write_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            while True:
                try:
                    snapshot = self._write_queue.get_nowait()
                except queue.Empty:
                    break
            try:
                self._write_snapshot(snapshot)
            except Exception:
                with self._lock:
                    self._failed = True
                LOGGER.exception(
                    "Operator status write failed; disabling the panel without affecting automation"
                )
                self._writer_stop.set()
            finally:
                with self._writer_condition:
                    self._last_written_sequence = max(
                        self._last_written_sequence,
                        int(snapshot.get("sequence", 0)),
                    )
                    self._writer_condition.notify_all()

    def _write_snapshot(self, snapshot: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(
            snapshot,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n"
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                prefix=f".{self._path.name}.",
                suffix=".tmp",
                dir=self._path.parent,
                delete=False,
            ) as handle:
                temporary = handle.name
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            for attempt in range(25):
                try:
                    os.replace(temporary, self._path)
                    temporary = None
                    break
                except PermissionError:
                    if attempt == 24:
                        raise
                    time.sleep(0.01)
        finally:
            if temporary and os.path.exists(temporary):
                try:
                    os.unlink(temporary)
                except OSError:
                    pass
