from __future__ import annotations

import ctypes
import logging
import os
import queue
import random
import threading
import time
from collections.abc import Callable
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ReadinessConfig, TypingConfig
from .models import AnswerTask
from .utils import sanitize_answer

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ForegroundWindow:
    process_name: str
    title: str


class ForegroundWindowGuard:
    """Prevents synthetic input from leaking into an unrelated foreground app."""

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

    def __init__(self, config: TypingConfig) -> None:
        self._process_names = {
            name.casefold() for name in config.expected_process_names if name
        }
        self._title_fragment = config.expected_window_title_contains.casefold()

    def current(self) -> ForegroundWindow | None:
        if os.name != "nt":
            return None
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        user32.GetForegroundWindow.argtypes = []
        user32.GetForegroundWindow.restype = wintypes.HWND
        user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
        user32.GetWindowTextLengthW.restype = ctypes.c_int
        user32.GetWindowTextW.argtypes = [
            wintypes.HWND,
            wintypes.LPWSTR,
            ctypes.c_int,
        ]
        user32.GetWindowTextW.restype = ctypes.c_int
        user32.GetWindowThreadProcessId.argtypes = [
            wintypes.HWND,
            ctypes.POINTER(wintypes.DWORD),
        ]
        user32.GetWindowThreadProcessId.restype = wintypes.DWORD
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.QueryFullProcessImageNameW.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.LPWSTR,
            ctypes.POINTER(wintypes.DWORD),
        ]
        kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return None

        title_length = user32.GetWindowTextLengthW(hwnd)
        title_buffer = ctypes.create_unicode_buffer(title_length + 1)
        user32.GetWindowTextW(hwnd, title_buffer, len(title_buffer))

        process_id = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))
        handle = kernel32.OpenProcess(
            self.PROCESS_QUERY_LIMITED_INFORMATION, False, process_id.value
        )
        process_name = ""
        if handle:
            try:
                size = wintypes.DWORD(32768)
                path_buffer = ctypes.create_unicode_buffer(size.value)
                if kernel32.QueryFullProcessImageNameW(
                    handle, 0, path_buffer, ctypes.byref(size)
                ):
                    process_name = Path(path_buffer.value).name
            finally:
                kernel32.CloseHandle(handle)
        return ForegroundWindow(process_name=process_name, title=title_buffer.value)

    def allowed(self) -> tuple[bool, str]:
        window = self.current()
        if window is None:
            return False, "foreground window could not be identified"
        if (
            self._process_names
            and window.process_name.casefold() not in self._process_names
        ):
            return False, f"foreground process is {window.process_name!r}"
        if self._title_fragment and self._title_fragment not in window.title.casefold():
            return False, f"foreground title is {window.title!r}"
        return True, f"{window.process_name}: {window.title}"


class ActivePromptState:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._signature: str | None = None
        self._uncertain = False
        self._readiness = "unknown"
        self._generation = 0

    def update(
        self,
        signature: str | None,
        readiness: str = "unknown",
        generation: int | None = None,
    ) -> bool:
        with self._condition:
            next_generation = self._generation if generation is None else generation
            if next_generation < self._generation:
                return False
            self._generation = next_generation
            self._signature = signature
            self._uncertain = False
            self._readiness = readiness if signature is not None else "unknown"
            self._condition.notify_all()
            return True

    def mark_uncertain(self, generation: int | None = None) -> None:
        with self._condition:
            next_generation = self._generation if generation is None else generation
            if next_generation < self._generation:
                return
            self._generation = next_generation
            if self._signature is not None:
                self._uncertain = True
            self._condition.notify_all()

    def is_current(self, signature: str) -> bool:
        with self._condition:
            return self._signature == signature and not self._uncertain

    def wait_current(
        self,
        signature: str,
        stop_event: threading.Event,
        timeout: float = 2.0,
    ) -> bool:
        deadline = time.monotonic() + timeout
        with self._condition:
            while (
                self._signature == signature
                and self._uncertain
                and not stop_event.is_set()
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(min(remaining, 0.05))
            return (
                not stop_event.is_set()
                and self._signature == signature
                and not self._uncertain
            )

    def wait_ready(
        self,
        signature: str,
        stop_event: threading.Event,
        timeout: float,
    ) -> bool:
        deadline = time.monotonic() + timeout
        with self._condition:
            while (
                self._signature == signature
                and (self._uncertain or self._readiness != "ready")
                and not stop_event.is_set()
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(min(remaining, 0.05))
            return (
                not stop_event.is_set()
                and self._signature == signature
                and not self._uncertain
                and self._readiness == "ready"
            )

    def wait_current_ready(
        self,
        signature: str,
        stop_event: threading.Event,
        timeout: float = 2.0,
    ) -> bool:
        """Wait through an in-flight frame, but fail closed on observed non-ready."""
        deadline = time.monotonic() + timeout
        with self._condition:
            while (
                self._signature == signature
                and self._uncertain
                and not stop_event.is_set()
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(min(remaining, 0.05))
            return (
                not stop_event.is_set()
                and self._signature == signature
                and not self._uncertain
                and self._readiness == "ready"
            )

    def get(self) -> str | None:
        with self._condition:
            return self._signature


def humanize_typing(
    controller: Any,
    text: str,
    delays: list[float],
    should_continue: Callable[[], bool],
    on_character_typed: Callable[[], None] | None = None,
) -> bool:
    """Type one character at a time with a caller-generated human delay profile."""
    for character, delay in zip(text, delays, strict=True):
        if not should_continue():
            return False
        controller.type(character)
        if on_character_typed is not None:
            on_character_typed()
        time.sleep(delay)
    return True


class SafeKeyboardExecutor:
    def __init__(
        self,
        config: TypingConfig,
        readiness_config: ReadinessConfig,
        active_prompt: ActivePromptState,
        stop_event: threading.Event,
    ) -> None:
        self._config = config
        self._readiness_config = readiness_config
        self._active_prompt = active_prompt
        self._stop_event = stop_event
        self._guard = ForegroundWindowGuard(config)
        try:
            from pynput.keyboard import Controller, Key
        except ImportError as exc:
            raise RuntimeError("pynput is required for keyboard execution") from exc
        self._controller = Controller()
        self._enter_key = Key.enter
        self._backspace_key = Key.backspace
        self._partial_characters = 0

    def _clear_partial_if_safe(self) -> bool:
        if self._partial_characters <= 0:
            return True
        allowed, reason = self._guard.allowed()
        if not allowed:
            LOGGER.warning(
                "Partial Discord input retained until focus returns: %s", reason
            )
            return False
        for _ in range(self._partial_characters):
            self._controller.press(self._backspace_key)
            self._controller.release(self._backspace_key)
        LOGGER.info(
            "Cleared %d partially typed macro characters", self._partial_characters
        )
        self._partial_characters = 0
        return True

    def _still_valid(self, signature: str, *, check_window: bool = True) -> bool:
        if self._readiness_config.require_green_outline:
            state_valid = self._active_prompt.wait_current_ready(
                signature,
                self._stop_event,
            )
        else:
            state_valid = self._active_prompt.wait_current(
                signature,
                self._stop_event,
            )
        if not state_valid:
            return False
        if check_window:
            allowed, reason = self._guard.allowed()
            if not allowed:
                LOGGER.warning("Typing aborted: %s", reason)
                return False
        return True

    def _wait_until(self, deadline: float, signature: str) -> bool:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return self._still_valid(signature, check_window=False)
            if self._stop_event.wait(min(remaining, 0.05)):
                return False
            if self._readiness_config.require_green_outline:
                state_valid = self._active_prompt.wait_current_ready(
                    signature,
                    self._stop_event,
                )
            else:
                state_valid = self._active_prompt.wait_current(
                    signature,
                    self._stop_event,
                )
            if not state_valid:
                return False

    def execute(self, task: AnswerTask) -> bool:
        answer = sanitize_answer(task.answer, self._config.max_answer_characters)
        if answer is None:
            LOGGER.warning("Rejected unsafe/empty answer: %r", task.answer)
            return False
        if self._readiness_config.require_green_outline:
            LOGGER.info(
                "Answer resolved while locked; waiting for green outline on %s",
                task.question_label or "the active round",
            )
            if not self._active_prompt.wait_ready(
                task.prompt_signature,
                self._stop_event,
                self._readiness_config.ready_wait_timeout_seconds,
            ):
                LOGGER.warning("Green-outline gate timed out or the round changed")
                return False
            ready_observed_at = time.monotonic()
            LOGGER.info("Green outline confirmed; humanized typing may begin")
        else:
            ready_observed_at = time.monotonic()

        if not self._config.enabled:
            LOGGER.info(
                "DRY RUN [%s] %s -> %s", task.source, task.question_label or "?", answer
            )
            return True

        allowed, reason = self._guard.allowed()
        if not allowed:
            LOGGER.warning("Typing skipped: %s", reason)
            return False
        if not self._clear_partial_if_safe():
            return False

        pre_delay = random.uniform(*self._config.pre_delay_seconds)
        delays = [random.uniform(*self._config.key_delay_seconds) for _ in answer]
        if self._readiness_config.require_green_outline:
            countdown = 0.0
            answer_open_at = ready_observed_at
            start_at = ready_observed_at + pre_delay
        else:
            countdown = (
                task.countdown_seconds
                if self._config.respect_detected_countdown
                and task.countdown_seconds is not None
                else self._config.fallback_answer_open_delay_seconds
            )
            answer_open_at = task.detected_at + max(0.0, countdown)
            earliest_start = time.monotonic() + pre_delay
            timed_start = answer_open_at - sum(delays)
            start_at = max(earliest_start, timed_start)
        LOGGER.info(
            "Humanized submit scheduled: answer=%r pre-delay=%.3fs type≈%.3fs open-delay=%.3fs",
            answer,
            pre_delay,
            sum(delays),
            countdown,
        )
        if not self._wait_until(start_at, task.prompt_signature):
            LOGGER.info(
                "Submission canceled before typing because the active prompt changed"
            )
            return False

        def should_continue() -> bool:
            return self._still_valid(task.prompt_signature, check_window=True)

        if not should_continue():
            return False

        def record_character() -> None:
            self._partial_characters += 1

        if not humanize_typing(
            self._controller,
            answer,
            delays,
            should_continue,
            on_character_typed=record_character,
        ):
            LOGGER.info("Submission canceled while typing")
            self._clear_partial_if_safe()
            return False

        enter_at = answer_open_at + self._config.enter_after_open_slack_seconds
        if not self._wait_until(enter_at, task.prompt_signature):
            LOGGER.info(
                "Submission canceled before Enter because the active prompt changed"
            )
            self._clear_partial_if_safe()
            return False
        if not should_continue():
            self._clear_partial_if_safe()
            return False
        self._controller.press(self._enter_key)
        self._controller.release(self._enter_key)
        self._partial_characters = 0
        LOGGER.info(
            "Submitted [%s] %s -> %s", task.source, task.question_label or "?", answer
        )
        return True


class AnswerDispatcher:
    """Serializes typing and suppresses duplicate answers for one active prompt."""

    def __init__(
        self,
        executor: SafeKeyboardExecutor,
        active_prompt: ActivePromptState,
        stop_event: threading.Event,
    ) -> None:
        self._executor = executor
        self._active_prompt = active_prompt
        self._stop_event = stop_event
        self._queue: queue.Queue[AnswerTask] = queue.Queue(maxsize=4)
        self._lock = threading.Lock()
        self._pending: set[str] = set()
        self._last_answered: str | None = None
        self._thread = threading.Thread(
            target=self._run, name="answer-dispatcher", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def observe_prompt(
        self,
        signature: str | None,
        readiness: str = "unknown",
        generation: int | None = None,
    ) -> bool:
        if not self._active_prompt.update(signature, readiness, generation):
            return False
        with self._lock:
            # A single transient OCR miss (None) must cancel pending typing but
            # must not re-arm an already answered clue. Rearm only after a
            # different real prompt has been observed.
            if (
                self._last_answered is not None
                and signature is not None
                and signature != self._last_answered
            ):
                self._last_answered = None
        return True

    def submit(self, task: AnswerTask) -> bool:
        with self._lock:
            if (
                task.prompt_signature in self._pending
                or task.prompt_signature == self._last_answered
            ):
                return False
            self._pending.add(task.prompt_signature)
        try:
            self._queue.put_nowait(task)
            return True
        except queue.Full:
            with self._lock:
                self._pending.discard(task.prompt_signature)
            LOGGER.warning("Answer queue full; dropping stale candidate")
            return False

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                task = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            succeeded = False
            try:
                succeeded = self._executor.execute(task)
            except Exception:
                LOGGER.exception("Keyboard execution failed")
            finally:
                with self._lock:
                    self._pending.discard(task.prompt_signature)
                    if succeeded:
                        self._last_answered = task.prompt_signature

    def join(self, timeout: float = 5.0) -> None:
        self._thread.join(timeout=timeout)


class EmergencyStopListener:
    def __init__(self, key_name: str, callback: Callable[[], None]) -> None:
        try:
            from pynput.keyboard import Key, Listener
        except ImportError as exc:
            raise RuntimeError(
                "pynput is required for the global emergency stop key"
            ) from exc
        expected = getattr(Key, key_name, None)
        if expected is None:
            raise ValueError(f"Unsupported stop key: {key_name}")

        def on_press(key: Any) -> bool | None:
            if key == expected:
                LOGGER.warning("Emergency stop key pressed")
                callback()
                return False
            return None

        self._listener = Listener(on_press=on_press)

    def start(self) -> None:
        self._listener.start()

    def stop(self) -> None:
        self._listener.stop()
