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
from .discord import DiscordComposer, DiscordComposerLocator
from .models import AnswerTask
from .status import NullStatus, OperatorStatus
from .utils import sanitize_answer

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ForegroundWindow:
    hwnd: int
    process_id: int
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

    @staticmethod
    def _windows_api() -> tuple[Any, Any]:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
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
        return user32, kernel32

    def _describe(self, hwnd: int) -> ForegroundWindow | None:
        if not hwnd or os.name != "nt":
            return None
        user32, kernel32 = self._windows_api()
        title_length = user32.GetWindowTextLengthW(hwnd)
        title_buffer = ctypes.create_unicode_buffer(title_length + 1)
        user32.GetWindowTextW(hwnd, title_buffer, len(title_buffer))

        process_id = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))
        if not process_id.value:
            return None
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
        return ForegroundWindow(
            hwnd=int(hwnd),
            process_id=int(process_id.value),
            process_name=process_name,
            title=title_buffer.value,
        )

    def current(self) -> ForegroundWindow | None:
        if os.name != "nt":
            return None
        user32, _kernel32 = self._windows_api()
        user32.GetForegroundWindow.argtypes = []
        user32.GetForegroundWindow.restype = wintypes.HWND
        hwnd = user32.GetForegroundWindow()
        return self._describe(int(hwnd)) if hwnd else None

    def visible_windows(self) -> tuple[ForegroundWindow, ...]:
        """Read visible top-level windows without activating or focusing any of them."""

        if os.name != "nt":
            return ()
        user32, _kernel32 = self._windows_api()
        user32.IsWindowVisible.argtypes = [wintypes.HWND]
        user32.IsWindowVisible.restype = wintypes.BOOL
        handles: list[int] = []
        callback_type = ctypes.WINFUNCTYPE(
            wintypes.BOOL, wintypes.HWND, wintypes.LPARAM
        )

        @callback_type
        def collect(hwnd: int, _lparam: int) -> bool:
            if user32.IsWindowVisible(hwnd):
                handles.append(int(hwnd))
            return True

        user32.EnumWindows.argtypes = [callback_type, wintypes.LPARAM]
        user32.EnumWindows.restype = wintypes.BOOL
        if not user32.EnumWindows(collect, 0):
            return ()
        windows = []
        for hwnd in handles:
            window = self._describe(hwnd)
            if window is not None:
                windows.append(window)
        return tuple(windows)

    def expected_window(self) -> ForegroundWindow | None:
        """Find one unambiguous Discord window without changing foreground state."""

        foreground = self.current()
        allowed, _reason = self.validate(foreground)
        if allowed and foreground is not None:
            return foreground

        matches = [
            window for window in self.visible_windows() if self.validate(window)[0]
        ]
        unique = {window.hwnd: window for window in matches}
        if len(unique) != 1:
            if len(unique) > 1:
                LOGGER.warning(
                    "Background Discord read skipped: %d matching windows",
                    len(unique),
                )
            return None
        return next(iter(unique.values()))

    def validate(self, window: ForegroundWindow | None) -> tuple[bool, str]:
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

    def allowed(self) -> tuple[bool, str]:
        return self.validate(self.current())


class ActivePromptState:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._signature: str | None = None
        self._uncertain = False
        self._readiness = "unknown"
        self._generation = 0
        self._clue_fingerprint = ""

    def update(
        self,
        signature: str | None,
        readiness: str = "unknown",
        generation: int | None = None,
        clue_fingerprint: str | None = None,
    ) -> bool:
        with self._condition:
            next_generation = self._generation if generation is None else generation
            if next_generation < self._generation:
                return False
            previous_signature = self._signature
            self._generation = next_generation
            self._signature = signature
            if clue_fingerprint is not None:
                self._clue_fingerprint = clue_fingerprint
            elif signature != previous_signature:
                self._clue_fingerprint = ""
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

    def is_current(self, signature: str, clue_fingerprint: str = "") -> bool:
        with self._condition:
            return (
                self._signature == signature
                and (not clue_fingerprint or self._clue_fingerprint == clue_fingerprint)
                and not self._uncertain
            )

    def wait_current(
        self,
        signature: str,
        stop_event: threading.Event,
        timeout: float = 2.0,
        clue_fingerprint: str = "",
    ) -> bool:
        deadline = time.monotonic() + timeout
        with self._condition:
            while (
                self._signature == signature
                and (not clue_fingerprint or self._clue_fingerprint == clue_fingerprint)
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
                and (not clue_fingerprint or self._clue_fingerprint == clue_fingerprint)
                and not self._uncertain
            )

    def wait_ready(
        self,
        signature: str,
        stop_event: threading.Event,
        timeout: float,
        clue_fingerprint: str = "",
    ) -> bool:
        deadline = time.monotonic() + timeout
        with self._condition:
            while (
                self._signature == signature
                and (not clue_fingerprint or self._clue_fingerprint == clue_fingerprint)
                and (self._uncertain or self._readiness != "ready")
                and self._readiness != "closed"
                and not stop_event.is_set()
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(min(remaining, 0.05))
            return (
                not stop_event.is_set()
                and self._signature == signature
                and (not clue_fingerprint or self._clue_fingerprint == clue_fingerprint)
                and not self._uncertain
                and self._readiness == "ready"
            )

    def wait_current_ready(
        self,
        signature: str,
        stop_event: threading.Event,
        timeout: float = 2.0,
        clue_fingerprint: str = "",
    ) -> bool:
        """Wait through an in-flight frame, but fail closed on observed non-ready."""
        deadline = time.monotonic() + timeout
        with self._condition:
            while (
                self._signature == signature
                and (not clue_fingerprint or self._clue_fingerprint == clue_fingerprint)
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
                and (not clue_fingerprint or self._clue_fingerprint == clue_fingerprint)
                and not self._uncertain
                and self._readiness == "ready"
            )

    def wait_current_open(
        self,
        signature: str,
        stop_event: threading.Event,
        timeout: float = 2.0,
        clue_fingerprint: str = "",
    ) -> bool:
        """Wait through OCR uncertainty and accept only a live red/green card."""

        deadline = time.monotonic() + timeout
        with self._condition:
            while (
                self._signature == signature
                and (not clue_fingerprint or self._clue_fingerprint == clue_fingerprint)
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
                and (not clue_fingerprint or self._clue_fingerprint == clue_fingerprint)
                and not self._uncertain
                and self._readiness in {"locked", "ready"}
            )

    def is_open(self, signature: str, clue_fingerprint: str = "") -> bool:
        with self._condition:
            return (
                self._signature == signature
                and (not clue_fingerprint or self._clue_fingerprint == clue_fingerprint)
                and not self._uncertain
                and self._readiness in {"locked", "ready"}
            )

    def is_ready(self, signature: str, clue_fingerprint: str = "") -> bool:
        with self._condition:
            return (
                self._signature == signature
                and (not clue_fingerprint or self._clue_fingerprint == clue_fingerprint)
                and not self._uncertain
                and self._readiness == "ready"
            )

    def execute_if_ready(
        self,
        signature: str,
        stop_event: threading.Event,
        callback: Callable[[], bool],
        clue_fingerprint: str = "",
    ) -> bool:
        """Hold the prompt-state lock across the last check and Enter dispatch."""

        with self._condition:
            if (
                stop_event.is_set()
                or self._signature != signature
                or (
                    clue_fingerprint
                    and self._clue_fingerprint != clue_fingerprint
                )
                or self._uncertain
                or self._readiness != "ready"
            ):
                return False
            return bool(callback())

    def get(self) -> str | None:
        with self._condition:
            return self._signature


class SafeKeyboardExecutor:
    """Atomically own one complete Discord answer and submit only on green."""

    def __init__(
        self,
        config: TypingConfig,
        readiness_config: ReadinessConfig,
        active_prompt: ActivePromptState,
        stop_event: threading.Event,
        status: OperatorStatus | NullStatus | None = None,
    ) -> None:
        self._config = config
        self._readiness_config = readiness_config
        self._active_prompt = active_prompt
        self._stop_event = stop_event
        self._status = status or NullStatus()
        self._guard = ForegroundWindowGuard(config)
        self._composer_locator = DiscordComposerLocator(
            config.composer_name_prefix,
            config.composer_class_fragment,
        )
        try:
            from pynput.keyboard import Controller, Key
        except ImportError as exc:
            raise RuntimeError("pynput is required for keyboard execution") from exc
        self._controller = Controller()
        self._enter_key = Key.enter
        self._orphaned_draft: str | None = None
        self._suppression_lock = threading.Lock()
        self._suppressed_rounds: set[str] = set()

    @staticmethod
    def _suppression_key(task: AnswerTask) -> str:
        return task.round_token or task.prompt_signature

    def suppress_task(self, task: AnswerTask, reason: str) -> None:
        with self._suppression_lock:
            self._suppressed_rounds.add(self._suppression_key(task))
        LOGGER.info("Automation suppressed for this round: %s", reason)

    def is_suppressed(self, task: AnswerTask) -> bool:
        with self._suppression_lock:
            return self._suppression_key(task) in self._suppressed_rounds

    def observe_prompt(
        self, signature: str | None, clue_fingerprint: str | None
    ) -> None:
        """Clear bounded per-quiz suppression when the tracked quiz ends."""

        with self._suppression_lock:
            if signature is None:
                self._suppressed_rounds.clear()

    def _remember_orphan(self, expected_prefix: str) -> None:
        if expected_prefix:
            self._orphaned_draft = expected_prefix
            LOGGER.warning(
                "Retaining exact draft ownership for safe cleanup when Discord returns"
            )

    def _cleanup_orphan(self) -> bool:
        expected = self._orphaned_draft
        if not expected:
            return True
        window = self._guard.current()
        allowed, _reason = self._guard.validate(window)
        window = window if allowed else None
        if window is None:
            return False
        composer = self._composer_locator.find(window.hwnd, window.process_id)
        if composer is None:
            return False
        current = composer.value()
        if current == "":
            self._orphaned_draft = None
            return True
        if current != expected:
            LOGGER.warning(
                "Orphaned draft was changed by the user; leaving it untouched"
            )
            self._orphaned_draft = None
            return False
        if self._clear_owned_draft(composer, expected):
            self._orphaned_draft = None
            return True
        return False

    def service_orphan(self) -> None:
        """Proactively clear an unchanged owned draft when Discord returns."""

        if self._orphaned_draft:
            self._cleanup_orphan()

    def _claim_empty_composer(
        self, window: ForegroundWindow
    ) -> tuple[DiscordComposer | None, bool]:
        if not self._config.verify_composer:
            return None, False
        composer = self._composer_locator.find(window.hwnd, window.process_id)
        if composer is None:
            return None, False
        if composer.value() != "":
            LOGGER.warning("Typing skipped: Discord composer is not empty")
            return None, True
        if not composer.focused() and self._config.auto_focus_composer:
            current = self._guard.current()
            allowed, _reason = self._guard.validate(current)
            if not allowed or current is None or current.hwnd != window.hwnd:
                LOGGER.warning("Composer focus blocked because Discord lost foreground")
                return None, False
            composer.set_focus()
            deadline = time.monotonic() + 0.25
            while time.monotonic() < deadline and not composer.focused():
                time.sleep(0.01)
        if not composer.focused():
            LOGGER.warning("Typing skipped: Discord composer is not focused")
            return None, False
        if composer.value() != "":
            LOGGER.warning("Typing skipped: Discord composer changed while claiming it")
            return None, True
        current = self._guard.current()
        allowed, _reason = self._guard.validate(current)
        if not allowed or current is None or current.hwnd != window.hwnd:
            LOGGER.warning("Composer claim canceled because Discord lost foreground")
            return None, False
        LOGGER.info("Claimed empty Discord composer %r", composer.name)
        return composer, False

    def _clear_owned_draft(
        self, composer: DiscordComposer | None, expected_prefix: str
    ) -> bool:
        if not expected_prefix:
            return True
        if composer is None:
            LOGGER.warning("Cannot safely clear an unverified composer draft")
            return False
        window = self._guard.current()
        allowed, reason = self._guard.validate(window)
        if not allowed:
            LOGGER.warning("Owned draft retained because %s", reason)
            return False
        try:
            if composer.value() != expected_prefix:
                LOGGER.warning(
                    "Composer diverged from the macro-owned draft; no user text was erased"
                )
                return False
            composer.clear_owned_value()
            if composer.value() != "":
                LOGGER.warning("Targeted Discord draft cleanup did not clear the editor")
                return False
            LOGGER.info("Cleared exact macro-owned draft after cancellation")
            return True
        except Exception:
            LOGGER.exception("Could not safely clear the macro-owned draft")
            return False

    def _clear_or_remember(
        self, composer: DiscordComposer | None, expected_prefix: str
    ) -> None:
        if composer is not None:
            try:
                if composer.value() != expected_prefix:
                    return
            except Exception:
                pass
        if not self._clear_owned_draft(composer, expected_prefix):
            self._remember_orphan(expected_prefix)

    def _wait_pre_delay(
        self,
        deadline: float,
        task: AnswerTask,
        window: ForegroundWindow,
        composer: DiscordComposer | None,
    ) -> str:
        """Monitor manual ownership before the one atomic composer write."""

        while not self._stop_event.is_set():
            prompt_valid = (
                self._active_prompt.is_ready(
                    task.prompt_signature, task.clue_fingerprint
                )
                if self._readiness_config.require_green_outline
                else self._active_prompt.is_open(
                    task.prompt_signature, task.clue_fingerprint
                )
            )
            if not prompt_valid:
                return "retry"
            current = self._guard.current()
            allowed, _reason = self._guard.validate(current)
            if not allowed or current is None or current.hwnd != window.hwnd:
                return "retry"
            if composer is not None:
                try:
                    value = composer.value()
                    if value != "":
                        return "manual"
                    if not composer.focused():
                        return "retry"
                except Exception:
                    LOGGER.debug(
                        "Discord composer rerendered during pre-delay",
                        exc_info=True,
                    )
                    return "retry"
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return "ready"
            if self._stop_event.wait(min(remaining, 0.005)):
                return "stopped"
        return "stopped"

    def _mark_manual_round(
        self, task: AnswerTask, answer: str, event_token: str
    ) -> None:
        self.suppress_task(task, "manual Discord text was detected")
        self._status.emit(
            "MANUAL",
            title=f"Manual answer active — {task.question_label or 'Question'}",
            detail="Your composer text is untouched; automation is off for this round",
            question=task.question_label or "Question",
            answer=answer,
            source=task.source,
            event_id=f"{event_token}:manual",
        )

    def _expire_safe_wait(
        self, task: AnswerTask, answer: str, event_token: str
    ) -> None:
        LOGGER.info("Foreground Discord wait expired")
        self.suppress_task(task, "the bounded foreground wait expired")
        self._status.emit(
            "ATTENTION",
            title="Solved answer expired",
            detail="Discord did not return before the safe wait ended",
            question=task.question_label or "Question",
            answer=answer,
            source=task.source,
            event_id=f"{event_token}:foreground-timeout",
        )

    def _wait_for_safe_composer(
        self,
        task: AnswerTask,
        answer: str,
        event_token: str,
    ) -> tuple[ForegroundWindow | None, DiscordComposer | None, bool]:
        """Wait passively for the exact live round, window, channel, and composer."""

        deadline = time.monotonic() + self._config.foreground_wait_timeout_seconds
        last_detail = ""
        while not self._stop_event.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._expire_safe_wait(task, answer, event_token)
                return None, None, False

            # If another app covers the calibrated card, capture becomes
            # uncertain. Wait through that interval and require a fresh exact
            # signature/fingerprint observation before any composer access.
            if self._readiness_config.require_green_outline:
                state_valid = self._active_prompt.wait_current_ready(
                    task.prompt_signature,
                    self._stop_event,
                    timeout=remaining,
                    clue_fingerprint=task.clue_fingerprint,
                )
            else:
                state_valid = self._active_prompt.wait_current_open(
                    task.prompt_signature,
                    self._stop_event,
                    timeout=remaining,
                    clue_fingerprint=task.clue_fingerprint,
                )
            if not state_valid:
                if (
                    not self._stop_event.is_set()
                    and time.monotonic() >= deadline
                ):
                    self._expire_safe_wait(task, answer, event_token)
                    return None, None, False
                LOGGER.info("Foreground wait canceled because the round changed")
                return None, None, False

            window = self._guard.current()
            allowed, reason = self._guard.validate(window)
            detail = "Continue manual research; return to the empty Discord composer"
            if allowed and window is not None:
                try:
                    if self._orphaned_draft:
                        self._cleanup_orphan()
                        if self._orphaned_draft:
                            detail = (
                                "Open #💜anime-chat so the prior owned draft can clear"
                            )
                        else:
                            # Cleanup either succeeded or discovered user-owned
                            # content. Inspect the fresh composer next iteration.
                            continue
                    elif not self._config.verify_composer:
                        return window, None, False
                    else:
                        composer, manual_text_detected = self._claim_empty_composer(
                            window
                        )
                        if manual_text_detected:
                            return window, None, True
                        if composer is not None:
                            # Recheck both logical identity and the exact HWND after
                            # UIA lookup/focus, before the first possible key.
                            if self._readiness_config.require_green_outline:
                                state_valid = self._active_prompt.wait_current_ready(
                                    task.prompt_signature,
                                    self._stop_event,
                                    timeout=min(remaining, 1.0),
                                    clue_fingerprint=task.clue_fingerprint,
                                )
                            else:
                                state_valid = self._active_prompt.wait_current_open(
                                    task.prompt_signature,
                                    self._stop_event,
                                    timeout=min(remaining, 1.0),
                                    clue_fingerprint=task.clue_fingerprint,
                                )
                            if not state_valid:
                                return None, None, False
                            current = self._guard.current()
                            current_allowed, _current_reason = self._guard.validate(
                                current
                            )
                            if (
                                current_allowed
                                and current is not None
                                and current.hwnd == window.hwnd
                            ):
                                LOGGER.info(
                                    "Foreground Discord composer is safe; resuming answer"
                                )
                                return window, composer, False
                        detail = "Switch to the empty #💜anime-chat message box"
                except Exception:
                    LOGGER.debug(
                        "Discord composer rerendered during safe wait",
                        exc_info=True,
                    )
                    detail = "Discord is re-rendering; keeping the answer pending"

            if detail != last_detail:
                LOGGER.info("Answer ready; waiting safely: %s (%s)", detail, reason)
                self._status.emit(
                    "WAITING_DISCORD",
                    title=f"Answer ready — {task.question_label or 'Question'}",
                    detail=detail,
                    question=task.question_label or "Question",
                    answer=answer,
                    source=task.source,
                    readiness="waiting-foreground",
                    event_id=f"{event_token}:foreground:{detail}",
                )
                last_detail = detail
            if self._stop_event.wait(min(remaining, 0.05)):
                return None, None, False
        return None, None, False

    def _commit_complete_answer(
        self,
        task: AnswerTask,
        answer: str,
        window: ForegroundWindow,
        composer: DiscordComposer | None,
    ) -> str:
        """Atomically place a complete answer while the exact prompt is green.

        Returns ``committed``, ``manual``, ``retry``, ``ambiguous``, or ``stale``.
        No character-at-a-time prefix is ever written.
        """

        outcome = "stale"

        def commit_if_owned() -> bool:
            nonlocal outcome
            current_window = self._guard.current()
            allowed, reason = self._guard.validate(current_window)
            if (
                not allowed
                or current_window is None
                or current_window.hwnd != window.hwnd
            ):
                LOGGER.info("Atomic commit deferred: %s", reason)
                outcome = "retry"
                return False
            if composer is None:
                LOGGER.warning("Atomic commit requires a verified Discord composer")
                outcome = "ambiguous"
                return False
            try:
                current_value = composer.value()
                if current_value != "":
                    outcome = "manual"
                    return False
                if not composer.focused():
                    outcome = "retry"
                    return False
            except Exception:
                LOGGER.debug(
                    "Discord composer rerendered before atomic commit",
                    exc_info=True,
                )
                outcome = "retry"
                return False

            try:
                composer.set_owned_value(answer)
            except Exception:
                LOGGER.warning(
                    "Atomic composer write raised; verifying the complete value",
                    exc_info=True,
                )
                try:
                    current_value = composer.value()
                except Exception:
                    self._remember_orphan(answer)
                    outcome = "ambiguous"
                    return False
                if current_value == answer:
                    LOGGER.info("Atomic composer write was verified after an exception")
                    outcome = "committed"
                    return True
                if current_value == "":
                    outcome = "ambiguous"
                    return False
                outcome = "manual"
                return False

            try:
                current_value = composer.value()
            except Exception:
                LOGGER.warning(
                    "Could not verify the complete atomic composer value",
                    exc_info=True,
                )
                self._remember_orphan(answer)
                outcome = "ambiguous"
                return False
            if current_value == answer:
                outcome = "committed"
                return True
            if current_value == "":
                outcome = "retry"
                return False
            outcome = "manual"
            return False

        if self._readiness_config.require_green_outline:
            committed = self._active_prompt.execute_if_ready(
                task.prompt_signature,
                self._stop_event,
                commit_if_owned,
                clue_fingerprint=task.clue_fingerprint,
            )
        elif self._active_prompt.is_open(
            task.prompt_signature, task.clue_fingerprint
        ):
            committed = commit_if_owned()
        else:
            committed = False
        return "committed" if committed else outcome

    def execute(self, task: AnswerTask) -> bool:
        event_token = task.round_token or task.prompt_signature
        if self.is_suppressed(task):
            return False
        answer = sanitize_answer(task.answer, self._config.max_answer_characters)
        if answer is None:
            LOGGER.warning("Rejected unsafe/empty answer: %r", task.answer)
            return False
        if not self._config.enabled:
            LOGGER.info(
                "DRY RUN [%s] %s -> %s", task.source, task.question_label or "?", answer
            )
            self._status.emit(
                "KNOWN",
                title=f"Dry run — {task.question_label or 'Question'}",
                detail="Resolved successfully; no keys are enabled",
                question=task.question_label or "Question",
                answer=answer,
                source=task.source,
                readiness="dry-run",
            )
            return True

        if self._readiness_config.require_green_outline:
            LOGGER.info(
                "Answer held in memory; Discord remains untouched until green"
            )
            self._status.emit(
                "WAITING_GREEN",
                title=f"Answer ready — {task.question_label or 'Question'}",
                detail="Held in memory; Discord stays untouched until answers open",
                question=task.question_label or "Question",
                answer=answer,
                source=task.source,
                readiness="locked",
                event_id=event_token,
            )
            if not self._active_prompt.wait_ready(
                task.prompt_signature,
                self._stop_event,
                self._readiness_config.ready_wait_timeout_seconds,
                clue_fingerprint=task.clue_fingerprint,
            ):
                LOGGER.warning("Green-outline gate timed out or the round changed")
                return False

        composer: DiscordComposer | None = None
        while not self._stop_event.is_set():
            window, composer, manual_text_detected = self._wait_for_safe_composer(
                task, answer, event_token
            )
            if window is None:
                return False
            if manual_text_detected:
                self._mark_manual_round(task, answer, event_token)
                return False

            pre_delay = random.uniform(*self._config.pre_delay_seconds)
            start_at = time.monotonic() + pre_delay
            LOGGER.info(
                "Atomic answer commit scheduled: answer=%r pre-delay=%.3fs",
                answer,
                pre_delay,
            )
            pre_delay_result = self._wait_pre_delay(
                start_at, task, window, composer
            )
            if pre_delay_result == "ready":
                commit_result = self._commit_complete_answer(
                    task, answer, window, composer
                )
                if commit_result == "committed":
                    break
                if commit_result == "manual":
                    self._mark_manual_round(task, answer, event_token)
                    return False
                if commit_result == "ambiguous":
                    self._status.emit(
                        "ATTENTION",
                        title="Atomic answer not verified",
                        detail="Enter is blocked; no partial typing was attempted",
                        question=task.question_label or "Question",
                        answer=answer,
                        source=task.source,
                        event_id=f"{event_token}:atomic-ambiguous",
                    )
                    self.suppress_task(task, "atomic composer write was ambiguous")
                    return False
                if commit_result == "stale":
                    LOGGER.info("Atomic commit canceled because the round changed")
                    return False
                LOGGER.info(
                    "Atomic commit lost safe ownership; returning to passive wait"
                )
                continue
            if pre_delay_result == "manual":
                self._mark_manual_round(task, answer, event_token)
                return False
            if pre_delay_result == "stopped":
                return False
            LOGGER.info("Pre-delay lost safe ownership; returning to passive wait")
        else:
            return False

        self._status.emit(
            "DRAFTING",
            title=f"Complete answer staged — {task.question_label or 'Question'}",
            detail="Committed atomically after green; verifying before Enter",
            question=task.question_label or "Question",
            answer=answer,
            source=task.source,
            readiness="ready",
            event_id=event_token,
            increment="drafts_started",
        )
        LOGGER.info("Complete answer committed atomically after green")
        if self._stop_event.wait(self._config.enter_after_open_slack_seconds):
            self._clear_or_remember(composer, answer)
            return False

        def dispatch_enter_if_owned() -> bool:
            window = self._guard.current()
            allowed, reason = self._guard.validate(window)
            if not allowed:
                LOGGER.warning("Typing aborted before Enter: %s", reason)
                return False
            if composer is not None:
                try:
                    if not composer.focused() or composer.value() != answer:
                        LOGGER.warning("Composer ownership was lost before Enter")
                        return False
                except Exception:
                    LOGGER.warning(
                        "Could not validate composer ownership before Enter",
                        exc_info=True,
                    )
                    return False

            # From the first keydown call onward the outcome is consumed. Even
            # an exception can mean Windows accepted the input, so rearming the
            # same round would risk a duplicate submission.
            try:
                self._controller.press(self._enter_key)
            except Exception:
                LOGGER.warning(
                    "Enter keydown outcome is unknown; suppressing duplicates",
                    exc_info=True,
                )
                return True
            try:
                self._controller.release(self._enter_key)
            except Exception:
                LOGGER.warning(
                    "Enter key release failed after keydown; suppressing duplicates",
                    exc_info=True,
                )
            return True

        if self._readiness_config.require_green_outline:
            dispatched = self._active_prompt.execute_if_ready(
                task.prompt_signature,
                self._stop_event,
                dispatch_enter_if_owned,
                clue_fingerprint=task.clue_fingerprint,
            )
        else:
            dispatched = dispatch_enter_if_owned()
        if not dispatched:
            self._clear_or_remember(composer, answer)
            self.suppress_task(task, "final ownership or green-state check failed")
            return False

        self._status.emit(
            "SUBMITTED",
            title=f"Enter sent — {task.question_label or 'Question'}",
            detail="Submission outcome consumed; duplicates are suppressed",
            question=task.question_label or "Question",
            answer=answer,
            source=task.source,
            readiness="ready",
            event_id=event_token,
            increment="submitted",
        )

        if composer is not None:
            try:
                deadline = time.monotonic() + 0.35
                while time.monotonic() < deadline and composer.value() != "":
                    time.sleep(0.01)
                cleared = composer.value() == ""
            except Exception:
                LOGGER.warning(
                    "Enter was sent and the Discord editor re-rendered; "
                    "suppressing any duplicate submission",
                    exc_info=True,
                )
                self._status.emit(
                    "SUBMITTED",
                    title="Enter sent — confirmation unavailable",
                    detail="Discord re-rendered; duplicate submission remains suppressed",
                    question=task.question_label or "Question",
                    answer=answer,
                    source=task.source,
                    readiness="ready",
                )
                return True
            if not cleared:
                LOGGER.warning(
                    "Enter was sent but Discord did not clear within 350ms; "
                    "suppressing any duplicate submission"
                )
                self._status.emit(
                    "SUBMITTED",
                    title="Enter sent — composer did not clear",
                    detail="The app will not retry this round",
                    question=task.question_label or "Question",
                    answer=answer,
                    source=task.source,
                    readiness="ready",
                )
                return True
        LOGGER.info(
            "Enter dispatched and composer cleared [%s] %s -> %s",
            task.source,
            task.question_label or "?",
            answer,
        )
        self._status.emit(
            "SUBMITTED",
            title=f"Enter sent — {task.question_label or 'Question'}",
            detail="Composer cleared; bot acceptance is confirmed only by its reveal",
            question=task.question_label or "Question",
            answer=answer,
            source=task.source,
            readiness="ready",
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
        # One worker may be waiting on Chrome/Gemini. Keep only the newest
        # queued observation behind it so OCR corrections cannot fill a FIFO
        # with stale fingerprints or crowd out the current clue.
        self._queue: queue.Queue[AnswerTask] = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._pending: set[tuple[str, str]] = set()
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
        clue_fingerprint: str | None = None,
    ) -> bool:
        if not self._active_prompt.update(
            signature, readiness, generation, clue_fingerprint
        ):
            return False
        self._executor.observe_prompt(signature, clue_fingerprint)
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
        if self._executor.is_suppressed(task):
            return False
        if not self._active_prompt.is_open(
            task.prompt_signature, task.clue_fingerprint
        ):
            LOGGER.debug("Rejected answer task for a non-live prompt")
            return False
        task_key = (task.prompt_signature, task.clue_fingerprint)
        with self._lock:
            if (
                task_key in self._pending
                or task.prompt_signature == self._last_answered
            ):
                return False
            self._pending.add(task_key)
        try:
            self._queue.put_nowait(task)
            return True
        except queue.Full:
            try:
                stale = self._queue.get_nowait()
            except queue.Empty:
                stale = None
            with self._lock:
                if stale is not None:
                    self._pending.discard(
                        (stale.prompt_signature, stale.clue_fingerprint)
                    )
            try:
                self._queue.put_nowait(task)
            except queue.Full:
                with self._lock:
                    self._pending.discard(task_key)
                LOGGER.warning("Latest answer slot raced; candidate will retry")
                return False
            LOGGER.debug("Replaced one stale queued answer with the latest clue")
            return True

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                task = self._queue.get(timeout=0.2)
            except queue.Empty:
                try:
                    self._executor.service_orphan()
                except Exception:
                    LOGGER.debug("Orphan-draft service failed", exc_info=True)
                continue
            succeeded = False
            with self._lock:
                already_answered = task.prompt_signature == self._last_answered
            if already_answered or self._executor.is_suppressed(task):
                with self._lock:
                    self._pending.discard(
                        (task.prompt_signature, task.clue_fingerprint)
                    )
                continue
            if not self._active_prompt.is_open(
                task.prompt_signature, task.clue_fingerprint
            ):
                with self._lock:
                    self._pending.discard(
                        (task.prompt_signature, task.clue_fingerprint)
                    )
                continue
            try:
                succeeded = self._executor.execute(task)
            except Exception:
                LOGGER.exception("Keyboard execution failed")
            finally:
                with self._lock:
                    self._pending.discard(
                        (task.prompt_signature, task.clue_fingerprint)
                    )
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
