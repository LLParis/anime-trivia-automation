from __future__ import annotations

import ctypes
import logging
import os
import random
import threading
import time
from collections import deque
from collections.abc import Callable
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ReadinessConfig, TypingConfig
from .discord import DiscordComposer, DiscordComposerLocator
from .models import AnswerTask
from .status import NullStatus, OperatorStatus
from .utils import normalize_question, sanitize_answer

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

    @staticmethod
    def idle_milliseconds() -> int:
        """Milliseconds since the operator's last keyboard or mouse input."""

        if os.name != "nt":
            return 0

        class LastInputInfo(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD)]

        user32 = ctypes.WinDLL("user32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        info = LastInputInfo()
        info.cbSize = ctypes.sizeof(LastInputInfo)
        if not user32.GetLastInputInfo(ctypes.byref(info)):
            return 0
        kernel32.GetTickCount.restype = wintypes.DWORD
        return int((kernel32.GetTickCount() - info.dwTime) & 0xFFFFFFFF)

    def activate(self, hwnd: int) -> bool:
        """Bring one window to the foreground; True when Windows honoured it.

        A background process may only call SetForegroundWindow after it has
        generated input; the documented workaround is one ALT tap through
        keybd_event. AttachThreadInput to the current foreground thread is the
        second fallback.
        """

        if not hwnd or os.name != "nt":
            return False
        user32, kernel32 = self._windows_api()
        kernel32.GetCurrentThreadId.restype = wintypes.DWORD
        user32.GetForegroundWindow.argtypes = []
        user32.GetForegroundWindow.restype = wintypes.HWND
        user32.SetForegroundWindow.argtypes = [wintypes.HWND]
        user32.SetForegroundWindow.restype = wintypes.BOOL
        user32.IsIconic.argtypes = [wintypes.HWND]
        user32.IsIconic.restype = wintypes.BOOL
        user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
        user32.ShowWindow.restype = wintypes.BOOL
        user32.keybd_event.argtypes = [
            wintypes.BYTE,
            wintypes.BYTE,
            wintypes.DWORD,
            ctypes.c_void_p,
        ]
        user32.AttachThreadInput.argtypes = [wintypes.DWORD, wintypes.DWORD, wintypes.BOOL]
        user32.AttachThreadInput.restype = wintypes.BOOL

        if int(user32.GetForegroundWindow() or 0) == int(hwnd):
            return True
        if user32.IsIconic(hwnd):
            user32.ShowWindow(hwnd, 9)  # SW_RESTORE
        vk_menu = 0x12
        user32.keybd_event(vk_menu, 0, 0, None)
        user32.keybd_event(vk_menu, 0, 0x0002, None)  # KEYEVENTF_KEYUP
        user32.SetForegroundWindow(hwnd)
        deadline = time.monotonic() + 0.25
        while time.monotonic() < deadline:
            if int(user32.GetForegroundWindow() or 0) == int(hwnd):
                return True
            time.sleep(0.01)

        previous = int(user32.GetForegroundWindow() or 0)
        if previous:
            foreground_thread = user32.GetWindowThreadProcessId(previous, None)
            current_thread = kernel32.GetCurrentThreadId()
            if foreground_thread and user32.AttachThreadInput(
                current_thread, foreground_thread, True
            ):
                try:
                    user32.SetForegroundWindow(hwnd)
                finally:
                    user32.AttachThreadInput(current_thread, foreground_thread, False)
        deadline = time.monotonic() + 0.25
        while time.monotonic() < deadline:
            if int(user32.GetForegroundWindow() or 0) == int(hwnd):
                return True
            time.sleep(0.01)
        return False


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
        # Foreground window we displaced to reach Discord for the current
        # task; restored after Enter so manual research can continue.
        self._activated_from: int | None = None

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
            if (not allowed or window is None) and self._config.auto_activate_discord:
                activation = self._try_activate_discord(window)
                if activation == "activated":
                    continue
                if activation:
                    detail = activation
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

    def _try_activate_discord(self, foreground: ForegroundWindow | None) -> str:
        """Bring the one Discord window forward once the operator is idle.

        Returns ``activated`` on success, otherwise a short operator-facing
        reason the wait is still passive.
        """

        idle_ms = self._guard.idle_milliseconds()
        if idle_ms < self._config.activation_idle_ms:
            return "Answer ready; waiting for you to pause typing before opening Discord"
        target = self._guard.expected_window()
        if target is None:
            return "Answer ready; Discord window not uniquely found for activation"
        if not self._guard.activate(target.hwnd):
            LOGGER.warning("Windows refused to bring Discord to the foreground")
            return "Answer ready; Windows refused to raise Discord, return manually"
        if foreground is not None and foreground.hwnd != target.hwnd:
            self._activated_from = foreground.hwnd
        LOGGER.info(
            "Raised Discord to the foreground after %d ms of operator idle time",
            idle_ms,
        )
        return "activated"

    def _restore_foreground(self) -> None:
        previous, self._activated_from = self._activated_from, None
        if previous is None or not self._config.restore_previous_foreground:
            return
        try:
            if self._guard.activate(previous):
                LOGGER.info("Returned focus to the previous window after Enter")
        except Exception:
            LOGGER.debug("Could not restore the previous foreground window", exc_info=True)

    def _type_complete_answer(
        self,
        task: AnswerTask,
        answer: str,
        window: ForegroundWindow,
        composer: DiscordComposer | None,
    ) -> str:
        """Type the complete answer with real keystrokes while the card is green.

        Returns ``committed``, ``manual``, ``retry``, ``ambiguous``, or ``stale``.
        Every character is verified against the composer so a human edit or a
        round change cancels the draft without erasing anyone else's text.
        """

        def prompt_live() -> bool:
            if self._readiness_config.require_green_outline:
                return self._active_prompt.is_ready(
                    task.prompt_signature, task.clue_fingerprint
                )
            return self._active_prompt.is_open(
                task.prompt_signature, task.clue_fingerprint
            )

        if not prompt_live():
            return "stale"
        current_window = self._guard.current()
        allowed, reason = self._guard.validate(current_window)
        if not allowed or current_window is None or current_window.hwnd != window.hwnd:
            LOGGER.info("Keystroke commit deferred: %s", reason)
            return "retry"
        if composer is None:
            LOGGER.warning("Keystroke commit requires a verified Discord composer")
            return "ambiguous"
        try:
            if composer.value() != "":
                return "manual"
            if not composer.focused():
                return "retry"
        except Exception:
            LOGGER.debug("Discord composer rerendered before typing", exc_info=True)
            return "retry"

        typed = 0
        delays = [random.uniform(*self._config.key_delay_seconds) for _ in answer]
        for character, delay in zip(answer, delays, strict=True):
            if self._stop_event.is_set() or not prompt_live():
                self._clear_or_remember(composer, answer[:typed])
                return "stale"
            try:
                if not composer.focused() or composer.value() != answer[:typed]:
                    LOGGER.warning("Composer diverged from the owned prefix while typing")
                    self._clear_or_remember(composer, answer[:typed])
                    return "manual"
            except Exception:
                LOGGER.warning("Composer verification failed while typing", exc_info=True)
                self._remember_orphan(answer[:typed])
                return "ambiguous"
            try:
                self._controller.type(character)
            except Exception:
                LOGGER.warning("Character injection outcome is ambiguous", exc_info=True)
                self._remember_orphan(answer[: typed + 1])
                return "ambiguous"
            typed += 1
            if delay > 0:
                time.sleep(delay)
        try:
            final_value = composer.value()
        except Exception:
            LOGGER.warning("Could not verify the completed keystroke draft", exc_info=True)
            self._remember_orphan(answer)
            return "ambiguous"
        if final_value == answer:
            return "committed"
        if final_value.startswith(answer):
            # Discord appended something (autocomplete/emoji) after the last
            # key; treat it as a manual divergence rather than sending it.
            LOGGER.warning("Composer contains more than the owned answer; not sending")
            return "manual"
        self._clear_or_remember(composer, final_value if answer.startswith(final_value) else "")
        return "retry" if final_value == "" else "manual"

    def _commit_complete_answer(
        self,
        task: AnswerTask,
        answer: str,
        window: ForegroundWindow,
        composer: DiscordComposer | None,
    ) -> str:
        """Place the complete answer in the composer while the exact prompt is green.

        Dispatches to real keystrokes (``typing.composer_write_mode = "type"``)
        or the UI Automation ValuePattern write (``"uia"``).
        """

        if self._config.composer_write_mode == "type":
            return self._type_complete_answer(task, answer, window, composer)
        return self._set_complete_answer(task, answer, window, composer)

    def _set_complete_answer(
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
                "Answer commit scheduled (%s, guess %d): answer=%r pre-delay=%.3fs",
                self._config.composer_write_mode,
                task.guess_index,
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
            detail="Composer holds the complete answer after green; sending Enter",
            question=task.question_label or "Question",
            answer=answer,
            source=task.source,
            readiness="ready",
            event_id=f"{event_token}:guess{task.guess_index}",
            increment="drafts_started",
        )
        LOGGER.info("Complete answer staged after green (guess %d)", task.guess_index)
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
            detail=(
                "Submission outcome consumed; duplicates are suppressed"
                if task.guess_index == 1
                else f"Follow-up guess {task.guess_index} sent while the card stayed green"
            ),
            question=task.question_label or "Question",
            answer=answer,
            source=task.source,
            readiness="ready",
            event_id=f"{event_token}:guess{task.guess_index}",
            increment="submitted",
        )
        self._restore_foreground()

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
    """Serializes composer access and runs the per-round guess ladder.

    Wrong guesses cost nothing in Anime Soul, so one round may receive several
    distinct answers: the first as soon as it is known, each later one only
    after ``guess_gap_seconds`` and only while the same card is still open.
    """

    def __init__(
        self,
        executor: SafeKeyboardExecutor,
        active_prompt: ActivePromptState,
        stop_event: threading.Event,
        *,
        max_guesses_per_round: int = 3,
        guess_gap_seconds: float = 1.5,
    ) -> None:
        self._executor = executor
        self._active_prompt = active_prompt
        self._stop_event = stop_event
        self._max_guesses = max(1, int(max_guesses_per_round))
        self._guess_gap = max(0.0, float(guess_gap_seconds))
        self._condition = threading.Condition()
        self._queue: deque[AnswerTask] = deque()
        # Distinct normalized answers already sent per prompt signature, and
        # when the last one was sent. Cleared as soon as a different live
        # prompt is observed so the next quiz can reuse round signatures.
        self._answered: dict[str, set[str]] = {}
        self._last_sent_at: dict[str, float] = {}
        self._in_flight: AnswerTask | None = None
        self._thread = threading.Thread(
            target=self._run, name="answer-dispatcher", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    @staticmethod
    def _answer_key(task: AnswerTask) -> str:
        return normalize_question(task.answer)

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
        with self._condition:
            if signature is not None:
                # A single transient OCR miss (None) must not re-arm an answered
                # round; only a different real prompt retires the old record.
                for stale in [key for key in self._answered if key != signature]:
                    self._answered.pop(stale, None)
                    self._last_sent_at.pop(stale, None)
                if clue_fingerprint is not None:
                    # Queued OCR variants of the same round with an older
                    # fingerprint can never pass the executor's identity check.
                    self._queue = deque(
                        task
                        for task in self._queue
                        if task.prompt_signature != signature
                        or task.clue_fingerprint == clue_fingerprint
                    )
            self._condition.notify_all()
        return True

    def submit(self, task: AnswerTask) -> bool:
        if self._executor.is_suppressed(task):
            return False
        if not self._active_prompt.is_open(
            task.prompt_signature, task.clue_fingerprint
        ):
            LOGGER.debug("Rejected answer task for a non-live prompt")
            return False
        answer_key = self._answer_key(task)
        if not answer_key:
            return False
        with self._condition:
            sent = self._answered.get(task.prompt_signature, set())
            if answer_key in sent:
                return False
            queued_keys = {
                self._answer_key(item)
                for item in self._queue
                if item.prompt_signature == task.prompt_signature
            }
            in_flight = self._in_flight
            if (
                in_flight is not None
                and in_flight.prompt_signature == task.prompt_signature
            ):
                queued_keys.add(self._answer_key(in_flight))
            if answer_key in queued_keys:
                return False
            if len(sent) + len(queued_keys) >= self._max_guesses:
                LOGGER.info(
                    "Guess ladder full for %s; not queueing %r",
                    task.question_label or task.prompt_signature,
                    task.answer,
                )
                return False
            self._queue.append(task)
            self._condition.notify_all()
        return True

    def _wait_for_guess_gap(self, task: AnswerTask) -> bool:
        """Space follow-up guesses; abandon them when the round stops being open."""

        with self._condition:
            last_sent = self._last_sent_at.get(task.prompt_signature)
        if last_sent is None:
            return True
        deadline = last_sent + self._guess_gap
        while not self._stop_event.is_set():
            if not self._active_prompt.is_open(
                task.prompt_signature, task.clue_fingerprint
            ):
                return False
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return True
            if self._stop_event.wait(min(remaining, 0.02)):
                return False
        return False

    def _run(self) -> None:
        while not self._stop_event.is_set():
            with self._condition:
                while not self._queue and not self._stop_event.is_set():
                    self._condition.wait(0.2)
                    if not self._queue:
                        try:
                            self._executor.service_orphan()
                        except Exception:
                            LOGGER.debug("Orphan-draft service failed", exc_info=True)
                if self._stop_event.is_set():
                    return
                task = self._queue.popleft()
                already_sent = self._answer_key(task) in self._answered.get(
                    task.prompt_signature, set()
                )
                self._in_flight = task
            try:
                if already_sent or self._executor.is_suppressed(task):
                    continue
                if not self._active_prompt.is_open(
                    task.prompt_signature, task.clue_fingerprint
                ):
                    continue
                if not self._wait_for_guess_gap(task):
                    LOGGER.info(
                        "Dropped follow-up guess %r because the round is no longer open",
                        task.answer,
                    )
                    continue
                succeeded = False
                try:
                    succeeded = self._executor.execute(task)
                except Exception:
                    LOGGER.exception("Keyboard execution failed")
                if succeeded:
                    with self._condition:
                        self._answered.setdefault(task.prompt_signature, set()).add(
                            self._answer_key(task)
                        )
                        self._last_sent_at[task.prompt_signature] = time.monotonic()
            finally:
                with self._condition:
                    self._in_flight = None

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
