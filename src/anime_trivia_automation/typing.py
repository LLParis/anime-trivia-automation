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
        return ForegroundWindow(
            hwnd=int(hwnd),
            process_id=int(process_id.value),
            process_name=process_name,
            title=title_buffer.value,
        )

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

    def wait_current_open(
        self,
        signature: str,
        stop_event: threading.Event,
        timeout: float = 2.0,
    ) -> bool:
        """Wait through OCR uncertainty and accept only a live red/green card."""

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
                and self._readiness in {"locked", "ready"}
            )

    def is_open(self, signature: str) -> bool:
        with self._condition:
            return (
                self._signature == signature
                and not self._uncertain
                and self._readiness in {"locked", "ready"}
            )

    def execute_if_ready(
        self,
        signature: str,
        stop_event: threading.Event,
        callback: Callable[[], bool],
    ) -> bool:
        """Hold the prompt-state lock across the last check and Enter dispatch."""

        with self._condition:
            if (
                stop_event.is_set()
                or self._signature != signature
                or self._uncertain
                or self._readiness != "ready"
            ):
                return False
            return bool(callback())

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
    """Own and verify one Discord draft, then press Enter only on green."""

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
    ) -> DiscordComposer | None:
        if not self._config.verify_composer:
            return None
        composer = self._composer_locator.find(window.hwnd, window.process_id)
        if composer is None:
            return None
        if composer.value() != "":
            LOGGER.warning("Typing skipped: Discord composer is not empty")
            return None
        if not composer.focused() and self._config.auto_focus_composer:
            composer.set_focus()
            deadline = time.monotonic() + 0.25
            while time.monotonic() < deadline and not composer.focused():
                time.sleep(0.01)
        if not composer.focused():
            LOGGER.warning("Typing skipped: Discord composer is not focused")
            return None
        if composer.value() != "":
            LOGGER.warning("Typing skipped: Discord composer changed while claiming it")
            return None
        LOGGER.info("Claimed empty Discord composer %r", composer.name)
        return composer

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

    def _still_open(self, signature: str, *, check_window: bool = True) -> bool:
        state_valid = self._active_prompt.wait_current_open(
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

    def _wait_until_open(self, deadline: float, signature: str) -> bool:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return self._still_open(signature, check_window=False)
            if self._stop_event.wait(min(remaining, 0.05)):
                return False
            if not self._active_prompt.wait_current_open(
                signature, self._stop_event
            ):
                return False

    def execute(self, task: AnswerTask) -> bool:
        event_token = task.round_token or task.prompt_signature
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

        if not self._cleanup_orphan():
            LOGGER.warning("Typing skipped until the previous owned draft is resolved")
            self._status.emit(
                "ATTENTION",
                title="Waiting for safe draft cleanup",
                detail="A prior owned draft must be resolved before typing",
                question=task.question_label or "Question",
                answer=answer,
                source=task.source,
                event_id=f"{event_token}:orphan",
            )
            return False

        if (
            self._readiness_config.require_green_outline
            and not self._config.draft_while_locked
        ):
            if not self._active_prompt.wait_ready(
                task.prompt_signature,
                self._stop_event,
                self._readiness_config.ready_wait_timeout_seconds,
            ):
                LOGGER.warning("Green-outline gate timed out or the round changed")
                return False

        window = self._guard.current()
        allowed, reason = self._guard.validate(window)
        if not allowed or window is None:
            LOGGER.warning("Typing skipped: %s", reason)
            self._status.emit(
                "ATTENTION",
                title="Discord is not ready",
                detail=reason,
                question=task.question_label or "Question",
                answer=answer,
                source=task.source,
                event_id=f"{event_token}:foreground",
            )
            return False
        composer = self._claim_empty_composer(window)
        if self._config.verify_composer and composer is None:
            self._status.emit(
                "ATTENTION",
                title="Composer safety check blocked typing",
                detail="Use the empty #💜anime-chat message box",
                question=task.question_label or "Question",
                answer=answer,
                source=task.source,
                event_id=f"{event_token}:composer",
            )
            return False

        pre_delay = random.uniform(*self._config.pre_delay_seconds)
        delays = [random.uniform(*self._config.key_delay_seconds) for _ in answer]
        start_at = time.monotonic() + pre_delay
        LOGGER.info(
            "Humanized draft scheduled: answer=%r pre-delay=%.3fs type≈%.3fs",
            answer,
            pre_delay,
            sum(delays),
        )
        if not self._wait_until_open(start_at, task.prompt_signature):
            LOGGER.info("Draft canceled before typing because the prompt changed")
            return False

        typed_characters = 0

        def should_continue() -> bool:
            if not self._still_open(task.prompt_signature, check_window=True):
                return False
            if composer is None:
                return True
            try:
                return (
                    composer.focused()
                    and composer.value() == answer[:typed_characters]
                )
            except Exception:
                LOGGER.exception("Discord composer validation failed")
                return False

        def record_character() -> None:
            nonlocal typed_characters
            typed_characters += 1

        if not should_continue():
            return False
        self._status.emit(
            "DRAFTING",
            title=f"Drafting — {task.question_label or 'Question'}",
            detail="Typing the verified answer during the red reading window",
            question=task.question_label or "Question",
            answer=answer,
            source=task.source,
            readiness="locked",
            event_id=event_token,
            increment="drafts_started",
        )
        try:
            draft_completed = humanize_typing(
                self._controller,
                answer,
                delays,
                should_continue,
                on_character_typed=record_character,
            )
        except Exception:
            LOGGER.warning(
                "Character injection outcome is ambiguous; Enter is blocked",
                exc_info=True,
            )
            if composer is not None:
                try:
                    current = composer.value()
                except Exception:
                    current = ""
                if current and answer.startswith(current):
                    self._remember_orphan(current)
            self._status.emit(
                "ATTENTION",
                title="Draft interrupted",
                detail="Character input was ambiguous; Enter is blocked",
                question=task.question_label or "Question",
                answer=answer,
                source=task.source,
                event_id=f"{event_token}:character-ambiguous",
            )
            return False
        if not draft_completed:
            LOGGER.info("Draft canceled while typing")
            self._clear_or_remember(composer, answer[:typed_characters])
            return False
        if composer is not None:
            try:
                complete_draft = composer.value() == answer
            except Exception:
                LOGGER.warning(
                    "Could not verify the completed Discord draft",
                    exc_info=True,
                )
                self._remember_orphan(answer)
                return False
            if not complete_draft:
                LOGGER.warning("Composer did not contain the exact completed macro draft")
                self._clear_or_remember(composer, answer[:typed_characters])
                return False

        if self._readiness_config.require_green_outline:
            LOGGER.info("Draft complete; waiting for green outline before Enter")
            self._status.emit(
                "WAITING_GREEN",
                title=f"Draft ready — {task.question_label or 'Question'}",
                detail="Answer is complete; Enter remains blocked until green",
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
            ):
                LOGGER.warning("Green-outline gate timed out or the round changed")
                self._clear_or_remember(composer, answer)
                return False
            if self._stop_event.wait(
                self._config.enter_after_open_slack_seconds
            ):
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
            )
        else:
            dispatched = dispatch_enter_if_owned()
        if not dispatched:
            self._clear_or_remember(composer, answer)
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
            "Discord composer accepted [%s] %s -> %s",
            task.source,
            task.question_label or "?",
            answer,
        )
        self._status.emit(
            "SUBMITTED",
            title=f"Submitted — {task.question_label or 'Question'}",
            detail="Discord cleared the composer after Enter",
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
        if not self._active_prompt.is_open(task.prompt_signature):
            LOGGER.debug("Rejected answer task for a non-live prompt")
            return False
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
                try:
                    self._executor.service_orphan()
                except Exception:
                    LOGGER.debug("Orphan-draft service failed", exc_info=True)
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
