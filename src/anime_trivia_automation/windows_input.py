from __future__ import annotations

import ctypes
import os
import threading
from collections.abc import Callable, Sequence
from ctypes import wintypes
from typing import Any


INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
INPUT_HARDWARE = 2

KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class _HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class _INPUTUNION(ctypes.Union):
    _fields_ = [
        ("mi", _MOUSEINPUT),
        ("ki", _KEYBDINPUT),
        ("hi", _HARDWAREINPUT),
    ]


class _INPUT(ctypes.Structure):
    _anonymous_ = ("value",)
    _fields_ = [
        ("type", wintypes.DWORD),
        ("value", _INPUTUNION),
    ]


SendInputCallable = Callable[[int, Any, int], int]
LastErrorCallable = Callable[[], int]


class BatchedWindowsInput:
    """Inject complete Unicode text with one bounded native batch.

    Text is converted to UTF-16 code units and submitted as one ``SendInput``
    call containing a key-down/key-up pair for every unit. This never uses the
    clipboard and never exposes a character-at-a-time call boundary to Python.
    """

    def __init__(
        self,
        *,
        send_input: SendInputCallable | None = None,
        last_error: LastErrorCallable | None = None,
    ) -> None:
        if send_input is None:
            if os.name != "nt":
                raise RuntimeError("BatchedWindowsInput requires Windows")
            user32 = ctypes.WinDLL("user32", use_last_error=True)
            native_send_input = user32.SendInput
            native_send_input.argtypes = [
                wintypes.UINT,
                ctypes.POINTER(_INPUT),
                ctypes.c_int,
            ]
            native_send_input.restype = wintypes.UINT
            send_input = native_send_input
            last_error = ctypes.get_last_error
        self._send_input = send_input
        self._last_error = last_error or (lambda: 0)
        self._lock = threading.Lock()

    @staticmethod
    def _keyboard_event(*, virtual_key: int, scan_code: int, flags: int) -> _INPUT:
        event = _INPUT()
        event.type = INPUT_KEYBOARD
        event.ki = _KEYBDINPUT(
            wVk=virtual_key,
            wScan=scan_code,
            dwFlags=flags,
            time=0,
            dwExtraInfo=0,
        )
        return event

    @classmethod
    def unicode_events(cls, text: str) -> tuple[_INPUT, ...]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        try:
            encoded = text.encode("utf-16-le", errors="strict")
        except UnicodeEncodeError as exc:
            raise ValueError("text contains an unpaired UTF-16 surrogate") from exc
        units = [
            int.from_bytes(encoded[index : index + 2], "little")
            for index in range(0, len(encoded), 2)
        ]
        if any(unit == 0 for unit in units):
            raise ValueError("text cannot contain a NUL code unit")
        events: list[_INPUT] = []
        for unit in units:
            events.append(
                cls._keyboard_event(
                    virtual_key=0,
                    scan_code=unit,
                    flags=KEYEVENTF_UNICODE,
                )
            )
            events.append(
                cls._keyboard_event(
                    virtual_key=0,
                    scan_code=unit,
                    flags=KEYEVENTF_UNICODE | KEYEVENTF_KEYUP,
                )
            )
        return tuple(events)

    def send_text(self, text: str) -> int:
        events = self.unicode_events(text)
        if not events:
            return 0
        return self._send_exact(events, operation="Unicode text")

    def _send_exact(self, events: Sequence[_INPUT], *, operation: str) -> int:
        expected = len(events)
        if expected < 1:
            return 0
        array_type = _INPUT * expected
        inputs = array_type(*events)
        with self._lock:
            sent = int(self._send_input(expected, inputs, ctypes.sizeof(_INPUT)))
        if sent != expected:
            error = int(self._last_error())
            detail = f"{operation} SendInput accepted {sent} of {expected} events"
            if error:
                raise OSError(error, detail)
            raise OSError(detail)
        return sent


__all__ = [
    "BatchedWindowsInput",
    "KEYEVENTF_KEYUP",
    "KEYEVENTF_UNICODE",
]
