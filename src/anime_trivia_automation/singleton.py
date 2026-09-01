from __future__ import annotations

import ctypes
import os
from ctypes import wintypes


class WorkerAlreadyRunningError(RuntimeError):
    pass


class WorkerMutex:
    """One live/dry-run automation worker per interactive Windows session."""

    ERROR_ALREADY_EXISTS = 183

    def __init__(self, name: str = "Local\\LLParis.AnimeTriviaAutomation.Worker") -> None:
        self._handle: int | None = None
        if os.name != "nt":
            return
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateMutexW.argtypes = [
            ctypes.c_void_p,
            wintypes.BOOL,
            wintypes.LPCWSTR,
        ]
        kernel32.CreateMutexW.restype = wintypes.HANDLE
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        ctypes.set_last_error(0)
        handle = kernel32.CreateMutexW(None, False, name)
        error = ctypes.get_last_error()
        if not handle:
            raise OSError(ctypes.get_last_error(), "Could not create worker mutex")
        if error == self.ERROR_ALREADY_EXISTS:
            kernel32.CloseHandle(handle)
            raise WorkerAlreadyRunningError(
                "Anime Trivia is already running. Use F12 in the existing worker first."
            )
        self._kernel32 = kernel32
        self._handle = int(handle)

    def close(self) -> None:
        if self._handle is None:
            return
        self._kernel32.CloseHandle(self._handle)
        self._handle = None

    def __enter__(self) -> WorkerMutex:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
