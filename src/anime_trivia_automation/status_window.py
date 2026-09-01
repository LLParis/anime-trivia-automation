from __future__ import annotations

import argparse
import ctypes
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


PHASE_COLORS = {
    "STARTING": "#8b93a7",
    "LOADING": "#5b8def",
    "ARMED": "#28c6d9",
    "RED": "#ed4245",
    "KNOWN": "#57f287",
    "UNKNOWN": "#f0b232",
    "RESOLVING": "#5b8def",
    "NOVEL": "#57f287",
    "DRAFTING": "#8f73ff",
    "WAITING_DISCORD": "#f0b232",
    "MANUAL": "#f0b232",
    "WAITING_GREEN": "#fee75c",
    "GREEN": "#2ecc70",
    "SUBMITTED": "#2ecc70",
    "LEARNED": "#1abc9c",
    "CLOSED": "#8b93a7",
    "QUIZ_COMPLETE": "#28c6d9",
    "ATTENTION": "#f0b232",
    "ERROR": "#ed4245",
    "STALE": "#ed4245",
    "STOPPING": "#8b93a7",
    "STOPPED": "#8b93a7",
}


def _read_status(path: Path) -> dict[str, Any]:
    """Read while explicitly allowing the worker's atomic replace on Windows."""

    if os.name != "nt":
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    kernel32 = ctypes.windll.kernel32
    GENERIC_READ = 0x80000000
    FILE_SHARE_READ = 0x00000001
    FILE_SHARE_WRITE = 0x00000002
    FILE_SHARE_DELETE = 0x00000004
    OPEN_EXISTING = 3
    FILE_ATTRIBUTE_NORMAL = 0x00000080
    kernel32.CreateFileW.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    kernel32.CreateFileW.restype = ctypes.c_void_p
    kernel32.GetFileSizeEx.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_longlong),
    ]
    kernel32.GetFileSizeEx.restype = ctypes.c_bool
    kernel32.ReadFile.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_void_p,
    ]
    kernel32.ReadFile.restype = ctypes.c_bool
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_bool
    handle = kernel32.CreateFileW(
        str(path),
        GENERIC_READ,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        None,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        None,
    )
    invalid = ctypes.c_void_p(-1).value
    if handle == invalid:
        raise OSError(ctypes.get_last_error(), "Could not open status snapshot")
    try:
        size = ctypes.c_longlong()
        if not kernel32.GetFileSizeEx(handle, ctypes.byref(size)):
            raise OSError(ctypes.get_last_error(), "Could not size status snapshot")
        buffer = ctypes.create_string_buffer(max(1, size.value))
        read = ctypes.c_uint32()
        if size.value and not kernel32.ReadFile(
            handle,
            buffer,
            size.value,
            ctypes.byref(read),
            None,
        ):
            raise OSError(ctypes.get_last_error(), "Could not read status snapshot")
        return json.loads(buffer.raw[: read.value].decode("utf-8"))
    finally:
        kernel32.CloseHandle(handle)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status-path", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--worker-pid", required=True, type=int)
    parser.add_argument("--width", type=int, default=560)
    parser.add_argument("--height", type=int, default=310)
    parser.add_argument("--margin-x", type=int, default=32)
    parser.add_argument("--margin-y", type=int, default=32)
    parser.add_argument("--opacity", type=float, default=0.96)
    parser.add_argument("--poll-ms", type=int, default=100)
    parser.add_argument("--stale-after", type=float, default=5.0)
    parser.add_argument("--auto-close", type=float, default=4.0)
    parser.add_argument("--error-close", type=float, default=15.0)
    parser.add_argument("--avoid-region", nargs=4, type=int)
    parser.add_argument("--topmost", action="store_true")
    parser.add_argument("--click-through", action="store_true")
    return parser.parse_args()


def _worker_alive(pid: int) -> bool:
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    kernel32 = ctypes.windll.kernel32
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259
    kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_bool, ctypes.c_uint32]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.GetExitCodeProcess.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    kernel32.GetExitCodeProcess.restype = ctypes.c_bool
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return False
    try:
        code = ctypes.c_uint32()
        return bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(code))) and (
            code.value == STILL_ACTIVE
        )
    finally:
        kernel32.CloseHandle(handle)


def _apply_no_activate_style(
    root: Any, *, topmost: bool, click_through: bool
) -> int | None:
    if os.name != "nt":
        return None
    root.update_idletasks()
    user32 = ctypes.windll.user32
    hwnd = int(root.winfo_id())
    parent = int(user32.GetParent(hwnd))
    if parent:
        hwnd = parent
    get_style = user32.GetWindowLongPtrW
    set_style = user32.SetWindowLongPtrW
    get_style.argtypes = [ctypes.c_void_p, ctypes.c_int]
    get_style.restype = ctypes.c_ssize_t
    set_style.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_ssize_t]
    set_style.restype = ctypes.c_ssize_t
    GWL_EXSTYLE = -20
    WS_EX_TOOLWINDOW = 0x00000080
    WS_EX_TRANSPARENT = 0x00000020
    WS_EX_NOACTIVATE = 0x08000000
    style = int(get_style(hwnd, GWL_EXSTYLE)) | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE
    if click_through:
        style |= WS_EX_TRANSPARENT
    set_style(hwnd, GWL_EXSTYLE, style)
    HWND_TOPMOST = -1
    HWND_TOP = 0
    SWP_NOSIZE = 0x0001
    SWP_NOMOVE = 0x0002
    SWP_NOACTIVATE = 0x0010
    SWP_FRAMECHANGED = 0x0020
    user32.SetWindowPos(
        hwnd,
        HWND_TOPMOST if topmost else HWND_TOP,
        0,
        0,
        0,
        0,
        SWP_NOSIZE | SWP_NOMOVE | SWP_NOACTIVATE | SWP_FRAMECHANGED,
    )
    return hwnd


def _show_without_activation(root: Any, hwnd: int | None) -> None:
    if os.name != "nt" or hwnd is None:
        root.deiconify()
        return
    user32 = ctypes.windll.user32
    SW_SHOWNOACTIVATE = 4
    user32.ShowWindow.argtypes = [ctypes.c_void_p, ctypes.c_int]
    user32.ShowWindow.restype = ctypes.c_bool
    user32.ShowWindow(hwnd, SW_SHOWNOACTIVATE)


def main() -> int:
    args = _parse_args()
    import tkinter as tk

    root = tk.Tk()
    # Never map a normal activating window, even for a single startup frame.
    root.withdraw()
    root.title("Anime Trivia Status")
    root.overrideredirect(True)
    root.configure(bg="#11131a")
    root.attributes("-alpha", args.opacity)
    if args.topmost:
        root.attributes("-topmost", True)

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    candidates = [
        (screen_width - args.width - args.margin_x, args.margin_y),
        (
            screen_width - args.width - args.margin_x,
            screen_height - args.height - args.margin_y,
        ),
        (args.margin_x, args.margin_y),
        (args.margin_x, screen_height - args.height - args.margin_y),
    ]

    def intersects_avoid_region(candidate_x: int, candidate_y: int) -> bool:
        if args.avoid_region is None:
            return False
        scale = max(0.5, float(root.winfo_fpixels("1i")) / 96.0)
        left, top, right, bottom = (value / scale for value in args.avoid_region)
        return not (
            candidate_x + args.width <= left
            or candidate_x >= right
            or candidate_y + args.height <= top
            or candidate_y >= bottom
        )

    placement = next(
        (
            (max(0, candidate_x), max(0, candidate_y))
            for candidate_x, candidate_y in candidates
            if not intersects_avoid_region(candidate_x, candidate_y)
        ),
        None,
    )
    if placement is None:
        root.withdraw()
        x, y = 0, 0
    else:
        x, y = placement
    root.geometry(f"{args.width}x{args.height}+{x}+{y}")

    accent = tk.Frame(root, bg="#8b93a7", width=8)
    accent.pack(side="left", fill="y")
    body = tk.Frame(root, bg="#11131a", padx=18, pady=14)
    body.pack(side="left", fill="both", expand=True)

    header_row = tk.Frame(body, bg="#11131a")
    header_row.pack(fill="x")
    phase_label = tk.Label(
        header_row,
        text="STARTING",
        fg="#11131a",
        bg="#8b93a7",
        font=("Segoe UI Semibold", 11),
        padx=10,
        pady=3,
    )
    phase_label.pack(side="left")
    mode_label = tk.Label(
        header_row,
        text="LIVE",
        fg="#aeb4c5",
        bg="#11131a",
        font=("Segoe UI Semibold", 10),
    )
    mode_label.pack(side="right")

    title_label = tk.Label(
        body,
        text="Starting Anime Trivia",
        anchor="w",
        fg="#f4f5f8",
        bg="#11131a",
        font=("Segoe UI Semibold", 18),
    )
    title_label.pack(fill="x", pady=(10, 2))
    detail_label = tk.Label(
        body,
        text="Opening the operator panel",
        anchor="w",
        fg="#aeb4c5",
        bg="#11131a",
        font=("Segoe UI", 10),
    )
    detail_label.pack(fill="x")

    clue_label = tk.Label(
        body,
        text="Waiting for startup",
        anchor="nw",
        justify="left",
        wraplength=args.width - 58,
        fg="#d7daea",
        bg="#181b24",
        font=("Segoe UI", 11),
        padx=10,
        pady=8,
    )
    clue_label.pack(fill="x", pady=(10, 7))

    answer_label = tk.Label(
        body,
        text="ANSWER  —",
        anchor="w",
        fg="#f4f5f8",
        bg="#11131a",
        font=("Segoe UI Semibold", 14),
    )
    answer_label.pack(fill="x")
    source_label = tk.Label(
        body,
        text="SOURCE  —",
        anchor="w",
        fg="#8b93a7",
        bg="#11131a",
        font=("Segoe UI", 9),
    )
    source_label.pack(fill="x", pady=(1, 8))

    counters_label = tk.Label(
        body,
        text="Seen 0   Known 0   Unknown 0   Sent 0   Learned 0",
        anchor="w",
        fg="#aeb4c5",
        bg="#11131a",
        font=("Consolas", 9),
    )
    counters_label.pack(fill="x")
    footer_label = tk.Label(
        body,
        text="F12 stops • Do not type while a draft is active",
        anchor="w",
        fg="#676d7f",
        bg="#11131a",
        font=("Segoe UI", 8),
    )
    footer_label.pack(fill="x", pady=(5, 0))

    if placement is not None:
        hwnd = _apply_no_activate_style(
            root,
            topmost=args.topmost,
            click_through=args.click_through,
        )
        _show_without_activation(root, hwnd)

    last_sequence = -1
    last_state: dict[str, Any] | None = None
    terminal_at: float | None = None

    def render(state: dict[str, Any], *, stale: bool = False) -> None:
        phase = "STALE" if stale else str(state.get("phase", "STARTING"))
        color = PHASE_COLORS.get(phase, "#8b93a7")
        accent.configure(bg=color)
        phase_label.configure(text=phase.replace("_", " "), bg=color)
        mode_label.configure(text=str(state.get("mode", "LIVE")))
        title_label.configure(text=str(state.get("title", phase.title())))
        detail = str(state.get("detail", ""))
        if stale:
            detail = "No status update from the automation process"
        detail_label.configure(text=detail)
        question = str(state.get("question", "—"))
        clue = str(state.get("clue", "—"))
        clue_label.configure(text=f"{question}  {clue}" if question != "—" else clue)
        answer_label.configure(text=f"ANSWER  {state.get('answer', '—')}")
        source_label.configure(
            text=f"SOURCE  {state.get('source', '—')}   •   {state.get('readiness', 'unknown')}"
        )
        counters = state.get("counters", {})
        counters_label.configure(
            text=(
                f"Seen {counters.get('rounds_seen', 0)}   "
                f"Known {counters.get('known', 0)}   "
                f"Unknown {counters.get('unknown', 0)}   "
                f"Sent {counters.get('submitted', 0)}   "
                f"Learned {counters.get('learned', 0)}"
            )
        )

    def poll() -> None:
        nonlocal last_sequence, last_state, terminal_at
        try:
            state = _read_status(args.status_path)
            if state.get("run_id") != args.run_id:
                root.destroy()
                return
            sequence = int(state.get("sequence", 0))
            if sequence != last_sequence:
                last_sequence = sequence
                last_state = state
                render(state)
            updated = datetime.fromisoformat(str(state["updated_at"]))
            if updated.tzinfo is None:
                updated = updated.replace(tzinfo=UTC)
            age = (datetime.now(UTC) - updated).total_seconds()
            worker_alive = _worker_alive(args.worker_pid)
            phase = str(state.get("phase", "STARTING"))
            # CUDA/OCR cold starts can legitimately take tens of seconds.  The
            # panel may report a stale stream, but it must never disappear while
            # its owning worker is still alive and may recover.
            stale_after = (
                max(args.stale_after, 60.0)
                if phase in {"STARTING", "LOADING"}
                else args.stale_after
            )
            stale = age > stale_after or not worker_alive
            if stale and state.get("phase") not in {"STOPPED", "ERROR"}:
                render(state, stale=True)
            if state.get("phase") == "STOPPED":
                if terminal_at is None:
                    terminal_at = time.monotonic()
                if time.monotonic() - terminal_at >= args.auto_close:
                    root.destroy()
                    return
            elif not worker_alive:
                if terminal_at is None:
                    terminal_at = time.monotonic()
                if time.monotonic() - terminal_at >= args.error_close:
                    root.destroy()
                    return
            else:
                terminal_at = None
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            worker_alive = _worker_alive(args.worker_pid)
            if last_state is not None:
                render(last_state, stale=True)
            elif not worker_alive:
                render(
                    {
                        "phase": "STALE",
                        "title": "Automation process ended",
                        "detail": "The status stream never became available",
                        "readiness": "closed",
                    },
                    stale=True,
                )
            if not worker_alive:
                if terminal_at is None:
                    terminal_at = time.monotonic()
                if time.monotonic() - terminal_at >= args.error_close:
                    root.destroy()
                    return
        root.after(args.poll_ms, poll)

    poll()
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
