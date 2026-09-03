from __future__ import annotations

import argparse
import ctypes
import logging
import os
from pathlib import Path

from .app import AnimeTriviaAutomation, inspect_image, print_inspection
from .config import load_config
from .report import write_quiz_report
from .singleton import WorkerAlreadyRunningError, WorkerMutex
from .status import NullStatus, OperatorStatus
from .utils import configure_logging


def _enable_physical_pixel_coordinates() -> None:
    if os.name != "nt":
        return
    try:
        # PROCESS_PER_MONITOR_DPI_AWARE. This keeps config coordinates aligned
        # with the physical pixels consumed by Desktop Duplication.
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anime-trivia",
        description="GPU-first Anime Soul trivia capture, lookup, and humanized answer typing.",
    )
    parser.add_argument(
        "--config", default="config.json", help="Path to the JSON configuration"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve answers but never send keystrokes",
    )
    parser.add_argument(
        "--validate-config", action="store_true", help="Validate configuration and exit"
    )
    parser.add_argument(
        "--print-outputs",
        action="store_true",
        help="Print DXcam device/output information and exit",
    )
    parser.add_argument(
        "--inspect-image",
        type=Path,
        help="Analyze a saved screenshot instead of capturing the desktop",
    )
    parser.add_argument(
        "--use-vlm",
        action="store_true",
        help="Allow the large local VLM slow path during --inspect-image",
    )
    parser.add_argument(
        "--rehearse",
        action="store_true",
        help=(
            "Run the full live path (activation, composer claim, typing, verification) "
            "but withhold Enter so the typed answer can be inspected and deleted"
        ),
    )
    parser.add_argument(
        "--report",
        nargs="?",
        const=1,
        type=int,
        metavar="RUNS",
        help="Print a per-round report of the last RUNS launches from the ledger and exit",
    )
    return parser


def main() -> int:
    _enable_physical_pixel_coordinates()
    args = build_parser().parse_args()
    if args.print_outputs:
        try:
            import dxcam
        except ImportError as exc:
            raise SystemExit(
                "dxcam is not installed; run scripts/install_windows.ps1"
            ) from exc
        print(dxcam.device_info())
        print(dxcam.output_info())
        return 0

    config = load_config(args.config)
    if args.report is not None:
        if config.runtime.ledger_path is None or not config.runtime.ledger_path.exists():
            print("No round ledger exists yet.")
            return 1
        text, out = write_quiz_report(
            config.runtime.ledger_path,
            config.runtime.log_dir or config.runtime.ledger_path.parent,
            runs=int(args.report),
        )
        print(text)
        print(f"report: {out}")
        return 0
    configure_logging(
        config.runtime.log_level,
        log_dir=None if (args.validate_config or args.inspect_image) else config.runtime.log_dir,
    )
    if args.validate_config:
        logging.getLogger(__name__).info(
            "Configuration is valid: %s", Path(args.config).resolve()
        )
        return 0
    if args.inspect_image:
        result = inspect_image(
            config, args.inspect_image.resolve(), use_vlm=args.use_vlm
        )
        print_inspection(result)
        return 0

    try:
        worker_mutex = WorkerMutex()
    except WorkerAlreadyRunningError as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 2
    try:
        status: OperatorStatus | NullStatus
        if config.status.enabled:
            status = OperatorStatus(
                config.status,
                config.runtime.status_path,
                dry_run=args.dry_run,
                # The panel is placed on the primary display.  Output-local
                # capture coordinates are still conservative here: they either
                # describe that display or make us avoid extra primary space.
                avoid_region=config.capture.region,
                ledger_path=config.runtime.ledger_path,
            )
        else:
            status = NullStatus()
        status.launch()
        status.emit(
            "LOADING",
            title="Loading OCR",
            detail="Initializing CUDA capture, OCR, and verified history",
            readiness="unknown",
        )
        try:
            if args.rehearse:
                from dataclasses import replace as _replace

                config = _replace(config, typing=_replace(config.typing, press_enter=False))
                logging.getLogger(__name__).warning(
                    "REHEARSAL: answers will be typed into Discord but Enter is withheld"
                )
            AnimeTriviaAutomation(
                config,
                dry_run=args.dry_run,
                status=status,
            ).run()
        except Exception as exc:
            status.emit(
                "ERROR",
                title="Startup or runtime failure",
                detail=f"{type(exc).__name__}: {exc}",
                readiness="closed",
                event_id=f"fatal:{type(exc).__name__}:{exc}",
                increment="fatal_errors",
            )
            raise
        finally:
            status.close()
    finally:
        worker_mutex.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
