from __future__ import annotations

import argparse
import ctypes
import logging
import os
from pathlib import Path

from .app import AnimeTriviaAutomation, inspect_image, print_inspection
from .config import load_config
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
    configure_logging(config.runtime.log_level)
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

    AnimeTriviaAutomation(config, dry_run=args.dry_run).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
