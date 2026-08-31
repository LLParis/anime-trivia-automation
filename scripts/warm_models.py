from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from anime_trivia_automation.config import load_config
from anime_trivia_automation.ocr import PaddleOCREngine
from anime_trivia_automation.utils import configure_logging
from anime_trivia_automation.vlm import LazyQwenResolver


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download, load, and warm all live models before play."
    )
    parser.add_argument("--config", default=str(REPO_ROOT / "config.json"))
    parser.add_argument("--skip-vlm", action="store_true", help="Warm only PaddleOCR")
    args = parser.parse_args()
    config = load_config(args.config)
    configure_logging(config.runtime.log_level)
    PaddleOCREngine(config.ocr)
    if not args.skip_vlm and config.vlm.enabled:
        LazyQwenResolver(config.vlm).ensure_loaded()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
