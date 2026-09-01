from __future__ import annotations

import hashlib
import logging
import queue
import re
import unicodedata
from pathlib import Path
from typing import Generic, TypeVar

T = TypeVar("T")


def normalize_question(text: str) -> str:
    """Normalize OCR/cache text without losing words important to fuzzy matching."""
    text = unicodedata.normalize("NFKC", text).casefold()
    text = "".join(character if character.isalnum() else " " for character in text)
    return " ".join(text.split())


def normalize_accessible_clue(text: str) -> str:
    """Normalize exact Discord accessibility text while preserving emoji/ZWJ clues."""

    text = unicodedata.normalize("NFKC", text).casefold()
    text = text.replace("\ufe0f", "")
    return " ".join(text.split()).strip()


def short_signature(namespace: str, value: str) -> str:
    payload = f"{namespace}\0{value}".encode("utf-8", errors="replace")
    return f"{namespace}:{hashlib.sha256(payload).hexdigest()[:24]}"


def sanitize_answer(text: str, max_characters: int) -> str | None:
    if not text:
        return None
    first_line = unicodedata.normalize("NFKC", text).splitlines()[0]
    answer = re.sub(
        r"^\s*(?:final\s+)?answer\s*:\s*", "", first_line, flags=re.IGNORECASE
    )
    answer = answer.strip().strip("`*_\"'").strip()
    answer = re.sub(r"\s+", " ", answer)
    if answer.endswith(".") and answer.count(".") == 1:
        answer = answer[:-1].rstrip()
    if not answer or answer.casefold() in {
        "unknown",
        "unsure",
        "i don't know",
        "i do not know",
    }:
        return None
    if len(answer) > max_characters or any(ord(character) < 32 for character in answer):
        return None
    return answer


class LatestMailbox(Generic[T]):
    """A size-one overwrite queue: inference always receives the newest scene."""

    def __init__(self) -> None:
        self._queue: queue.Queue[T] = queue.Queue(maxsize=1)

    def put(self, item: T) -> None:
        try:
            self._queue.put_nowait(item)
            return
        except queue.Full:
            pass
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass
        self._queue.put_nowait(item)

    def get(self, timeout: float) -> T:
        return self._queue.get(timeout=timeout)


def configure_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s.%(msecs)03d %(levelname)-8s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
