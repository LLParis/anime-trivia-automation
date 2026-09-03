from __future__ import annotations

import hashlib
import logging
import queue
import re
import time
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


def describe_emoji(clue: str) -> str:
    """Spell out emoji as lower-case Unicode names so a rebus reads as words.

    ``"🏍️❄️🏚️🥫"`` becomes ``"motorcycle, snowflake, derelict house, canned
    food"``; text-only models resolve that far more reliably than raw glyphs.
    """

    names: list[str] = []
    seen: set[str] = set()
    for character in unicodedata.normalize("NFKC", clue).replace("️", ""):
        category = unicodedata.category(character)
        if character == "‍" or not (category.startswith("S") or ord(character) > 0xFFFF):
            continue
        name = unicodedata.name(character, "").replace("_", " ").lower()
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    return ", ".join(names)


def humanize_answer(answer: str, *, lowercase: bool = True, strip_punctuation: bool = True) -> str:
    """The form a person types: lowercase, no punctuation, single spaces.

    Anime Soul accepted "girls last tour", "steins gate", and "one piece" from
    players on 2026-09-02; nobody types "Girls' Last Tour". Digits and letters
    (any script) are kept, everything else becomes a space.
    """

    value = unicodedata.normalize("NFKC", answer)
    if strip_punctuation:
        # Apostrophes vanish ("natsumes"), every other mark becomes a space
        # ("one punch man", "steins gate").
        value = value.replace("'", "").replace("\u2019", "")
        value = "".join(character if character.isalnum() else " " for character in value)
    if lowercase:
        value = value.casefold()
    value = " ".join(value.split())
    return value or answer


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


def configure_logging(level: str, log_dir: Path | None = None) -> Path | None:
    """Log to the console and, when ``log_dir`` is set, to one file per launch.

    The console launcher closes with the quiz, so a persistent per-launch file
    is the only record of what every round decided. Returns the file path.
    """

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    log_format = "%(asctime)s.%(msecs)03d %(levelname)-8s %(name)s | %(message)s"
    logging.basicConfig(level=numeric_level, format=log_format, datefmt="%H:%M:%S")
    if log_dir is None:
        return None
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        log_path = log_dir / f"anime-trivia-{stamp}.log"
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setLevel(numeric_level)
        handler.setFormatter(logging.Formatter(log_format, "%H:%M:%S"))
        logging.getLogger().addHandler(handler)
        logging.getLogger(__name__).info("Session log file: %s", log_path)
        return log_path
    except OSError:
        logging.getLogger(__name__).exception("Could not open the session log file")
        return None


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
