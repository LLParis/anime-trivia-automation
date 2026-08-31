from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import MatchConfig
from .models import CacheHit
from .utils import normalize_question

LOGGER = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class TriviaCache:
    """In-memory fuzzy/pHash indexes backed by an atomically replaced JSON file."""

    def __init__(
        self, path: Path, config: MatchConfig, seed_path: Path | None = None
    ) -> None:
        self._path = path
        self._seed_path = seed_path
        self._config = config
        self._lock = threading.RLock()
        self._data: dict[str, Any] = {}
        self._text_index: dict[str, tuple[str, str]] = {}
        self._image_index: dict[str, Any] = {}
        self._load()

    @property
    def path(self) -> Path:
        return self._path

    def _load(self) -> None:
        with self._lock:
            should_create_local = not self._path.exists()
            source_path = self._path
            if (
                should_create_local
                and self._seed_path is not None
                and self._seed_path.exists()
            ):
                source_path = self._seed_path
            if source_path.exists():
                with source_path.open("r", encoding="utf-8") as handle:
                    raw = json.load(handle)
            else:
                raw = {
                    "schema_version": 1,
                    "text_questions": {},
                    "image_hashes": {},
                    "metadata": {"text_questions": {}, "image_hashes": {}},
                }
            if not isinstance(raw, dict) or raw.get("schema_version") != 1:
                raise ValueError(f"Unsupported cache schema in {self._path}")
            text_questions = raw.get("text_questions", {})
            image_hashes = raw.get("image_hashes", {})
            if not isinstance(text_questions, dict) or not isinstance(
                image_hashes, dict
            ):
                raise TypeError("Cache answer maps must be JSON objects")
            for key, value in [*text_questions.items(), *image_hashes.items()]:
                if (
                    not isinstance(key, str)
                    or not isinstance(value, str)
                    or not value.strip()
                ):
                    raise ValueError("Cache keys and answers must be non-empty strings")
            metadata = raw.setdefault(
                "metadata", {"text_questions": {}, "image_hashes": {}}
            )
            if not isinstance(metadata, dict):
                raise TypeError("Cache metadata must be a JSON object")
            metadata.setdefault("text_questions", {})
            metadata.setdefault("image_hashes", {})
            if not isinstance(metadata["text_questions"], dict) or not isinstance(
                metadata["image_hashes"], dict
            ):
                raise TypeError("Cache metadata maps must be JSON objects")
            self._data = raw
            self._rebuild_indexes_locked()
            if should_create_local:
                self._save_locked()
            LOGGER.info(
                "Cache loaded: %d text questions, %d image hashes (%s)",
                len(self._text_index),
                len(self._image_index),
                self._path,
            )

    def _rebuild_indexes_locked(self) -> None:
        self._text_index.clear()
        for original_key, answer in self._data["text_questions"].items():
            normalized = normalize_question(original_key)
            if normalized:
                previous = self._text_index.get(normalized)
                if previous is not None and previous[0] != original_key:
                    raise ValueError(
                        "Text cache contains keys that normalize identically: "
                        f"{previous[0]!r} and {original_key!r}"
                    )
                self._text_index[normalized] = (original_key, answer)

        self._image_index.clear()
        try:
            import imagehash
        except ImportError as exc:
            raise RuntimeError("ImageHash is required to load the image cache") from exc
        expected_hex_length = self._config.phash_size * self._config.phash_size // 4
        for hash_text, answer in self._data["image_hashes"].items():
            if len(hash_text) != expected_hex_length:
                LOGGER.warning("Skipping pHash with wrong length: %s", hash_text)
                continue
            try:
                parsed = imagehash.hex_to_hash(hash_text)
            except ValueError:
                LOGGER.warning("Skipping invalid pHash key: %s", hash_text)
                continue
            self._image_index[hash_text] = (parsed, answer)

    def match_text(self, question: str) -> CacheHit | None:
        query = normalize_question(question)
        if not query:
            return None
        with self._lock:
            direct = self._text_index.get(query)
            if direct is not None:
                original_key, answer = direct
                return CacheHit(
                    kind="text", key=original_key, answer=answer, score=100.0
                )
            choices = list(self._text_index.keys())
        if not choices:
            return None

        try:
            from rapidfuzz import fuzz, process
        except ImportError as exc:
            raise RuntimeError("RapidFuzz is required for text cache matching") from exc

        ranked = process.extract(
            query,
            choices,
            scorer=fuzz.WRatio,
            limit=2,
        )
        if not ranked:
            return None
        best_key, best_score, _ = ranked[0]
        if float(best_score) < self._config.text_score_threshold:
            return None
        runner_score = float(ranked[1][1]) if len(ranked) > 1 else None
        if (
            runner_score is not None
            and float(best_score) - runner_score < self._config.text_score_margin
        ):
            LOGGER.warning(
                "Ambiguous text cache match rejected: best=%.1f runner-up=%.1f",
                best_score,
                runner_score,
            )
            return None
        with self._lock:
            original_key, answer = self._text_index[best_key]
        return CacheHit(
            kind="text",
            key=original_key,
            answer=answer,
            score=float(best_score),
            runner_up_score=runner_score,
        )

    def match_image(self, hash_text: str) -> CacheHit | None:
        try:
            import imagehash
        except ImportError as exc:
            raise RuntimeError(
                "ImageHash is required for image cache matching"
            ) from exc
        try:
            query = imagehash.hex_to_hash(hash_text)
        except ValueError:
            return None
        with self._lock:
            candidates = [
                (int(query - parsed), key, answer)
                for key, (parsed, answer) in self._image_index.items()
            ]
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        best_distance, best_key, answer = candidates[0]
        runner_distance = candidates[1][0] if len(candidates) > 1 else None
        if best_distance > self._config.phash_max_distance:
            return None
        if (
            runner_distance is not None
            and runner_distance - best_distance < self._config.phash_distance_margin
        ):
            LOGGER.warning(
                "Ambiguous image cache match rejected: best=%d runner-up=%d",
                best_distance,
                runner_distance,
            )
            return None
        return CacheHit(
            kind="image",
            key=best_key,
            answer=answer,
            score=float(best_distance),
            runner_up_score=float(runner_distance)
            if runner_distance is not None
            else None,
        )

    def add_text(self, question: str, answer: str, *, source: str) -> None:
        key = normalize_question(question)
        if not key:
            raise ValueError("Cannot cache an empty text question")
        with self._lock:
            existing = self._data["text_questions"].get(key)
            if existing is not None and existing != answer:
                LOGGER.warning("Preserving existing cache answer for text key %r", key)
                return
            previous_data = deepcopy(self._data)
            try:
                self._data["text_questions"][key] = answer
                self._data["metadata"]["text_questions"][key] = {
                    "source": source,
                    "created_utc": _utc_now(),
                }
                self._rebuild_indexes_locked()
                self._save_locked()
            except Exception:
                self._data = previous_data
                self._rebuild_indexes_locked()
                raise
        LOGGER.info("Cached novel text clue -> %s", answer)

    def add_image(self, hash_text: str, answer: str, *, source: str) -> None:
        with self._lock:
            existing = self._data["image_hashes"].get(hash_text)
            if existing is not None and existing != answer:
                LOGGER.warning(
                    "Preserving existing cache answer for image hash %s", hash_text
                )
                return
            previous_data = deepcopy(self._data)
            try:
                self._data["image_hashes"][hash_text] = answer
                self._data["metadata"]["image_hashes"][hash_text] = {
                    "source": source,
                    "hash_size": self._config.phash_size,
                    "created_utc": _utc_now(),
                }
                self._rebuild_indexes_locked()
                self._save_locked()
            except Exception:
                self._data = previous_data
                self._rebuild_indexes_locked()
                raise
        LOGGER.info("Cached novel visual clue %s -> %s", hash_text, answer)

    def _save_locked(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        serialized = (
            json.dumps(self._data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                prefix=f".{self._path.name}.",
                suffix=".tmp",
                dir=self._path.parent,
                delete=False,
            ) as handle:
                temporary_path = handle.name
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self._path)
        finally:
            if temporary_path and os.path.exists(temporary_path):
                try:
                    os.unlink(temporary_path)
                except OSError:
                    pass
