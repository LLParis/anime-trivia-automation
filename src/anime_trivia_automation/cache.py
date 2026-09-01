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
from .utils import normalize_accessible_clue, normalize_question

LOGGER = logging.getLogger(__name__)

# Versions before 0.5.0 promoted a single model guess and a spatially
# unscoped OCR reveal directly into the durable cache.  A failed live round
# proved that both sources can be wrong.  Purge those legacy records whenever
# an existing local cache is opened; trusted seed data is overlaid afterward.
UNTRUSTED_LEGACY_SOURCES = {
    "qwen3-vl",
    "authoritative-round-reveal",
}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class TriviaCache:
    """In-memory fuzzy/pHash indexes backed by an atomically replaced JSON file."""

    def __init__(
        self,
        path: Path,
        config: MatchConfig,
        seed_path: Path | None = None,
        history_path: Path | None = None,
    ) -> None:
        self._path = path
        self._seed_path = seed_path
        self._history_path = history_path
        self._config = config
        self._lock = threading.RLock()
        self._data: dict[str, Any] = {}
        self._text_index: dict[str, tuple[str, str]] = {}
        self._image_index: dict[str, Any] = {}
        self._semantic_index: dict[tuple[str, str], tuple[str, str]] = {}
        self._history_exact: dict[tuple[str, str], str] = {}
        self._history_text: dict[tuple[str, str], str] = {}
        self._load()
        self._load_history()

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
                    "semantic_questions": {},
                    "metadata": {
                        "text_questions": {},
                        "image_hashes": {},
                        "semantic_questions": {},
                    },
                }
            if not isinstance(raw, dict) or raw.get("schema_version") != 1:
                raise ValueError(f"Unsupported cache schema in {self._path}")
            text_questions = raw.get("text_questions", {})
            image_hashes = raw.get("image_hashes", {})
            semantic_questions = raw.setdefault("semantic_questions", {})
            if not isinstance(text_questions, dict) or not isinstance(
                image_hashes, dict
            ):
                raise TypeError("Cache answer maps must be JSON objects")
            if not isinstance(semantic_questions, dict):
                raise TypeError("Semantic cache map must be a JSON object")
            for key, value in [
                *text_questions.items(),
                *image_hashes.items(),
                *semantic_questions.items(),
            ]:
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
            metadata.setdefault("semantic_questions", {})
            if not isinstance(metadata["text_questions"], dict) or not isinstance(
                metadata["image_hashes"], dict
            ):
                raise TypeError("Cache metadata maps must be JSON objects")
            if not isinstance(metadata["semantic_questions"], dict):
                raise TypeError("Semantic cache metadata must be a JSON object")

            changed = False
            for namespace in ("text_questions", "image_hashes"):
                namespace_metadata = metadata[namespace]
                for key, record in list(namespace_metadata.items()):
                    source = record.get("source") if isinstance(record, dict) else None
                    if source not in UNTRUSTED_LEGACY_SOURCES:
                        continue
                    raw[namespace].pop(key, None)
                    namespace_metadata.pop(key, None)
                    changed = True
                    LOGGER.warning(
                        "Removed untrusted legacy cache entry %s:%s (%s)",
                        namespace,
                        key,
                        source,
                    )

            # The repository seed is reviewed data and is authoritative over
            # the mutable runtime file.  Always overlay it so a corrected seed
            # repairs an already-created local cache on the next launch.
            if self._seed_path is not None and self._seed_path.exists():
                with self._seed_path.open("r", encoding="utf-8") as handle:
                    seed = json.load(handle)
                if not isinstance(seed, dict) or seed.get("schema_version") != 1:
                    raise ValueError(f"Unsupported cache schema in {self._seed_path}")
                seed_metadata = seed.get("metadata", {})
                for namespace in ("text_questions", "image_hashes"):
                    seed_values = seed.get(namespace, {})
                    if not isinstance(seed_values, dict):
                        raise TypeError("Seed cache answer maps must be JSON objects")
                    before = dict(raw[namespace])
                    raw[namespace].update(seed_values)
                    changed = changed or before != raw[namespace]
                    source_metadata = (
                        seed_metadata.get(namespace, {})
                        if isinstance(seed_metadata, dict)
                        else {}
                    )
                    if isinstance(source_metadata, dict):
                        before_metadata = dict(metadata[namespace])
                        managed_sources = {
                            record.get("source")
                            for record in source_metadata.values()
                            if isinstance(record, dict) and record.get("source")
                        }
                        for key, record in list(metadata[namespace].items()):
                            source = (
                                record.get("source")
                                if isinstance(record, dict)
                                else None
                            )
                            if source in managed_sources and key not in seed_values:
                                raw[namespace].pop(key, None)
                                metadata[namespace].pop(key, None)
                                changed = True
                        metadata[namespace].update(source_metadata)
                        changed = changed or before_metadata != metadata[namespace]

            self._data = raw
            self._rebuild_indexes_locked()
            if should_create_local or changed:
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

        self._semantic_index.clear()
        for stored_key, answer in self._data["semantic_questions"].items():
            answer_type, separator, normalized = stored_key.partition(":")
            if separator and answer_type in {"character", "anime_title"} and normalized:
                self._semantic_index[(answer_type, normalized)] = (stored_key, answer)

    def _load_history(self) -> None:
        if self._history_path is None or not self._history_path.exists():
            return
        with self._history_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict) or raw.get("schema_version") != 1:
            raise ValueError(f"Unsupported history schema in {self._history_path}")
        pairs = raw.get("pairs", [])
        if not isinstance(pairs, list):
            raise TypeError("History pairs must be a JSON array")
        for pair in pairs:
            if not isinstance(pair, dict):
                raise TypeError("Each history pair must be a JSON object")
            clue = pair.get("clue")
            expected_type = pair.get("type")
            answer = pair.get("answer")
            if (
                not isinstance(clue, str)
                or expected_type not in {"character", "anime_title"}
                or not isinstance(answer, str)
                or not answer.strip()
            ):
                raise ValueError("History pairs require clue, type, and answer")
            exact_key = (expected_type, normalize_accessible_clue(clue))
            text_key = (expected_type, normalize_question(clue))
            indexes = [(self._history_exact, exact_key)]
            if text_key[1]:
                indexes.append((self._history_text, text_key))
            for index, key in indexes:
                existing = index.get(key)
                if existing is not None and existing != answer:
                    raise ValueError(f"Conflicting history answers for {clue!r}")
                index[key] = answer
        LOGGER.info(
            "Authoritative Discord history loaded: %d clues (%s)",
            len(self._history_exact),
            self._history_path,
        )

    def match_history(
        self, clue: str, expected_answer_type: str
    ) -> CacheHit | None:
        exact_key = (
            expected_answer_type,
            normalize_accessible_clue(clue),
        )
        runtime = self._semantic_index.get(exact_key)
        if runtime is not None:
            stored_key, answer = runtime
            return CacheHit(
                kind="history",
                key=stored_key,
                answer=answer,
                score=100.0,
            )
        answer = self._history_exact.get(exact_key)
        if answer is not None:
            return CacheHit(
                kind="history",
                key=exact_key[1],
                answer=answer,
                score=100.0,
            )

        normalized = normalize_question(clue)
        if not normalized:
            return None
        choices = [
            key
            for answer_type, key in self._history_text
            if answer_type == expected_answer_type
        ]
        if not choices:
            return None
        try:
            from rapidfuzz import fuzz, process
        except ImportError as exc:
            raise RuntimeError("RapidFuzz is required for history matching") from exc
        ranked = process.extract(
            normalized,
            choices,
            scorer=fuzz.WRatio,
            limit=2,
        )
        if not ranked or float(ranked[0][1]) < self._config.text_score_threshold:
            return None
        best_key, best_score, _ = ranked[0]
        runner_score = float(ranked[1][1]) if len(ranked) > 1 else None
        if (
            runner_score is not None
            and float(best_score) - runner_score < self._config.text_score_margin
        ):
            return None
        answer = self._history_text[(expected_answer_type, best_key)]
        return CacheHit(
            kind="history",
            key=best_key,
            answer=answer,
            score=float(best_score),
            runner_up_score=runner_score,
        )

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

    def add_semantic(
        self,
        clue: str,
        expected_answer_type: str,
        answer: str,
        *,
        source: str,
    ) -> None:
        if expected_answer_type not in {"character", "anime_title"}:
            raise ValueError("Semantic clue answer type is invalid")
        normalized = normalize_accessible_clue(clue)
        if not normalized:
            raise ValueError("Cannot cache an empty semantic clue")
        key = f"{expected_answer_type}:{normalized}"
        with self._lock:
            existing = self._data["semantic_questions"].get(key)
            if existing is not None and existing != answer:
                LOGGER.warning("Preserving existing semantic answer for %r", clue)
                return
            previous_data = deepcopy(self._data)
            try:
                self._data["semantic_questions"][key] = answer
                self._data["metadata"]["semantic_questions"][key] = {
                    "source": source,
                    "created_utc": _utc_now(),
                    "clue": clue,
                }
                self._rebuild_indexes_locked()
                self._save_locked()
            except Exception:
                self._data = previous_data
                self._rebuild_indexes_locked()
                raise
        LOGGER.info("Cached authoritative semantic clue -> %s", answer)

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
