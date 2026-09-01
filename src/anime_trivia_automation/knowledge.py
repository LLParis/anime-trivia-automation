from __future__ import annotations

import html
import logging
import re
import sqlite3
import threading
import unicodedata
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger(__name__)

_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "her",
    "his",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "she",
    "that",
    "the",
    "their",
    "they",
    "this",
    "to",
    "who",
    "with",
}


def normalize_text(value: str) -> str:
    """Normalize exact quote/title keys consistently across builder and reader."""

    value = unicodedata.normalize("NFKC", value).casefold()
    value = "".join(character if character.isalnum() else " " for character in value)
    return " ".join(value.split())


def strip_html(value: str) -> str:
    """Collapse API HTML fragments into compact, searchable plain text."""

    value = re.sub(r"(?i)<br\s*/?>", "\n", value or "")
    value = re.sub(r"<[^>]+>", " ", value)
    return " ".join(html.unescape(value).split())


class KnowledgeIndex:
    """Thread-safe, read-only access to the generated local anime index."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path).resolve()
        self._lock = threading.RLock()
        self._connection: sqlite3.Connection | None = None
        self._available = False
        self._count = 0
        try:
            if not self._path.is_file():
                return
            uri = self._path.as_uri() + "?mode=ro"
            connection = sqlite3.connect(
                uri,
                uri=True,
                check_same_thread=False,
                timeout=1.0,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA query_only=ON")
            schema = connection.execute(
                "SELECT value FROM schema_info WHERE key='schema_version'"
            ).fetchone()
            fts = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='record_fts'"
            ).fetchone()
            if schema is None or str(schema[0]) != "1" or fts is None:
                connection.close()
                return
            self._count = int(
                connection.execute("SELECT COUNT(*) FROM records").fetchone()[0]
            )
            self._connection = connection
            self._available = True
        except (OSError, sqlite3.Error, TypeError, ValueError):
            LOGGER.exception("Local anime knowledge index is unavailable")
            self._connection = None
            self._available = False
            self._count = 0

    @property
    def available(self) -> bool:
        return self._available and self._connection is not None

    @property
    def count(self) -> int:
        return self._count if self.available else 0

    def exact_quote(self, clue: str) -> dict[str, Any] | None:
        """Return the source-attributed exact normalized quote answer, if unique."""

        normalized = normalize_text(clue)
        if not normalized:
            return None
        with self._lock:
            connection = self._connection
            if not self._available or connection is None:
                return None
            try:
                rows = connection.execute(
                    """
                    SELECT q.anime_title, q.character_name, q.quote_text, q.url,
                           s.source_name, s.license_name, s.license_url
                    FROM quotes AS q
                    JOIN sources AS s ON s.source_key = q.source_key
                    WHERE q.normalized_quote = ?
                    ORDER BY q.anime_title, q.character_name
                    LIMIT 3
                    """,
                    (normalized,),
                ).fetchall()
                if not rows:
                    return None
                answers = {normalize_text(str(row["anime_title"])) for row in rows}
                if len(answers) != 1:
                    LOGGER.warning("Conflicting exact quote answers rejected: %r", clue)
                    return None
                row = rows[0]
                character = str(row["character_name"] or "").strip()
                quote = str(row["quote_text"] or "").strip()
                snippet = f'“{quote}”'
                if character:
                    snippet += f" — {character}"
                return self._evidence(
                    row,
                    title=str(row["anime_title"]),
                    snippet=snippet,
                    answer=str(row["anime_title"]),
                    answer_type="anime_title",
                    character=character,
                )
            except (sqlite3.Error, TypeError, ValueError):
                LOGGER.exception("Exact local quote lookup failed")
                return None

    def search(
        self,
        query: str,
        answer_type: str = "unknown",
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        """Search exact aliases first, then FTS5, returning attributed evidence."""

        normalized = normalize_text(query)
        if not normalized:
            return []
        expected = (
            answer_type if answer_type in {"anime_title", "character"} else "unknown"
        )
        maximum = max(1, min(int(limit), 50))
        with self._lock:
            connection = self._connection
            if not self._available or connection is None:
                return []
            try:
                results: list[dict[str, Any]] = []
                seen: set[tuple[str, str, str]] = set()
                sql = """
                    SELECT r.*, s.source_name, s.license_name, s.license_url
                    FROM alias_lookup AS a
                    JOIN records AS r ON r.record_id = a.record_id
                    JOIN sources AS s ON s.source_key = r.source_key
                    WHERE a.normalized_alias = ?
                """
                parameters: list[Any] = [normalized]
                if expected != "unknown":
                    sql += " AND r.answer_type = ?"
                    parameters.append(expected)
                sql += " ORDER BY r.source_key, r.title LIMIT ?"
                parameters.append(maximum)
                for row in connection.execute(sql, parameters):
                    self._append_result(results, seen, row)

                if len(results) < maximum:
                    match = self._fts_query(normalized)
                    if match:
                        sql = """
                            SELECT r.*, s.source_name, s.license_name, s.license_url,
                                   bm25(record_fts, 8.0, 5.0, 1.0, 2.0) AS rank
                            FROM record_fts
                            JOIN records AS r ON r.record_id = record_fts.rowid
                            JOIN sources AS s ON s.source_key = r.source_key
                            WHERE record_fts MATCH ?
                        """
                        parameters = [match]
                        if expected != "unknown":
                            sql += " AND r.answer_type = ?"
                            parameters.append(expected)
                        sql += " ORDER BY rank, r.record_id LIMIT ?"
                        parameters.append(maximum * 3)
                        for row in connection.execute(sql, parameters):
                            self._append_result(results, seen, row)
                            if len(results) >= maximum:
                                break
                return results[:maximum]
            except (sqlite3.Error, TypeError, ValueError):
                LOGGER.exception("Local anime knowledge search failed")
                return []

    def canonical_answer(self, answer: str, answer_type: str) -> str | None:
        """Resolve an exact normalized alias only when it identifies one entity."""

        normalized = normalize_text(answer)
        if not normalized or answer_type not in {"anime_title", "character"}:
            return None
        with self._lock:
            connection = self._connection
            if not self._available or connection is None:
                return None
            try:
                rows = connection.execute(
                    """
                    SELECT DISTINCT r.title
                    FROM alias_lookup AS a
                    JOIN records AS r ON r.record_id = a.record_id
                    WHERE a.normalized_alias = ? AND r.answer_type = ?
                    ORDER BY r.title
                    LIMIT 3
                    """,
                    (normalized, answer_type),
                ).fetchall()
                titles = {
                    str(row["title"]).strip()
                    for row in rows
                    if str(row["title"]).strip()
                }
                if len(titles) == 1:
                    return next(iter(titles))
                return None
            except (sqlite3.Error, TypeError, ValueError):
                LOGGER.exception("Local canonical-answer lookup failed")
                return None

    def close(self) -> None:
        with self._lock:
            connection = self._connection
            self._connection = None
            self._available = False
            self._count = 0
            if connection is not None:
                try:
                    connection.close()
                except sqlite3.Error:
                    LOGGER.debug("Could not close knowledge index", exc_info=True)

    @staticmethod
    def _fts_query(normalized: str) -> str:
        terms: list[str] = []
        seen: set[str] = set()
        for token in normalized.split():
            if token in _STOP_WORDS or len(token) < 2 or token in seen:
                continue
            seen.add(token)
            terms.append(token.replace('"', '""'))
            if len(terms) >= 16:
                break
        return " OR ".join(f'"{term}"' for term in terms)

    @staticmethod
    def _evidence(
        row: sqlite3.Row,
        *,
        title: str,
        snippet: str,
        answer: str,
        answer_type: str,
        character: str = "",
    ) -> dict[str, Any]:
        return {
            "source": str(row["source_name"] or "Local knowledge"),
            "title": title,
            "snippet": snippet,
            "url": str(row["url"] or ""),
            "answer": answer,
            "answer_type": answer_type,
            "character": character,
            "license": str(row["license_name"] or ""),
            "license_url": str(row["license_url"] or ""),
        }

    def _append_result(
        self,
        results: list[dict[str, Any]],
        seen: set[tuple[str, str, str]],
        row: sqlite3.Row,
    ) -> None:
        identity = (
            str(row["source_key"]),
            str(row["answer_type"]),
            normalize_text(str(row["title"])),
        )
        if identity in seen:
            return
        seen.add(identity)
        results.append(
            self._evidence(
                row,
                title=str(row["title"]),
                snippet=str(row["snippet"] or ""),
                answer=str(row["title"]),
                answer_type=str(row["answer_type"]),
            )
        )

    def __enter__(self) -> "KnowledgeIndex":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()
