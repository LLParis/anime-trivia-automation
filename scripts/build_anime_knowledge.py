from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import sys
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anime_trivia_automation.knowledge import normalize_text, strip_html

SCHEMA = """
PRAGMA foreign_keys=ON;
CREATE TABLE schema_info (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE sources (
    source_key TEXT PRIMARY KEY,
    source_name TEXT NOT NULL,
    source_url TEXT NOT NULL DEFAULT '',
    license_name TEXT NOT NULL,
    license_url TEXT NOT NULL DEFAULT '',
    local_path TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    modified_ns INTEGER NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE anime (
    anime_id INTEGER PRIMARY KEY,
    source_key TEXT NOT NULL REFERENCES sources(source_key),
    source_record_id TEXT NOT NULL,
    canonical_title TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    url TEXT NOT NULL DEFAULT '',
    year INTEGER,
    UNIQUE(source_key, source_record_id)
);
CREATE TABLE anime_aliases (
    anime_id INTEGER NOT NULL REFERENCES anime(anime_id),
    source_key TEXT NOT NULL REFERENCES sources(source_key),
    alias TEXT NOT NULL,
    normalized_alias TEXT NOT NULL,
    alias_kind TEXT NOT NULL,
    UNIQUE(anime_id, normalized_alias)
);
CREATE TABLE characters (
    character_id INTEGER PRIMARY KEY,
    source_key TEXT NOT NULL REFERENCES sources(source_key),
    source_record_id TEXT NOT NULL,
    canonical_name TEXT NOT NULL,
    native_name TEXT NOT NULL DEFAULT '',
    biography TEXT NOT NULL DEFAULT '',
    url TEXT NOT NULL DEFAULT '',
    UNIQUE(source_key, source_record_id)
);
CREATE TABLE character_aliases (
    character_id INTEGER NOT NULL REFERENCES characters(character_id),
    source_key TEXT NOT NULL REFERENCES sources(source_key),
    alias TEXT NOT NULL,
    normalized_alias TEXT NOT NULL,
    alias_kind TEXT NOT NULL,
    UNIQUE(character_id, normalized_alias)
);
CREATE TABLE character_media (
    character_id INTEGER NOT NULL REFERENCES characters(character_id),
    anime_id INTEGER NOT NULL REFERENCES anime(anime_id),
    source_key TEXT NOT NULL REFERENCES sources(source_key),
    role TEXT NOT NULL,
    UNIQUE(character_id, anime_id, role)
);
CREATE TABLE anime_relations (
    relation_id INTEGER PRIMARY KEY,
    anime_id INTEGER NOT NULL REFERENCES anime(anime_id),
    source_key TEXT NOT NULL REFERENCES sources(source_key),
    relation_type TEXT NOT NULL,
    related_source_record_id TEXT NOT NULL DEFAULT '',
    related_title TEXT NOT NULL DEFAULT '',
    related_type TEXT NOT NULL DEFAULT '',
    related_url TEXT NOT NULL DEFAULT '',
    UNIQUE(anime_id, relation_type, related_source_record_id, related_url)
);
CREATE TABLE provider_links (
    anime_id INTEGER NOT NULL REFERENCES anime(anime_id),
    source_key TEXT NOT NULL REFERENCES sources(source_key),
    provider TEXT NOT NULL,
    url TEXT NOT NULL,
    UNIQUE(anime_id, url)
);
CREATE TABLE quotes (
    quote_id INTEGER PRIMARY KEY,
    source_key TEXT NOT NULL REFERENCES sources(source_key),
    source_record_id TEXT NOT NULL,
    normalized_quote TEXT NOT NULL,
    quote_text TEXT NOT NULL,
    anime_title TEXT NOT NULL,
    character_name TEXT NOT NULL DEFAULT '',
    url TEXT NOT NULL DEFAULT '',
    UNIQUE(source_key, normalized_quote, anime_title, character_name)
);
CREATE TABLE records (
    record_id INTEGER PRIMARY KEY,
    source_key TEXT NOT NULL REFERENCES sources(source_key),
    entity_type TEXT NOT NULL,
    answer_type TEXT NOT NULL,
    source_record_id TEXT NOT NULL,
    title TEXT NOT NULL,
    aliases TEXT NOT NULL DEFAULT '',
    snippet TEXT NOT NULL DEFAULT '',
    tags TEXT NOT NULL DEFAULT '',
    url TEXT NOT NULL DEFAULT '',
    UNIQUE(source_key, entity_type, source_record_id)
);
CREATE TABLE alias_lookup (
    normalized_alias TEXT NOT NULL,
    record_id INTEGER NOT NULL REFERENCES records(record_id),
    source_key TEXT NOT NULL REFERENCES sources(source_key),
    alias TEXT NOT NULL,
    PRIMARY KEY(normalized_alias, record_id)
);
CREATE VIRTUAL TABLE record_fts USING fts5(
    title,
    aliases,
    snippet,
    tags,
    content='records',
    content_rowid='record_id',
    tokenize='unicode61 remove_diacritics 2'
);
CREATE INDEX idx_quotes_normalized ON quotes(normalized_quote);
CREATE INDEX idx_anime_source ON anime(source_key, source_record_id);
CREATE INDEX idx_anime_alias_normalized ON anime_aliases(normalized_alias);
CREATE INDEX idx_character_source ON characters(source_key, source_record_id);
CREATE INDEX idx_character_alias_normalized ON character_aliases(normalized_alias);
CREATE INDEX idx_character_media_character ON character_media(character_id);
CREATE INDEX idx_character_media_anime ON character_media(anime_id);
CREATE INDEX idx_relation_anime ON anime_relations(anime_id);
CREATE INDEX idx_alias_lookup_normalized ON alias_lookup(normalized_alias);
CREATE INDEX idx_records_answer_type ON records(answer_type);
"""


def compact(value: Any, maximum: int = 12_000) -> str:
    text = " ".join(str(value or "").split())
    return text[:maximum]


def unique_strings(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = compact(value, 500)
        normalized = normalize_text(text)
        if text and normalized and normalized not in seen:
            seen.add(normalized)
            result.append(text)
    return result


def parse_json(value: str, default: Any) -> Any:
    if not value or value in {"[]", "{}", "nan", "NaN"}:
        return default
    try:
        parsed = json.loads(value)
        return parsed
    except (json.JSONDecodeError, TypeError):
        return default


def integer(value: Any) -> int | None:
    try:
        return int(float(value)) if str(value).strip() else None
    except (TypeError, ValueError):
        return None


def provider(url: str) -> str:
    lowered = url.casefold()
    for name in ("anilist", "anime-planet", "kitsu", "myanimelist"):
        if name in lowered:
            return name
    return "other"


def insert_source(
    connection: sqlite3.Connection,
    key: str,
    name: str,
    source_url: str,
    license_name: str,
    license_url: str,
    path: Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    stat = path.stat()
    connection.execute(
        """
        INSERT INTO sources(
            source_key, source_name, source_url, license_name, license_url,
            local_path, size_bytes, modified_ns, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source_key) DO UPDATE SET
            source_name=excluded.source_name,
            source_url=excluded.source_url,
            license_name=excluded.license_name,
            license_url=excluded.license_url,
            metadata_json=excluded.metadata_json
        """,
        (
            key,
            name,
            source_url,
            license_name,
            license_url,
            str(path.resolve()),
            stat.st_size,
            stat.st_mtime_ns,
            json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True),
        ),
    )


def insert_record(
    connection: sqlite3.Connection,
    *,
    source_key: str,
    entity_type: str,
    answer_type: str,
    source_record_id: str,
    title: str,
    aliases: Iterable[str],
    snippet: str,
    tags: Iterable[str],
    url: str,
) -> int:
    alias_values = unique_strings([title, *aliases])
    tag_values = unique_strings(tags)
    connection.execute(
        """
        INSERT OR IGNORE INTO records(
            source_key, entity_type, answer_type, source_record_id,
            title, aliases, snippet, tags, url
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            source_key,
            entity_type,
            answer_type,
            source_record_id,
            compact(title, 500),
            " | ".join(alias_values),
            compact(snippet),
            " | ".join(tag_values),
            compact(url, 2_000),
        ),
    )
    row = connection.execute(
        """
        SELECT record_id FROM records
        WHERE source_key=? AND entity_type=? AND source_record_id=?
        """,
        (source_key, entity_type, source_record_id),
    ).fetchone()
    if row is None:
        raise RuntimeError("record insert failed")
    record_id = int(row[0])
    connection.executemany(
        "INSERT OR IGNORE INTO alias_lookup VALUES (?, ?, ?, ?)",
        [
            (normalize_text(alias), record_id, source_key, alias)
            for alias in alias_values
            if normalize_text(alias)
        ],
    )
    return record_id


def insert_anime(
    connection: sqlite3.Connection,
    *,
    source_key: str,
    source_record_id: str,
    canonical_title: str,
    description: str,
    url: str,
    year: int | None,
    aliases: list[str],
    alias_kind: str = "title",
) -> int:
    connection.execute(
        """
        INSERT OR IGNORE INTO anime(
            source_key, source_record_id, canonical_title, description, url, year
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (source_key, source_record_id, canonical_title, description, url, year),
    )
    row = connection.execute(
        "SELECT anime_id FROM anime WHERE source_key=? AND source_record_id=?",
        (source_key, source_record_id),
    ).fetchone()
    if row is None:
        raise RuntimeError("anime insert failed")
    anime_id = int(row[0])
    connection.executemany(
        """
        INSERT OR IGNORE INTO anime_aliases(
            anime_id, source_key, alias, normalized_alias, alias_kind
        ) VALUES (?, ?, ?, ?, ?)
        """,
        [
            (anime_id, source_key, alias, normalize_text(alias), alias_kind)
            for alias in unique_strings([canonical_title, *aliases])
        ],
    )
    return anime_id


def build_anilist(connection: sqlite3.Connection, path: Path) -> dict[str, int]:
    stats = {"rows": 0, "anime": 0, "characters_seen": 0, "bad_json": 0}
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            break
        except OverflowError:
            limit //= 10

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            stats["rows"] += 1
            source_record_id = compact(row.get("id"), 80)
            titles = unique_strings(
                [
                    row.get("title_english"),
                    row.get("title_userPreferred"),
                    row.get("title_romaji"),
                    row.get("title_native"),
                ]
            )
            synonyms = parse_json(row.get("synonyms", ""), [])
            if not isinstance(synonyms, list):
                synonyms = []
                stats["bad_json"] += 1
            aliases = unique_strings([*titles, *synonyms])
            if not source_record_id or not aliases:
                continue
            canonical = compact(row.get("title_english")) or aliases[0]
            description = compact(strip_html(row.get("description", "")))
            url = compact(row.get("siteUrl"), 2_000)
            anime_id = insert_anime(
                connection,
                source_key="anilist",
                source_record_id=source_record_id,
                canonical_title=canonical,
                description=description,
                url=url,
                year=integer(row.get("startDate_year")),
                aliases=aliases,
            )
            stats["anime"] += 1

            genres = parse_json(row.get("genres", ""), [])
            genres = genres if isinstance(genres, list) else []
            tags_raw = parse_json(row.get("tags", ""), [])
            tags_raw = tags_raw if isinstance(tags_raw, list) else []
            tag_names: list[str] = []
            tag_evidence: list[str] = []
            for tag in tags_raw:
                if not isinstance(tag, dict):
                    continue
                name = compact(tag.get("name"), 200)
                if name:
                    tag_names.append(name)
                    detail = compact(strip_html(str(tag.get("description", ""))), 400)
                    if detail and len(tag_evidence) < 16:
                        tag_evidence.append(f"{name}: {detail}")

            relation_evidence: list[str] = []
            relations = parse_json(row.get("relations", ""), [])
            if not isinstance(relations, list):
                relations = []
                stats["bad_json"] += 1
            for relation in relations:
                if not isinstance(relation, dict):
                    continue
                node = relation.get("node") or {}
                node = node if isinstance(node, dict) else {}
                related_titles = node.get("title") or {}
                related_titles = related_titles if isinstance(related_titles, dict) else {}
                related = compact(
                    related_titles.get("english")
                    or related_titles.get("romaji")
                    or related_titles.get("native")
                )
                related_id = compact(node.get("id"), 80)
                related_type = compact(node.get("type"), 80)
                relation_type = compact(relation.get("relationType"), 80)
                related_url = (
                    f"https://anilist.co/{related_type.casefold()}/{related_id}"
                    if related_type and related_id
                    else ""
                )
                connection.execute(
                    """
                    INSERT OR IGNORE INTO anime_relations(
                        anime_id, source_key, relation_type, related_source_record_id,
                        related_title, related_type, related_url
                    ) VALUES (?, 'anilist', ?, ?, ?, ?, ?)
                    """,
                    (
                        anime_id,
                        relation_type,
                        related_id,
                        related,
                        related_type,
                        related_url,
                    ),
                )
                if related and len(relation_evidence) < 20:
                    relation_evidence.append(f"{relation_type}: {related}")

            snippet = " ".join(
                part
                for part in [
                    description,
                    "Tags: " + "; ".join(tag_evidence) if tag_evidence else "",
                    "Relations: " + "; ".join(relation_evidence)
                    if relation_evidence
                    else "",
                ]
                if part
            )
            insert_record(
                connection,
                source_key="anilist",
                entity_type="anime",
                answer_type="anime_title",
                source_record_id=source_record_id,
                title=canonical,
                aliases=aliases,
                snippet=snippet,
                tags=[*genres, *tag_names],
                url=url,
            )

            characters = parse_json(row.get("characters", ""), [])
            if not isinstance(characters, list):
                characters = []
                stats["bad_json"] += 1
            for edge in characters:
                if not isinstance(edge, dict):
                    continue
                node = edge.get("node") or {}
                node = node if isinstance(node, dict) else {}
                character_source_id = compact(node.get("id"), 80)
                names = node.get("name") or {}
                names = names if isinstance(names, dict) else {}
                alternatives = names.get("alternative") or []
                alternatives = alternatives if isinstance(alternatives, list) else []
                full_name = compact(names.get("full"), 500)
                native_name = compact(names.get("native"), 500)
                aliases_char = unique_strings([full_name, native_name, *alternatives])
                if not character_source_id or not aliases_char:
                    continue
                canonical_name = full_name or aliases_char[0]
                biography = compact(strip_html(str(node.get("description", ""))))
                character_url = f"https://anilist.co/character/{character_source_id}"
                connection.execute(
                    """
                    INSERT INTO characters(
                        source_key, source_record_id, canonical_name, native_name,
                        biography, url
                    ) VALUES ('anilist', ?, ?, ?, ?, ?)
                    ON CONFLICT(source_key, source_record_id) DO UPDATE SET
                        canonical_name=CASE
                            WHEN length(excluded.canonical_name) > 0
                            THEN excluded.canonical_name ELSE characters.canonical_name END,
                        native_name=CASE
                            WHEN length(excluded.native_name) > 0
                            THEN excluded.native_name ELSE characters.native_name END,
                        biography=CASE
                            WHEN length(excluded.biography) > length(characters.biography)
                            THEN excluded.biography ELSE characters.biography END
                    """,
                    (
                        character_source_id,
                        canonical_name,
                        native_name,
                        biography,
                        character_url,
                    ),
                )
                character_id = int(
                    connection.execute(
                        """
                        SELECT character_id FROM characters
                        WHERE source_key='anilist' AND source_record_id=?
                        """,
                        (character_source_id,),
                    ).fetchone()[0]
                )
                connection.executemany(
                    """
                    INSERT OR IGNORE INTO character_aliases(
                        character_id, source_key, alias, normalized_alias, alias_kind
                    ) VALUES (?, 'anilist', ?, ?, ?)
                    """,
                    [
                        (
                            character_id,
                            alias,
                            normalize_text(alias),
                            "native" if alias == native_name else "name",
                        )
                        for alias in aliases_char
                    ],
                )
                connection.execute(
                    """
                    INSERT OR IGNORE INTO character_media(
                        character_id, anime_id, source_key, role
                    ) VALUES (?, ?, 'anilist', ?)
                    """,
                    (character_id, anime_id, compact(edge.get("role"), 80)),
                )
                stats["characters_seen"] += 1

            if stats["rows"] % 500 == 0:
                print(f"AniList rows: {stats['rows']:,}", flush=True)
    return stats


def build_character_records(connection: sqlite3.Connection) -> int:
    count = 0
    query = """
        SELECT c.character_id, c.source_record_id, c.canonical_name, c.biography, c.url,
               COALESCE(a.aliases, '') AS aliases,
               COALESCE(m.media, '') AS media
        FROM characters AS c
        LEFT JOIN (
            SELECT character_id, GROUP_CONCAT(alias, ' | ') AS aliases
            FROM character_aliases GROUP BY character_id
        ) AS a ON a.character_id = c.character_id
        LEFT JOIN (
            SELECT cm.character_id,
                   GROUP_CONCAT(DISTINCT cm.role || ' in ' || an.canonical_title) AS media
            FROM character_media AS cm
            JOIN anime AS an ON an.anime_id = cm.anime_id
            GROUP BY cm.character_id
        ) AS m ON m.character_id = c.character_id
        ORDER BY c.character_id
    """
    for row in connection.execute(query):
        aliases = str(row["aliases"] or "").split(" | ")
        media = str(row["media"] or "").replace(",", "; ")
        snippet = " ".join(
            part
            for part in [str(row["biography"] or ""), f"Media roles: {media}" if media else ""]
            if part
        )
        insert_record(
            connection,
            source_key="anilist",
            entity_type="character",
            answer_type="character",
            source_record_id=str(row["source_record_id"]),
            title=str(row["canonical_name"]),
            aliases=aliases,
            snippet=snippet,
            tags=[media] if media else [],
            url=str(row["url"]),
        )
        count += 1
    return count


def build_quotes(connection: sqlite3.Connection, path: Path) -> dict[str, int]:
    stats = {"rows": 0, "quotes": 0}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            stats["rows"] += 1
            quote = compact(row.get("Quote"))
            anime = compact(row.get("Anime"), 500)
            character = compact(row.get("Character"), 500)
            if anime.startswith("(") and anime.endswith(")"):
                anime = anime[1:-1].strip()
            normalized = normalize_text(quote)
            if not normalized or not anime:
                continue
            source_record_id = compact(row.get("id"), 80) or str(stats["rows"])
            connection.execute(
                """
                INSERT OR IGNORE INTO quotes(
                    source_key, source_record_id, normalized_quote, quote_text,
                    anime_title, character_name, url
                ) VALUES ('animequotes', ?, ?, ?, ?, ?, '')
                """,
                (source_record_id, normalized, quote, anime, character),
            )
            insert_record(
                connection,
                source_key="animequotes",
                entity_type="quote",
                answer_type="anime_title",
                source_record_id=source_record_id,
                title=anime,
                aliases=[anime, character],
                snippet=f'“{quote}”' + (f" — {character}" if character else ""),
                tags=[character] if character else [],
                url="",
            )
            stats["quotes"] += 1
    return stats


def build_manami(connection: sqlite3.Connection, path: Path) -> dict[str, int]:
    stats = {"lines": 0, "anime": 0, "bad_json": 0}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stats["lines"] += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats["bad_json"] += 1
                continue
            if not isinstance(record, dict):
                continue
            if "$schema" in record:
                license_data = record.get("license") or {}
                license_data = license_data if isinstance(license_data, dict) else {}
                insert_source(
                    connection,
                    "manami",
                    "Manami anime-offline-database",
                    compact(record.get("repository"), 2_000),
                    compact(license_data.get("name"))
                    or "ODbL 1.0 + DbCL 1.0",
                    compact(license_data.get("url"), 2_000),
                    path,
                    record,
                )
                continue

            title = compact(record.get("title"), 500)
            if not title:
                continue
            source_record_id = str(line_number)
            synonyms = record.get("synonyms") or []
            synonyms = synonyms if isinstance(synonyms, list) else []
            aliases = unique_strings([title, *synonyms])
            sources = record.get("sources") or []
            sources = sources if isinstance(sources, list) else []
            sources = unique_strings(sources)
            tags = record.get("tags") or []
            tags = tags if isinstance(tags, list) else []
            tags = unique_strings(tags)
            related = record.get("relatedAnime") or []
            related = related if isinstance(related, list) else []
            related = unique_strings(related)
            season = record.get("animeSeason") or {}
            season = season if isinstance(season, dict) else {}
            year = integer(season.get("year"))
            url = sources[0] if sources else ""
            snippet = " ".join(
                part
                for part in [
                    f"Type: {compact(record.get('type'), 80)}",
                    f"Season: {compact(season.get('season'), 80)} {year or ''}".strip(),
                    "Tags: " + "; ".join(tags) if tags else "",
                    "Cross-provider sources: " + "; ".join(sources) if sources else "",
                    "Related records: " + "; ".join(related) if related else "",
                ]
                if part
            )
            anime_id = insert_anime(
                connection,
                source_key="manami",
                source_record_id=source_record_id,
                canonical_title=title,
                description=snippet,
                url=url,
                year=year,
                aliases=aliases,
            )
            connection.executemany(
                """
                INSERT OR IGNORE INTO provider_links(
                    anime_id, source_key, provider, url
                ) VALUES (?, 'manami', ?, ?)
                """,
                [(anime_id, provider(link), link) for link in sources],
            )
            connection.executemany(
                """
                INSERT OR IGNORE INTO anime_relations(
                    anime_id, source_key, relation_type, related_url
                ) VALUES (?, 'manami', 'RELATED', ?)
                """,
                [(anime_id, link) for link in related],
            )
            insert_record(
                connection,
                source_key="manami",
                entity_type="anime",
                answer_type="anime_title",
                source_record_id=source_record_id,
                title=title,
                aliases=aliases,
                snippet=snippet,
                tags=tags,
                url=url,
            )
            stats["anime"] += 1
            if stats["anime"] % 5_000 == 0:
                print(f"Manami anime: {stats['anime']:,}", flush=True)
    return stats


def create_index(
    anilist_path: Path,
    quotes_path: Path,
    manami_path: Path,
    index_path: Path,
) -> dict[str, Any]:
    for source in (anilist_path, quotes_path, manami_path):
        if not source.is_file():
            raise FileNotFoundError(source)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{index_path.name}.",
        suffix=".tmp",
        dir=index_path.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    handle.close()
    started = time.perf_counter()
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(str(temporary))
        connection.row_factory = sqlite3.Row
        connection.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            PRAGMA temp_store=MEMORY;
            PRAGMA cache_size=-65536;
            PRAGMA locking_mode=EXCLUSIVE;
            """
        )
        connection.executescript(SCHEMA)
        connection.executemany(
            "INSERT INTO schema_info(key, value) VALUES (?, ?)",
            [
                ("schema_version", "1"),
                ("built_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
                ("builder", "build_anime_knowledge.py"),
            ],
        )
        insert_source(
            connection,
            "anilist",
            "AniList API anime dataset",
            "https://anilist.co/graphiql",
            "License not supplied with local dataset snapshot",
            "https://anilist.co/terms",
            anilist_path,
            {"boundary": "AniList-derived rows; verify upstream terms before redistribution"},
        )
        insert_source(
            connection,
            "animequotes",
            "ewgsta English Anime Quotes",
            "https://huggingface.co/datasets/ewgsta/animequotes",
            "MIT",
            "https://huggingface.co/datasets/ewgsta/animequotes/blob/main/README.md",
            quotes_path,
            {
                "boundary": "Quote text retained only in the ewgsta source partition; underlying quotations may have separate copyright"
            },
        )
        insert_source(
            connection,
            "manami",
            "Manami anime-offline-database",
            "https://github.com/manami-project/anime-offline-database",
            "ODbL 1.0 + DbCL 1.0",
            "https://github.com/manami-project/anime-offline-database/blob/2026-27/LICENSE",
            manami_path,
            {"boundary": "Manami data remains source-attributed under ODbL/DbCL"},
        )

        connection.commit()
        connection.execute("BEGIN")
        anilist_stats = build_anilist(connection, anilist_path)
        character_records = build_character_records(connection)
        quote_stats = build_quotes(connection, quotes_path)
        manami_stats = build_manami(connection, manami_path)
        print("Building FTS5 index...", flush=True)
        connection.execute(
            """
            INSERT INTO record_fts(rowid, title, aliases, snippet, tags)
            SELECT record_id, title, aliases, snippet, tags FROM records
            """
        )
        connection.execute("ANALYZE")
        connection.execute("INSERT INTO record_fts(record_fts) VALUES('optimize')")
        connection.commit()
        counts = {
            table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in (
                "sources",
                "anime",
                "anime_aliases",
                "characters",
                "character_aliases",
                "character_media",
                "anime_relations",
                "provider_links",
                "quotes",
                "records",
                "alias_lookup",
            )
        }
        connection.close()
        connection = None
        # Windows rejects fsync on a read-only descriptor. The temporary
        # database is ours, so open it read/write solely for the durability
        # barrier immediately before the atomic replacement.
        descriptor = os.open(temporary, os.O_RDWR)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, index_path)
        elapsed = time.perf_counter() - started
        return {
            "schema_version": 1,
            "index": str(index_path.resolve()),
            "index_bytes": index_path.stat().st_size,
            "build_seconds": round(elapsed, 3),
            "counts": counts,
            "source_stats": {
                "anilist": anilist_stats,
                "character_records": character_records,
                "animequotes": quote_stats,
                "manami": manami_stats,
            },
        }
    except Exception:
        if connection is not None:
            connection.close()
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the local source-attributed anime knowledge index."
    )
    parser.add_argument("--anilist", required=True, type=Path)
    parser.add_argument("--quotes", required=True, type=Path)
    parser.add_argument("--manami", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    args = parser.parse_args()
    result = create_index(
        args.anilist.resolve(),
        args.quotes.resolve(),
        args.manami.resolve(),
        args.index.resolve(),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
