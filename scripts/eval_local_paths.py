"""Measure what the local data can answer, offline, with no model calls.

Every answer path that does not call the cloud solver is worth exactly what it
measures, so this is the harness that decides whether one ships. It evaluates
against the committed history, which is the complete quiz record.

    .venv\\Scripts\\python.exe scripts\\eval_local_paths.py --lane quotes
    .venv\\Scripts\\python.exe scripts\\eval_local_paths.py --lane emoji
    .venv\\Scripts\\python.exe scripts\\eval_local_paths.py --lane prose

Baselines recorded 2026-09-03, for anything new to beat:

    quotes  66 clues  exact + fragment match   19 right, 0 wrong, 2.6 ms
    emoji   58 clues  affinity @0.50/0.06      80% precision, fires on 10
    prose   62 clues  SQLite FTS5 BM25         15% rank-1, 26% top-5

Also measured and rejected: dense retrieval with BAAI/bge-base-en-v1.5 over all
158,232 records reached only 11% rank-1 on prose, worse than BM25, and 71%
precision on the quotations exact matching misses. Exact GPU search over the
whole corpus costs 0.4 ms, so an approximate index buys nothing at this size.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anime_trivia_automation.cache import TriviaCache, is_emoji_clue  # noqa: E402
from anime_trivia_automation.config import MatchConfig, load_config  # noqa: E402
from anime_trivia_automation.knowledge import KnowledgeIndex  # noqa: E402
from anime_trivia_automation.utils import (  # noqa: E402
    normalize_question,
    quiz_answer_form,
)

STOP = {
    "a", "an", "the", "who", "whose", "that", "which", "with", "and", "or", "of",
    "in", "on", "to", "for", "his", "her", "their", "its", "he", "she", "they",
    "is", "are", "was", "were", "be", "been", "as", "at", "by", "from", "this",
    "more", "than", "while", "after", "before", "into", "over", "out", "up",
}


def is_quote(clue: str) -> bool:
    s = clue.strip()
    return s.startswith(('"', "“")) and s.endswith(('"', "”", '."'))


def is_prose(clue: str) -> bool:
    return sum(c.isalpha() for c in clue) >= 12 and not is_quote(clue)


def same_answer(want: str, got: str) -> bool:
    """'Canute' and 'Canute Svenson' are one character; prefix match both ways."""

    a, b = normalize_question(want), normalize_question(got or "")
    if not a or not b:
        return False
    if a == b:
        return True
    aw, bw = a.split(), b.split()
    return aw == bw[: len(aw)] or bw == aw[: len(bw)]


def history(config) -> list[dict]:
    return json.loads(config.runtime.history_path.read_text(encoding="utf-8"))["pairs"]


def lane_quotes(config, pairs: list[dict]) -> None:
    index = KnowledgeIndex(config.novel.knowledge_index_path)
    if not index.available:
        print("local knowledge index unavailable", file=sys.stderr)
        return
    quotes = [p for p in pairs if is_quote(p["clue"])]
    # Measure what actually ships: the app rewrites a corpus title into a
    # spelling this quiz has revealed before.
    known_forms = tuple(sorted({str(p["answer"]).strip() for p in pairs}))
    right = wrong = silent = 0
    timings = []
    for pair in quotes:
        started = time.perf_counter()
        got = index.match_quote(pair["clue"])
        timings.append((time.perf_counter() - started) * 1000)
        if got is None:
            silent += 1
            continue
        answer = quiz_answer_form(got[0], known_forms)
        if same_answer(pair["answer"], answer):
            right += 1
        else:
            wrong += 1
            print(f"   WRONG {pair['clue'][:44]!r} -> {answer} (want {pair['answer']})")
    timings.sort()
    fired = right + wrong
    print(
        f"\nquotes: {len(quotes)} clues | right {right} | wrong {wrong} | silent {silent}\n"
        f"  precision {right / max(1, fired):.0%} | coverage {right / max(1, len(quotes)):.0%} "
        f"of quotes, {right / max(1, len(pairs)):.0%} of ALL clues\n"
        f"  lookup {timings[len(timings) // 2]:.2f} ms median"
    )
    index.close()


def lane_emoji(config, pairs: list[dict]) -> None:
    """Leave-one-out over every real rebus, sweeping threshold and margin."""

    import tempfile

    rebuses = [p for p in pairs if is_emoji_clue(p["clue"])]
    answers = Counter(p["answer"] for p in rebuses)
    winnable = [p for p in rebuses if answers[p["answer"]] > 1]
    print(
        f"rebuses {len(rebuses)} | winnable (answer has another rebus) {len(winnable)}\n"
        f"ceiling {len(winnable)}/{len(rebuses)} = {len(winnable) / len(rebuses):.0%}\n"
    )
    root = Path(tempfile.mkdtemp())
    print(f"{'thresh':>7}{'margin':>8}{'fires':>7}{'right':>7}{'wrong':>7}{'precision':>11}{'recall':>8}")
    for threshold in (0.45, 0.50, 0.55, 0.60):
        for margin in (0.0, 0.06):
            right = wrong = fired = 0
            for index, held in enumerate(rebuses):
                rest = [p for p in pairs if p is not held]
                path = root / f"h{index}.json"
                path.write_text(
                    json.dumps({"schema_version": 1, "pairs": rest}), encoding="utf-8"
                )
                cache = TriviaCache(
                    root / f"c{index}.json",
                    MatchConfig(
                        emoji_affinity_threshold=threshold,
                        emoji_affinity_margin=margin,
                    ),
                    history_path=path,
                )
                hit = cache.match_emoji_affinity(held["clue"], held["type"])
                if hit is None:
                    continue
                fired += 1
                if same_answer(held["answer"], hit.answer):
                    right += 1
                else:
                    wrong += 1
            print(
                f"{threshold:>7.2f}{margin:>8.2f}{fired:>7}{right:>7}{wrong:>7}"
                f"{right / max(1, fired):>11.0%}{right / max(1, len(winnable)):>8.0%}"
            )


def lane_prose(config, pairs: list[dict]) -> None:
    """BM25 over the local corpus, the baseline dense retrieval failed to beat."""

    connection = sqlite3.connect(str(config.novel.knowledge_index_path))
    connection.row_factory = sqlite3.Row
    prose = [p for p in pairs if is_prose(p["clue"])]
    top1 = top5 = 0
    for pair in prose:
        words = [
            w for w in normalize_question(pair["clue"]).split()
            if w not in STOP and len(w) > 3
        ]
        terms = sorted(set(words), key=lambda w: -len(w))[:12]
        if len(terms) < 3:
            continue
        rows = connection.execute(
            "select r.title from record_fts f join records r on r.record_id = f.rowid "
            "where record_fts match ? and r.answer_type = ? "
            "order by bm25(record_fts) limit 5",
            (" OR ".join(terms), pair["type"]),
        ).fetchall()
        titles = [r["title"] or "" for r in rows]
        if titles and same_answer(pair["answer"], titles[0]):
            top1 += 1
        elif any(same_answer(pair["answer"], t) for t in titles):
            top5 += 1
    total = max(1, len(prose))
    print(
        f"prose: {len(prose)} clues\n"
        f"  rank-1 correct : {top1}/{total} = {top1 / total:.0%}\n"
        f"  top-5 correct  : {top1 + top5}/{total} = {(top1 + top5) / total:.0%}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    parser.add_argument(
        "--lane", choices=("quotes", "emoji", "prose", "all"), default="all"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    pairs = history(config)
    print(f"history: {len(pairs)} clues\n")
    lanes = (
        [args.lane] if args.lane != "all" else ["quotes", "emoji", "prose"]
    )
    for lane in lanes:
        print(f"===== {lane} =====")
        {"quotes": lane_quotes, "emoji": lane_emoji, "prose": lane_prose}[lane](
            config, pairs
        )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
