"""Offline accuracy/latency check of the answer providers on reviewed history.

Runs the real Gemini provider (and optionally the managed local Qwen resolver)
over clue/answer pairs from ``data/trivia_history.seed.json`` and reports how
often the best answer or one of its follow-up guesses matches the bot's own
reveal. Nothing here touches Discord or the keyboard.

Examples::

    .venv\\Scripts\\python.exe scripts\\eval_resolvers.py --config config.json --limit 24
    .venv\\Scripts\\python.exe scripts\\eval_resolvers.py --config config.json --provider qwen --limit 12
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anime_trivia_automation.config import load_config
from anime_trivia_automation.gemini import GeminiProvider, GeminiRequest
from anime_trivia_automation.novel import NovelAnswerResolver
from anime_trivia_automation.utils import normalize_question


def matches(expected: str, produced: str | None) -> bool:
    if not produced:
        return False
    want = normalize_question(expected)
    got = normalize_question(produced)
    if not want or not got:
        return False
    if want == got:
        return True
    # Anime Soul accepts any single name token ("fuu", "Kogami") and common
    # title short forms; count a produced answer that contains the full
    # expected answer, or shares a >=4-character name token, as accepted.
    if want in got or got in want:
        return True
    want_tokens = {token for token in want.split() if len(token) >= 4}
    got_tokens = {token for token in got.split() if len(token) >= 4}
    return bool(want_tokens & got_tokens)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--provider", choices=("gemini", "qwen"), default="qwen")
    parser.add_argument("--limit", type=int, default=20, help="most recent N pairs")
    parser.add_argument("--offset", type=int, default=0, help="skip the newest N pairs")
    parser.add_argument("--pause", type=float, default=4.0, help="seconds between calls")
    parser.add_argument("--out", default=None, help="JSON report path")
    args = parser.parse_args()

    if args.provider == "gemini" and args.limit != 1:
        parser.error(
            "Gemini Developer API evaluations are hard-capped at --limit 1 "
            "to protect project quota"
        )

    config = load_config(args.config)
    pairs = json.loads(config.runtime.history_path.read_text(encoding="utf-8"))["pairs"]
    if args.offset:
        pairs = pairs[: -args.offset]
    pairs = pairs[-args.limit :] if args.limit else pairs

    rows: list[dict[str, object]] = []
    if args.provider == "gemini":
        provider = GeminiProvider(config.gemini)
        availability = asyncio.run(provider.preflight())
        print(f"gemini preflight: {availability.phase} ({availability.detail})")
        if not availability.available:
            return 2

        async def run_one(clue: str, answer_type: str):
            return await provider.resolve(
                GeminiRequest(
                    clue=clue,
                    expected_answer_type=answer_type,  # type: ignore[arg-type]
                    deadline=time.perf_counter() + config.gemini.text_timeout_seconds,
                )
            )

        for pair in pairs:
            started = time.perf_counter()
            result = asyncio.run(run_one(pair["clue"], pair["type"]))
            elapsed = (time.perf_counter() - started) * 1000.0
            guesses = [result.answer, *result.alternatives] if result.answer else []
            hit_index = next(
                (index for index, guess in enumerate(guesses, 1) if matches(pair["answer"], guess)),
                0,
            )
            rows.append(
                {
                    "clue": pair["clue"],
                    "type": pair["type"],
                    "expected": pair["answer"],
                    "status": result.status,
                    "answer": result.answer,
                    "alternatives": list(result.alternatives),
                    "confidence": round(result.confidence, 3),
                    "latency_ms": round(elapsed),
                    "hit_index": hit_index,
                }
            )
            mark = "OK " if hit_index == 1 else ("alt" if hit_index else "MISS")
            print(
                f"{mark} {elapsed:6.0f}ms conf={result.confidence:.2f} "
                f"{pair['answer']!r} <- {result.answer!r} {list(result.alternatives)} "
                f"| {pair['clue'][:70]}"
            )
            time.sleep(args.pause)
        asyncio.run(provider.close())
    else:
        resolver = NovelAnswerResolver(config.novel)
        if not resolver.ensure_ready():
            print(f"qwen not ready: {resolver.last_detail}")
            return 2
        try:
            for pair in pairs:
                started = time.perf_counter()
                ranked = resolver.resolve_ranked(pair["clue"], pair["type"])
                elapsed = (time.perf_counter() - started) * 1000.0
                guesses = [ranked.answer, *ranked.alternatives] if ranked else []
                hit_index = next(
                    (
                        index
                        for index, guess in enumerate(guesses, 1)
                        if matches(pair["answer"], guess)
                    ),
                    0,
                )
                rows.append(
                    {
                        "clue": pair["clue"],
                        "type": pair["type"],
                        "expected": pair["answer"],
                        "answer": ranked.answer if ranked else None,
                        "alternatives": list(ranked.alternatives) if ranked else [],
                        "confidence": round(resolver.last_confidence, 3),
                        "detail": resolver.last_detail,
                        "latency_ms": round(elapsed),
                        "hit_index": hit_index,
                    }
                )
                mark = "OK " if hit_index == 1 else ("alt" if hit_index else "MISS")
                print(
                    f"{mark} {elapsed:6.0f}ms conf={resolver.last_confidence:.2f} "
                    f"{pair['answer']!r} <- {ranked.answer if ranked else None!r} "
                    f"{list(ranked.alternatives) if ranked else []} | {pair['clue'][:70]}"
                )
        finally:
            resolver.close()

    total = len(rows)
    first = sum(1 for row in rows if row["hit_index"] == 1)
    any_hit = sum(1 for row in rows if row["hit_index"])
    answered = sum(1 for row in rows if row["answer"])
    latencies = sorted(int(row["latency_ms"]) for row in rows)  # type: ignore[arg-type]
    median = latencies[len(latencies) // 2] if latencies else 0
    print(
        f"\n{args.provider}: {total} clues | answered {answered} | first-guess correct {first} "
        f"| correct within ladder {any_hit} | median latency {median} ms"
    )
    out = Path(args.out) if args.out else Path("runtime") / f"eval-{args.provider}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"provider": args.provider, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
