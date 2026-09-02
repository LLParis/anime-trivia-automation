from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from anime_trivia_automation.config import (
    AppConfig,
    CaptureConfig,
    NovelConfig,
    validate_config,
)
from anime_trivia_automation.novel import NovelAnswerResolver


class NovelAnswerResolverTests(unittest.TestCase):
    def test_endpoint_validation_rejects_loopback_userinfo_spoof(self) -> None:
        config = AppConfig(
            capture=CaptureConfig(region=(0, 0, 1920, 1080), calibrated=True),
            novel=NovelConfig(
                enabled=True,
                endpoint="http://127.0.0.1:123@evil.example:80",
            ),
        )
        with self.assertRaisesRegex(ValueError, "loopback-only"):
            validate_config(config)

    def make_resolver(self, directory: str) -> NovelAnswerResolver:
        catalog = Path(directory) / "answers.json"
        catalog.write_text(
            json.dumps({"answers": ["Attack on Titan", "Girls' Last Tour"]}),
            encoding="utf-8",
        )
        resolver = NovelAnswerResolver(
            NovelConfig(
                enabled=True,
                endpoint="http://127.0.0.1:65530",
                answer_catalog_path=catalog,
                knowledge_index_path=None,
                min_confidence=0.72,
            )
        )
        resolver._ready = True
        return resolver

    def test_query_planning_is_deterministic_and_needs_no_model_call(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            resolver = self.make_resolver(directory)
            with mock.patch.object(resolver, "_chat_json") as chat:
                queries = resolver._plan_queries(
                    "A young woman who recruits two wandering swordsmen.",
                    "character",
                    "",
                    time.perf_counter() + 5.0,
                )
            self.assertEqual(chat.call_count, 0)
            self.assertGreaterEqual(len(queries), 2)
            self.assertIn("anime character", queries[0])

    def test_grounded_answer_requires_typed_evidence_and_is_not_cached(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            resolver = self.make_resolver(directory)
            with (
                mock.patch.object(resolver, "ensure_ready", return_value=True),
                mock.patch.object(
                    resolver, "_plan_queries", return_value=["query one", "query two"]
                ),
                mock.patch.object(
                    resolver,
                    "_search_web",
                    return_value=[
                        {
                            "source": "Wikipedia",
                            "title": "Attack on Titan",
                            "snippet": "Dedicate your hearts, Survey Corps",
                            "url": "https://example.test/aot",
                            "answer": "Attack on Titan",
                            "answer_type": "anime_title",
                        }
                    ],
                ),
                mock.patch.object(
                    resolver,
                    "_synthesize",
                    return_value={
                        "answer": "attack on titan",
                        "alternatives": ["Girls Last Tour", "attack on titan"],
                        "confidence": 0.98,
                    },
                ) as synthesize,
            ):
                first = resolver.resolve_ranked('"Dedicate your hearts!"', "anime_title")
                second = resolver.resolve('"Dedicate your hearts!"', "anime_title")
            assert first is not None
            self.assertEqual(first.answer, "Attack on Titan")
            # Alternatives are canonicalized and deduplicated against the answer.
            self.assertEqual(first.alternatives, ("Girls' Last Tour",))
            self.assertEqual(second, "Attack on Titan")
            self.assertEqual(synthesize.call_count, 2)

    def test_low_confidence_answer_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            resolver = self.make_resolver(directory)
            with (
                mock.patch.object(resolver, "ensure_ready", return_value=True),
                mock.patch.object(
                    resolver, "_plan_queries", return_value=["query one", "query two"]
                ),
                mock.patch.object(resolver, "_search_web", return_value=[]),
                mock.patch.object(
                    resolver,
                    "_synthesize",
                    return_value={"answer": "Wrong", "alternatives": [], "confidence": 0.60},
                ),
            ):
                self.assertIsNone(resolver.resolve("ambiguous clue", "anime_title"))

    def test_emoji_semantics_include_anime_specific_search_terms(self) -> None:
        description = NovelAnswerResolver._describe_symbols("🚗 ⛰️ 🥤 💨")
        self.assertIn("mountain pass", description)
        self.assertIn("cup of water", description)
        self.assertIn("drifting", description)

    def test_absolute_deadline_is_carried_into_the_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            resolver = NovelAnswerResolver(
                NovelConfig(
                    enabled=True,
                    endpoint="http://127.0.0.1:65530",
                    answer_catalog_path=None,
                    knowledge_index_path=None,
                    total_timeout_seconds=0.05,
                )
            )
            resolver._ready = True

            def slow_plan(_clue, _answer_type, _symbols, deadline):
                time.sleep(0.06)
                resolver._bounded_timeout(1.0, deadline)
                return ["unused", "unused two"]

            with (
                mock.patch.object(resolver, "ensure_ready", return_value=True),
                mock.patch.object(resolver, "_plan_queries", side_effect=slow_plan),
            ):
                started = time.perf_counter()
                with self.assertLogs(
                    "anime_trivia_automation.novel", level="ERROR"
                ):
                    self.assertIsNone(resolver.resolve("new clue", "anime_title"))
                self.assertLess(time.perf_counter() - started, 0.25)

    def test_failed_startup_opens_a_no_retry_circuit(self) -> None:
        resolver = NovelAnswerResolver(
            NovelConfig(
                enabled=True,
                endpoint="http://127.0.0.1:65530",
                manage_server=False,
                answer_catalog_path=None,
                knowledge_index_path=None,
            )
        )
        with mock.patch.object(resolver, "_health_ok", return_value=False) as health:
            with self.assertLogs(
                "anime_trivia_automation.novel", level="ERROR"
            ):
                self.assertFalse(resolver.ensure_ready())
            self.assertFalse(resolver.ensure_ready())
        self.assertEqual(health.call_count, 1)

    def test_managed_mode_refuses_an_unowned_occupied_endpoint(self) -> None:
        resolver = NovelAnswerResolver(
            NovelConfig(
                enabled=True,
                endpoint="http://127.0.0.1:65530",
                manage_server=True,
                answer_catalog_path=None,
                knowledge_index_path=None,
            )
        )
        with mock.patch.object(resolver, "_health_ok", return_value=True):
            with self.assertLogs(
                "anime_trivia_automation.novel", level="ERROR"
            ):
                self.assertFalse(resolver.ensure_ready())
        self.assertIn("unowned process", resolver.last_detail)

    def test_untyped_web_title_cannot_validate_a_character_answer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            resolver = self.make_resolver(directory)
            answer, accepted = resolver._canonicalize_from_evidence(
                "Samurai Champloo",
                "character",
                [
                    {
                        "source": "Web",
                        "title": "Samurai Champloo",
                        "snippet": "Anime series",
                        "url": "https://example.test",
                    }
                ],
                "anime series character",
            )
            self.assertEqual(answer, "Samurai Champloo")
            self.assertFalse(accepted)


if __name__ == "__main__":
    unittest.main()
