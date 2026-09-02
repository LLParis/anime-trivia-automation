from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from anime_trivia_automation.cache import TriviaCache
from anime_trivia_automation.config import MatchConfig


class CachePrecedenceTests(unittest.TestCase):
    def test_reviewed_history_overrides_conflicting_mutable_semantic_entry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history_path = root / "history.json"
            history_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "pairs": [
                            {
                                "clue": '"Who decides limits? And based on what?"',
                                "type": "anime_title",
                                "answer": "One-Punch Man",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            cache = TriviaCache(
                root / "mutable.json",
                MatchConfig(),
                history_path=history_path,
            )
            cache.add_semantic(
                '"Who decides limits? And based on what?"',
                "anime_title",
                "Wrong Runtime Answer",
                source="test-conflict",
            )

            hit = cache.match_history(
                '"Who decides limits? And based on what?"',
                "anime_title",
            )
            self.assertIsNotNone(hit)
            assert hit is not None
            self.assertEqual(hit.answer, "One-Punch Man")


if __name__ == "__main__":
    unittest.main()
