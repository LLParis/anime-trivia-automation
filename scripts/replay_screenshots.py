"""Replay saved Anime Soul screenshots through one warmed OCR/cache pipeline."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from anime_trivia_automation.cache import TriviaCache
from anime_trivia_automation.config import load_config
from anime_trivia_automation.models import Scene
from anime_trivia_automation.ocr import PaddleOCREngine, PromptExtractor


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()

    config = load_config(args.config)
    ocr = PaddleOCREngine(config.ocr)
    extractor = PromptExtractor(config.prompt, config.matching, config.readiness)
    cache = TriviaCache(
        config.runtime.cache_path,
        config.matching,
        config.runtime.seed_cache_path,
        config.runtime.history_path,
    )
    results: list[dict[str, object]] = []
    for generation, path in enumerate(args.images, start=1):
        frame = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            results.append({"image": str(path), "error": "unreadable"})
            continue
        scene = Scene(
            generation=generation,
            frame=frame,
            captured_at=time.perf_counter(),
            detected_at=time.monotonic(),
            mean_delta=1.0,
            changed_ratio=1.0,
        )
        spans = ocr.recognize(frame)
        observation = extractor.extract(scene, spans)
        if observation is None:
            results.append({"image": str(path), "error": "prompt-not-detected"})
            continue
        hit = cache.match_history(
            observation.hint_text,
            observation.expected_answer_type,
        )
        if hit is None and observation.prompt_kind == "text":
            hit = cache.match_text(observation.hint_text)
        if hit is None and observation.prompt_kind == "visual":
            hit = cache.match_image(observation.perceptual_hash or "")
        results.append(
            {
                "image": str(path),
                "question": observation.question_label,
                "kind": observation.prompt_kind,
                "readiness": observation.readiness,
                "hint": observation.hint_text,
                "hash": observation.perceptual_hash,
                "answer": hit.answer if hit else None,
                "source": hit.kind if hit else None,
            }
        )
    print(json.dumps(results, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
