from __future__ import annotations

import json
import logging
import queue
import re
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from .cache import TriviaCache
from .capture import DXCapture, GpuFrameChangeGate
from .config import AppConfig
from .models import AnswerTask, PromptObservation, Scene
from .ocr import PaddleOCREngine, PromptExtractor
from .typing import (
    ActivePromptState,
    AnswerDispatcher,
    EmergencyStopListener,
    SafeKeyboardExecutor,
)
from .utils import LatestMailbox, ensure_directory, sanitize_answer
from .vlm import LazyQwenResolver

LOGGER = logging.getLogger(__name__)


class AnimeTriviaAutomation:
    """End-to-end latest-scene pipeline for Anime Soul trivia cards."""

    def __init__(self, config: AppConfig, *, dry_run: bool = False) -> None:
        if not config.capture.calibrated:
            raise RuntimeError(
                "capture.calibrated is false. Run scripts/calibrate_region.py "
                "before starting live desktop capture."
            )
        if dry_run:
            config = replace(config, typing=replace(config.typing, enabled=False))
        self._config = config
        self._stop_event = threading.Event()
        self._stop_lock = threading.Lock()
        self._stopped = False
        self._mailbox: LatestMailbox[Scene] = LatestMailbox()
        self._active_prompt = ActivePromptState()

        capture_width = config.capture.region[2] - config.capture.region[0]
        capture_height = config.capture.region[3] - config.capture.region[1]
        self._change_gate = GpuFrameChangeGate(
            config.change_detection,
            capture_size=(capture_width, capture_height),
            on_change=self._on_visual_change,
        )
        self._ocr = PaddleOCREngine(config.ocr)
        self._extractor = PromptExtractor(
            config.prompt,
            config.matching,
            config.readiness,
        )
        self._cache = TriviaCache(
            config.runtime.cache_path,
            config.matching,
            seed_path=config.runtime.seed_cache_path,
        )
        self._pending_visual_hashes: dict[str, str] = {}
        self._vlm = LazyQwenResolver(config.vlm)

        self._keyboard = SafeKeyboardExecutor(
            config.typing,
            config.readiness,
            self._active_prompt,
            self._stop_event,
        )
        self._dispatcher = AnswerDispatcher(
            self._keyboard, self._active_prompt, self._stop_event
        )
        self._capture = DXCapture(config.capture, self._on_frame, self._stop_event)
        self._processor_thread = threading.Thread(
            target=self._processing_loop,
            name="trivia-inference",
            daemon=True,
        )
        self._emergency_stop = EmergencyStopListener(config.typing.stop_key, self.stop)

    def _on_visual_change(self, generation: int) -> None:
        # Pause any in-progress input immediately. OCR will either revalidate
        # the same logical round (for a harmless status edit) or replace it.
        self._active_prompt.mark_uncertain(generation)
        LOGGER.debug("Visual scene changed; generation %d awaiting OCR", generation)

    def _on_frame(self, frame: Any, captured_at: float) -> None:
        try:
            scene = self._change_gate.observe(frame, captured_at)
            if scene is not None:
                self._mailbox.put(scene)
                LOGGER.debug(
                    "Stable scene %d: MAD=%.5f changed=%.3f%%",
                    scene.generation,
                    scene.mean_delta,
                    scene.changed_ratio * 100.0,
                )
        except Exception:
            LOGGER.exception("Frame-change pipeline failed")
            self.stop()

    def _processing_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                scene = self._mailbox.get(timeout=0.2)
            except queue.Empty:
                continue
            for attempt in range(self._config.runtime.scene_retry_limit + 1):
                try:
                    self._process_scene(scene)
                    break
                except Exception:
                    can_retry = (
                        attempt < self._config.runtime.scene_retry_limit
                        and self._change_gate.generation == scene.generation
                        and not self._stop_event.is_set()
                    )
                    LOGGER.exception(
                        "Scene %d processing failed%s",
                        scene.generation,
                        "; retrying once" if can_retry else "",
                    )
                    if not can_retry:
                        break
                    if self._stop_event.wait(0.05):
                        break

    def _process_scene(self, scene: Scene) -> None:
        stage_start = time.perf_counter()
        ocr_scene, prelocated = self._extractor.crop_to_active_card(scene)
        spans = self._ocr.recognize(ocr_scene.frame)
        observation = self._extractor.extract(ocr_scene, spans)
        if observation is None and prelocated:
            # Fail back to the broad calibrated band if an unrelated vertical
            # colored component ever passes the strict prelocator geometry.
            spans = self._ocr.recognize(scene.frame)
            observation = self._extractor.extract(scene, spans)
        ocr_ms = (time.perf_counter() - stage_start) * 1000.0

        extract_start = time.perf_counter()
        extract_ms = (time.perf_counter() - extract_start) * 1000.0
        if observation is None:
            # A partial render or one OCR miss must never reactivate an older
            # card or re-arm an answered round. Keep the state uncertain until
            # a complete current-generation card is observed.
            self._active_prompt.mark_uncertain(scene.generation)
            LOGGER.debug(
                "Scene %d contains no complete Anime Soul prompt", scene.generation
            )
            return

        if not self._dispatcher.observe_prompt(
            observation.signature,
            observation.readiness,
            scene.generation,
        ):
            LOGGER.debug(
                "Discarded stale OCR observation for scene %d", scene.generation
            )
            return
        self._save_prompt_crop(observation)
        self._learn_from_authoritative_reveal(observation, spans)
        LOGGER.info(
            "Prompt %s kind=%s type=%s readiness=%s (red=%d green=%d) hint=%r "
            "OCR=%.1fms extract/hash=%.1fms",
            observation.question_label or "?",
            observation.prompt_kind,
            observation.expected_answer_type,
            observation.readiness,
            observation.red_outline_pixels,
            observation.green_outline_pixels,
            observation.hint_text or "<visual/emoji>",
            ocr_ms,
            extract_ms,
        )

        lookup_start = time.perf_counter()
        if observation.prompt_kind == "text":
            hit = self._cache.match_text(observation.hint_text)
        else:
            hit = self._cache.match_image(observation.perceptual_hash or "")
        lookup_ms = (time.perf_counter() - lookup_start) * 1000.0

        answer: str | None
        source: str
        vlm_ms = 0.0
        if hit is not None:
            answer = hit.answer
            source = f"{hit.kind}-cache"
            metric = (
                f"score={hit.score:.1f}"
                if hit.kind == "text"
                else f"distance={hit.score:.0f}"
            )
            LOGGER.info("Fast-path %s hit (%s) -> %s", hit.kind, metric, answer)
        else:
            if (
                observation.prompt_kind == "visual"
                and not self._config.vlm.allow_novel_visual_submission
            ):
                answer = None
                source = "visual-unverified"
                if observation.question_label and observation.perceptual_hash:
                    self._pending_visual_hashes[observation.question_label] = (
                        observation.perceptual_hash
                    )
                LOGGER.warning(
                    "Novel visual clue held for authoritative reveal learning"
                )
            else:
                vlm_start = time.perf_counter()
                answer = self._vlm.resolve(observation)
                vlm_ms = (time.perf_counter() - vlm_start) * 1000.0
                source = "qwen3-vl"
            if answer is not None:
                # Cache even if a later scene arrived while the slow path ran; it
                # becomes a fast hit next time. Stale-scene protection still blocks typing.
                try:
                    if observation.prompt_kind == "text":
                        self._cache.add_text(
                            observation.hint_text, answer, source=source
                        )
                    elif observation.perceptual_hash:
                        self._cache.add_image(
                            observation.perceptual_hash, answer, source=source
                        )
                except (OSError, TypeError, ValueError):
                    # A disk/cache failure must not throw away an answer that is
                    # already resolved for the still-live round.
                    LOGGER.exception("Could not persist the learned answer")

        if answer is None:
            LOGGER.warning(
                "No confident answer for prompt %s", observation.question_label or "?"
            )
            return
        # A generative miss can take seconds. The capture gate continues to
        # advance meanwhile, so require both logical-prompt and visual-scene
        # freshness before queuing input. If this was merely a status edit of
        # the same question, the newest scene will immediately hit the cache.
        if (
            self._change_gate.generation != scene.generation
            or not self._active_prompt.is_current(observation.signature)
        ):
            LOGGER.info(
                "Resolved stale prompt; answer cached but not queued for typing"
            )
            return

        timings = {
            "ocr": ocr_ms,
            "extract_hash": extract_ms,
            "lookup": lookup_ms,
            "vlm": vlm_ms,
        }
        task = AnswerTask(
            answer=answer,
            prompt_signature=observation.signature,
            expected_answer_type=observation.expected_answer_type,
            question_label=observation.question_label,
            detected_at=scene.detected_at,
            countdown_seconds=observation.countdown_seconds,
            source=source,
            stage_timings_ms=timings,
        )
        if self._dispatcher.submit(task):
            LOGGER.info(
                "Answer queued: %s (processing %.1fms, slow-path %.1fms)",
                answer,
                ocr_ms + extract_ms + lookup_ms,
                vlm_ms,
            )

    def _learn_from_authoritative_reveal(
        self,
        observation: PromptObservation,
        spans: tuple[Any, ...],
    ) -> None:
        if observation.readiness != "closed" or not observation.question_label:
            return
        pending_hash = self._pending_visual_hashes.get(observation.question_label)
        if pending_hash is None:
            return
        for span in spans:
            match = re.search(
                r"\bthe answer was\s+(.+?)(?:[.!?](?:\s|$)|$)",
                span.text,
                flags=re.IGNORECASE,
            )
            if match is None:
                continue
            answer = sanitize_answer(
                match.group(1),
                self._config.typing.max_answer_characters,
            )
            if answer is None:
                continue
            try:
                self._cache.add_image(
                    pending_hash,
                    answer,
                    source="authoritative-round-reveal",
                )
            except (OSError, TypeError, ValueError):
                LOGGER.exception("Could not persist authoritative visual answer")
                return
            self._pending_visual_hashes.pop(observation.question_label, None)
            LOGGER.info(
                "Learned visual clue %s from authoritative reveal -> %s",
                observation.question_label,
                answer,
            )
            return

    def _save_prompt_crop(self, observation: PromptObservation) -> None:
        if not self._config.runtime.save_prompt_crops:
            return
        try:
            import cv2
        except ImportError:
            LOGGER.warning("Cannot save prompt crops because OpenCV is unavailable")
            return
        ensure_directory(self._config.runtime.debug_dir)
        label = (
            observation.question_label.replace("/", "-")
            if observation.question_label
            else "unknown"
        )
        path = (
            self._config.runtime.debug_dir
            / f"scene-{observation.scene.generation}-{label}.png"
        )
        cv2.imwrite(str(path), observation.prompt_crop)

    def run(self) -> None:
        LOGGER.info(
            "Starting Anime Trivia Automation%s. Keep the Discord message box focused. Press %s to stop.",
            " in DRY RUN mode" if not self._config.typing.enabled else "",
            self._config.typing.stop_key.upper(),
        )
        if self._config.vlm.enabled and self._config.vlm.ready_before_capture:
            LOGGER.info(
                "Loading Qwen3-VL before arming capture so the slow path is ready"
            )
            self._vlm.ensure_loaded()

        self._dispatcher.start()
        self._processor_thread.start()
        self._emergency_stop.start()
        if (
            self._config.vlm.enabled
            and not self._config.vlm.ready_before_capture
            and self._config.vlm.preload_in_background
        ):
            threading.Thread(
                target=self._vlm.preload, name="vlm-preload", daemon=True
            ).start()
        self._capture.start()

        try:
            while not self._stop_event.wait(0.25):
                pass
        except KeyboardInterrupt:
            LOGGER.warning("Ctrl+C received")
            self.stop()
        finally:
            self.stop()
            self._capture.join()
            self._processor_thread.join(timeout=5.0)
            self._dispatcher.join(timeout=5.0)
            LOGGER.info("Anime Trivia Automation stopped")

    def stop(self) -> None:
        with self._stop_lock:
            if self._stopped:
                return
            self._stopped = True
            self._stop_event.set()
        self._capture.request_stop()
        try:
            self._emergency_stop.stop()
        except RuntimeError:
            LOGGER.debug("Emergency-stop listener was already stopped", exc_info=True)


def inspect_image(
    config: AppConfig, image_path: Path, *, use_vlm: bool
) -> dict[str, Any]:
    """Offline inspection path for the user's saved Anime Soul screenshots."""
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("OpenCV and NumPy are required for --inspect-image") from exc
    # cv2.imread does not reliably accept non-ASCII Windows paths (for example
    # the user's Imágenes folder); decode bytes read through Python instead.
    frame = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    ocr = PaddleOCREngine(config.ocr)
    extractor = PromptExtractor(config.prompt, config.matching, config.readiness)
    cache = TriviaCache(
        config.runtime.cache_path,
        config.matching,
        seed_path=config.runtime.seed_cache_path,
    )
    scene = Scene(
        generation=1,
        frame=frame,
        captured_at=time.perf_counter(),
        detected_at=time.monotonic(),
        mean_delta=1.0,
        changed_ratio=1.0,
    )
    spans = ocr.recognize(frame)
    observation = extractor.extract(scene, spans)
    if observation is None:
        return {"prompt_detected": False, "ocr": [span.text for span in spans]}
    hit = (
        cache.match_text(observation.hint_text)
        if observation.prompt_kind == "text"
        else cache.match_image(observation.perceptual_hash or "")
    )
    answer = hit.answer if hit else None
    source = f"{hit.kind}-cache" if hit else None
    if answer is None and use_vlm:
        answer = LazyQwenResolver(config.vlm).resolve(observation)
        source = "qwen3-vl" if answer else None
    return {
        "prompt_detected": True,
        "question": observation.question_label,
        "kind": observation.prompt_kind,
        "expected_answer_type": observation.expected_answer_type,
        "readiness": observation.readiness,
        "red_outline_pixels": observation.red_outline_pixels,
        "green_outline_pixels": observation.green_outline_pixels,
        "hint_text": observation.hint_text,
        "perceptual_hash": observation.perceptual_hash,
        "answer": answer,
        "source": source,
        "ocr": [span.text for span in spans],
    }


def print_inspection(result: dict[str, Any]) -> None:
    # Windows PowerShell often uses cp1252; escaped JSON remains printable even
    # when unrelated chat OCR contains Japanese or other Unicode text.
    print(json.dumps(result, ensure_ascii=True, indent=2))
