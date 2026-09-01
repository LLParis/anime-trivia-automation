from __future__ import annotations

import json
import logging
import queue
import re
import threading
import time
from collections import Counter
from dataclasses import replace
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .cache import TriviaCache
from .capture import DXCapture, GpuFrameChangeGate
from .config import AppConfig
from .discord import DiscordQuestionLocator
from .models import AnswerTask, CacheHit, PromptObservation, Scene
from .ocr import PaddleOCREngine, PromptExtractor
from .typing import (
    ActivePromptState,
    AnswerDispatcher,
    EmergencyStopListener,
    ForegroundWindowGuard,
    SafeKeyboardExecutor,
)
from .utils import (
    LatestMailbox,
    ensure_directory,
    normalize_accessible_clue,
    normalize_question,
    sanitize_answer,
)
from .vlm import LazyQwenResolver

LOGGER = logging.getLogger(__name__)


@dataclass
class PendingRound:
    signature: str
    question_label: str
    expected_answer_type: str
    prompt_kind: str
    clue: str
    hashes: set[str] = field(default_factory=set)
    baseline_reveals: Counter[str] = field(default_factory=Counter)
    saw_ready: bool = False


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
            history_path=config.runtime.history_path,
        )
        self._pending_round: PendingRound | None = None
        self._ephemeral_answer: tuple[str, str, str] | None = None
        self._quiz_ended = False
        self._accessible_round: tuple[str, str] | None = None
        self._vlm = LazyQwenResolver(config.vlm)
        self._foreground_guard = ForegroundWindowGuard(config.typing)
        self._question_locator = DiscordQuestionLocator()

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
        quiz_complete = "quiz complete" in normalize_question(
            " ".join(span.text for span in spans)
        )
        observation = self._extractor.extract(ocr_scene, spans)
        if observation is None and prelocated:
            # Fail back to the broad calibrated band if an unrelated vertical
            # colored component ever passes the strict prelocator geometry.
            spans = self._ocr.recognize(scene.frame)
            quiz_complete = "quiz complete" in normalize_question(
                " ".join(span.text for span in spans)
            )
            observation = self._extractor.extract(scene, spans)
        ocr_ms = (time.perf_counter() - stage_start) * 1000.0

        extract_start = time.perf_counter()
        extract_ms = (time.perf_counter() - extract_start) * 1000.0
        if observation is None:
            if quiz_complete:
                self._finish_quiz()
                return
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

        # Closed cards are terminal observations.  They may complete one
        # strictly tracked visual-learning transaction, but they can never
        # resolve, re-arm, or queue an answer.  This also makes post-quiz
        # scrolling harmless.
        if observation.readiness == "closed":
            if self._quiz_ended:
                return
            self._learn_from_authoritative_reveal(observation, spans)
            if quiz_complete:
                self._finish_quiz()
            return
        if observation.readiness not in {"locked", "ready"}:
            return
        if self._quiz_ended:
            LOGGER.info("Fresh live card observed; starting a new quiz session")
            self._quiz_ended = False
        if (
            self._pending_round is not None
            and self._pending_round.signature != observation.signature
        ):
            LOGGER.info(
                "Expired unlearned visual transaction for %s",
                self._pending_round.question_label,
            )
            self._pending_round = None
        if (
            self._ephemeral_answer is not None
            and self._ephemeral_answer[0] != observation.signature
        ):
            self._ephemeral_answer = None

        lookup_start = time.perf_counter()
        hit = self._match_authoritative_history(observation)
        if hit is None and observation.prompt_kind == "text":
            hit = self._cache.match_history(
                observation.hint_text,
                observation.expected_answer_type,
            ) or self._cache.match_text(observation.hint_text)
        elif hit is None:
            hit = self._cache.match_image(observation.perceptual_hash or "")
        lookup_ms = (time.perf_counter() - lookup_start) * 1000.0

        answer: str | None
        source: str
        vlm_ms = 0.0
        if hit is not None:
            answer = hit.answer
            source = f"{hit.kind}-cache"
            metric = (
                f"distance={hit.score:.0f}"
                if hit.kind == "image"
                else f"score={hit.score:.1f}"
            )
            LOGGER.info("Fast-path %s hit (%s) -> %s", hit.kind, metric, answer)
        elif (
            self._ephemeral_answer is not None
            and self._ephemeral_answer[0] == observation.signature
        ):
            _signature, answer, source = self._ephemeral_answer
            LOGGER.info("Reusing verified in-memory round answer -> %s", answer)
        else:
            self._arm_pending_round(observation, scene)
            if (
                not self._config.vlm.allow_unverified_submission
                or (
                    observation.prompt_kind == "visual"
                    and not self._config.vlm.allow_novel_visual_submission
                )
            ):
                answer = None
                source = "model-unverified"
                LOGGER.warning(
                    "Unverified model submission disabled; waiting for an authoritative reveal"
                )
            else:
                vlm_start = time.perf_counter()
                answer = self._vlm.resolve(observation)
                vlm_ms = (time.perf_counter() - vlm_start) * 1000.0
                source = "local-model-consensus"
            if answer is not None:
                # A model result can serve only this live round.  It is never
                # promoted into the durable cache until a correctly associated
                # Anime Soul reveal verifies it.
                self._ephemeral_answer = (
                    observation.signature,
                    answer,
                    source,
                )

        if answer is None:
            LOGGER.warning(
                "No confident answer for prompt %s", observation.question_label or "?"
            )
            return
        # Chat animations can advance the capture generation while the model
        # works.  Accept the answer only if OCR has re-confirmed the same live
        # logical card; never rely on exact frame-generation equality.
        if not self._active_prompt.wait_current_open(
            observation.signature, self._stop_event, timeout=1.0
        ):
            LOGGER.info("Resolved answer belongs to a stale or closed prompt")
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

    def _finish_quiz(self) -> None:
        if self._quiz_ended:
            return
        self._quiz_ended = True
        self._pending_round = None
        self._ephemeral_answer = None
        self._accessible_round = None
        self._dispatcher.observe_prompt(
            None,
            "closed",
            self._change_gate.generation,
        )
        LOGGER.info("Quiz-complete latch armed; historical cards are inert")

    def _match_authoritative_history(
        self, observation: PromptObservation
    ) -> CacheHit | None:
        if (
            self._accessible_round is not None
            and self._accessible_round[0] == observation.signature
        ):
            clue = self._accessible_round[1]
            return self._cache.match_history(
                clue,
                observation.expected_answer_type,
            )

        allowed, _reason = self._foreground_guard.allowed()
        window = self._foreground_guard.current() if allowed else None
        if window is None:
            return None
        try:
            accessible = self._question_locator.find_question(
                window.hwnd,
                window.process_id,
            )
        except Exception:
            LOGGER.debug("Discord semantic clue read failed", exc_info=True)
            return None
        if (
            accessible is None
            or accessible.question_label != observation.question_label
            or accessible.expected_answer_type != observation.expected_answer_type
        ):
            return None
        self._accessible_round = (observation.signature, accessible.clue)
        hit = self._cache.match_history(
            accessible.clue,
            observation.expected_answer_type,
        )
        if hit is not None:
            LOGGER.info("Authoritative Discord clue hit: %r", accessible.clue)
        return hit

    def _reveal_records(
        self, spans: tuple[Any, ...]
    ) -> list[tuple[str, str, int, int]]:
        correct_markers = [
            span
            for span in spans
            if normalize_question(span.text).startswith("correct")
            or normalize_question(span.text).startswith("time s up")
        ]
        records: list[tuple[str, str, int, int]] = []
        for span in spans:
            match = re.search(
                r"\bthe answer was\s+(.+?)(?:\s*\+\d+\s+AS\s+Points.*|$)",
                span.text,
                flags=re.IGNORECASE,
            )
            if match is None:
                continue
            has_result_header = any(
                0 <= span.top - marker.top <= 220
                and abs(span.left - marker.left) <= 420
                for marker in correct_markers
            )
            if not has_result_header:
                continue
            raw_answer = match.group(1).strip()
            if raw_answer.endswith("."):
                raw_answer = raw_answer[:-1].rstrip()
            answer = sanitize_answer(
                raw_answer,
                self._config.typing.max_answer_characters,
            )
            normalized = normalize_question(answer or "")
            if answer is not None and normalized:
                records.append(
                    (normalized, answer, int(span.top), int(span.left))
                )
        return records

    def _arm_pending_round(
        self, observation: PromptObservation, full_scene: Scene
    ) -> None:
        if not observation.question_label:
            return
        clue = observation.hint_text
        if (
            self._accessible_round is not None
            and self._accessible_round[0] == observation.signature
        ):
            clue = self._accessible_round[1]
        if (
            self._pending_round is None
            or self._pending_round.signature != observation.signature
        ):
            full_spans = self._ocr.recognize(full_scene.frame)
            baseline = Counter(
                normalized
                for normalized, _answer, _top, _left in self._reveal_records(full_spans)
            )
            self._pending_round = PendingRound(
                signature=observation.signature,
                question_label=observation.question_label,
                expected_answer_type=observation.expected_answer_type,
                prompt_kind=observation.prompt_kind,
                clue=clue,
                hashes=(
                    {observation.perceptual_hash}
                    if observation.perceptual_hash
                    else set()
                ),
                baseline_reveals=baseline,
                saw_ready=observation.readiness == "ready",
            )
            LOGGER.info(
                "Armed authoritative reveal transaction for %s with %d baseline result(s)",
                observation.question_label,
                sum(baseline.values()),
            )
            return
        if observation.perceptual_hash and len(self._pending_round.hashes) < 6:
            self._pending_round.hashes.add(observation.perceptual_hash)
        if observation.readiness == "ready":
            self._pending_round.saw_ready = True

    def _learn_from_authoritative_reveal(
        self,
        observation: PromptObservation,
        spans: tuple[Any, ...],
    ) -> None:
        pending = self._pending_round
        if (
            observation.readiness != "closed"
            or pending is None
            or observation.signature != pending.signature
            or observation.question_label != pending.question_label
            or not pending.saw_ready
            or observation.card_box is None
        ):
            return

        semantic_continuity = False
        allowed, _reason = self._foreground_guard.allowed()
        window = self._foreground_guard.current() if allowed else None
        if window is not None and pending.clue:
            try:
                current = self._question_locator.find_question(
                    window.hwnd, window.process_id
                )
                semantic_continuity = bool(
                    current is not None
                    and current.question_label == pending.question_label
                    and current.expected_answer_type == pending.expected_answer_type
                    and normalize_accessible_clue(current.clue)
                    == normalize_accessible_clue(pending.clue)
                )
            except Exception:
                LOGGER.debug("Closed-card semantic continuity failed", exc_info=True)

        if pending.prompt_kind == "visual" and not semantic_continuity:
            if observation.perceptual_hash is None or not pending.hashes:
                return
            try:
                import imagehash

                closed_hash = imagehash.hex_to_hash(observation.perceptual_hash)
                nearest_locked_distance = min(
                    int(closed_hash - imagehash.hex_to_hash(hash_text))
                    for hash_text in pending.hashes
                )
            except (ImportError, ValueError):
                LOGGER.exception("Could not compare the closed visual clue hash")
                return
            if nearest_locked_distance > self._config.matching.phash_max_distance:
                LOGGER.warning(
                    "Closed visual crop does not match its locked clue (distance=%d)",
                    nearest_locked_distance,
                )
                return
        elif pending.prompt_kind == "text" and not semantic_continuity:
            from rapidfuzz import fuzz

            continuity_score = fuzz.WRatio(
                normalize_question(pending.clue),
                normalize_question(observation.hint_text),
            )
            if continuity_score < self._config.matching.text_score_threshold:
                return

        records = self._reveal_records(spans)
        current_counts = Counter(
            normalized for normalized, _answer, _top, _left in records
        )
        card_left, _card_top, card_right, card_bottom = observation.card_box
        new_answers: dict[str, str] = {}
        for normalized, answer, top, left in records:
            if not card_bottom < top <= card_bottom + 900:
                continue
            if not card_left - 160 <= left <= card_right + 160:
                continue
            if current_counts[normalized] <= pending.baseline_reveals[normalized]:
                continue
            new_answers[normalized] = answer
        if len(new_answers) != 1:
            LOGGER.debug(
                "Visual reveal transaction still waiting: %d eligible answer(s)",
                len(new_answers),
            )
            return

        answer = next(iter(new_answers.values()))
        try:
            if pending.clue:
                self._cache.add_semantic(
                    pending.clue,
                    pending.expected_answer_type,
                    answer,
                    source="authoritative-card-reveal-v2",
                )
            if pending.prompt_kind == "text" and pending.clue:
                self._cache.add_text(
                    pending.clue,
                    answer,
                    source="authoritative-card-reveal-v2",
                )
            else:
                for hash_text in sorted(pending.hashes):
                    self._cache.add_image(
                        hash_text,
                        answer,
                        source="authoritative-card-reveal-v2",
                    )
        except (OSError, TypeError, ValueError):
            LOGGER.exception("Could not persist authoritative round answer")
            return
        self._pending_round = None
        LOGGER.info(
            "Learned clue %s from its paired reveal -> %s",
            observation.question_label,
            answer,
        )

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
            "Starting Anime Trivia Automation%s. Keep Anime Soul Discord in the foreground; the composer is verified automatically. Press %s to stop.",
            " in DRY RUN mode" if not self._config.typing.enabled else "",
            self._config.typing.stop_key.upper(),
        )
        if self._config.vlm.enabled and self._config.vlm.ready_before_capture:
            LOGGER.info(
                "Loading the explicitly enabled experimental VLM before capture"
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
        history_path=config.runtime.history_path,
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
    hit = cache.match_history(
        observation.hint_text,
        observation.expected_answer_type,
    )
    if hit is None:
        hit = (
            cache.match_text(observation.hint_text)
            if observation.prompt_kind == "text"
            else cache.match_image(observation.perceptual_hash or "")
        )
    answer = hit.answer if hit else None
    source = f"{hit.kind}-cache" if hit else None
    if answer is None and use_vlm:
        answer = LazyQwenResolver(config.vlm).resolve(observation)
        source = "experimental-vlm" if answer else None
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
