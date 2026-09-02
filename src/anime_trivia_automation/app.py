from __future__ import annotations

import asyncio
import json
import logging
import queue
import re
import threading
import time
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .antigravity import (
    AntigravityProvider,
    AntigravityRequest,
)
from .cache import TriviaCache
from .capture import DXCapture, GpuFrameChangeGate
from .config import AppConfig
from .discord import DiscordQuestionLocator
from .gemini import GeminiProvider, GeminiRequest, GeminiResult
from .models import AnswerTask, CacheHit, PromptObservation, Scene
from .novel import NovelAnswerResolver
from .ocr import PaddleOCREngine, PromptExtractor
from .status import NullStatus, OperatorStatus
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
    clue_fingerprint: str
    hashes: set[str] = field(default_factory=set)
    baseline_reveals: Counter[str] = field(default_factory=Counter)
    baseline_reveal_ids: set[str] = field(default_factory=set)
    saw_ready: bool = False


ResolutionKey = tuple[str, str]


@dataclass(frozen=True)
class ResolutionRequest:
    key: ResolutionKey
    round_token: str
    signature: str
    clue_fingerprint: str
    clue: str
    observation: PromptObservation
    started_at: float
    semantic_clue: bool = False


@dataclass(frozen=True)
class ProviderResolution:
    key: ResolutionKey
    provider: str
    source: str
    answer: str | None
    confidence: float
    elapsed_ms: float
    detail: str = ""
    error: str | None = None
    # Lower-ranked answers the provider also considered plausible. They become
    # follow-up guesses because a wrong guess costs nothing in Anime Soul.
    alternatives: tuple[str, ...] = ()


@dataclass
class AsyncResolutionRound:
    request: ResolutionRequest
    providers: set[str]
    pending: set[str]
    ocr_ms: float
    extract_ms: float
    lookup_ms: float
    results: dict[str, ProviderResolution] = field(default_factory=dict)
    # Distinct answers in arrival order. guesses[0] is the first submission;
    # the rest are typed only while the same card is still green.
    guesses: list[ProviderResolution] = field(default_factory=list)
    queued_answers: set[str] = field(default_factory=set)
    candidate: ProviderResolution | None = None
    queued: bool = False
    unknown_emitted: bool = False
    fallback_started: bool = False
    retired: bool = False


class AnimeTriviaAutomation:
    """End-to-end latest-scene pipeline for Anime Soul trivia cards."""

    def __init__(
        self,
        config: AppConfig,
        *,
        dry_run: bool = False,
        status: OperatorStatus | NullStatus | None = None,
    ) -> None:
        if not config.capture.calibrated:
            raise RuntimeError(
                "capture.calibrated is false. Run scripts/calibrate_region.py "
                "before starting live desktop capture."
            )
        if dry_run:
            config = replace(config, typing=replace(config.typing, enabled=False))
        self._config = config
        self._status = status or NullStatus()
        self._stop_event = threading.Event()
        self._stop_lock = threading.Lock()
        self._stopped = False
        self._mailbox: LatestMailbox[Scene] = LatestMailbox()
        self._active_prompt = ActivePromptState()
        self._status_session_id = 1
        self._status_round_id = 0
        self._active_status_signature: str | None = None
        self._active_status_question_label: str | None = None
        self._active_status_clue_key: str | None = None
        self._active_status_token: str | None = None
        self._active_status_closed = False
        self._status_resolution: dict[str, str] = {}
        self._resolution_rounds: dict[ResolutionKey, AsyncResolutionRound] = {}
        self._active_resolution_key: ResolutionKey | None = None
        self._resolution_results: queue.Queue[ProviderResolution] = queue.Queue(
            maxsize=16
        )
        self._resolution_executor = ThreadPoolExecutor(
            max_workers=3,
            thread_name_prefix="trivia-resolver",
        )
        # No more than six provider calls can be running or waiting. A normal
        # round submits at most local Qwen plus Gemini, so this covers a whole
        # live card transition without permitting an unbounded executor queue.
        self._resolution_slots = threading.BoundedSemaphore(6)
        self._provider_locks: dict[str, Any] = {
            "qwen": threading.Lock(),
            "gemini": threading.BoundedSemaphore(2),
            "antigravity": threading.Lock(),
            "vlm": threading.Lock(),
        }
        self._resolution_shutdown_lock = threading.Lock()
        self._resolution_shutdown = False

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
        self._status.emit(
            "LOADING",
            title="Reviewed history ready",
            detail=f"Loaded {self._cache.history_count} reviewed clues; preparing capture",
            history_entries=self._cache.history_count,
        )
        self._pending_round: PendingRound | None = None
        self._ephemeral_answer: tuple[str, str, str, str] | None = None
        self._quiz_ended = False
        self._accessible_round: tuple[str, str] | None = None
        self._vlm = LazyQwenResolver(config.vlm)
        self._novel = NovelAnswerResolver(config.novel)
        self._gemini = GeminiProvider(config.gemini)
        self._antigravity = AntigravityProvider(config.antigravity)
        self._gemini_loop: asyncio.AbstractEventLoop | None = None
        self._gemini_loop_thread: threading.Thread | None = None
        self._gemini_loop_ready = threading.Event()
        self._foreground_guard = ForegroundWindowGuard(config.typing)
        self._question_locator = DiscordQuestionLocator()

        self._keyboard = SafeKeyboardExecutor(
            config.typing,
            config.readiness,
            self._active_prompt,
            self._stop_event,
            status=self._status,
        )
        self._dispatcher = AnswerDispatcher(
            self._keyboard,
            self._active_prompt,
            self._stop_event,
            max_guesses_per_round=config.typing.max_guesses_per_round,
            guess_gap_seconds=config.typing.guess_gap_seconds,
        )
        self._capture = DXCapture(
            config.capture,
            self._on_frame,
            self._stop_event,
            on_started=self._on_capture_started,
            on_error=self._on_capture_error,
        )
        self._processor_thread = threading.Thread(
            target=self._processing_loop,
            name="trivia-inference",
            daemon=True,
        )
        self._emergency_stop = EmergencyStopListener(config.typing.stop_key, self.stop)

    def _on_capture_started(self) -> None:
        self._status.emit(
            "ARMED",
            title="Armed and monitoring",
            detail="Waiting for the next Anime Soul trivia card",
            question="—",
            clue="Monitoring #💜anime-chat for a red trivia card",
            answer="—",
            source=f"{self._cache.history_count} reviewed local clues",
            readiness="idle",
            history_entries=self._cache.history_count,
            event_id=f"armed:{self._status_session_id}",
            new_round=True,
        )

    def _on_capture_error(self, detail: str) -> None:
        self._status.emit(
            "ERROR",
            title="Capture failure",
            detail=detail,
            readiness="closed",
            event_id=f"capture-error:{detail}",
            increment="fatal_errors",
        )

    def _round_token(
        self, observation: PromptObservation, *, live: bool
    ) -> tuple[str, bool]:
        clue_key = self._status_clue_key(observation)
        same_round = False
        if self._active_status_token is not None and not self._active_status_closed:
            if observation.question_label and self._active_status_question_label:
                same_round = (
                    observation.question_label == self._active_status_question_label
                )
            elif observation.question_label and not self._active_status_question_label:
                # The footer is occasionally missed on the first OCR pass.  A
                # later question label upgrades the active fallback identity;
                # it must not make one physical card count as two rounds.
                same_round = clue_key == self._active_status_clue_key
            elif not observation.question_label and self._active_status_question_label:
                # Likewise, tolerate a transient footer miss after the card's
                # stable numbered identity has already been established.
                same_round = clue_key == self._active_status_clue_key
            else:
                same_round = observation.signature == self._active_status_signature

        if live and not same_round:
            self._status_round_id += 1
            self._active_status_signature = observation.signature
            self._active_status_question_label = observation.question_label
            self._active_status_clue_key = clue_key
            self._active_status_token = (
                f"session-{self._status_session_id}:round-{self._status_round_id}:"
                f"{observation.question_label or observation.signature}"
            )
            self._active_status_closed = False
            return self._active_status_token, True
        if same_round and self._active_status_token is not None:
            self._active_status_signature = observation.signature
            if observation.question_label:
                self._active_status_question_label = observation.question_label
            self._active_status_clue_key = clue_key
            return self._active_status_token, False
        return (
            f"session-{self._status_session_id}:untracked:{observation.signature}",
            False,
        )

    @staticmethod
    def _status_clue_key(observation: PromptObservation) -> str:
        if observation.hint_text:
            return f"text:{normalize_question(observation.hint_text)}"
        if observation.perceptual_hash:
            return f"visual:{observation.perceptual_hash}"
        return f"signature:{observation.signature}"

    def _display_clue(self, observation: PromptObservation) -> str:
        if (
            self._accessible_round is not None
            and self._accessible_round[0] == observation.signature
        ):
            return self._accessible_round[1]
        return observation.hint_text or "Visual / emoji clue"

    def _clue_fingerprint(self, observation: PromptObservation) -> str:
        accessible_clue = None
        if (
            self._accessible_round is not None
            and self._accessible_round[0] == observation.signature
        ):
            accessible_clue = self._accessible_round[1]
        clue = accessible_clue or observation.hint_text or "Visual / emoji clue"
        if (
            accessible_clue is None
            and observation.prompt_kind == "visual"
            and observation.perceptual_hash
        ):
            return f"visual:{observation.perceptual_hash}"
        effective_kind = self._effective_prompt_kind(observation.prompt_kind, clue)
        if effective_kind == "text":
            return "text:" + normalize_question(self._canonical_text_clue(clue))
        if accessible_clue is None and observation.perceptual_hash:
            return f"visual:{observation.perceptual_hash}"
        return "semantic:" + normalize_accessible_clue(clue)

    @staticmethod
    def _canonical_text_clue(clue: str) -> str:
        """Remove only a punctuation-delimited single-glyph OCR suffix."""

        # OCR occasionally appends one stray glyph after a complete sentence
        # (`...scum. T` / `...scum. 1`). Meaningful endings such as Team 7,
        # Class A, Level E, and Dragon Ball Z do not match this narrow shape.
        artifact = re.fullmatch(
            r"(?s)(.+[.!?][\"\u201d\u2019']?)\s+[A-Za-z0-9]",
            clue.strip(),
        )
        return artifact.group(1) if artifact is not None else clue

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
            self._status.emit(
                "ERROR",
                title="Frame pipeline failure",
                detail="Capture frame processing stopped",
                readiness="closed",
                event_id=f"frame-error:{self._change_gate.generation}",
                increment="fatal_errors",
            )
            self.stop()

    def _processing_loop(self) -> None:
        while not self._stop_event.is_set():
            self._drain_resolution_results()
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
                        self._status.emit(
                            "ATTENTION",
                            title="Scene processing failed",
                            detail=f"Could not process scene {scene.generation}; monitoring continues",
                            event_id=f"scene-error:{scene.generation}",
                        )
                        break
                    if self._stop_event.wait(0.05):
                        break
            self._drain_resolution_results()

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
            if self._should_finish_without_card(quiz_complete):
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

        if self._quiz_ended and observation.readiness in {"locked", "ready"}:
            # A red or green card is always the live round: finished rounds are
            # grey. Any live card therefore starts the next quiz session, even
            # when the worker was launched mid-quiz. (A locked-Q1-only latch
            # silently dropped Q7-Q10 of the 2026-09-02 7 AM quiz.)
            LOGGER.info(
                "Live card %s observed after quiz completion; starting a new session",
                observation.question_label or "?",
            )
            self._quiz_ended = False
        self._save_prompt_crop(observation)
        # A round signature is intentionally stable across OCR edits, so a UIA
        # clue cached under that signature must never survive into this fresh
        # stable observation. Re-read Discord semantics before every lookup.
        self._accessible_round = None
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
        round_token, new_round = self._round_token(
            observation,
            live=observation.readiness in {"locked", "ready"},
        )
        if new_round:
            self._retire_other_resolutions(round_token)
        if observation.readiness in {"locked", "ready"}:
            if new_round:
                self._status.emit(
                    "RED" if observation.readiness == "locked" else "GREEN",
                    title=(
                        f"{observation.question_label or 'Question'} — "
                        f"{'Get Ready' if observation.readiness == 'locked' else 'Answer Now'}"
                    ),
                    detail=(
                        "Resolving while answers are locked"
                        if observation.readiness == "locked"
                        else "First observed after answers opened"
                    ),
                    question=observation.question_label or "Question",
                    clue=self._display_clue(observation),
                    readiness=observation.readiness,
                    event_id=round_token,
                    increment="rounds_seen",
                    new_round=True,
                )

        # Closed cards are terminal observations.  They may complete one
        # strictly tracked visual-learning transaction, but they can never
        # resolve, re-arm, or queue an answer.  This also makes post-quiz
        # scrolling harmless.
        if observation.readiness == "closed":
            if self._quiz_ended:
                return
            if not self._dispatcher.observe_prompt(
                observation.signature,
                "closed",
                scene.generation,
                self._clue_fingerprint(observation),
            ):
                return
            active_round_closed = self._active_status_token == round_token
            if active_round_closed:
                self._status.emit(
                    "CLOSED",
                    title=f"{observation.question_label or 'Question'} — Round over",
                    detail="Submission gate is closed",
                    question=observation.question_label or "Question",
                    clue=self._display_clue(observation),
                    readiness="closed",
                    event_id=round_token,
                    increment="closed",
                )
                self._active_status_closed = True
                self._active_resolution_key = None
                state = self._resolution_rounds.get(
                    (round_token, self._clue_fingerprint(observation))
                )
                if state is not None:
                    state.retired = True
            self._learn_from_authoritative_reveal(observation, spans)
            if (
                quiz_complete
                and active_round_closed
                and self._active_status_closed
                and self._is_final_question(observation.question_label)
            ):
                self._finish_quiz()
            return
        if observation.readiness not in {"locked", "ready"}:
            return
        if (
            self._pending_round is not None
            and self._pending_round.signature != observation.signature
        ):
            LOGGER.info(
                "Expired unlearned visual transaction for %s",
                self._pending_round.question_label,
            )
            self._pending_round = None
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
        live_clue = self._display_clue(observation)
        clue_fingerprint = self._clue_fingerprint(observation)
        if not self._dispatcher.observe_prompt(
            observation.signature,
            observation.readiness,
            scene.generation,
            clue_fingerprint,
        ):
            return
        self._active_resolution_key = (round_token, clue_fingerprint)
        if self._ephemeral_answer is not None and (
            self._ephemeral_answer[0] != observation.signature
            or self._ephemeral_answer[1] != clue_fingerprint
        ):
            LOGGER.info("Discarded a candidate after the round clue changed")
            self._ephemeral_answer = None

        # Track every live round through green, including cache/model hits.
        # Previously only unresolved observations refreshed this transaction,
        # so all model-solved rounds missed saw_ready and could not learn their
        # official bot reveal.
        self._arm_pending_round(observation, scene)

        if hit is not None:
            answer = hit.answer
            source = f"{hit.kind}-cache"
            metric = (
                f"distance={hit.score:.0f}"
                if hit.kind == "image"
                else f"score={hit.score:.1f}"
            )
            LOGGER.info("Fast-path %s hit (%s) -> %s", hit.kind, metric, answer)
            previous_resolution = self._status_resolution.get(round_token)
            self._status_resolution[round_token] = "known"
            self._status.emit(
                "KNOWN",
                title=f"Known answer — {observation.question_label or 'Question'}",
                detail=f"Verified {source.replace('-', ' ')} hit ({metric})",
                question=observation.question_label or "Question",
                clue=self._display_clue(observation),
                answer=answer,
                source=source,
                readiness=observation.readiness,
                event_id=round_token,
                increment="known" if previous_resolution != "known" else None,
                decrement="unknown" if previous_resolution == "unknown" else None,
            )
            self._queue_answer_if_current(
                answer=answer,
                source=source,
                observation=observation,
                clue_fingerprint=clue_fingerprint,
                round_token=round_token,
                detected_at=scene.detected_at,
                ocr_ms=ocr_ms,
                extract_ms=extract_ms,
                lookup_ms=lookup_ms,
                provider_ms=0.0,
            )
            return

        if (
            self._ephemeral_answer is not None
            and self._ephemeral_answer[0] == observation.signature
            and self._ephemeral_answer[1] == clue_fingerprint
        ):
            _signature, _fingerprint, answer, source = self._ephemeral_answer
            LOGGER.info("Reusing verified in-memory round answer -> %s", answer)
            state = self._resolution_rounds.get((round_token, clue_fingerprint))
            if state is not None:
                self._queue_resolution_candidate(state)
            else:
                self._queue_answer_if_current(
                    answer=answer,
                    source=source,
                    observation=observation,
                    clue_fingerprint=clue_fingerprint,
                    round_token=round_token,
                    detected_at=scene.detected_at,
                    ocr_ms=ocr_ms,
                    extract_ms=extract_ms,
                    lookup_ms=lookup_ms,
                    provider_ms=0.0,
                )
            return

        key = (round_token, clue_fingerprint)
        state = self._resolution_rounds.get(key)
        if state is None:
            state = self._start_async_resolution(
                key=key,
                round_token=round_token,
                clue_fingerprint=clue_fingerprint,
                clue=live_clue,
                observation=observation,
                ocr_ms=ocr_ms,
                extract_ms=extract_ms,
                lookup_ms=lookup_ms,
            )
        self._drain_resolution_results()
        self._queue_resolution_candidate(state)
        self._emit_unknown_if_complete(state)

    def _enabled_resolution_providers(
        self, observation: PromptObservation, clue: str
    ) -> set[str]:
        providers: set[str] = set()
        placeholders = {
            "Visual / emoji clue",
            "<visual/emoji>",
        }
        semantic_clue = bool(
            observation.prompt_kind == "text"
            or (
                self._accessible_round is not None
                and self._accessible_round[0] == observation.signature
                and clue not in placeholders
            )
        )
        antigravity_ready = bool(
            self._config.antigravity.enabled
            and self._antigravity.availability.available
        )
        gemini_config = getattr(self._config, "gemini", None)
        gemini_ready = bool(
            gemini_config is not None
            and gemini_config.enabled
            and getattr(self, "_gemini", None) is not None
            and self._gemini.availability.available
            and not self._gemini.rate_limited
        )
        # Account-auth Gemini 3.7 is the measured low-latency primary for any
        # clue Discord exposes semantically, including emoji sequences. Local
        # Qwen is retained as a secondary only when this lane abstains or is
        # unavailable. The rate-limited API remains a raw-image fallback.
        if semantic_clue:
            if antigravity_ready:
                providers.add("antigravity")
            if self._config.novel.enabled and self._novel.ready_for_resolve:
                providers.add("qwen")
            if not providers and gemini_ready:
                providers.add("gemini")
        elif gemini_ready:
            providers.add("gemini")
        if self._config.vlm.allow_unverified_submission and not (
            observation.prompt_kind == "visual"
            and not self._config.vlm.allow_novel_visual_submission
        ):
            providers.add("vlm")
        return providers

    def _start_async_resolution(
        self,
        *,
        key: ResolutionKey,
        round_token: str,
        clue_fingerprint: str,
        clue: str,
        observation: PromptObservation,
        ocr_ms: float,
        extract_ms: float,
        lookup_ms: float,
    ) -> AsyncResolutionRound:
        providers = self._enabled_resolution_providers(observation, clue)
        semantic_clue = bool(
            observation.prompt_kind == "text"
            or (
                self._accessible_round is not None
                and self._accessible_round[0] == observation.signature
                and clue not in {"Visual / emoji clue", "<visual/emoji>"}
            )
        )
        request = ResolutionRequest(
            key=key,
            round_token=round_token,
            signature=observation.signature,
            clue_fingerprint=clue_fingerprint,
            clue=clue,
            observation=observation,
            started_at=time.perf_counter(),
            semantic_clue=semantic_clue,
        )
        state = AsyncResolutionRound(
            request=request,
            providers=set(providers),
            pending=set(providers),
            ocr_ms=ocr_ms,
            extract_ms=extract_ms,
            lookup_ms=lookup_ms,
            fallback_started="antigravity" in providers,
        )
        self._resolution_rounds[key] = state
        if not providers:
            if self._maybe_start_antigravity_fallback(
                state, reason="No primary resolver is currently available"
            ):
                return state
            LOGGER.warning(
                "No verified resolver is enabled for prompt %s",
                observation.question_label or "?",
            )
            return state

        provider_label = " + ".join(sorted(providers))
        self._status.emit(
            "RESOLVING",
            title=f"Researching — {observation.question_label or 'Question'}",
            detail=f"Resolving concurrently with {provider_label}",
            question=observation.question_label or "Question",
            clue=clue,
            answer="—",
            source=provider_label,
            readiness=observation.readiness,
            event_id=round_token,
        )
        for provider in sorted(providers):
            self._submit_resolution_provider(provider, request)
        return state

    def _submit_resolution_provider(
        self, provider: str, request: ResolutionRequest
    ) -> None:
        if self._stop_event.is_set() or not self._resolution_slots.acquire(
            blocking=False
        ):
            self._put_resolution_result(
                ProviderResolution(
                    key=request.key,
                    provider=provider,
                    source=f"{provider}-resolver",
                    answer=None,
                    confidence=0.0,
                    elapsed_ms=0.0,
                    error="resolver capacity unavailable",
                )
            )
            return
        try:
            future = self._resolution_executor.submit(
                self._run_resolution_provider, provider, request
            )
        except RuntimeError as exc:
            self._resolution_slots.release()
            self._put_resolution_result(
                ProviderResolution(
                    key=request.key,
                    provider=provider,
                    source=f"{provider}-resolver",
                    answer=None,
                    confidence=0.0,
                    elapsed_ms=0.0,
                    error=str(exc),
                )
            )
            return
        future.add_done_callback(
            lambda completed, name=provider, key=request.key: self._provider_done(
                name, key, completed
            )
        )

    def _run_resolution_provider(
        self, provider: str, request: ResolutionRequest
    ) -> ProviderResolution:
        started = time.perf_counter()
        answer: str | None = None
        alternatives: tuple[str, ...] = ()
        confidence = 0.0
        detail = ""
        source = f"{provider}-resolver"
        error: str | None = None
        try:
            with self._provider_locks[provider]:
                if self._stop_event.is_set():
                    raise RuntimeError("automation is stopping")
                if provider == "qwen":
                    source = "qwen38-retrieval-consensus"
                    if (
                        time.perf_counter()
                        >= request.started_at
                        + float(self._config.novel.total_timeout_seconds)
                    ):
                        raise TimeoutError("local resolver request expired in queue")
                    ranked = self._novel.resolve_ranked(
                        request.clue,
                        request.observation.expected_answer_type,
                    )
                    if ranked is not None:
                        answer = ranked.answer
                        alternatives = ranked.alternatives
                    confidence = self._novel.last_confidence
                    detail = self._novel.last_detail
                elif provider == "gemini":
                    source = "gemini-3.7-structured"
                    gemini_result = self._resolve_with_gemini(request)
                    answer = gemini_result.answer if gemini_result.accepted else None
                    alternatives = tuple(gemini_result.alternatives)
                    confidence = gemini_result.confidence
                    detail = gemini_result.detail
                elif provider == "antigravity":
                    source = "antigravity-account-3.7-low"
                    if (
                        self._active_resolution_key != request.key
                        or self._active_status_token != request.round_token
                        or self._active_status_closed
                        or not self._active_prompt.is_open(
                            request.signature, request.clue_fingerprint
                        )
                    ):
                        error = "stale"
                        detail = "Antigravity request became stale before launch"
                    else:
                        antigravity_result = asyncio.run(
                            self._antigravity.resolve(
                                AntigravityRequest(
                                    clue=request.clue,
                                    expected_answer_type=(
                                        request.observation.expected_answer_type
                                    ),
                                    prompt_kind=(
                                        "text"
                                        if request.semantic_clue
                                        else request.observation.prompt_kind
                                    ),
                                    deadline=(
                                        request.started_at
                                        + float(
                                            self._config.antigravity.total_timeout_seconds
                                        )
                                    ),
                                )
                            )
                        )
                        answer = (
                            antigravity_result.answer
                            if antigravity_result.accepted
                            else None
                        )
                        confidence = antigravity_result.confidence
                        detail = antigravity_result.detail
                elif provider == "vlm":
                    source = "local-model-consensus"
                    answer = self._vlm.resolve(request.observation)
                else:
                    raise ValueError(f"unknown resolver provider {provider!r}")
        except Exception as exc:
            # Provider exceptions may contain transport/request metadata.
            # Preserve only the exception class so credentials can never
            # escape through the app-level result queue or logs.
            error = type(exc).__name__
            LOGGER.warning(
                "%s resolver failed closed for %s",
                provider,
                request.observation.question_label or "?",
            )
        return ProviderResolution(
            key=request.key,
            provider=provider,
            source=source,
            answer=answer,
            confidence=confidence,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            detail=detail,
            error=error,
            alternatives=alternatives if answer is not None else (),
        )

    def _resolve_with_gemini(self, request: ResolutionRequest) -> GeminiResult:
        """Run Gemini on its one persistent event loop, never the OCR loop."""

        image_bytes: bytes | None = None
        image_mime_type: str | None = None
        crop = request.observation.prompt_crop
        if request.observation.prompt_kind == "visual" and crop is not None:
            try:
                import cv2

                encoded, buffer = cv2.imencode(".png", crop)
                if encoded:
                    image_bytes = bytes(buffer)
                    image_mime_type = "image/png"
            except Exception:
                LOGGER.debug("Gemini prompt-crop encoding failed", exc_info=True)
        if request.observation.prompt_kind == "visual" and image_bytes is None:
            raise ValueError("visual resolver has no encoded prompt crop")
        request_timeout = (
            float(self._config.gemini.total_timeout_seconds)
            if request.observation.prompt_kind == "visual"
            else float(self._config.gemini.text_timeout_seconds)
        )
        gemini_request = GeminiRequest(
            clue=request.clue,
            expected_answer_type=request.observation.expected_answer_type,
            image_bytes=image_bytes,
            image_mime_type=image_mime_type,
            deadline=request.started_at + request_timeout,
        )
        return self._run_gemini_coroutine(
            self._gemini.resolve(gemini_request),
            timeout=request_timeout + 1.0,
        )

    def _start_gemini_event_loop(self) -> None:
        if self._gemini_loop_thread is not None:
            return
        loop = asyncio.new_event_loop()
        self._gemini_loop = loop

        def run_loop() -> None:
            asyncio.set_event_loop(loop)
            self._gemini_loop_ready.set()
            loop.run_forever()
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            loop.close()

        self._gemini_loop_thread = threading.Thread(
            target=run_loop,
            name="gemini-event-loop",
            daemon=True,
        )
        self._gemini_loop_thread.start()
        if not self._gemini_loop_ready.wait(2.0):
            raise RuntimeError("Gemini event loop failed to start")

    def _run_gemini_coroutine(
        self,
        coroutine: Any,
        *,
        timeout: float,
        cancel_on_stop: bool = True,
    ) -> Any:
        loop = self._gemini_loop
        if loop is None or not loop.is_running():
            raise RuntimeError("Gemini event loop is not running")
        future = asyncio.run_coroutine_threadsafe(coroutine, loop)
        deadline = time.monotonic() + timeout
        while True:
            if cancel_on_stop and self._stop_event.is_set():
                future.cancel()
                raise RuntimeError("automation is stopping")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                future.cancel()
                raise TimeoutError("Gemini coroutine exceeded its outer deadline")
            try:
                return future.result(timeout=min(remaining, 0.1))
            except FutureTimeoutError:
                continue

    def _provider_done(
        self,
        provider: str,
        key: ResolutionKey,
        future: Future[ProviderResolution],
    ) -> None:
        try:
            result = future.result()
        except Exception as exc:
            result = ProviderResolution(
                key=key,
                provider=provider,
                source=f"{provider}-resolver",
                answer=None,
                confidence=0.0,
                elapsed_ms=0.0,
                error=type(exc).__name__,
            )
        finally:
            self._resolution_slots.release()
        self._put_resolution_result(result)

    def _put_resolution_result(self, result: ProviderResolution) -> None:
        try:
            self._resolution_results.put_nowait(result)
        except queue.Full:
            # Capacity is larger than the submission semaphore, so this can
            # occur only during abnormal shutdown or test injection.
            LOGGER.error("Resolution result queue overflow; dropping %s", result.provider)

    def _drain_resolution_results(self) -> None:
        while True:
            try:
                result = self._resolution_results.get_nowait()
            except queue.Empty:
                return
            self._accept_resolution_result(result)

    def _accept_resolution_result(self, result: ProviderResolution) -> None:
        state = self._resolution_rounds.get(result.key)
        if state is None or result.provider not in state.pending:
            LOGGER.info("Discarded duplicate or unknown %s resolver result", result.provider)
            return
        state.pending.remove(result.provider)
        state.results[result.provider] = result
        if result.provider == "antigravity" and result.error == "stale":
            state.results.pop("antigravity", None)
            state.providers.discard("antigravity")
            state.fallback_started = False
            self._maybe_start_antigravity_fallback(
                state, reason="The clue returned after a stale queued fallback"
            )
            return
        if (
            state.retired
            or self._quiz_ended
            or self._active_status_token != state.request.round_token
            or self._active_status_closed
        ):
            LOGGER.info(
                "Discarded late %s result for stale round %s",
                result.provider,
                state.request.round_token,
            )
            state.retired = True
            if not state.pending:
                self._resolution_rounds.pop(result.key, None)
            return

        # A wrong guess costs nothing in Anime Soul, while waiting for the
        # slowest provider costs the round. The first answer to arrive is
        # queued immediately; every later distinct answer (from another
        # provider or from a provider's own alternatives) becomes a follow-up
        # guess that the dispatcher spaces out while the card stays green.
        # Primary answers ladder immediately. A provider's own runner-up
        # alternatives are weaker than another provider's primary answer, so
        # they are held back until every provider has reported; otherwise a
        # fast local guess plus its alternatives could fill the guess cap
        # before the slower, stronger account lane delivers its answer.
        added = self._add_resolution_guesses(state, result, include_alternatives=False)
        if not state.pending:
            for reported in state.results.values():
                added.extend(
                    self._add_resolution_guesses(
                        state, reported, include_alternatives=True, primary=False
                    )
                )
        if added:
            self._announce_resolution_guesses(state, added)
            self._queue_resolution_candidate(state)
        if state.pending:
            return
        if not state.guesses:
            if (
                "antigravity" in state.providers
                and self._maybe_start_qwen_fallback(
                    state, reason="Antigravity abstained or was unavailable"
                )
            ):
                return
            if self._maybe_start_antigravity_fallback(
                state, reason="Primary resolvers abstained or were unavailable"
            ):
                return
        self._emit_unknown_if_complete(state)

    def _add_resolution_guesses(
        self,
        state: AsyncResolutionRound,
        result: ProviderResolution,
        *,
        include_alternatives: bool,
        primary: bool = True,
    ) -> list[ProviderResolution]:
        """Append the result's distinct answers to the round's guess ladder."""

        if result.answer is None:
            return []
        added: list[ProviderResolution] = []
        seen = {normalize_question(guess.answer or "") for guess in state.guesses}
        ordered: list[tuple[str, float, str]] = []
        if primary:
            ordered.append((result.answer, result.confidence, ""))
        if include_alternatives:
            for index, alternative in enumerate(result.alternatives, start=1):
                ordered.append((alternative, 0.0, f"alternative {index}"))
        for answer, confidence, note in ordered:
            cleaned = sanitize_answer(answer, self._config.typing.max_answer_characters)
            if cleaned is None:
                continue
            normalized = normalize_question(cleaned)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            guess = replace(
                result,
                answer=cleaned,
                confidence=confidence,
                alternatives=(),
                detail=f"{result.detail} ({note})" if note else result.detail,
            )
            state.guesses.append(guess)
            added.append(guess)
        if state.candidate is None and state.guesses:
            state.candidate = state.guesses[0]
        return added

    def _announce_resolution_guesses(
        self, state: AsyncResolutionRound, added: list[ProviderResolution]
    ) -> None:
        if self._active_resolution_key != state.request.key:
            return
        first = state.guesses[0]
        assert first.answer is not None
        if self._ephemeral_answer is None or self._ephemeral_answer[0:2] != (
            state.request.signature,
            state.request.clue_fingerprint,
        ):
            self._ephemeral_answer = (
                state.request.signature,
                state.request.clue_fingerprint,
                first.answer,
                first.source,
            )
        previous_resolution = self._status_resolution.get(state.request.round_token)
        self._status_resolution[state.request.round_token] = "known"
        question = state.request.observation.question_label or "Question"
        for guess in added:
            position = state.guesses.index(guess) + 1
            confidence = (
                f", {guess.confidence:.0%}" if guess.confidence > 0 else ""
            )
            if position == 1:
                self._status.emit(
                    "NOVEL",
                    title=f"New answer — {question}",
                    detail=(
                        f"{guess.provider} answered in {guess.elapsed_ms:.0f} ms"
                        f"{confidence}"
                    ),
                    question=question,
                    clue=state.request.clue,
                    answer=guess.answer,
                    source=guess.source,
                    readiness=state.request.observation.readiness,
                    event_id=state.request.round_token,
                    increment="known" if previous_resolution != "known" else None,
                    decrement=(
                        "unknown" if previous_resolution == "unknown" else None
                    ),
                )
                previous_resolution = "known"
            else:
                self._status.emit(
                    "NOVEL",
                    title=f"Follow-up guess {position} — {question}",
                    detail=(
                        f"{guess.provider} disagrees or offered an alternative"
                        f"{confidence}; queued behind the first answer"
                    ),
                    question=question,
                    clue=state.request.clue,
                    answer=guess.answer,
                    source=guess.source,
                    readiness=state.request.observation.readiness,
                    event_id=f"{state.request.round_token}:guess{position}",
                )

    def _maybe_start_antigravity_fallback(
        self, state: AsyncResolutionRound, *, reason: str
    ) -> bool:
        if (
            state.fallback_started
            or state.retired
            or self._stop_event.is_set()
            or self._quiz_ended
            or self._active_resolution_key != state.request.key
            or self._active_status_token != state.request.round_token
            or self._active_status_closed
            or not self._active_prompt.is_open(
                state.request.signature, state.request.clue_fingerprint
            )
            or not state.request.semantic_clue
            or not self._config.antigravity.enabled
            or not self._antigravity.availability.available
        ):
            return False
        state.fallback_started = True
        state.providers.add("antigravity")
        state.pending.add("antigravity")
        self._status.emit(
            "RESOLVING",
            title=(
                f"Using account fallback — "
                f"{state.request.observation.question_label or 'Question'}"
            ),
            detail=f"{reason}; asking Antigravity Gemini 3.7 Low",
            question=state.request.observation.question_label or "Question",
            clue=state.request.clue,
            answer="—",
            source="Antigravity account quota",
            readiness=state.request.observation.readiness,
            event_id=f"{state.request.round_token}:antigravity",
        )
        # Give the fallback its own absolute queue+execution budget. This
        # prevents an old queued request from starting a fresh six-second call
        # after the round has already moved on.
        fallback_request = replace(state.request, started_at=time.perf_counter())
        self._submit_resolution_provider("antigravity", fallback_request)
        return True

    def _maybe_start_qwen_fallback(
        self, state: AsyncResolutionRound, *, reason: str
    ) -> bool:
        if (
            "qwen" in state.providers
            or state.retired
            or self._stop_event.is_set()
            or self._quiz_ended
            or self._active_resolution_key != state.request.key
            or self._active_status_token != state.request.round_token
            or self._active_status_closed
            or not self._active_prompt.is_open(
                state.request.signature, state.request.clue_fingerprint
            )
            or not state.request.semantic_clue
            or not self._config.novel.enabled
        ):
            return False
        state.providers.add("qwen")
        state.pending.add("qwen")
        self._status.emit(
            "RESOLVING",
            title=(
                f"Using local fallback — "
                f"{state.request.observation.question_label or 'Question'}"
            ),
            detail=f"{reason}; asking local Qwen3.8",
            question=state.request.observation.question_label or "Question",
            clue=state.request.clue,
            answer="—",
            source="local Qwen fallback",
            readiness=state.request.observation.readiness,
            event_id=f"{state.request.round_token}:qwen-fallback",
        )
        fallback_request = replace(state.request, started_at=time.perf_counter())
        self._submit_resolution_provider("qwen", fallback_request)
        return True

    def _queue_resolution_candidate(self, state: AsyncResolutionRound) -> bool:
        """Queue every not-yet-queued guess for the still-live round."""

        if state.retired or not state.guesses:
            return False
        if (
            self._active_resolution_key != state.request.key
            or self._active_status_token != state.request.round_token
            or self._active_status_closed
            or not self._active_prompt.is_open(
                state.request.signature,
                state.request.clue_fingerprint,
            )
        ):
            return False
        queued_any = False
        for position, guess in enumerate(state.guesses, start=1):
            assert guess.answer is not None
            normalized = normalize_question(guess.answer)
            if normalized in state.queued_answers:
                continue
            if position > self._config.typing.max_guesses_per_round:
                break
            queued = self._queue_answer_if_current(
                answer=guess.answer,
                source=guess.source,
                observation=state.request.observation,
                clue_fingerprint=state.request.clue_fingerprint,
                round_token=state.request.round_token,
                detected_at=state.request.observation.scene.detected_at,
                ocr_ms=state.ocr_ms,
                extract_ms=state.extract_ms,
                lookup_ms=state.lookup_ms,
                provider_ms=guess.elapsed_ms,
                guess_index=position,
            )
            if not queued:
                break
            state.queued_answers.add(normalized)
            queued_any = True
        state.queued = bool(state.queued_answers)
        return queued_any

    def _queue_answer_if_current(
        self,
        *,
        answer: str,
        source: str,
        observation: PromptObservation,
        clue_fingerprint: str,
        round_token: str,
        detected_at: float,
        ocr_ms: float,
        extract_ms: float,
        lookup_ms: float,
        provider_ms: float,
        guess_index: int = 1,
    ) -> bool:
        if (
            self._active_status_token != round_token
            or self._active_status_closed
            or not self._active_prompt.is_open(
                observation.signature, clue_fingerprint
            )
        ):
            return False
        timings = {
            "ocr": ocr_ms,
            "extract_hash": extract_ms,
            "lookup": lookup_ms,
            "vlm": provider_ms,
            "novel": provider_ms if source == "qwen38-retrieval-consensus" else 0.0,
        }
        task = AnswerTask(
            answer=answer,
            prompt_signature=observation.signature,
            expected_answer_type=observation.expected_answer_type,
            question_label=observation.question_label,
            detected_at=detected_at,
            countdown_seconds=observation.countdown_seconds,
            source=source,
            stage_timings_ms=timings,
            round_token=round_token,
            clue_fingerprint=clue_fingerprint,
            guess_index=guess_index,
        )
        queued = self._dispatcher.submit(task)
        if queued:
            LOGGER.info(
                "Answer queued (guess %d): %s (processing %.1fms, provider %.1fms)",
                guess_index,
                answer,
                ocr_ms + extract_ms + lookup_ms,
                provider_ms,
            )
        return queued

    def _emit_unknown_if_complete(self, state: AsyncResolutionRound) -> None:
        if state.retired or state.pending or state.guesses or state.unknown_emitted:
            return
        if (
            self._active_resolution_key != state.request.key
            or self._active_status_token != state.request.round_token
            or self._active_status_closed
            or not self._active_prompt.is_open(
                state.request.signature,
                state.request.clue_fingerprint,
            )
        ):
            return
        if self._maybe_start_antigravity_fallback(
            state, reason="Primary resolvers completed while this clue was inactive"
        ):
            return
        state.unknown_emitted = True
        observation = state.request.observation
        LOGGER.warning(
            "No confident answer for prompt %s after all providers completed",
            observation.question_label or "?",
        )
        previous_resolution = self._status_resolution.get(state.request.round_token)
        if previous_resolution == "known":
            return
        self._status_resolution[state.request.round_token] = "unknown"
        self._status.emit(
            "UNKNOWN",
            title=f"Unknown — {observation.question_label or 'Question'}",
            detail="All enabled resolvers abstained; waiting to learn the reveal",
            question=observation.question_label or "Question",
            clue=state.request.clue,
            answer="SKIP",
            source="no verified match",
            readiness=observation.readiness,
            event_id=state.request.round_token,
            increment="unknown" if previous_resolution is None else None,
        )

    def _retire_other_resolutions(self, round_token: str) -> None:
        for key, state in list(self._resolution_rounds.items()):
            if state.request.round_token != round_token:
                if state.pending:
                    state.retired = True
                else:
                    self._resolution_rounds.pop(key, None)

    def _finish_quiz(self) -> None:
        if self._quiz_ended:
            return
        self._quiz_ended = True
        self._pending_round = None
        self._ephemeral_answer = None
        self._accessible_round = None
        self._active_resolution_key = None
        self._resolution_rounds.clear()
        self._dispatcher.observe_prompt(
            None,
            "closed",
            self._change_gate.generation,
        )
        self._status.emit(
            "QUIZ_COMPLETE",
            title="Quiz complete",
            detail="Historical cards are inert; waiting for the next live quiz",
            readiness="closed",
            event_id=f"quiz-complete:{self._status_session_id}",
        )
        self._status_session_id += 1
        self._active_status_signature = None
        self._active_status_question_label = None
        self._active_status_clue_key = None
        self._active_status_token = None
        self._active_status_closed = False
        LOGGER.info("Quiz-complete latch armed; historical cards are inert")

    @staticmethod
    def _is_final_question(question_label: str | None) -> bool:
        """Only let a quiz-complete marker close the card it belongs to."""

        if not question_label:
            return False
        match = re.search(r"\b(\d+)\s*/\s*(\d+)\b", question_label)
        return bool(match and int(match.group(1)) == int(match.group(2)))

    @staticmethod
    def _effective_prompt_kind(prompt_kind: str, clue: str) -> str:
        if clue and sum(character.isalpha() for character in clue) >= 3:
            return "text"
        return prompt_kind

    def _should_finish_without_card(self, quiz_complete: bool) -> bool:
        """Accept a viewport marker only after our tracked final round closed."""

        return (
            quiz_complete
            and self._active_status_closed
            and self._is_final_question(self._active_status_question_label)
        )

    def _match_authoritative_history(
        self, observation: PromptObservation
    ) -> CacheHit | None:
        window = self._foreground_guard.expected_window()
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

    def _semantic_reveal_records(self) -> list[tuple[str, str, str, int]]:
        """Read official loaded bot answers directly from Discord accessibility."""

        foreground_guard = getattr(self, "_foreground_guard", None)
        question_locator = getattr(self, "_question_locator", None)
        if foreground_guard is None or question_locator is None:
            return []
        window = foreground_guard.expected_window()
        if window is None:
            return []
        try:
            raw_records = question_locator.find_reveal_records(
                window.hwnd, window.process_id
            )
        except Exception:
            LOGGER.debug("Discord semantic reveal read failed", exc_info=True)
            return []
        records: list[tuple[str, str, str, int]] = []
        for raw_record in raw_records:
            answer = sanitize_answer(
                raw_record.answer,
                self._config.typing.max_answer_characters,
            )
            normalized = normalize_question(answer or "")
            if answer is not None and normalized:
                records.append(
                    (
                        normalized,
                        answer,
                        raw_record.identity,
                        raw_record.screen_top,
                    )
                )
        return records

    @staticmethod
    def _select_new_semantic_answers(
        pending: PendingRound,
        records: list[tuple[str, str, str, int]],
        card_screen_bottom: int,
    ) -> dict[str, str]:
        """Select only newly appended official results below the tracked card."""

        return {
            normalized: answer
            for normalized, answer, identity, screen_top in records
            if identity not in pending.baseline_reveal_ids
            and card_screen_bottom < screen_top <= card_screen_bottom + 900
        }

    def _arm_pending_round(
        self, observation: PromptObservation, full_scene: Scene
    ) -> None:
        if not observation.question_label:
            return
        clue = observation.hint_text
        semantic_clue = False
        if (
            self._accessible_round is not None
            and self._accessible_round[0] == observation.signature
        ):
            clue = self._accessible_round[1]
            semantic_clue = True
        clue_fingerprint = self._clue_fingerprint(observation)
        effective_prompt_kind = (
            self._effective_prompt_kind(observation.prompt_kind, clue)
            if semantic_clue
            else observation.prompt_kind
        )
        if effective_prompt_kind == "text":
            clue = self._canonical_text_clue(clue)
        if (
            self._pending_round is None
            or self._pending_round.signature != observation.signature
            or self._pending_round.clue_fingerprint != clue_fingerprint
        ):
            full_spans = self._ocr.recognize(full_scene.frame)
            baseline = Counter(
                normalized
                for normalized, _answer, _top, _left in self._reveal_records(full_spans)
            )
            semantic_baseline = self._semantic_reveal_records()
            baseline.update(normalized for normalized, *_rest in semantic_baseline)
            self._pending_round = PendingRound(
                signature=observation.signature,
                question_label=observation.question_label,
                expected_answer_type=observation.expected_answer_type,
                prompt_kind=effective_prompt_kind,
                clue=clue,
                clue_fingerprint=clue_fingerprint,
                hashes=(
                    {observation.perceptual_hash}
                    if observation.perceptual_hash
                    else set()
                ),
                baseline_reveals=baseline,
                baseline_reveal_ids={
                    identity for _normalized, _answer, identity, _top in semantic_baseline
                },
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

        current_accessible = None
        window = self._foreground_guard.expected_window()
        if window is not None:
            try:
                current_accessible = self._question_locator.find_question(
                    window.hwnd, window.process_id
                )
            except Exception:
                LOGGER.debug("Closed-card semantic continuity failed", exc_info=True)

        semantic_records = self._semantic_reveal_records()
        semantic_new_answers: dict[str, str] = {}
        if (
            current_accessible is not None
            and current_accessible.question_label == pending.question_label
            and current_accessible.expected_answer_type
            == pending.expected_answer_type
        ):
            semantic_new_answers = self._select_new_semantic_answers(
                pending,
                semantic_records,
                current_accessible.screen_bottom,
            )
        if len(semantic_new_answers) > 1:
            LOGGER.debug(
                "Semantic reveal transaction is ambiguous: %d new answers",
                len(semantic_new_answers),
            )
            return

        if len(semantic_new_answers) == 1:
            # The bot's semantic result is stronger continuity evidence than a
            # closed-card pHash. This is the primary visual-learning path.
            answer = next(iter(semantic_new_answers.values()))
        else:
            semantic_continuity = bool(
                current_accessible is not None
                and current_accessible.question_label == pending.question_label
                and current_accessible.expected_answer_type
                == pending.expected_answer_type
                and normalize_accessible_clue(current_accessible.clue)
                == normalize_accessible_clue(pending.clue)
            )

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
            for normalized, candidate_answer, top, left in records:
                if not card_bottom < top <= card_bottom + 900:
                    continue
                if not card_left - 160 <= left <= card_right + 160:
                    continue
                if current_counts[normalized] <= pending.baseline_reveals[normalized]:
                    continue
                new_answers[normalized] = candidate_answer
            if len(new_answers) != 1:
                LOGGER.debug(
                    "OCR reveal transaction still waiting: %d eligible answer(s)",
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
        learned_token = self._active_status_token or (
            f"session-{self._status_session_id}:{pending.signature}"
        )
        self._status.emit(
            "LEARNED",
            title=f"Learned — {pending.question_label}",
            detail="The bot reveal is now a verified local fast-path answer",
            question=pending.question_label,
            clue=pending.clue or "Visual / emoji clue",
            answer=answer,
            source="authoritative bot reveal",
            readiness="closed",
            event_id=learned_token,
            increment="learned",
        )
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
            "Starting Anime Trivia Automation%s with Gemini 3.7 API, local Qwen, and the account fallback. Keep Anime Soul visible; manual research remains available. Press %s to stop.",
            " in DRY RUN mode" if not self._config.typing.enabled else "",
            self._config.typing.stop_key.upper(),
        )
        if self._config.vlm.enabled and self._config.vlm.ready_before_capture:
            LOGGER.info(
                "Loading the explicitly enabled experimental VLM before capture"
            )
            self._vlm.ensure_loaded()
        if self._config.novel.enabled:
            self._status.emit(
                "LOADING",
                title="Loading Qwen3.8 anime solver",
                detail="Starting the retrieval-grounded local model before capture",
                readiness="unknown",
            )
            if not self._novel.ensure_ready():
                LOGGER.warning(
                    "Qwen3.8 fallback unavailable: %s", self._novel.last_detail
                )
                self._status.emit(
                    "ATTENTION",
                    title="Local Qwen unavailable — account solver remains active",
                    detail=self._novel.last_detail,
                    readiness="unknown",
                    event_id="qwen-unavailable",
                )
        if self._config.gemini.enabled:
            self._start_gemini_event_loop()
            self._status.emit(
                "LOADING",
                title="Checking Gemini API solver",
                detail="Verifying credentials and Gemini 3.7 Flash availability",
                readiness="unknown",
            )
            availability = self._run_gemini_coroutine(
                self._gemini.preflight(),
                timeout=float(self._config.gemini.preflight_timeout_seconds) + 1.0,
            )
            if not availability.available:
                LOGGER.warning("Gemini resolver unavailable: %s", availability.detail)
                self._status.emit(
                    "ATTENTION",
                    title="Gemini unavailable — local solver remains active",
                    detail=availability.detail,
                    readiness="unknown",
                    event_id="gemini-unavailable",
                )
        if self._config.antigravity.enabled:
            self._status.emit(
                "LOADING",
                title="Checking Antigravity account fallback",
                detail="Verifying cached account auth and Gemini 3.7 Low access",
                readiness="unknown",
            )
            availability = asyncio.run(self._antigravity.preflight())
            if not availability.available:
                LOGGER.warning(
                    "Antigravity fallback unavailable: %s", availability.detail
                )
                self._status.emit(
                    "ATTENTION",
                    title="Antigravity unavailable — primary solvers remain active",
                    detail=availability.detail,
                    readiness="unknown",
                    event_id="antigravity-unavailable",
                )

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
                self._status.heartbeat()
        except KeyboardInterrupt:
            LOGGER.warning("Ctrl+C received")
            self.stop()
        finally:
            self.stop()
            self._capture.join()
            self._processor_thread.join(timeout=5.0)
            self._shutdown_resolution_workers()
            self._dispatcher.join(timeout=5.0)
            LOGGER.info("Anime Trivia Automation stopped")

    def stop(self) -> None:
        with self._stop_lock:
            if self._stopped:
                return
            self._stopped = True
            self._stop_event.set()
        self._status.emit(
            "STOPPING",
            title="Stopping safely",
            detail="Closing capture and preserving any owned draft safely",
            readiness="closed",
            event_id="stopping",
        )
        self._capture.request_stop()
        try:
            self._emergency_stop.stop()
        except RuntimeError:
            LOGGER.debug("Emergency-stop listener was already stopped", exc_info=True)

    def _shutdown_resolution_workers(self) -> None:
        with self._resolution_shutdown_lock:
            if self._resolution_shutdown:
                return
            self._resolution_shutdown = True
        # Kill an exact in-flight account-auth CLI process before waiting on
        # resolver threads, so F12 cannot inherit the provider's full timeout.
        try:
            asyncio.run(self._antigravity.close())
        except Exception:
            LOGGER.debug("Antigravity provider close failed", exc_info=True)
        self._resolution_executor.shutdown(wait=True, cancel_futures=True)
        self._novel.close()
        loop = self._gemini_loop
        thread = self._gemini_loop_thread
        if loop is not None and loop.is_running():
            try:
                self._run_gemini_coroutine(
                    self._gemini.close(),
                    timeout=3.0,
                    cancel_on_stop=False,
                )
            except Exception:
                LOGGER.debug("Gemini resolver close failed", exc_info=True)
            loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join(timeout=5.0)
        self._gemini_loop = None
        self._gemini_loop_thread = None


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
