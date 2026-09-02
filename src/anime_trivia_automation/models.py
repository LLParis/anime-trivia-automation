from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ReadinessState = Literal["locked", "ready", "closed", "unknown"]


@dataclass(frozen=True)
class Scene:
    generation: int
    frame: Any
    captured_at: float
    detected_at: float
    mean_delta: float
    changed_ratio: float


@dataclass(frozen=True)
class OcrSpan:
    text: str
    score: float
    box: tuple[int, int, int, int]

    @property
    def left(self) -> int:
        return self.box[0]

    @property
    def top(self) -> int:
        return self.box[1]

    @property
    def right(self) -> int:
        return self.box[2]

    @property
    def bottom(self) -> int:
        return self.box[3]

    @property
    def center_y(self) -> float:
        return (self.top + self.bottom) / 2.0


@dataclass(frozen=True)
class PromptObservation:
    scene: Scene
    spans: tuple[OcrSpan, ...]
    full_text: str
    hint_text: str
    expected_answer_type: Literal["character", "anime_title", "unknown"]
    prompt_kind: Literal["text", "visual"]
    prompt_crop: Any
    countdown_seconds: float | None
    question_label: str | None
    signature: str
    readiness: ReadinessState
    red_outline_pixels: int = 0
    green_outline_pixels: int = 0
    perceptual_hash: str | None = None
    card_box: tuple[int, int, int, int] | None = None


@dataclass(frozen=True)
class CacheHit:
    kind: Literal["text", "image", "history"]
    key: str
    answer: str
    score: float
    runner_up_score: float | None = None


@dataclass(frozen=True)
class AnswerTask:
    answer: str
    prompt_signature: str
    expected_answer_type: str
    question_label: str | None
    detected_at: float
    countdown_seconds: float | None
    source: str
    stage_timings_ms: dict[str, float] = field(default_factory=dict)
    round_token: str | None = None
    clue_fingerprint: str = ""
    # 1 = first answer for this round; higher values are follow-up guesses that
    # the dispatcher spaces out while the same card stays green.
    guess_index: int = 1
