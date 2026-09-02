from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import os
import re
import time
import unicodedata
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr, ValidationError
from pydantic.functional_validators import model_validator

from .config import GeminiConfig

LOGGER = logging.getLogger(__name__)

AnswerType = Literal["character", "anime_title"]
GeminiLane = Literal["primary", "scout"]
GeminiConfidence = Literal["high", "medium", "low"]
GeminiResultStatus = Literal[
    "answered",
    "abstained",
    "timeout",
    "unavailable",
    "invalid_request",
    "invalid_response",
    "error",
]
GeminiAvailabilityPhase = Literal[
    "not_checked",
    "disabled",
    "missing_key",
    "checking",
    "ready",
    "unavailable",
    "closed",
]

_ALLOWED_IMAGE_MIME_TYPES = frozenset(
    {"image/png", "image/jpeg", "image/webp", "image/heic", "image/heif"}
)
_UNKNOWN_ANSWERS = frozenset(
    {
        "unknown",
        "unsure",
        "abstain",
        "none",
        "n/a",
        "i don't know",
        "i do not know",
    }
)
_BLOCKED_ANSWER_FRAGMENTS = (
    "http://",
    "https://",
    "discord.gg/",
    "@everyone",
    "@here",
    "```",
)


@dataclass(frozen=True)
class GeminiRequest:
    """One bounded cloud-resolution request.

    ``deadline`` is an absolute ``time.perf_counter()`` value. The provider
    always applies the earlier of this deadline and its configured timeout.
    Image bytes must contain only the already-cropped trivia card or clue.
    """

    clue: str
    expected_answer_type: AnswerType
    image_bytes: bytes | None = None
    image_mime_type: str | None = None
    lane: GeminiLane = "primary"
    deadline: float | None = None


@dataclass(frozen=True)
class GeminiResult:
    status: GeminiResultStatus
    answer: str | None
    answer_type: AnswerType
    confidence: float
    confidence_label: GeminiConfidence | None
    model: str
    lane: GeminiLane
    latency_ms: float
    detail: str

    @property
    def accepted(self) -> bool:
        return self.status == "answered" and self.answer is not None


@dataclass(frozen=True)
class GeminiAvailability:
    phase: GeminiAvailabilityPhase
    available: bool
    model: str
    detail: str
    checked_at: float | None = None
    latency_ms: float = 0.0


class _TriviaPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: StrictStr | None
    answer_type: Literal["character", "anime_title"]
    confidence: float = Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False)
    confidence_label: Literal["high", "medium", "low"]
    abstain: StrictBool
    evidence: StrictStr = Field(min_length=1, max_length=320)

    @model_validator(mode="after")
    def validate_semantics(self) -> _TriviaPayload:
        if self.abstain and self.answer is not None:
            raise ValueError("an abstention cannot contain an answer")
        if not self.abstain and self.answer is None:
            raise ValueError("a non-abstention must contain an answer")
        if self.confidence_label == "high" and self.confidence < 0.75:
            raise ValueError("high confidence is inconsistent with its score")
        if self.confidence_label == "medium" and not 0.35 <= self.confidence < 0.90:
            raise ValueError("medium confidence is inconsistent with its score")
        if self.confidence_label == "low" and self.confidence >= 0.75:
            raise ValueError("low confidence is inconsistent with its score")
        return self


ClientFactory = Callable[[str, int], Any]


class GeminiProvider:
    """Production Gemini resolver with a startup circuit and hard deadlines.

    The provider never enables Google Search or any other tool. It performs no
    live retry, never reads an API key from configuration, and never logs an
    exception message that could contain request metadata. Call ``preflight``
    during application startup; ``resolve`` fails closed until it succeeds.
    """

    def __init__(
        self,
        config: GeminiConfig,
        *,
        environ: Mapping[str, str] | None = None,
        client_factory: ClientFactory | None = None,
    ) -> None:
        self._config = config
        self._environ = os.environ if environ is None else environ
        self._client_factory = client_factory or self._default_client_factory
        self._client: Any | None = None
        self._preflight_attempted = False
        self._preflight_lock = asyncio.Lock()
        if not config.enabled:
            self._availability = GeminiAvailability(
                phase="disabled",
                available=False,
                model=config.primary_model,
                detail="Gemini provider is disabled",
            )
        else:
            self._availability = GeminiAvailability(
                phase="not_checked",
                available=False,
                model=config.primary_model,
                detail="Startup preflight has not run",
            )

    @property
    def availability(self) -> GeminiAvailability:
        return self._availability

    async def preflight(
        self, *, force: bool = False, deadline: float | None = None
    ) -> GeminiAvailability:
        """Verify authentication and primary-model availability once."""

        if not self._config.enabled:
            return self._availability
        async with self._preflight_lock:
            if self._preflight_attempted and not force:
                return self._availability
            if force:
                await self._close_client()

            self._preflight_attempted = True
            key = self._read_api_key()
            if key is None:
                self._availability = GeminiAvailability(
                    phase="missing_key",
                    available=False,
                    model=self._config.primary_model,
                    detail=(
                        f"Environment variable {self._config.api_key_env} is not set"
                    ),
                    checked_at=time.time(),
                )
                return self._availability

            started = time.perf_counter()
            effective_deadline = self._effective_deadline(
                float(self._config.preflight_timeout_seconds), deadline
            )
            self._availability = GeminiAvailability(
                phase="checking",
                available=False,
                model=self._config.primary_model,
                detail="Checking Gemini authentication and model availability",
                checked_at=time.time(),
            )
            try:
                self._client = self._client_factory(
                    key,
                    max(
                        1,
                        math.ceil(
                            max(
                                self._config.preflight_timeout_seconds,
                                self._config.total_timeout_seconds,
                                10.0,
                            )
                            * 1000.0
                        ),
                    ),
                )
                model_record = await self._get_model(
                    self._config.primary_model, effective_deadline
                )
                model_name = str(getattr(model_record, "name", ""))
                if not model_name.endswith(self._config.primary_model):
                    raise ValueError("Gemini preflight returned the wrong model")
            except (TimeoutError, asyncio.TimeoutError):
                await self._close_client()
                self._availability = GeminiAvailability(
                    phase="unavailable",
                    available=False,
                    model=self._config.primary_model,
                    detail="Gemini startup preflight timed out",
                    checked_at=time.time(),
                    latency_ms=self._elapsed_ms(started),
                )
            except Exception as exc:
                await self._close_client()
                LOGGER.warning(
                    "Gemini startup preflight failed (%s)", type(exc).__name__
                )
                self._availability = GeminiAvailability(
                    phase="unavailable",
                    available=False,
                    model=self._config.primary_model,
                    detail=f"Gemini startup preflight failed ({type(exc).__name__})",
                    checked_at=time.time(),
                    latency_ms=self._elapsed_ms(started),
                )
            else:
                self._availability = GeminiAvailability(
                    phase="ready",
                    available=True,
                    model=self._config.primary_model,
                    detail="Gemini authentication and 3.7 Flash access are ready",
                    checked_at=time.time(),
                    latency_ms=self._elapsed_ms(started),
                )
                LOGGER.info(
                    "Gemini provider ready: model=%s latency_ms=%.0f",
                    self._config.primary_model,
                    self._availability.latency_ms,
                )
            return self._availability

    async def resolve(self, request: GeminiRequest) -> GeminiResult:
        """Resolve one clue without retries, search, or hidden model fallback."""

        started = time.perf_counter()
        model, thinking_level, lane_error = self._select_lane(request.lane)
        if lane_error is not None:
            return self._result(
                request,
                model=model,
                status="unavailable",
                started=started,
                detail=lane_error,
            )
        if not self._availability.available or self._client is None:
            return self._result(
                request,
                model=model,
                status="unavailable",
                started=started,
                detail=self._availability.detail,
            )

        try:
            clue = self._sanitize_clue(request.clue)
            input_value = self._build_input(request, clue)
            deadline = self._effective_deadline(
                float(self._config.total_timeout_seconds), request.deadline
            )
        except (TypeError, ValueError) as exc:
            return self._result(
                request,
                model=model,
                status="invalid_request",
                started=started,
                detail=str(exc),
            )

        try:
            interaction = await self._create_interaction(
                model=model,
                thinking_level=thinking_level,
                input_value=input_value,
                response_schema=_TriviaPayload.model_json_schema(),
                deadline=deadline,
            )
            payload = _TriviaPayload.model_validate(self._decode_output(interaction))
        except (TimeoutError, asyncio.TimeoutError):
            return self._result(
                request,
                model=model,
                status="timeout",
                started=started,
                detail="Gemini resolution exceeded its absolute deadline",
            )
        except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
            LOGGER.warning("Gemini returned an invalid response (%s)", type(exc).__name__)
            return self._result(
                request,
                model=model,
                status="invalid_response",
                started=started,
                detail=f"Gemini response failed validation ({type(exc).__name__})",
            )
        except Exception as exc:
            LOGGER.warning("Gemini request failed (%s)", type(exc).__name__)
            return self._result(
                request,
                model=model,
                status="error",
                started=started,
                detail=f"Gemini request failed ({type(exc).__name__})",
            )

        if payload.answer_type != request.expected_answer_type:
            return self._result(
                request,
                model=model,
                status="invalid_response",
                started=started,
                confidence=payload.confidence,
                confidence_label=payload.confidence_label,
                detail="Gemini returned the wrong answer type",
            )
        if payload.abstain:
            return self._result(
                request,
                model=model,
                status="abstained",
                started=started,
                confidence=payload.confidence,
                confidence_label=payload.confidence_label,
                detail="Gemini abstained because the clue was not decisive",
            )

        answer = self._sanitize_answer(payload.answer)
        if answer is None:
            return self._result(
                request,
                model=model,
                status="invalid_response",
                started=started,
                confidence=payload.confidence,
                confidence_label=payload.confidence_label,
                detail="Gemini answer failed the safety sanitizer",
            )
        if (
            payload.confidence_label != "high"
            or payload.confidence < float(self._config.min_confidence)
        ):
            return self._result(
                request,
                model=model,
                status="abstained",
                started=started,
                confidence=payload.confidence,
                confidence_label=payload.confidence_label,
                detail="Gemini answer was below the live confidence threshold",
            )
        return self._result(
            request,
            model=model,
            status="answered",
            answer=answer,
            started=started,
            confidence=payload.confidence,
            confidence_label=payload.confidence_label,
            detail="Gemini returned one high-confidence canonical answer",
        )

    async def close(self) -> None:
        await self._close_client()
        self._preflight_attempted = False
        self._availability = GeminiAvailability(
            phase="closed",
            available=False,
            model=self._config.primary_model,
            detail="Gemini provider is closed",
            checked_at=time.time(),
        )

    def _read_api_key(self) -> str | None:
        value = self._environ.get(self._config.api_key_env)
        if value is None and os.name == "nt":
            # A process cannot see User-scope environment changes made after it
            # started. Read the same value from HKCU without exposing it, so an
            # already-running Codex/Explorer parent can still launch the worker.
            try:
                import winreg

                with winreg.OpenKey(
                    winreg.HKEY_CURRENT_USER,
                    r"Environment",
                    access=winreg.KEY_READ,
                ) as environment_key:
                    registry_value, registry_type = winreg.QueryValueEx(
                        environment_key, self._config.api_key_env
                    )
                if registry_type in {winreg.REG_SZ, winreg.REG_EXPAND_SZ} and isinstance(
                    registry_value, str
                ):
                    value = registry_value
            except (FileNotFoundError, OSError, TypeError):
                value = None
        if value is None:
            return None
        value = value.strip()
        return value or None

    def _select_lane(self, lane: GeminiLane) -> tuple[str, str, str | None]:
        if lane == "primary":
            return (
                self._config.primary_model,
                self._config.primary_thinking_level,
                None,
            )
        if lane == "scout" and not self._config.scout_enabled:
            return (
                self._config.scout_model,
                self._config.scout_thinking_level,
                "Gemini Flash-Lite scout is disabled",
            )
        if lane == "scout":
            return (
                self._config.scout_model,
                self._config.scout_thinking_level,
                None,
            )
        return self._config.primary_model, self._config.primary_thinking_level, (
            "Unknown Gemini model lane"
        )

    def _build_input(self, request: GeminiRequest, clue: str) -> str | list[dict[str, str]]:
        answer_instruction = (
            "the canonical full character name"
            if request.expected_answer_type == "character"
            else "the canonical English anime title"
        )
        prompt = (
            "You are a precision anime and manga trivia solver. This request is "
            "ungrounded: do not use tools, web search, or external actions. Identify exactly "
            f"{answer_instruction}. Never combine alternatives or return commentary in the "
            "answer field. Inspect the attached cropped quiz clue when present. If the clue "
            "is ambiguous, the type is uncertain, or you cannot identify one answer, set "
            "abstain=true and answer=null. Confidence must reflect the evidence, not "
            "popularity.\n\n"
            f"Required answer_type: {request.expected_answer_type}\n"
            f"Quiz clue: {clue}"
        )
        if request.image_bytes is None:
            if request.image_mime_type is not None:
                raise ValueError("image_mime_type requires image_bytes")
            return prompt
        if not isinstance(request.image_bytes, bytes):
            raise TypeError("image_bytes must be bytes")
        if not request.image_bytes:
            raise ValueError("image_bytes cannot be empty")
        if len(request.image_bytes) > int(self._config.max_image_bytes):
            raise ValueError("cropped image exceeds gemini.max_image_bytes")
        mime_type = (request.image_mime_type or "").casefold()
        if mime_type not in _ALLOWED_IMAGE_MIME_TYPES:
            raise ValueError("image_mime_type is not supported by Gemini")
        return [
            {"type": "text", "text": prompt},
            {
                "type": "image",
                "data": base64.b64encode(request.image_bytes).decode("ascii"),
                "mime_type": mime_type,
            },
        ]

    def _sanitize_clue(self, clue: str) -> str:
        if not isinstance(clue, str):
            raise TypeError("clue must be text")
        if any(unicodedata.category(character) == "Cc" and not character.isspace() for character in clue):
            raise ValueError("clue contains control characters")
        value = " ".join(unicodedata.normalize("NFKC", clue).split())
        if not value:
            raise ValueError("clue cannot be empty")
        if len(value) > int(self._config.max_clue_characters):
            raise ValueError("clue exceeds gemini.max_clue_characters")
        return value

    def _sanitize_answer(self, answer: str | None) -> str | None:
        if answer is None or not isinstance(answer, str):
            return None
        if "\n" in answer or "\r" in answer:
            return None
        value = unicodedata.normalize("NFKC", answer).strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1].strip()
        value = re.sub(r"\s+", " ", value)
        folded = value.casefold()
        if (
            not value
            or folded in _UNKNOWN_ANSWERS
            or len(value) > int(self._config.max_answer_characters)
            or any(fragment in folded for fragment in _BLOCKED_ANSWER_FRAGMENTS)
            or any(character in value for character in ("@", "<", ">", "`", "\\"))
            or re.match(r"^(?:final\s+)?answer\s*:", value, flags=re.IGNORECASE)
        ):
            return None
        if not any(character.isalnum() for character in value):
            return None
        for character in value:
            category = unicodedata.category(character)
            if category[0] in {"L", "N", "P"} or category == "Zs":
                continue
            if character in {"&", "+", "×"}:
                continue
            return None
        return value

    async def _create_interaction(
        self,
        *,
        model: str,
        thinking_level: str,
        input_value: str | list[dict[str, str]],
        response_schema: dict[str, Any],
        deadline: float,
    ) -> Any:
        if self._client is None:
            raise RuntimeError("Gemini client is not initialized")
        remaining = deadline - time.perf_counter()
        if remaining <= 0.01:
            raise TimeoutError("Gemini request deadline already expired")
        call = self._client.aio.interactions.create(
            model=model,
            input=input_value,
            generation_config={"thinking_level": thinking_level},
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": response_schema,
            },
            # The Interactions API rejects manually supplied deadlines below
            # 10 seconds. asyncio.wait_for remains the authoritative, shorter
            # live deadline and cancels/discards a late response.
            timeout=max(10.0, remaining),
        )
        try:
            return await asyncio.wait_for(call, timeout=remaining)
        except asyncio.TimeoutError as exc:
            raise TimeoutError("Gemini request exceeded its absolute deadline") from exc

    async def _get_model(self, model: str, deadline: float) -> Any:
        if self._client is None:
            raise RuntimeError("Gemini client is not initialized")
        remaining = deadline - time.perf_counter()
        if remaining <= 0.01:
            raise TimeoutError("Gemini preflight deadline already expired")
        call = self._client.aio.models.get(model=model)
        try:
            return await asyncio.wait_for(call, timeout=remaining)
        except asyncio.TimeoutError as exc:
            raise TimeoutError("Gemini preflight exceeded its deadline") from exc

    @staticmethod
    def _decode_output(interaction: Any) -> dict[str, Any]:
        output_text = getattr(interaction, "output_text", None)
        if not isinstance(output_text, str) or not output_text.strip():
            raise TypeError("Gemini interaction contains no output text")
        parsed = json.loads(output_text)
        if not isinstance(parsed, dict):
            raise TypeError("Gemini structured output is not a JSON object")
        return parsed

    async def _close_client(self) -> None:
        client, self._client = self._client, None
        if client is None:
            return
        try:
            await client.aio.aclose()
        except Exception as exc:
            LOGGER.debug("Gemini client close failed (%s)", type(exc).__name__)

    @staticmethod
    def _default_client_factory(api_key: str, timeout_ms: int) -> Any:
        from google import genai
        from google.genai import types

        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=timeout_ms,
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )
        # google-genai 2.21.0 coerces public attempts=0 to one retry. The
        # Interactions adapter exposes its effective pinned retry policy; set
        # and verify it explicitly so a timed trivia request is sent once.
        retry_config = client.aio.interactions.sdk_configuration.retry_config
        retry_config.max_retries = 0
        if retry_config.max_retries != 0:
            raise RuntimeError("Gemini SDK could not disable automatic retries")
        return client

    @staticmethod
    def _effective_deadline(configured_seconds: float, supplied: float | None) -> float:
        now = time.perf_counter()
        configured = now + max(0.01, configured_seconds)
        if supplied is None:
            return configured
        if not math.isfinite(supplied):
            raise ValueError("deadline must be finite")
        if supplied <= now:
            raise TimeoutError("deadline has already expired")
        return min(configured, supplied)

    @staticmethod
    def _elapsed_ms(started: float) -> float:
        return max(0.0, (time.perf_counter() - started) * 1000.0)

    def _result(
        self,
        request: GeminiRequest,
        *,
        model: str,
        status: GeminiResultStatus,
        started: float,
        detail: str,
        answer: str | None = None,
        confidence: float = 0.0,
        confidence_label: GeminiConfidence | None = None,
    ) -> GeminiResult:
        return GeminiResult(
            status=status,
            answer=answer,
            answer_type=request.expected_answer_type,
            confidence=max(0.0, min(1.0, float(confidence))),
            confidence_label=confidence_label,
            model=model,
            lane=request.lane,
            latency_ms=self._elapsed_ms(started),
            detail=detail,
        )


__all__ = [
    "GeminiAvailability",
    "GeminiProvider",
    "GeminiRequest",
    "GeminiResult",
]
