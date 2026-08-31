from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

Region = tuple[int, int, int, int]


def _region(
    value: Sequence[int] | None, name: str, *, optional: bool = False
) -> Region | None:
    if value is None and optional:
        return None
    if value is None or len(value) != 4:
        raise ValueError(f"{name} must contain [left, top, right, bottom]")
    result = tuple(int(v) for v in value)
    left, top, right, bottom = result
    if left < 0 or top < 0 or right <= left or bottom <= top:
        raise ValueError(f"{name} is invalid: {result}")
    return result


def _tuple2(value: Sequence[int], name: str) -> tuple[int, int]:
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly two integers")
    result = (int(value[0]), int(value[1]))
    if result[0] <= 0 or result[1] <= 0:
        raise ValueError(f"{name} values must be positive")
    return result


def _section(data: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, Mapping):
        raise TypeError(f"Configuration section '{name}' must be a JSON object")
    return value


@dataclass(frozen=True)
class CaptureConfig:
    # Coordinates are physical pixels local to the selected DXGI output.
    region: Region
    calibrated: bool = False
    device_idx: int | None = 0
    output_idx: int | None = None
    backend: str = "dxgi"
    processor_backend: str = "cv2"
    fps: int = 60
    video_mode: bool = True
    max_buffer_len: int = 4


@dataclass(frozen=True)
class ChangeDetectionConfig:
    device: str = "cuda:0"
    require_cuda: bool = True
    thumbnail_size: tuple[int, int] = (96, 96)
    mean_absolute_threshold: float = 0.012
    changed_pixel_threshold: float = 0.055
    changed_pixel_ratio: float = 0.012
    stable_frames: int = 3
    ignore_regions: tuple[Region, ...] = ()


@dataclass(frozen=True)
class PromptConfig:
    # Optional region relative to the capture crop. When null, OCR boxes locate
    # the band between "Get Ready" and "Answer with" dynamically.
    static_hint_roi: Region | None = None
    min_text_characters: int = 10
    min_alpha_characters: int = 6
    visual_min_stddev: float = 4.0
    crop_padding_x: int = 14
    crop_padding_y: int = 3
    header_markers: tuple[str, ...] = ("anime guessing game", "get ready")
    answer_markers: tuple[str, ...] = ("answer with", "first correct guess")


@dataclass(frozen=True)
class OcrConfig:
    device: str = "gpu:0"
    require_cuda: bool = True
    engine: str = "paddle_static"
    text_detection_model_name: str = "PP-OCRv6_small_det"
    text_recognition_model_name: str = "PP-OCRv6_small_rec"
    recognition_score_threshold: float = 0.35
    detection_limit_side_len: int = 960
    warmup_smoke_test: bool = True


@dataclass(frozen=True)
class MatchConfig:
    text_score_threshold: float = 88.0
    text_score_margin: float = 4.0
    phash_size: int = 16
    phash_max_distance: int = 10
    phash_distance_margin: int = 3


@dataclass(frozen=True)
class VlmConfig:
    enabled: bool = True
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct"
    device: str = "cuda:0"
    quantization: str = "nf4"
    local_files_only: bool = False
    ready_before_capture: bool = True
    preload_in_background: bool = False
    max_image_side: int = 768
    max_new_tokens: int = 24
    max_answer_characters: int = 96


@dataclass(frozen=True)
class TypingConfig:
    enabled: bool = True
    expected_process_names: tuple[str, ...] = ("Discord.exe",)
    expected_window_title_contains: str = "Discord"
    pre_delay_seconds: tuple[float, float] = (0.4, 1.1)
    key_delay_seconds: tuple[float, float] = (0.03, 0.08)
    respect_detected_countdown: bool = True
    fallback_answer_open_delay_seconds: float = 5.0
    enter_after_open_slack_seconds: float = 0.06
    max_answer_characters: int = 96
    stop_key: str = "f12"


@dataclass(frozen=True)
class RuntimeConfig:
    cache_path: Path = Path("data/trivia_cache.json")
    seed_cache_path: Path | None = Path("data/trivia_cache.seed.json")
    debug_dir: Path = Path("debug")
    save_prompt_crops: bool = False
    scene_retry_limit: int = 1
    log_level: str = "INFO"


@dataclass(frozen=True)
class AppConfig:
    capture: CaptureConfig
    change_detection: ChangeDetectionConfig = field(
        default_factory=ChangeDetectionConfig
    )
    prompt: PromptConfig = field(default_factory=PromptConfig)
    ocr: OcrConfig = field(default_factory=OcrConfig)
    matching: MatchConfig = field(default_factory=MatchConfig)
    vlm: VlmConfig = field(default_factory=VlmConfig)
    typing: TypingConfig = field(default_factory=TypingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, Mapping):
        raise TypeError("The configuration root must be a JSON object")

    capture_raw = _section(raw, "capture")
    change_raw = _section(raw, "change_detection")
    prompt_raw = _section(raw, "prompt")
    ocr_raw = _section(raw, "ocr")
    match_raw = _section(raw, "matching")
    vlm_raw = _section(raw, "vlm")
    typing_raw = _section(raw, "typing")
    runtime_raw = _section(raw, "runtime")

    if "region" not in capture_raw:
        raise ValueError("capture.region is required")
    capture = CaptureConfig(
        region=_region(capture_raw["region"], "capture.region"),  # type: ignore[arg-type]
        calibrated=bool(capture_raw.get("calibrated", False)),
        device_idx=capture_raw.get("device_idx", 0),
        output_idx=capture_raw.get("output_idx"),
        backend=str(capture_raw.get("backend", "dxgi")),
        processor_backend=str(capture_raw.get("processor_backend", "cv2")),
        fps=int(capture_raw.get("fps", 60)),
        video_mode=bool(capture_raw.get("video_mode", True)),
        max_buffer_len=int(capture_raw.get("max_buffer_len", 4)),
    )

    ignore_regions = tuple(
        _region(item, f"change_detection.ignore_regions[{index}]")  # type: ignore[arg-type]
        for index, item in enumerate(change_raw.get("ignore_regions", []))
    )
    change_detection = ChangeDetectionConfig(
        device=str(change_raw.get("device", "cuda:0")),
        require_cuda=bool(change_raw.get("require_cuda", True)),
        thumbnail_size=_tuple2(
            change_raw.get("thumbnail_size", [96, 96]),
            "change_detection.thumbnail_size",
        ),
        mean_absolute_threshold=float(change_raw.get("mean_absolute_threshold", 0.012)),
        changed_pixel_threshold=float(change_raw.get("changed_pixel_threshold", 0.055)),
        changed_pixel_ratio=float(change_raw.get("changed_pixel_ratio", 0.012)),
        stable_frames=int(change_raw.get("stable_frames", 3)),
        ignore_regions=ignore_regions,  # type: ignore[arg-type]
    )

    prompt = PromptConfig(
        static_hint_roi=_region(
            prompt_raw.get("static_hint_roi"), "prompt.static_hint_roi", optional=True
        ),
        min_text_characters=int(prompt_raw.get("min_text_characters", 10)),
        min_alpha_characters=int(prompt_raw.get("min_alpha_characters", 6)),
        visual_min_stddev=float(prompt_raw.get("visual_min_stddev", 4.0)),
        crop_padding_x=int(prompt_raw.get("crop_padding_x", 14)),
        crop_padding_y=int(prompt_raw.get("crop_padding_y", 3)),
        header_markers=tuple(
            str(v).casefold()
            for v in prompt_raw.get(
                "header_markers", ["anime guessing game", "get ready"]
            )
        ),
        answer_markers=tuple(
            str(v).casefold()
            for v in prompt_raw.get(
                "answer_markers", ["answer with", "first correct guess"]
            )
        ),
    )

    ocr = OcrConfig(
        device=str(ocr_raw.get("device", "gpu:0")),
        require_cuda=bool(ocr_raw.get("require_cuda", True)),
        engine=str(ocr_raw.get("engine", "paddle_static")),
        text_detection_model_name=str(
            ocr_raw.get("text_detection_model_name", "PP-OCRv6_small_det")
        ),
        text_recognition_model_name=str(
            ocr_raw.get("text_recognition_model_name", "PP-OCRv6_small_rec")
        ),
        recognition_score_threshold=float(
            ocr_raw.get("recognition_score_threshold", 0.35)
        ),
        detection_limit_side_len=int(ocr_raw.get("detection_limit_side_len", 960)),
        warmup_smoke_test=bool(ocr_raw.get("warmup_smoke_test", True)),
    )

    matching = MatchConfig(
        text_score_threshold=float(match_raw.get("text_score_threshold", 88.0)),
        text_score_margin=float(match_raw.get("text_score_margin", 4.0)),
        phash_size=int(match_raw.get("phash_size", 16)),
        phash_max_distance=int(match_raw.get("phash_max_distance", 10)),
        phash_distance_margin=int(match_raw.get("phash_distance_margin", 3)),
    )

    vlm = VlmConfig(
        enabled=bool(vlm_raw.get("enabled", True)),
        model_id=str(vlm_raw.get("model_id", "Qwen/Qwen3-VL-8B-Instruct")),
        device=str(vlm_raw.get("device", "cuda:0")),
        quantization=str(vlm_raw.get("quantization", "nf4")),
        local_files_only=bool(vlm_raw.get("local_files_only", False)),
        ready_before_capture=bool(vlm_raw.get("ready_before_capture", True)),
        preload_in_background=bool(vlm_raw.get("preload_in_background", False)),
        max_image_side=int(vlm_raw.get("max_image_side", 768)),
        max_new_tokens=int(vlm_raw.get("max_new_tokens", 24)),
        max_answer_characters=int(vlm_raw.get("max_answer_characters", 96)),
    )

    pre_delay = typing_raw.get("pre_delay_seconds", [0.4, 1.1])
    key_delay = typing_raw.get("key_delay_seconds", [0.03, 0.08])
    typing = TypingConfig(
        enabled=bool(typing_raw.get("enabled", True)),
        expected_process_names=tuple(
            str(v) for v in typing_raw.get("expected_process_names", ["Discord.exe"])
        ),
        expected_window_title_contains=str(
            typing_raw.get("expected_window_title_contains", "Discord")
        ),
        pre_delay_seconds=(float(pre_delay[0]), float(pre_delay[1])),
        key_delay_seconds=(float(key_delay[0]), float(key_delay[1])),
        respect_detected_countdown=bool(
            typing_raw.get("respect_detected_countdown", True)
        ),
        fallback_answer_open_delay_seconds=float(
            typing_raw.get("fallback_answer_open_delay_seconds", 5.0)
        ),
        enter_after_open_slack_seconds=float(
            typing_raw.get("enter_after_open_slack_seconds", 0.06)
        ),
        max_answer_characters=int(typing_raw.get("max_answer_characters", 96)),
        stop_key=str(typing_raw.get("stop_key", "f12")).casefold(),
    )

    base_dir = config_path.parent

    def resolve_local(value: str) -> Path:
        candidate = Path(value).expanduser()
        return (
            candidate.resolve()
            if candidate.is_absolute()
            else (base_dir / candidate).resolve()
        )

    seed_cache_value = runtime_raw.get("seed_cache_path", "data/trivia_cache.seed.json")
    runtime = RuntimeConfig(
        cache_path=resolve_local(
            str(runtime_raw.get("cache_path", "data/trivia_cache.json"))
        ),
        seed_cache_path=(
            resolve_local(str(seed_cache_value))
            if seed_cache_value is not None
            else None
        ),
        debug_dir=resolve_local(str(runtime_raw.get("debug_dir", "debug"))),
        save_prompt_crops=bool(runtime_raw.get("save_prompt_crops", False)),
        scene_retry_limit=int(runtime_raw.get("scene_retry_limit", 1)),
        log_level=str(runtime_raw.get("log_level", "INFO")).upper(),
    )

    config = AppConfig(
        capture=capture,
        change_detection=change_detection,
        prompt=prompt,
        ocr=ocr,
        matching=matching,
        vlm=vlm,
        typing=typing,
        runtime=runtime,
    )
    validate_config(config)
    return config


def validate_config(config: AppConfig) -> None:
    capture_width = config.capture.region[2] - config.capture.region[0]
    capture_height = config.capture.region[3] - config.capture.region[1]
    if not 1 <= config.capture.fps <= 240:
        raise ValueError("capture.fps must be between 1 and 240")
    for name, value in (
        ("device_idx", config.capture.device_idx),
        ("output_idx", config.capture.output_idx),
    ):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int)
        ):
            raise TypeError(f"capture.{name} must be an integer or null")
        if value is not None and value < 0:
            raise ValueError(f"capture.{name} must be nonnegative")
    if config.capture.max_buffer_len < 1:
        raise ValueError("capture.max_buffer_len must be at least 1")
    if config.capture.backend not in {"dxgi", "winrt"}:
        raise ValueError("capture.backend must be 'dxgi' or 'winrt'")
    if config.change_detection.stable_frames < 1:
        raise ValueError("change_detection.stable_frames must be at least 1")
    if (
        config.change_detection.require_cuda
        and not config.change_detection.device.startswith("cuda")
    ):
        raise ValueError(
            "change_detection.device must be CUDA when require_cuda is true"
        )
    for name, value in (
        ("mean_absolute_threshold", config.change_detection.mean_absolute_threshold),
        ("changed_pixel_threshold", config.change_detection.changed_pixel_threshold),
        ("changed_pixel_ratio", config.change_detection.changed_pixel_ratio),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"change_detection.{name} must be between 0 and 1")
    if config.prompt.static_hint_roi is not None:
        _left, _top, right, bottom = config.prompt.static_hint_roi
        if right > capture_width or bottom > capture_height:
            raise ValueError("prompt.static_hint_roi must fit inside capture.region")
    if not config.prompt.header_markers or not config.prompt.answer_markers:
        raise ValueError("prompt header/answer marker lists cannot be empty")
    if config.prompt.min_text_characters < 1 or config.prompt.min_alpha_characters < 1:
        raise ValueError("prompt text thresholds must be positive")
    for index, region in enumerate(config.change_detection.ignore_regions):
        if region is None:
            continue
        _left, _top, right, bottom = region
        if right > capture_width or bottom > capture_height:
            raise ValueError(
                f"change_detection.ignore_regions[{index}] must fit inside capture.region"
            )
    if not 0.0 <= config.ocr.recognition_score_threshold <= 1.0:
        raise ValueError("ocr.recognition_score_threshold must be between 0 and 1")
    if config.ocr.require_cuda and not config.ocr.device.startswith("gpu"):
        raise ValueError("ocr.device must be a GPU when ocr.require_cuda is true")
    if not 0.0 <= config.matching.text_score_threshold <= 100.0:
        raise ValueError("matching.text_score_threshold must be between 0 and 100")
    if not 0.0 <= config.matching.text_score_margin <= 100.0:
        raise ValueError("matching.text_score_margin must be between 0 and 100")
    if config.matching.phash_size < 4:
        raise ValueError("matching.phash_size must be at least 4")
    phash_bits = config.matching.phash_size**2
    if not 0 <= config.matching.phash_max_distance <= phash_bits:
        raise ValueError("matching.phash_max_distance is outside the hash bit range")
    if not 0 <= config.matching.phash_distance_margin <= phash_bits:
        raise ValueError("matching.phash_distance_margin is outside the hash bit range")
    if config.vlm.enabled:
        if not config.vlm.device.startswith("cuda"):
            raise ValueError("vlm.device must be CUDA when the VLM is enabled")
        if config.vlm.max_image_side < 64 or config.vlm.max_new_tokens < 1:
            raise ValueError("VLM image/token limits are invalid")
        if config.vlm.max_answer_characters < 1:
            raise ValueError("vlm.max_answer_characters must be positive")
    if config.typing.pre_delay_seconds[0] > config.typing.pre_delay_seconds[1]:
        raise ValueError("typing.pre_delay_seconds minimum exceeds maximum")
    if config.typing.pre_delay_seconds[0] < 0:
        raise ValueError("typing.pre_delay_seconds values must be nonnegative")
    if config.typing.key_delay_seconds[0] > config.typing.key_delay_seconds[1]:
        raise ValueError("typing.key_delay_seconds minimum exceeds maximum")
    if config.typing.key_delay_seconds[0] <= 0:
        raise ValueError("typing.key_delay_seconds values must be positive")
    if config.typing.fallback_answer_open_delay_seconds < 0:
        raise ValueError(
            "typing.fallback_answer_open_delay_seconds must be nonnegative"
        )
    if config.typing.enter_after_open_slack_seconds < 0:
        raise ValueError("typing.enter_after_open_slack_seconds must be nonnegative")
    timing_values = (
        *config.typing.pre_delay_seconds,
        *config.typing.key_delay_seconds,
        config.typing.fallback_answer_open_delay_seconds,
        config.typing.enter_after_open_slack_seconds,
    )
    if not all(math.isfinite(value) for value in timing_values):
        raise ValueError("typing timing values must be finite")
    if not config.typing.expected_process_names:
        raise ValueError("typing.expected_process_names cannot be empty")
    if not config.typing.expected_window_title_contains.strip():
        raise ValueError("typing.expected_window_title_contains cannot be empty")
    if config.typing.max_answer_characters < 1:
        raise ValueError("typing.max_answer_characters must be positive")
    if (
        config.runtime.seed_cache_path is not None
        and config.runtime.seed_cache_path == config.runtime.cache_path
    ):
        raise ValueError("runtime seed and mutable cache paths must be different")
    if not 0 <= config.runtime.scene_retry_limit <= 3:
        raise ValueError("runtime.scene_retry_limit must be between 0 and 3")
