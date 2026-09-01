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
    localized_tile_size: int = 12
    localized_changed_ratio: float = 0.08
    stable_frames: int = 3
    ignore_regions: tuple[Region, ...] = ()


@dataclass(frozen=True)
class PromptConfig:
    # Optional region relative to the capture crop. When null, OCR boxes locate
    # the band between "Anime Guessing Game" and "Answer with" dynamically.
    static_hint_roi: Region | None = None
    min_text_characters: int = 10
    min_alpha_characters: int = 6
    visual_min_stddev: float = 4.0
    visual_trim_threshold: float = 18.0
    visual_trim_padding: int = 8
    crop_padding_x: int = 14
    crop_padding_y: int = 3
    max_header_to_answer_pixels: int = 300
    max_answer_to_footer_pixels: int = 120
    horizontal_alignment_tolerance: int = 96
    header_markers: tuple[str, ...] = ("anime guessing game",)
    answer_markers: tuple[str, ...] = ("answer with", "first correct guess")


@dataclass(frozen=True)
class ReadinessConfig:
    # The Discord embed's colored left outline is authoritative. OCR status text
    # is recorded as corroborating evidence but cannot open the gate by default.
    require_green_outline: bool = True
    ready_wait_timeout_seconds: float = 20.0
    search_left_pixels: int = 96
    search_right_pixels: int = 16
    search_vertical_padding: int = 18
    min_color_pixels: int = 300
    min_outline_height_pixels: int = 80
    min_outline_width_pixels: int = 3
    max_outline_width_pixels: int = 14
    min_outline_aspect_ratio: float = 12.0
    min_outline_fill_ratio: float = 0.60
    prelocate_active_card: bool = True
    prelocate_right_extent_pixels: int = 960
    prelocate_padding_pixels: int = 12
    red_min_channel: int = 140
    green_min_channel: int = 100
    channel_dominance_ratio: float = 1.25
    state_dominance_ratio: float = 1.50
    allow_text_only_ready: bool = False
    ready_text_markers: tuple[str, ...] = (
        "answer now",
        "answers open you have",
    )
    locked_text_markers: tuple[str, ...] = (
        "get ready",
        "reading time",
    )
    closed_text_markers: tuple[str, ...] = ("round over",)


@dataclass(frozen=True)
class OcrConfig:
    device: str = "gpu:0"
    require_cuda: bool = True
    engine: str = "transformers"
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
    model_id: str = "Qwen/Qwen3-VL-32B-Instruct"
    device: str = "cuda:0"
    quantization: str = "nf4"
    allow_novel_visual_submission: bool = False
    allow_unverified_submission: bool = False
    local_files_only: bool = False
    ready_before_capture: bool = True
    preload_in_background: bool = False
    max_image_side: int = 768
    max_visual_upscale_factor: float = 4.0
    visual_transcription_tokens: int = 64
    visual_consensus_passes: int = 2
    max_new_tokens: int = 24
    max_answer_characters: int = 96


@dataclass(frozen=True)
class TypingConfig:
    enabled: bool = True
    expected_process_names: tuple[str, ...] = ("Discord.exe",)
    expected_window_title_contains: str = "Discord"
    pre_delay_seconds: tuple[float, float] = (0.4, 1.1)
    key_delay_seconds: tuple[float, float] = (0.03, 0.08)
    draft_while_locked: bool = True
    verify_composer: bool = True
    auto_focus_composer: bool = True
    composer_name_prefix: str = "Message #"
    composer_class_fragment: str = "slateTextArea"
    respect_detected_countdown: bool = True
    fallback_answer_open_delay_seconds: float = 5.0
    enter_after_open_slack_seconds: float = 0.06
    max_answer_characters: int = 96
    stop_key: str = "f12"


@dataclass(frozen=True)
class StatusConfig:
    enabled: bool = True
    topmost: bool = True
    click_through: bool = True
    width: int = 560
    height: int = 310
    margin_x: int = 32
    margin_y: int = 32
    opacity: float = 0.96
    poll_ms: int = 100
    stale_after_seconds: float = 5.0
    auto_close_seconds: float = 4.0
    error_close_seconds: float = 15.0


@dataclass(frozen=True)
class RuntimeConfig:
    cache_path: Path = Path("data/trivia_cache.json")
    seed_cache_path: Path | None = Path("data/trivia_cache.seed.json")
    history_path: Path | None = Path("data/trivia_history.seed.json")
    status_path: Path = Path("runtime/operator_status.json")
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
    readiness: ReadinessConfig = field(default_factory=ReadinessConfig)
    ocr: OcrConfig = field(default_factory=OcrConfig)
    matching: MatchConfig = field(default_factory=MatchConfig)
    vlm: VlmConfig = field(default_factory=VlmConfig)
    typing: TypingConfig = field(default_factory=TypingConfig)
    status: StatusConfig = field(default_factory=StatusConfig)
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
    readiness_raw = _section(raw, "readiness")
    ocr_raw = _section(raw, "ocr")
    match_raw = _section(raw, "matching")
    vlm_raw = _section(raw, "vlm")
    typing_raw = _section(raw, "typing")
    status_raw = _section(raw, "status")
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
        localized_tile_size=int(change_raw.get("localized_tile_size", 12)),
        localized_changed_ratio=float(change_raw.get("localized_changed_ratio", 0.08)),
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
        visual_trim_threshold=float(prompt_raw.get("visual_trim_threshold", 18.0)),
        visual_trim_padding=int(prompt_raw.get("visual_trim_padding", 8)),
        crop_padding_x=int(prompt_raw.get("crop_padding_x", 14)),
        crop_padding_y=int(prompt_raw.get("crop_padding_y", 3)),
        max_header_to_answer_pixels=int(
            prompt_raw.get("max_header_to_answer_pixels", 300)
        ),
        max_answer_to_footer_pixels=int(
            prompt_raw.get("max_answer_to_footer_pixels", 120)
        ),
        horizontal_alignment_tolerance=int(
            prompt_raw.get("horizontal_alignment_tolerance", 96)
        ),
        header_markers=tuple(
            str(v).casefold()
            for v in prompt_raw.get("header_markers", ["anime guessing game"])
        ),
        answer_markers=tuple(
            str(v).casefold()
            for v in prompt_raw.get(
                "answer_markers", ["answer with", "first correct guess"]
            )
        ),
    )

    readiness = ReadinessConfig(
        require_green_outline=bool(readiness_raw.get("require_green_outline", True)),
        ready_wait_timeout_seconds=float(
            readiness_raw.get("ready_wait_timeout_seconds", 20.0)
        ),
        search_left_pixels=int(readiness_raw.get("search_left_pixels", 96)),
        search_right_pixels=int(readiness_raw.get("search_right_pixels", 16)),
        search_vertical_padding=int(readiness_raw.get("search_vertical_padding", 18)),
        min_color_pixels=int(readiness_raw.get("min_color_pixels", 300)),
        min_outline_height_pixels=int(
            readiness_raw.get("min_outline_height_pixels", 80)
        ),
        min_outline_width_pixels=int(readiness_raw.get("min_outline_width_pixels", 3)),
        max_outline_width_pixels=int(readiness_raw.get("max_outline_width_pixels", 14)),
        min_outline_aspect_ratio=float(
            readiness_raw.get("min_outline_aspect_ratio", 12.0)
        ),
        min_outline_fill_ratio=float(readiness_raw.get("min_outline_fill_ratio", 0.60)),
        prelocate_active_card=bool(readiness_raw.get("prelocate_active_card", True)),
        prelocate_right_extent_pixels=int(
            readiness_raw.get("prelocate_right_extent_pixels", 960)
        ),
        prelocate_padding_pixels=int(readiness_raw.get("prelocate_padding_pixels", 12)),
        red_min_channel=int(readiness_raw.get("red_min_channel", 140)),
        green_min_channel=int(readiness_raw.get("green_min_channel", 100)),
        channel_dominance_ratio=float(
            readiness_raw.get("channel_dominance_ratio", 1.25)
        ),
        state_dominance_ratio=float(readiness_raw.get("state_dominance_ratio", 1.50)),
        allow_text_only_ready=bool(readiness_raw.get("allow_text_only_ready", False)),
        ready_text_markers=tuple(
            str(value).casefold()
            for value in readiness_raw.get(
                "ready_text_markers", ["answer now", "answers open you have"]
            )
        ),
        locked_text_markers=tuple(
            str(value).casefold()
            for value in readiness_raw.get(
                "locked_text_markers", ["get ready", "reading time"]
            )
        ),
        closed_text_markers=tuple(
            str(value).casefold()
            for value in readiness_raw.get("closed_text_markers", ["round over"])
        ),
    )

    ocr = OcrConfig(
        device=str(ocr_raw.get("device", "gpu:0")),
        require_cuda=bool(ocr_raw.get("require_cuda", True)),
        engine=str(ocr_raw.get("engine", "transformers")),
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
        model_id=str(vlm_raw.get("model_id", "Qwen/Qwen3-VL-32B-Instruct")),
        device=str(vlm_raw.get("device", "cuda:0")),
        quantization=str(vlm_raw.get("quantization", "nf4")),
        allow_novel_visual_submission=bool(
            vlm_raw.get("allow_novel_visual_submission", False)
        ),
        allow_unverified_submission=bool(
            vlm_raw.get("allow_unverified_submission", False)
        ),
        local_files_only=bool(vlm_raw.get("local_files_only", False)),
        ready_before_capture=bool(vlm_raw.get("ready_before_capture", True)),
        preload_in_background=bool(vlm_raw.get("preload_in_background", False)),
        max_image_side=int(vlm_raw.get("max_image_side", 768)),
        max_visual_upscale_factor=float(vlm_raw.get("max_visual_upscale_factor", 4.0)),
        visual_transcription_tokens=int(vlm_raw.get("visual_transcription_tokens", 64)),
        visual_consensus_passes=int(vlm_raw.get("visual_consensus_passes", 2)),
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
        draft_while_locked=bool(typing_raw.get("draft_while_locked", True)),
        verify_composer=bool(typing_raw.get("verify_composer", True)),
        auto_focus_composer=bool(typing_raw.get("auto_focus_composer", True)),
        composer_name_prefix=str(
            typing_raw.get("composer_name_prefix", "Message #")
        ),
        composer_class_fragment=str(
            typing_raw.get("composer_class_fragment", "slateTextArea")
        ),
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

    status = StatusConfig(
        enabled=bool(status_raw.get("enabled", True)),
        topmost=bool(status_raw.get("topmost", True)),
        click_through=bool(status_raw.get("click_through", True)),
        width=int(status_raw.get("width", 560)),
        height=int(status_raw.get("height", 310)),
        margin_x=int(status_raw.get("margin_x", 32)),
        margin_y=int(status_raw.get("margin_y", 32)),
        opacity=float(status_raw.get("opacity", 0.96)),
        poll_ms=int(status_raw.get("poll_ms", 100)),
        stale_after_seconds=float(status_raw.get("stale_after_seconds", 5.0)),
        auto_close_seconds=float(status_raw.get("auto_close_seconds", 4.0)),
        error_close_seconds=float(status_raw.get("error_close_seconds", 15.0)),
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
    history_value = runtime_raw.get("history_path", "data/trivia_history.seed.json")
    runtime = RuntimeConfig(
        cache_path=resolve_local(
            str(runtime_raw.get("cache_path", "data/trivia_cache.json"))
        ),
        seed_cache_path=(
            resolve_local(str(seed_cache_value))
            if seed_cache_value is not None
            else None
        ),
        history_path=(
            resolve_local(str(history_value)) if history_value is not None else None
        ),
        status_path=resolve_local(
            str(runtime_raw.get("status_path", "runtime/operator_status.json"))
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
        readiness=readiness,
        ocr=ocr,
        matching=matching,
        vlm=vlm,
        typing=typing,
        status=status,
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
        ("localized_changed_ratio", config.change_detection.localized_changed_ratio),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"change_detection.{name} must be between 0 and 1")
    if (
        not 1
        <= config.change_detection.localized_tile_size
        <= min(config.change_detection.thumbnail_size)
    ):
        raise ValueError("change_detection.localized_tile_size is invalid")
    if config.prompt.static_hint_roi is not None:
        _left, _top, right, bottom = config.prompt.static_hint_roi
        if right > capture_width or bottom > capture_height:
            raise ValueError("prompt.static_hint_roi must fit inside capture.region")
    if not config.prompt.header_markers or not config.prompt.answer_markers:
        raise ValueError("prompt header/answer marker lists cannot be empty")
    if config.prompt.min_text_characters < 1 or config.prompt.min_alpha_characters < 1:
        raise ValueError("prompt text thresholds must be positive")
    if (
        config.prompt.visual_trim_threshold <= 0
        or config.prompt.visual_trim_padding < 0
    ):
        raise ValueError("prompt visual trim settings are invalid")
    if (
        min(
            config.prompt.max_header_to_answer_pixels,
            config.prompt.max_answer_to_footer_pixels,
            config.prompt.horizontal_alignment_tolerance,
        )
        <= 0
    ):
        raise ValueError("prompt geometry limits must be positive")
    if config.readiness.ready_wait_timeout_seconds <= 0:
        raise ValueError("readiness.ready_wait_timeout_seconds must be positive")
    if (
        min(
            config.readiness.search_left_pixels,
            config.readiness.search_right_pixels,
            config.readiness.search_vertical_padding,
        )
        < 0
    ):
        raise ValueError("readiness search sizes must be nonnegative")
    if (
        min(
            config.readiness.min_color_pixels,
            config.readiness.min_outline_height_pixels,
            config.readiness.min_outline_width_pixels,
            config.readiness.max_outline_width_pixels,
        )
        <= 0
    ):
        raise ValueError("readiness component dimensions/counts must be positive")
    if (
        config.readiness.min_outline_width_pixels
        > config.readiness.max_outline_width_pixels
    ):
        raise ValueError("readiness outline width minimum exceeds maximum")
    if (
        not 0 <= config.readiness.red_min_channel <= 255
        or not 0 <= config.readiness.green_min_channel <= 255
    ):
        raise ValueError("readiness channel minimums must be between 0 and 255")
    if config.readiness.channel_dominance_ratio <= 1.0:
        raise ValueError("readiness.channel_dominance_ratio must exceed 1")
    if config.readiness.state_dominance_ratio <= 1.0:
        raise ValueError("readiness.state_dominance_ratio must exceed 1")
    if config.readiness.min_outline_aspect_ratio <= 1.0:
        raise ValueError("readiness.min_outline_aspect_ratio must exceed 1")
    if not 0.0 < config.readiness.min_outline_fill_ratio <= 1.0:
        raise ValueError("readiness.min_outline_fill_ratio must be between 0 and 1")
    if config.readiness.prelocate_right_extent_pixels < 200:
        raise ValueError("readiness.prelocate_right_extent_pixels is too small")
    if config.readiness.prelocate_padding_pixels < 0:
        raise ValueError("readiness.prelocate_padding_pixels must be nonnegative")
    if (
        not config.readiness.ready_text_markers
        or not config.readiness.locked_text_markers
        or not config.readiness.closed_text_markers
    ):
        raise ValueError("readiness text marker lists cannot be empty")
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
    if config.ocr.engine != "transformers":
        raise ValueError(
            "ocr.engine must be 'transformers' so OCR and Qwen share one CUDA runtime"
        )
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
        if config.vlm.max_visual_upscale_factor < 1.0:
            raise ValueError("vlm.max_visual_upscale_factor must be at least 1")
        if config.vlm.visual_transcription_tokens < 8:
            raise ValueError("vlm.visual_transcription_tokens is too small")
        if not 2 <= config.vlm.visual_consensus_passes <= 3:
            raise ValueError("vlm.visual_consensus_passes must be 2 or 3")
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
    if config.typing.verify_composer and (
        not config.typing.composer_name_prefix.strip()
        or not config.typing.composer_class_fragment.strip()
    ):
        raise ValueError("typing composer selectors cannot be empty")
    if config.typing.max_answer_characters < 1:
        raise ValueError("typing.max_answer_characters must be positive")
    if config.status.width < 360 or config.status.height < 220:
        raise ValueError("status panel dimensions are too small")
    if config.status.margin_x < 0 or config.status.margin_y < 0:
        raise ValueError("status panel margins must be nonnegative")
    if not 0.25 <= config.status.opacity <= 1.0:
        raise ValueError("status.opacity must be between 0.25 and 1.0")
    if not 50 <= config.status.poll_ms <= 2000:
        raise ValueError("status.poll_ms must be between 50 and 2000")
    if (
        config.status.stale_after_seconds <= 0
        or config.status.auto_close_seconds < 0
        or config.status.error_close_seconds <= 0
    ):
        raise ValueError("status timing values are invalid")
    if (
        config.runtime.seed_cache_path is not None
        and config.runtime.seed_cache_path == config.runtime.cache_path
    ):
        raise ValueError("runtime seed and mutable cache paths must be different")
    if not 0 <= config.runtime.scene_retry_limit <= 3:
        raise ValueError("runtime.scene_retry_limit must be between 0 and 3")
