from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .config import MatchConfig, OcrConfig, PromptConfig, ReadinessConfig
from .models import OcrSpan, PromptObservation, ReadinessState, Scene
from .utils import normalize_question, short_signature

LOGGER = logging.getLogger(__name__)


def _page_field(page: Any, key: str, default: Any) -> Any:
    try:
        return page[key]
    except (KeyError, TypeError, IndexError):
        pass
    payload = getattr(page, "json", None)
    if callable(payload):
        payload = payload()
    if isinstance(payload, Mapping):
        payload = payload.get("res", payload)
        if isinstance(payload, Mapping):
            return payload.get(key, default)
    return default


class PaddleOCREngine:
    """Current PaddleOCR 3.x GPU pipeline with one-time model initialization."""

    def __init__(self, config: OcrConfig) -> None:
        self._config = config
        try:
            import paddle
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError(
                "PaddleOCR runtime is missing. Run scripts/install_windows.ps1 first."
            ) from exc

        if config.require_cuda:
            compiled = bool(paddle.is_compiled_with_cuda())
            device_count = int(paddle.device.cuda.device_count()) if compiled else 0
            if not compiled or device_count < 1:
                raise RuntimeError(
                    "PaddlePaddle GPU support is unavailable. Install the official cu129 "
                    "paddlepaddle-gpu wheel and verify the NVIDIA driver."
                )
        LOGGER.info(
            "Loading PaddleOCR: engine=%s device=%s det=%s rec=%s",
            config.engine,
            config.device,
            config.text_detection_model_name,
            config.text_recognition_model_name,
        )
        self._ocr = PaddleOCR(
            device=config.device,
            engine=config.engine,
            text_detection_model_name=config.text_detection_model_name,
            text_recognition_model_name=config.text_recognition_model_name,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_recognition_batch_size=1,
            text_rec_score_thresh=config.recognition_score_threshold,
            text_det_limit_type="max",
            text_det_limit_side_len=config.detection_limit_side_len,
            use_tensorrt=False,
        )
        if config.warmup_smoke_test:
            self._warm_and_validate()

    def _warm_and_validate(self) -> None:
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV and NumPy are required for the OCR startup smoke check"
            ) from exc

        smoke = np.full((150, 900, 3), 22, dtype=np.uint8)
        cv2.putText(
            smoke,
            "ANIME TRIVIA READY 123",
            (24, 98),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.0,
            (245, 245, 245),
            4,
            cv2.LINE_AA,
        )
        spans = self.recognize(smoke)
        if not spans:
            raise RuntimeError(
                "PaddleOCR loaded on CUDA but returned zero boxes for a known-text smoke image. "
                "This can indicate an incompatible Windows/RTX 50-series Paddle wheel; do not "
                "run the macro until the OCR runtime is fixed."
            )
        LOGGER.info(
            "PaddleOCR warm-up passed: %s", " | ".join(span.text for span in spans)
        )

    def recognize(self, frame: Any) -> tuple[OcrSpan, ...]:
        pages = self._ocr.predict(input=frame)
        spans: list[OcrSpan] = []
        for page in pages:
            texts = list(_page_field(page, "rec_texts", []))
            scores = list(_page_field(page, "rec_scores", []))
            boxes = list(_page_field(page, "rec_boxes", []))
            polygons = list(_page_field(page, "rec_polys", []))
            for index, raw_text in enumerate(texts):
                text = str(raw_text).strip()
                score = float(scores[index]) if index < len(scores) else 0.0
                if not text or score < self._config.recognition_score_threshold:
                    continue
                if index < len(boxes):
                    raw_box = boxes[index]
                    box = tuple(int(value) for value in raw_box[:4])
                elif index < len(polygons):
                    polygon = polygons[index]
                    xs = [int(point[0]) for point in polygon]
                    ys = [int(point[1]) for point in polygon]
                    box = (min(xs), min(ys), max(xs), max(ys))
                else:
                    box = (0, index, 1, index + 1)
                spans.append(OcrSpan(text=text, score=score, box=box))
        spans.sort(key=lambda span: (span.top, span.left))
        return tuple(spans)


def _merge_line(spans: Sequence[OcrSpan]) -> OcrSpan:
    ordered = sorted(spans, key=lambda span: span.left)
    return OcrSpan(
        text=" ".join(span.text for span in ordered),
        score=sum(span.score for span in ordered) / len(ordered),
        box=(
            min(span.left for span in ordered),
            min(span.top for span in ordered),
            max(span.right for span in ordered),
            max(span.bottom for span in ordered),
        ),
    )


def _line_groups(spans: Iterable[OcrSpan]) -> list[OcrSpan]:
    ordered = sorted(spans, key=lambda span: (span.center_y, span.left))
    groups: list[list[OcrSpan]] = []
    for span in ordered:
        if not groups:
            groups.append([span])
            continue
        current = groups[-1]
        center = sum(item.center_y for item in current) / len(current)
        average_height = sum(max(1, item.bottom - item.top) for item in current) / len(
            current
        )
        if abs(span.center_y - center) <= max(8.0, average_height * 0.55):
            current.append(span)
        else:
            groups.append([span])
    return [_merge_line(group) for group in groups]


class PromptExtractor:
    """Finds the newest Anime Soul card and isolates only its hint band."""

    def __init__(
        self,
        prompt_config: PromptConfig,
        match_config: MatchConfig,
        readiness_config: ReadinessConfig | None = None,
    ) -> None:
        self._config = prompt_config
        self._match = match_config
        self._readiness = readiness_config or ReadinessConfig()

    def extract(
        self, scene: Scene, spans: tuple[OcrSpan, ...]
    ) -> PromptObservation | None:
        if not spans:
            return None
        frame = scene.frame
        frame_height, frame_width = frame.shape[:2]
        lines = _line_groups(spans)

        header_lines = [
            line
            for line in lines
            if self._contains_any(line.text, self._config.header_markers)
        ]
        answer_lines = [
            line
            for line in lines
            if self._contains_any(line.text, self._config.answer_markers)
        ]
        if not header_lines or not answer_lines:
            return None

        header: OcrSpan | None = None
        answer_line: OcrSpan | None = None
        ordered_headers = sorted(header_lines, key=lambda line: line.top)
        # Only the bottommost header can be the newest round. If it is still
        # rendering, fail closed rather than falling back to an older card.
        header = ordered_headers[-1]
        candidates = [
            line
            for line in answer_lines
            if line.top > header.bottom
            and line.top - header.bottom <= self._config.max_header_to_answer_pixels
            and abs(line.left - header.left)
            <= self._config.horizontal_alignment_tolerance
        ]
        if not candidates:
            return None
        answer_line = min(candidates, key=lambda line: line.top)
        footer_candidates = [
            line
            for line in lines
            if line.top > answer_line.bottom
            and line.top - answer_line.bottom
            <= self._config.max_answer_to_footer_pixels
            and abs(line.left - answer_line.left)
            <= self._config.horizontal_alignment_tolerance
            and re.search(
                r"question\s*\d+\s*/\s*\d+",
                line.text,
                flags=re.IGNORECASE,
            )
        ]
        selected_card_bottom = (
            min(footer_candidates, key=lambda line: line.top).bottom + 1
            if footer_candidates
            else min(frame_height + 1, answer_line.bottom + 100)
        )

        card_lines = [
            line
            for line in lines
            if line.top >= header.top and line.top < selected_card_bottom
        ]
        full_text = " ".join(line.text for line in card_lines)
        readiness, red_pixels, green_pixels = self._detect_readiness(
            frame,
            card_lines,
            header,
            answer_line,
            full_text,
        )

        if self._config.static_hint_roi is not None:
            left, top, right, bottom = self._config.static_hint_roi
        else:
            left = max(
                0, min(header.left, answer_line.left) - self._config.crop_padding_x
            )
            right = min(
                frame_width,
                max(header.right, answer_line.right) + self._config.crop_padding_x,
            )
            top = max(0, header.bottom + self._config.crop_padding_y)
            bottom = min(frame_height, answer_line.top - self._config.crop_padding_y)
        if right - left < 8 or bottom - top < 8:
            return None

        prompt_crop = frame[top:bottom, left:right].copy()
        hint_spans = [
            span
            for span in spans
            if top <= span.center_y <= bottom
            and left <= (span.left + span.right) / 2.0 <= right
            and not self._contains_any(span.text, self._config.header_markers)
            and not self._contains_any(span.text, self._config.answer_markers)
        ]
        hint_text = " ".join(line.text for line in _line_groups(hint_spans)).strip()
        normalized_hint = normalize_question(hint_text)
        alpha_count = sum(character.isalpha() for character in normalized_hint)
        prompt_kind = (
            "text"
            if len(normalized_hint) >= self._config.min_text_characters
            and alpha_count >= self._config.min_alpha_characters
            else "visual"
        )

        answer_text = normalize_question(answer_line.text)
        if "character" in answer_text:
            expected_type = "character"
        elif "anime" in answer_text and "title" in answer_text:
            expected_type = "anime_title"
        else:
            normalized_full = normalize_question(full_text)
            if "answer with the character name" in normalized_full:
                expected_type = "character"
            elif "answer with the anime title" in normalized_full:
                expected_type = "anime_title"
            else:
                expected_type = "unknown"

        countdown_match = re.search(
            r"answers?\s+open\s+in\s+(\d+(?:\.\d+)?)\s*s",
            full_text,
            flags=re.IGNORECASE,
        )
        countdown = float(countdown_match.group(1)) if countdown_match else None
        question_match = re.search(
            r"question\s*(\d+)\s*/\s*(\d+)", full_text, flags=re.IGNORECASE
        )
        question_label = (
            f"{question_match.group(1)}/{question_match.group(2)}"
            if question_match
            else None
        )
        round_signature = (
            f"round:{expected_type}:{question_label}" if question_label else None
        )

        perceptual_hash: str | None = None
        if prompt_kind == "text":
            signature = round_signature or short_signature(
                expected_type, normalized_hint
            )
        else:
            try:
                import imagehash
                import numpy as np
                from PIL import Image
            except ImportError as exc:
                raise RuntimeError(
                    "Pillow, NumPy, and ImageHash are required for visual prompts"
                ) from exc
            if float(np.std(prompt_crop)) < self._config.visual_min_stddev:
                LOGGER.debug(
                    "Ignoring blank/transition visual prompt (stddev below threshold)"
                )
                return None
            rgb = Image.fromarray(prompt_crop[:, :, ::-1].copy())
            perceptual_hash = str(
                imagehash.phash(rgb, hash_size=self._match.phash_size)
            )
            signature = round_signature or f"visual:{perceptual_hash}"

        return PromptObservation(
            scene=scene,
            spans=spans,
            full_text=full_text,
            hint_text=hint_text,
            expected_answer_type=expected_type,  # type: ignore[arg-type]
            prompt_kind=prompt_kind,  # type: ignore[arg-type]
            prompt_crop=prompt_crop,
            countdown_seconds=countdown,
            question_label=question_label,
            signature=signature,
            readiness=readiness,
            red_outline_pixels=red_pixels,
            green_outline_pixels=green_pixels,
            perceptual_hash=perceptual_hash,
        )

    def _detect_readiness(
        self,
        frame: Any,
        card_lines: Sequence[OcrSpan],
        header: OcrSpan,
        answer_line: OcrSpan,
        full_text: str,
    ) -> tuple[ReadinessState, int, int]:
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV and NumPy are required for outline readiness detection"
            ) from exc

        frame_height, frame_width = frame.shape[:2]
        question_lines = [
            line for line in card_lines if "question" in normalize_question(line.text)
        ]
        bottom_anchor = (
            max(line.bottom for line in question_lines)
            if question_lines
            else answer_line.bottom
        )
        text_left = min(
            [header.left, answer_line.left, *(line.left for line in question_lines)]
        )
        left = max(0, text_left - self._readiness.search_left_pixels)
        right = min(frame_width, text_left + self._readiness.search_right_pixels)
        top = max(0, header.top - self._readiness.search_vertical_padding)
        bottom = min(
            frame_height,
            bottom_anchor + self._readiness.search_vertical_padding,
        )
        strip = frame[top:bottom, left:right]
        if strip.size == 0:
            return "unknown", 0, 0

        blue = strip[:, :, 0].astype(np.float32)
        green = strip[:, :, 1].astype(np.float32)
        red = strip[:, :, 2].astype(np.float32)
        channel_ratio = self._readiness.channel_dominance_ratio
        red_mask = (
            (red >= self._readiness.red_min_channel)
            & (red >= green * channel_ratio)
            & (red >= blue * channel_ratio)
        )
        green_mask = (
            (green >= self._readiness.green_min_channel)
            & (green >= red * channel_ratio)
            & (green >= blue * channel_ratio)
        )
        red_pixels = self._largest_outline_component(cv2, red_mask)
        green_pixels = self._largest_outline_component(cv2, green_mask)
        minimum = self._readiness.min_color_pixels
        state_ratio = self._readiness.state_dominance_ratio

        if green_pixels >= minimum and green_pixels >= red_pixels * state_ratio:
            return "ready", red_pixels, green_pixels
        if red_pixels >= minimum and red_pixels >= green_pixels * state_ratio:
            return "locked", red_pixels, green_pixels

        normalized = normalize_question(full_text)
        text_ready = any(
            normalize_question(marker) in normalized
            for marker in self._readiness.ready_text_markers
        )
        text_locked = any(
            normalize_question(marker) in normalized
            for marker in self._readiness.locked_text_markers
        )
        if self._readiness.allow_text_only_ready and text_ready and not text_locked:
            return "ready", red_pixels, green_pixels
        if text_locked:
            return "locked", red_pixels, green_pixels
        return "unknown", red_pixels, green_pixels

    def _largest_outline_component(self, cv2: Any, mask: Any) -> int:
        component_count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
            mask.astype("uint8"),
            connectivity=8,
        )
        best_area = 0
        for index in range(1, component_count):
            width = int(stats[index, cv2.CC_STAT_WIDTH])
            height = int(stats[index, cv2.CC_STAT_HEIGHT])
            area = int(stats[index, cv2.CC_STAT_AREA])
            if width <= 0:
                continue
            if height < self._readiness.min_outline_height_pixels:
                continue
            if width < self._readiness.min_outline_width_pixels:
                continue
            if width > self._readiness.max_outline_width_pixels:
                continue
            if height / width < self._readiness.min_outline_aspect_ratio:
                continue
            if area / (width * height) < self._readiness.min_outline_fill_ratio:
                continue
            best_area = max(best_area, area)
        return best_area

    @staticmethod
    def _contains_any(text: str, markers: Sequence[str]) -> bool:
        normalized = normalize_question(text)
        return any(normalize_question(marker) in normalized for marker in markers)
