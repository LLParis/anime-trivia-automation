from __future__ import annotations

import logging
import threading
import time
from typing import Any

from .config import VlmConfig
from .models import PromptObservation
from .utils import sanitize_answer

LOGGER = logging.getLogger(__name__)


class LazyQwenResolver:
    """Quantized Qwen3-VL slow path, loaded once and kept resident."""

    def __init__(self, config: VlmConfig) -> None:
        self._config = config
        self._model: Any | None = None
        self._processor: Any | None = None
        self._torch: Any | None = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def preload(self) -> None:
        if not self._config.enabled:
            return
        try:
            self.ensure_loaded()
            LOGGER.info("VLM preload completed")
        except Exception:
            LOGGER.exception(
                "VLM background preload failed; cache fast paths remain available"
            )

    def ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            try:
                import torch
                from transformers import (
                    AutoModelForImageTextToText,
                    AutoProcessor,
                    BitsAndBytesConfig,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "Transformers, Accelerate, bitsandbytes, and CUDA PyTorch are required for VLM fallback"
                ) from exc
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "Qwen3-VL fallback requires a CUDA-capable PyTorch runtime"
                )

            LOGGER.info(
                "Loading local VLM %s (%s). A first-time model download is large and slow.",
                self._config.model_id,
                self._config.quantization,
            )
            load_options: dict[str, Any] = {
                "dtype": torch.bfloat16,
                "device_map": {"": int(self._config.device.split(":")[-1])},
                "attn_implementation": "sdpa",
                "low_cpu_mem_usage": True,
                "local_files_only": self._config.local_files_only,
            }
            if self._config.quantization.casefold() == "nf4":
                load_options["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            elif self._config.quantization.casefold() not in {"none", "bf16"}:
                raise ValueError(
                    f"Unsupported VLM quantization: {self._config.quantization}"
                )

            processor = AutoProcessor.from_pretrained(
                self._config.model_id,
                local_files_only=self._config.local_files_only,
            )
            model = AutoModelForImageTextToText.from_pretrained(
                self._config.model_id,
                **load_options,
            )
            model.eval()
            self._torch = torch
            self._processor = processor
            self._model = model
            LOGGER.info("VLM loaded on %s", self._config.device)

    def resolve(self, observation: PromptObservation) -> str | None:
        if not self._config.enabled:
            return None
        self.ensure_loaded()
        assert (
            self._torch is not None
            and self._processor is not None
            and self._model is not None
        )

        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow is required for VLM image input") from exc

        rgb = Image.fromarray(observation.prompt_crop[:, :, ::-1].copy()).convert("RGB")
        if max(rgb.size) > self._config.max_image_side:
            scale = self._config.max_image_side / max(rgb.size)
            rgb = rgb.resize(
                (max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale))),
                Image.Resampling.LANCZOS,
            )

        expected = {
            "character": "the character's canonical name",
            "anime_title": "the anime's canonical English or common romanized title",
            "unknown": "the short canonical anime trivia answer",
        }[observation.expected_answer_type]
        clue_text = (
            observation.hint_text
            if observation.hint_text
            else "The clue is visual or emoji-only."
        )
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You solve Anime Soul guessing-game clues. Clues can be prose descriptions, "
                            "character images, or emoji rebuses. Use anime/manga knowledge. Return only "
                            "the requested canonical answer, with no explanation, prefix, or punctuation. "
                            "If genuinely uncertain, return UNKNOWN."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": rgb},
                    {
                        "type": "text",
                        "text": (
                            f"Expected answer: {expected}.\n"
                            f"OCR clue: {clue_text}\n"
                            "Solve this exact clue. Output only the answer."
                        ),
                    },
                ],
            },
        ]

        started = time.perf_counter()
        with self._inference_lock:
            inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self._config.device)
            prompt_length = inputs["input_ids"].shape[1]
            with self._torch.inference_mode():
                generated = self._model.generate(
                    **inputs,
                    max_new_tokens=self._config.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
            raw_answer = self._processor.batch_decode(
                generated[:, prompt_length:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        elapsed = (time.perf_counter() - started) * 1000.0
        answer = sanitize_answer(raw_answer, self._config.max_answer_characters)
        LOGGER.info(
            "VLM fallback: %.1f ms -> %s", elapsed, answer or "UNKNOWN/rejected"
        )
        return answer
