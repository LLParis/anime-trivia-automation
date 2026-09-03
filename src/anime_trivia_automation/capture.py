from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

from .config import CaptureConfig, ChangeDetectionConfig
from .models import Scene

LOGGER = logging.getLogger(__name__)


class GpuFrameChangeGate:
    """CUDA MAD/pixel-ratio comparison followed by a short visual settle gate."""

    def __init__(
        self,
        config: ChangeDetectionConfig,
        capture_size: tuple[int, int],
        on_change: Callable[[int], None] | None = None,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch is required for the CUDA frame-change gate"
            ) from exc

        if config.require_cuda and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable; the configured GPU frame-change gate cannot start"
            )
        if config.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"Frame-change device {config.device!r} is unavailable")

        self._torch = torch
        self._config = config
        self._capture_width, self._capture_height = capture_size
        self._on_change = on_change
        self._previous: Any | None = None
        self._dirty = False
        self._stable_count = 0
        self._generation = 0
        self._last_mean_delta = 1.0
        self._last_changed_ratio = 1.0

        if config.device.startswith("cuda"):
            torch.cuda.set_device(config.device)
            LOGGER.info(
                "Frame change detection: %s (%s)",
                config.device,
                torch.cuda.get_device_name(config.device),
            )

    @property
    def generation(self) -> int:
        return self._generation

    def _thumbnail(self, frame: Any) -> Any:
        torch = self._torch
        # DXcam supplies BGR uint8. Transfer the crop, convert to luminance, and
        # downsample on CUDA. The tiny synchronized result is the only readback.
        image = torch.from_numpy(frame).to(
            self._config.device, dtype=torch.float32, non_blocking=True
        )
        image = image.permute(2, 0, 1).unsqueeze(0).div_(255.0)
        gray = (
            image[:, 0:1]
            .mul(0.114)
            .add_(image[:, 1:2], alpha=0.587)
            .add_(image[:, 2:3], alpha=0.299)
        )
        target_width, target_height = self._config.thumbnail_size
        gray = torch.nn.functional.interpolate(
            gray,
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        )

        for left, top, right, bottom in self._config.ignore_regions:
            thumb_left = max(0, int(left * target_width / self._capture_width))
            thumb_right = min(
                target_width,
                int(
                    (right * target_width + self._capture_width - 1)
                    / self._capture_width
                ),
            )
            thumb_top = max(0, int(top * target_height / self._capture_height))
            thumb_bottom = min(
                target_height,
                int(
                    (bottom * target_height + self._capture_height - 1)
                    / self._capture_height
                ),
            )
            gray[:, :, thumb_top:thumb_bottom, thumb_left:thumb_right] = 0.0
        return gray

    def observe(self, frame: Any, captured_at: float) -> Scene | None:
        torch = self._torch
        with torch.inference_mode():
            current = self._thumbnail(frame)
            if self._previous is None:
                self._previous = current
                self._generation = 1
                self._dirty = True
                self._stable_count = 0
                return None

            delta = (current - self._previous).abs()
            changed_map = (delta >= self._config.changed_pixel_threshold).float()
            tile_size = self._config.localized_tile_size
            localized_ratio = torch.nn.functional.avg_pool2d(
                changed_map,
                kernel_size=tile_size,
                stride=max(1, tile_size // 2),
            ).max()
            metrics = (
                torch.stack(
                    (
                        delta.mean(),
                        changed_map.mean(),
                        localized_ratio,
                    )
                )
                .detach()
                .cpu()
                .tolist()
            )
            mean_delta = float(metrics[0])
            changed_ratio = float(metrics[1])
            localized_changed_ratio = float(metrics[2])
            self._previous = current
            self._last_mean_delta = mean_delta
            self._last_changed_ratio = changed_ratio

        changed = (
            mean_delta >= self._config.mean_absolute_threshold
            or changed_ratio >= self._config.changed_pixel_ratio
            or localized_changed_ratio >= self._config.localized_changed_ratio
        )
        if changed:
            self._generation += 1
            self._dirty = True
            self._stable_count = 0
            if self._on_change is not None:
                self._on_change(self._generation)
            return None

        if not self._dirty:
            return None

        self._stable_count += 1
        if self._stable_count < self._config.stable_frames:
            return None

        self._dirty = False
        return Scene(
            generation=self._generation,
            frame=frame.copy(),
            captured_at=captured_at,
            detected_at=time.monotonic(),
            mean_delta=self._last_mean_delta,
            changed_ratio=self._last_changed_ratio,
        )

    def force_scene(self, frame: Any, captured_at: float) -> Scene:
        """Emit a scene now for a change the thumbnail gate would not notice.

        The card's accent strip flipping colour is a few thousand pixels in a
        2.25-megapixel region, well under the thumbnail thresholds, yet it is
        the one change that opens the answer window.
        """

        self._generation += 1
        self._dirty = False
        self._stable_count = 0
        if self._on_change is not None:
            self._on_change(self._generation)
        return Scene(
            generation=self._generation,
            frame=frame.copy(),
            captured_at=captured_at,
            detected_at=time.monotonic(),
            mean_delta=self._last_mean_delta,
            changed_ratio=self._last_changed_ratio,
        )


class DXCapture:
    """Owns the DXcam instance and feeds copied BGR frames to the gate."""

    def __init__(
        self,
        config: CaptureConfig,
        on_frame: Callable[[Any, float], None],
        stop_event: threading.Event,
        on_started: Callable[[], None] | None = None,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        self._config = config
        self._on_frame = on_frame
        self._stop_event = stop_event
        self._on_started = on_started
        self._on_error = on_error
        self._camera: Any | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("DXCapture has already been started")
        self._thread = threading.Thread(
            target=self._run, name="dxcam-consumer", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        try:
            import dxcam
        except ImportError as exc:
            LOGGER.exception("DXcam import failed")
            if self._on_error is not None:
                self._on_error("DXcam import failed")
            self._stop_event.set()
            raise RuntimeError("dxcam==0.3.0 is required on Windows") from exc

        create_options: dict[str, Any] = {
            "output_color": "BGR",
            "backend": self._config.backend,
            "processor_backend": self._config.processor_backend,
            "max_buffer_len": self._config.max_buffer_len,
        }
        if self._config.device_idx is not None:
            create_options["device_idx"] = self._config.device_idx
        if self._config.output_idx is not None:
            create_options["output_idx"] = self._config.output_idx
        try:
            camera = dxcam.create(**create_options)
            with self._lock:
                self._camera = camera
            camera.start(
                region=self._config.region,
                target_fps=self._config.fps,
                video_mode=self._config.video_mode,
            )
            LOGGER.info(
                "DXcam active: region=%s fps=%d backend=%s output=%s",
                self._config.region,
                self._config.fps,
                self._config.backend,
                self._config.output_idx
                if self._config.output_idx is not None
                else "primary",
            )
            if self._on_started is not None:
                self._on_started()

            while not self._stop_event.is_set():
                packet = camera.get_latest_frame(copy=True, with_timestamp=True)
                if packet is None:
                    continue
                frame, timestamp = packet
                if frame is None:
                    continue
                self._on_frame(frame, float(timestamp))
        except Exception:
            if not self._stop_event.is_set():
                LOGGER.exception("DXcam capture loop failed")
                if self._on_error is not None:
                    self._on_error("DXcam capture loop failed")
                self._stop_event.set()
        finally:
            with self._lock:
                camera = self._camera
                self._camera = None
            if camera is not None:
                try:
                    camera.stop()
                except (RuntimeError, AttributeError):
                    LOGGER.debug("DXcam stop during cleanup failed", exc_info=True)
                try:
                    camera.release()
                except (RuntimeError, AttributeError):
                    LOGGER.debug("DXcam release during cleanup failed", exc_info=True)
            LOGGER.info("DXcam stopped")

    def request_stop(self) -> None:
        with self._lock:
            camera = self._camera
        if camera is not None:
            try:
                camera.stop()
            except (RuntimeError, AttributeError):
                LOGGER.debug("DXcam stop request failed", exc_info=True)

    def join(self, timeout: float = 5.0) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)
