import os
import cv2
import logging
import threading
import time
from typing import Optional, Tuple
import numpy as np

# Prevent FFmpeg / OpenCV network buffer buildup for IP streams
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "fflags;nobuffer|flags;low_delay|max_delay;0")

try:
    from .live_config import (
        IP_CAMERA_URL,
        FRAME_MAX_DIM,
        JPEG_QUALITY,
    )
except (ImportError, ValueError):
    from live_config import (
        IP_CAMERA_URL,
        FRAME_MAX_DIM,
        JPEG_QUALITY,
    )

try:
    from .sos_manager import CameraBlackDetector
except (ImportError, ValueError):
    from sos_manager import CameraBlackDetector

logger = logging.getLogger("CameraStream")


class CameraStream:
    """Captures frames from an IP camera (or webcam) in a background thread.
    
    Using a dedicated reader thread avoids OpenCV buffer buildup, ensuring
    every captured frame sent to Gemini is the real-time, instantaneous view.
    """

    def __init__(
        self,
        source: Optional[str] = None,
        max_dim: int = FRAME_MAX_DIM,
        jpeg_quality: int = JPEG_QUALITY,
    ):
        self.source_str = str(source or IP_CAMERA_URL).strip()
        # Allow numeric string or int for local webcam fallback (e.g. '0' -> 0)
        if self.source_str.isdigit():
            self.source = int(self.source_str)
        else:
            self.source = self.source_str

        self.max_dim = max_dim
        self.jpeg_quality = jpeg_quality

        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

        self._raw_frame: Optional[np.ndarray] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_jpeg: Optional[bytes] = None
        self._last_frame_time: float = 0.0
        self._frame_count: int = 0
        self._black_detector = CameraBlackDetector()

    def start(self) -> bool:
        """Start capturing from the camera source in a background thread."""
        if self._running:
            return True

        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True, name="CameraReader")
        self._thread.start()
        logger.info(f"Camera capture thread started for source: {self.source}")
        return True

    def _reader_loop(self):
        """Continuously drain frames from VideoCapture to maintain zero latency."""
        reconnect_delay = 2.0

        while self._running:
            if self._cap is None or not self._cap.isOpened():
                try:
                    self._cap = cv2.VideoCapture(self.source)
                    if self._cap is not None and self._cap.isOpened():
                        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception as e:
                    logger.debug(f"Connection error: {e}")

            if self._cap is None or not self._cap.isOpened():
                time.sleep(reconnect_delay)
                continue

            success, frame = self._cap.read()
            if success and frame is not None:
                # Update local camera-black confirmation detector on instantaneous frame
                self._black_detector.process_frame(frame)

                resized = self._resize_if_needed(frame)
                jpeg_bytes = self._encode_jpeg(resized)

                with self._lock:
                    self._raw_frame = frame
                    self._latest_frame = resized
                    self._latest_jpeg = jpeg_bytes
                    self._last_frame_time = time.time()
                    self._frame_count += 1
            else:
                # Brief sleep if frame couldn't be read
                time.sleep(0.05)

    def _resize_if_needed(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame so the longest dimension is <= max_dim."""
        h, w = frame.shape[:2]
        longest = max(h, w)
        if longest <= self.max_dim:
            return frame

        scale = self.max_dim / float(longest)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _encode_jpeg(self, frame: np.ndarray) -> Optional[bytes]:
        """Encode OpenCV BGR frame to JPEG bytes."""
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        success, buffer = cv2.imencode(".jpg", frame, encode_params)
        if success:
            return buffer.tobytes()
        return None

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Return a copy of the latest decoded frame, or None."""
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def get_latest_jpeg(self) -> Optional[bytes]:
        """Return the latest pre-encoded JPEG bytes, or None."""
        with self._lock:
            return self._latest_jpeg

    def get_high_res_jpeg(self, max_dim: int = 1920, quality: int = 95) -> Optional[bytes]:
        """Return a crystal-clear high-resolution JPEG of the latest frame for reading notes and OCR."""
        with self._lock:
            if self._raw_frame is None:
                return self._latest_jpeg
            frame = self._raw_frame.copy()

        h, w = frame.shape[:2]
        longest = max(h, w)
        if longest > max_dim:
            scale = max_dim / float(longest)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)

        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, buffer = cv2.imencode(".jpg", frame, encode_params)
        return buffer.tobytes() if success else None

    def is_healthy(self, timeout_sec: float = 3.0) -> bool:
        """Check if fresh frames have been received within the timeout window."""
        with self._lock:
            if self._latest_frame is None:
                return False
            return (time.time() - self._last_frame_time) < timeout_sec

    @property
    def is_black_confirmed(self) -> bool:
        """Return True if sustained black/dark camera frames have been confirmed."""
        return self._black_detector.is_confirmed_black

    def set_simulated_black(self, simulated: Optional[bool]):
        """Manually override camera black state for safe unit/integration testing."""
        self._black_detector.set_simulation(simulated)

    def get_stats(self) -> dict:
        """Get operational status and stats."""
        with self._lock:
            return {
                "source": str(self.source),
                "running": self._running,
                "healthy": (time.time() - self._last_frame_time) < 3.0 if self._last_frame_time > 0 else False,
                "frame_count": self._frame_count,
                "last_frame_age_sec": round(time.time() - self._last_frame_time, 2) if self._last_frame_time > 0 else None,
            }

    def stop(self):
        """Stop capture loop and release camera resources."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._cap:
            self._cap.release()
            self._cap = None
        logger.info("Camera stopped.")
