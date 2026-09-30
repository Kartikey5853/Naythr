"""Configuration settings for live video call with Gemini."""

import os
from pathlib import Path


def _load_env_files():
    """Load key-value pairs from .env files."""
    candidates = [
        Path(__file__).resolve().parent / ".env",
        Path(__file__).resolve().parent.parent / ".env",
    ]
    for env_path in candidates:
        if env_path.is_file():
            try:
                for line in env_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, val = line.split("=", 1)
                    key, val = key.strip(), val.strip().strip('"').strip("'")
                    os.environ.setdefault(key, val)
            except Exception:
                pass


_load_env_files()

# Gemini Credentials & Model
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_LIVE_MODEL = os.getenv("GEMINI_LIVE_MODEL", "gemini-3.1-flash-live-preview")
FALLBACK_LIVE_MODEL = "gemini-3.8-live"

# Camera Configuration
IP_CAMERA_URL = os.getenv(
    "IP_CAMERA_URL",
    os.getenv("CAMERA_URL", "http://192.168.1.4:8080/video"),
)

# Streaming Parameters
FRAME_INTERVAL_SEC = float(os.getenv("FRAME_INTERVAL_SEC", "1.0"))  # ~1 FPS
FRAME_MAX_DIM = int(os.getenv("FRAME_MAX_DIM", "1280"))              # 1280px high-definition for text & details
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "92"))                  # High JPEG quality for OCR & reading notes

# Wake-Up Call Settings
ENABLE_WAKE_UP_CALL = os.getenv("ENABLE_WAKE_UP_CALL", "True").lower() in ("true", "1", "yes")
WAKE_UP_PROMPT = os.getenv(
    "WAKE_UP_PROMPT",
    "Wake up call: Greet the user in one short, natural sentence and confirm you can see their live camera feed."
)

# Default Assistant Persona
SYSTEM_INSTRUCTION = (
    "You are a helpful real-time AI visual assistant in a live video and voice call. "
    "You continuously receive real-time video frames from the user's camera and live audio from their microphone. "
    "When receiving a wake-up call or greeting, acknowledge concisely, confirm you are awake, and confirm you can see the camera feed. "
    "When asked questions or spoken to, respond naturally, concisely, and directly by voice to what you see and hear."
)
