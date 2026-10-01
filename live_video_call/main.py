import sys
if sys.platform == "win32":
    sys.coinit_flags = 0
    try:
        from bleak.backends.winrt.util import uninitialize_sta, allow_sta
        uninitialize_sta()
        allow_sta()
    except Exception:
        pass

"""Gemini Live Video Call - IP Camera + PC Mic/Speaker with 'Hey Marvin' Wake Word.

Open this folder and run:
    python main.py
"""

# ==============================================================================
# CONFIGURATION - CHANGE YOUR SETTINGS HERE
# ==============================================================================

# IP Camera URL (e.g. "http://192.168.1.4:8080/video", "rtsp://...", or "0" for webcam)
import os
CAMERA_URL = os.getenv(
    "CAMERA_URL",
    os.getenv("IP_CAMERA_URL", "http://10.10.10.128:8080/video"),
)

# Gemini API Key (leave empty "" to automatically load from .env in the project)


# Live Model & Frame Rate
MODEL_NAME = os.getenv("GEMINI_LIVE_MODEL", "gemini-3.1-flash-live-preview")  # Fast Multimodal Live model
FRAME_RATE = 1.0                              # Video context frames sent to Gemini (~1 FPS recommended)

# Wake-Up Call Settings
ENABLE_WAKE_UP_CALL = False                   # Waits for "Hey Marvin" trigger instead of auto-speaking on connect
WAKE_UP_PROMPT = "Wake up call: Greet the user in one short sentence as Marvin and confirm you can see their live camera feed."

# Microphone & Audio Settings
ENABLE_MIC = True                             # Enable PC microphone
ENABLE_SPEAKER = True                         # Enable PC speaker audio output
MIC_THRESHOLD_DB = 55.0                       # Baseline voice sensitivity threshold in dB
SILENCE_WAIT_SECONDS = 2.8                    # Pause tolerance: wait 2.8s of quiet before ending user turn
WAKE_THRESHOLD = 0.38                         # Confidence score threshold for "Hey Marvin" in noisy environments
VAD_THRESHOLD = 0.35                          # Silero VAD threshold to reject non-speech ambient noise

# Wearable BLE Obstacle Sensor Settings (in millimeters)
SENSOR_TRIGGER_DISTANCE_MM = int(os.getenv("SENSOR_TRIGGER_DISTANCE_MM", "600"))  # Trigger alert when distance < 600 mm
SENSOR_RESET_DISTANCE_MM = int(os.getenv("SENSOR_RESET_DISTANCE_MM", "800"))      # Reset alert state to NORMAL when distance > 800 mm
SENSOR_RE_ALERT_DISTANCE_CHANGE_MM = int(os.getenv("SENSOR_RE_ALERT_DISTANCE_CHANGE_MM", "150"))  # Re-alert when distance decreases by >= 150 mm (closer)

# SOS Emergency Confirmation Layer Settings
SOS_DISTANCE_MM = int(os.getenv("SOS_DISTANCE_MM", "300"))                             # Proximity threshold for all 3 sensors (< 300 mm)
SOS_CONFIRMATION_TIMEOUT_SECONDS = float(os.getenv("SOS_CONFIRMATION_TIMEOUT_SECONDS", "7.0"))  # Seconds to wait for user confirmation

# ==============================================================================

import argparse
import asyncio
import logging
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Optional, Callable

import numpy as np
import sounddevice as sd
from google.genai import types
from openwakeword.model import Model

# Windows console encoding fix for clean terminal output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure imports work whether run as `python main.py` or `python -m live_video_call.main`
_CURRENT_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _CURRENT_DIR.parent
if str(_CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(_CURRENT_DIR))
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

try:
    from .camera_stream import CameraStream
    from .audio_stream import AudioDevice
    from .gemini_live import GeminiLiveSession
    from .live_config import GEMINI_API_KEY as ENV_GEMINI_KEY
    from .sos_manager import SosManager, default_trigger_sos
    from .intents.intent_parser import IntentParser
    from .memory.memory_manager import MemoryManager
except (ImportError, ValueError):
    from camera_stream import CameraStream
    from audio_stream import AudioDevice
    from gemini_live import GeminiLiveSession
    from live_config import GEMINI_API_KEY as ENV_GEMINI_KEY
    from sos_manager import SosManager, default_trigger_sos
    try:
        from intents.intent_parser import IntentParser
        from memory.memory_manager import MemoryManager
    except ImportError:
        IntentParser = None
        MemoryManager = None

try:
    from bluetooth_test.ble_scan import (
        start_ble_sensor_thread,
        run_ble_sensor_client,
        parse_sensor_readings,
        format_sensor_prompt,
        SensorHysteresisManager,
        SENSOR_TRIGGER_DISTANCE_MM as DEFAULT_TRIGGER_MM,
        SENSOR_RESET_DISTANCE_MM as DEFAULT_RESET_MM,
        SENSOR_RE_ALERT_DISTANCE_CHANGE_MM as DEFAULT_RE_ALERT_MM,
    )
except ImportError:
    start_ble_sensor_thread = None
    run_ble_sensor_client = None
    parse_sensor_readings = None
    format_sensor_prompt = None
    SensorHysteresisManager = None
    DEFAULT_TRIGGER_MM = 600
    DEFAULT_RESET_MM = 800
    DEFAULT_RE_ALERT_MM = 150

# Silence verbose background logs
logging.basicConfig(level=logging.WARNING)

# Marvin ONNX Model Path
MARVIN_MODEL_PATH = _ROOT_DIR / "models" / "hey_marvin_v0.1.onnx"
if not MARVIN_MODEL_PATH.exists():
    MARVIN_MODEL_PATH = _CURRENT_DIR / "models" / "hey_marvin_v0.1.onnx"
if not MARVIN_MODEL_PATH.exists():
    MARVIN_MODEL_PATH = Path("models/hey_marvin_v0.1.onnx").resolve()

# Assistant Persona
MARVIN_SYSTEM_PROMPT = (
    "You are Marvin, a highly perceptive real-time AI visual assistant in a live video and voice call with the user. "
    "Your name is Marvin. When asked who you are, what your name is, or to introduce yourself, always state clearly that you are Marvin. "
    "You continuously receive real-time high-resolution video frames from the user's camera and live audio from their microphone. "
    "Carefully inspect the visual details. You have high-accuracy vision and OCR: "
    "When the user shows a note, paper, notebook, book, screen, label, sign, or currency, accurately read the exact visible text, handwriting, numbers, and content. "
    "If asked 'what do you see' or 'read this', focus on the main subject or note in the center of the frame and read it clearly and accurately. "
    "For navigation or walking questions, mention immediate hazards and direction. "
    "When you receive an automated '[SYSTEM SENSOR EVENT]' indicating an obstacle from wearable sensors (LEFT, CENTER, or RIGHT), "
    "immediately assess the CURRENT LIVE CAMERA VIEW and determine what is causing the detection. "
    "Use both the sensor direction/distance and the current camera view to identify the obstacle, and give ONE extremely short navigation instruction (approximately 5-12 words). "
    "Examples: 'Person on your left. Move right.', 'Wall on your left. Move right.', 'Person ahead. Stop.', 'Obstacle ahead. Move left.', 'Chair on your right. Move left.'. "
    "Prioritize immediate navigation information. Do not explain your reasoning. Do not mention the internal sensor event. "
    "When you receive an automated '[SYSTEM SOS CHECK]', speak ONLY the exact question: 'Are you okay?'. Do not explain technical details or sensor readings. "
    "When you receive an automated '[SYSTEM EMERGENCY SOS ACTIVATED]', immediately speak aloud to the user in a calm, clear, reassuring, and comforting voice: "
    "confirm that emergency SOS is active, let them know emergency contacts have been notified with their live location, and advise them to remain calm. Do not read raw URL strings. "
    "Respond naturally, directly, and concisely by voice to what you see and hear."
)


def get_api_key(cli_key: str = "") -> str:
    """Resolve API key: Top of file > CLI arg > .env file."""
    if GEMINI_API_KEY.strip():
        return GEMINI_API_KEY.strip()
    if cli_key and cli_key.strip():
        return cli_key.strip()
    if ENV_GEMINI_KEY and ENV_GEMINI_KEY.strip():
        return ENV_GEMINI_KEY.strip()
    return os.getenv("GEMINI_API_KEY", "").strip()


class MarvinWakeWordDetector:
    """Robust 'Hey Marvin' detector with Silero VAD to reject background crowd noise."""

    def __init__(
        self,
        model_path: str | Path,
        threshold: float = WAKE_THRESHOLD,
        vad_threshold: float = VAD_THRESHOLD,
    ):
        self.model_path = str(model_path)
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"Marvin model ONNX file not found at: {self.model_path}")

        print(f"[INIT] Loading Marvin wake word model ({Path(self.model_path).name}) with VAD={vad_threshold}...")
        self.model = Model(
            wakeword_models=[self.model_path],
            inference_framework="onnx",
            vad_threshold=vad_threshold,
        )
        self.model_key = "hey_marvin_v0.1"
        self.threshold = threshold
        self.cooldown = 0
        self.current_score: float = 0.0
        print("[INIT] Marvin wake word model ready!")

    def detect(self, samples: np.ndarray) -> bool:
        chunk = samples.flatten()
        prediction = self.model.predict(chunk)
        score = prediction.get(self.model_key, 0.0)
        if not score and prediction:
            score = next(iter(prediction.values()), 0.0)

        self.current_score = float(score)

        if self.cooldown > 0:
            self.cooldown -= 1
            return False

        if score >= self.threshold:
            self.cooldown = 32  # ~2.5s cooldown at 80ms/chunk
            return True

        return False


class GatedAudioDevice(AudioDevice):
    """Audio device that gates mic audio on 'Hey Marvin' and adapts to crowded noise floors."""

    STATE_WAITING = "WAITING FOR TRIGGER"
    STATE_SPEAK = "SPEAK"
    STATE_RESPONDING = "RESPONDING"

    def __init__(
        self,
        wake_detector: MarvinWakeWordDetector,
        threshold_db: float = MIC_THRESHOLD_DB,
        silence_wait_sec: float = SILENCE_WAIT_SECONDS,
        no_speech_timeout: float = 8.0,
        max_speak_duration: float = 25.0,
    ):
        # OpenWakeWord operates on 1280 samples at 16kHz (80 ms)
        super().__init__(
            input_sample_rate=16000,
            output_sample_rate=24000,
            chunk_size=1280,
            threshold_db=threshold_db,
        )
        self.wake_detector = wake_detector
        self.silence_wait_sec = silence_wait_sec
        self.no_speech_timeout = no_speech_timeout
        self.max_speak_duration = max_speak_duration

        self.state = self.STATE_WAITING
        self.user_has_spoken = False
        self.last_speech_time: Optional[float] = None
        self.speak_start_time: Optional[float] = None

        # Dynamic ambient noise floor tracking (exponential moving average)
        self.ambient_rms: float = 80.0
        self.speech_energy_ratio: float = 1.6  # Speech must be 1.6x ambient RMS

        # Callbacks dispatched thread-safely
        self.on_wake_detected: Optional[Callable[[], None]] = None
        self.on_speech_finished: Optional[Callable[[], None]] = None
        self.on_speech_timeout: Optional[Callable[[], None]] = None
        self.on_state_change: Optional[Callable[[str], None]] = None

    def set_state(self, new_state: str):
        """Thread-safe state change."""
        if self.state == new_state:
            return
        self.state = new_state
        if new_state == self.STATE_SPEAK:
            self.user_has_spoken = False
            self.last_speech_time = None
            self.speak_start_time = time.time()
        elif new_state == self.STATE_WAITING:
            self.user_has_spoken = False
            self.last_speech_time = None
            self.speak_start_time = None

        if self._loop and self.on_state_change and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self.on_state_change, new_state)

    def _mic_callback(self, indata, frames, time_info, status):
        """Callback for microphone chunks."""
        if not self._running:
            return

        samples = np.frombuffer(indata, dtype=np.int16)
        rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
        db = float(20.0 * np.log10(max(1.0, rms)))

        self.current_rms = rms
        self.current_db = db

        # Track ambient noise floor adaptively during quiet or waiting periods
        if self.state == self.STATE_WAITING or (self.state == self.STATE_SPEAK and not self.user_has_spoken):
            # Slow adaptation to current room volume
            self.ambient_rms = 0.96 * self.ambient_rms + 0.04 * rms

        # Speech threshold adapts dynamically to ambient noise
        adaptive_speech_threshold = max(220.0, self.ambient_rms * self.speech_energy_ratio)
        is_speech_active = rms >= adaptive_speech_threshold or db >= self.threshold_db
        self.is_voice_detected = is_speech_active

        chunk_bytes = bytes(indata)

        # --------------------------------------------------------------
        # STATE 1: WAITING FOR TRIGGER ("Hey Marvin")
        # --------------------------------------------------------------
        if self.state == self.STATE_WAITING:
            detected = self.wake_detector.detect(samples)
            if detected:
                self.set_state(self.STATE_SPEAK)
                if self._loop and self.on_wake_detected and not self._loop.is_closed():
                    self._loop.call_soon_threadsafe(self.on_wake_detected)
            return

        # --------------------------------------------------------------
        # STATE 2: SPEAK (Streaming user voice to Gemini)
        # --------------------------------------------------------------
        if self.state == self.STATE_SPEAK:
            now = time.time()

            # Stream audio chunk to Gemini Live
            if self._loop and self._in_queue and not self._loop.is_closed():
                try:
                    self._loop.call_soon_threadsafe(self._in_queue.put_nowait, chunk_bytes)
                except Exception:
                    pass

            # Voice activity check with adaptive threshold
            if is_speech_active:
                self.user_has_spoken = True
                self.last_speech_time = now

            if self.user_has_spoken and self.last_speech_time:
                silence_duration = now - self.last_speech_time
                # Only end turn after generous pause (2.8s) of quiet
                if silence_duration >= self.silence_wait_sec or (
                    self.speak_start_time and (now - self.speak_start_time) >= self.max_speak_duration
                ):
                    self.set_state(self.STATE_RESPONDING)
                    if self._loop and self.on_speech_finished and not self._loop.is_closed():
                        self._loop.call_soon_threadsafe(self.on_speech_finished)
            else:
                # Trigger fired but user did not speak within timeout window
                if self.speak_start_time and (now - self.speak_start_time) >= self.no_speech_timeout:
                    self.set_state(self.STATE_WAITING)
                    if self._loop and self.on_speech_timeout and not self._loop.is_closed():
                        self._loop.call_soon_threadsafe(self.on_speech_timeout)
            return

        # --------------------------------------------------------------
        # STATE 3: RESPONDING (Gemini is answering)
        # --------------------------------------------------------------
        if self.state == self.STATE_RESPONDING:
            # Microphone is completely muted to Gemini while Marvin responds
            # to prevent speaker audio feedback or false barge-in interruptions
            return


async def run_live_call(
    camera_source: str,
    api_key: str,
    model: str,
    fps: float,
    threshold_db: float = MIC_THRESHOLD_DB,
    silence_wait_sec: float = SILENCE_WAIT_SECONDS,
    enable_wake_up_call: bool = ENABLE_WAKE_UP_CALL,
    wake_up_prompt: str = WAKE_UP_PROMPT,
):
    """Main live session combining Camera, Marvin Wake Word, and Gemini Live."""
    frame_interval = 1.0 / max(0.1, fps)

    print("=" * 70)
    print(" [MARVIN LIVE CALL] Gemini Live + IP Camera + 'Hey Marvin' Trigger")
    print("=" * 70)

    # 1. Initialize Marvin Wake Word Detector with VAD for crowded rooms
    print(f"\n[1/3] Loading Wake Word Detector: {MARVIN_MODEL_PATH.name}...")
    wake_detector = MarvinWakeWordDetector(
        model_path=MARVIN_MODEL_PATH,
        threshold=WAKE_THRESHOLD,
        vad_threshold=VAD_THRESHOLD,
    )

    # 2. Initialize Camera
    print(f"\n[2/3] Connecting to Camera: {camera_source}")
    camera = CameraStream(source=camera_source)
    camera.start()

    for _ in range(6):
        if camera.is_healthy():
            break
        await asyncio.sleep(0.25)

    if camera.is_healthy():
        print("   [+] Camera connected and streaming live frames.")
    else:
        print(f"   [!] Camera at '{camera_source}' not detected yet (auto-reconnecting).")

    # 3. Initialize Gated Audio
    print(f"\n[3/3] Initializing Gated Audio (Silence Wait: {silence_wait_sec:.1f}s)...")
    audio = GatedAudioDevice(
        wake_detector=wake_detector,
        threshold_db=threshold_db,
        silence_wait_sec=silence_wait_sec,
    )

    # 4. Connect to Gemini Live Session
    print(f"\nConnecting to Gemini Live ({model})...")
    session = GeminiLiveSession(
        api_key=api_key,
        model=model,
        camera=camera,
        audio=audio,
        frame_interval_sec=frame_interval,
        system_instruction=MARVIN_SYSTEM_PROMPT,
    )

    intent_parser = IntentParser() if IntentParser else None
    memory_manager = MemoryManager() if MemoryManager else None
    sos_manager: Optional[SosManager] = None

    def on_user_speech(transcript: str):
        print(f"\nYou (Voice) > {transcript}")
        if sos_manager and sos_manager.state == SosManager.STATE_SOS_CHECKING:
            async def _check_sos_voice_response():
                handled = await sos_manager.handle_user_response(transcript)
                if handled and sos_manager.state == SosManager.STATE_NORMAL:
                    audio.set_state(GatedAudioDevice.STATE_WAITING)
                elif handled and sos_manager.state == SosManager.STATE_SOS_ACTIVE:
                    audio.set_state(GatedAudioDevice.STATE_RESPONDING)
            asyncio.create_task(_check_sos_voice_response())

        # Action + Memory Intent Extraction
        if intent_parser and memory_manager and transcript.strip():
            res = intent_parser.parse(transcript)
            if res.should_store:
                record = memory_manager.process_intent(res, source="user_voice")
                if record:
                    print(f"[ACTION/MEMORY] Recorded {res.category} ({res.intent}) -> ID: {record['id']}")
                    if res.requires_clarification and res.clarification_prompt:
                        print(f"[ACTION/MEMORY] Clarification needed: {res.clarification_prompt}")

    def on_gemini_speech(chunk: str):
        nonlocal gemini_is_speaking_turn
        if not gemini_is_speaking_turn:
            print("\nMarvin > ", end="", flush=True)
            gemini_is_speaking_turn = True
        print(chunk, end="", flush=True)

    def on_turn_complete():
        nonlocal gemini_is_speaking_turn
        if gemini_is_speaking_turn:
            print("\n")
            gemini_is_speaking_turn = False
        if sos_manager and sos_manager.state == SosManager.STATE_SOS_CHECKING:
            audio.set_state(GatedAudioDevice.STATE_SPEAK)
            print("[SOS] Microphone open - listening for response (say 'Yes', 'I'm okay', 'No', or 'Help')...")
        elif sos_manager and sos_manager.state == SosManager.STATE_SOS_ACTIVE:
            audio.set_state(GatedAudioDevice.STATE_SPEAK)
            print("[SOS ACTIVE] Emergency mode active - Marvin is listening (speak anytime without 'Hey Marvin')...")
        else:
            # Reset back to standby listening for the next trigger
            audio.set_state(GatedAudioDevice.STATE_WAITING)
            print('[STANDBY] Say "Hey Marvin"')

    def on_interrupted():
        nonlocal gemini_is_speaking_turn
        if gemini_is_speaking_turn:
            print("\n[Interrupted]")
            gemini_is_speaking_turn = False
        if sos_manager and sos_manager.state == SosManager.STATE_SOS_ACTIVE:
            audio.set_state(GatedAudioDevice.STATE_SPEAK)
        else:
            audio.set_state(GatedAudioDevice.STATE_WAITING)

    session.on_user_speech = on_user_speech
    session.on_gemini_speech = on_gemini_speech
    session.on_turn_complete = on_turn_complete
    session.on_interrupted = on_interrupted

    # State transition event handlers
    def handle_wake_detected():
        print("\n" + "=" * 50)
        print(" >>> [TRIGGER DETECTED] \"Hey Marvin\" <<<")
        print(" [SPEAK] Listening to your question (take your time)...")
        print("=" * 50 + "\n")
        # Send fresh high-resolution camera frame immediately for OCR and reading notes
        async def _push_fresh_frame():
            if camera and getattr(session, "_session", None) and session.is_connected:
                fresh_jpeg = (
                    camera.get_high_res_jpeg(max_dim=1920, quality=95)
                    if hasattr(camera, "get_high_res_jpeg")
                    else camera.get_latest_jpeg()
                )
                if fresh_jpeg:
                    try:
                        await session._session.send_realtime_input(
                            video=types.Blob(data=fresh_jpeg, mime_type="image/jpeg")
                        )
                    except Exception:
                        pass
        asyncio.create_task(_push_fresh_frame())

    def handle_speech_finished():
        nonlocal gemini_is_speaking_turn
        gemini_is_speaking_turn = False
        print("\n[RESPONDING] Processing your question...")
        async def _signal_speech_end():
            # 1. Drain any remaining chunks in the mic queue so no trailing audio is sent
            # to the server while Gemini is already speaking (which triggers false barge-in)
            if hasattr(session, "_mic_queue") and session._mic_queue:
                while not session._mic_queue.empty():
                    try:
                        session._mic_queue.get_nowait()
                        session._mic_queue.task_done()
                    except Exception:
                        break

            # 2. Tell Gemini the user has stopped speaking
            if getattr(session, "_session", None) and session.is_connected:
                try:
                    await session._session.send_realtime_input(audio_stream_end=True)
                except Exception:
                    pass
        asyncio.create_task(_signal_speech_end())

    def handle_speech_timeout():
        if sos_manager and sos_manager.state == SosManager.STATE_SOS_ACTIVE:
            print('[SOS ACTIVE] Still listening for user (mic remains open)...')
        else:
            print('[STANDBY] No speech detected after trigger. Say "Hey Marvin"')

    audio.on_wake_detected = handle_wake_detected
    audio.on_speech_finished = handle_speech_finished
    audio.on_speech_timeout = handle_speech_timeout

    try:
        await session.connect()
        print(f"[+] Connected to Gemini Live ({session.model})!\n")
    except Exception as e:
        print(f"[-] Failed to connect: {e}")
        audio.stop()
        camera.stop()
        return

    # Wearable Bluetooth ESP32 Sensor Integration with Hysteresis & Concurrency Lock
    sensor_send_lock = asyncio.Lock()
    manual_hysteresis = (
        SensorHysteresisManager(
            trigger_distance_mm=SENSOR_TRIGGER_DISTANCE_MM,
            reset_distance_mm=SENSOR_RESET_DISTANCE_MM,
            re_alert_distance_change_mm=SENSOR_RE_ALERT_DISTANCE_CHANGE_MM,
        )
        if SensorHysteresisManager
        else None
    )

    # 5. Initialize SOS Emergency Confirmation Manager
    sos_manager = SosManager(
        sos_distance_mm=SOS_DISTANCE_MM,
        timeout_seconds=SOS_CONFIRMATION_TIMEOUT_SECONDS,
        send_gemini_prompt=lambda p: session.send_text(p) if (getattr(session, "_session", None) and session.is_connected) else None,
        trigger_sos_callback=default_trigger_sos,
        unmute_mic_callback=lambda: audio.set_state(GatedAudioDevice.STATE_SPEAK),
    )

    async def handle_sensor_obstacle(direction: str, distance_mm: float = 400.0):
        nonlocal gemini_is_speaking_turn

        # If SOS is currently checking or active, suppress normal navigation obstacles
        if sos_manager and sos_manager.state in (SosManager.STATE_SOS_CHECKING, SosManager.STATE_SOS_ACTIVE):
            return

        # Prevent multiple simultaneous Gemini sensor requests
        if sensor_send_lock.locked():
            return

        async with sensor_send_lock:
            dir_clean = direction.upper().strip()
            dist_int = int(round(distance_mm))

            # Required console log: [GEMINI] Sending sensor event: LEFT
            print(f"[GEMINI] Sending sensor event: {dir_clean}")

            prompt = (
                format_sensor_prompt(dir_clean, distance_mm)
                if format_sensor_prompt
                else (
                    f"[SYSTEM SENSOR EVENT]\n\n"
                    f"The wearable's {dir_clean} sensor detected an obstacle at {dist_int} mm.\n\n"
                    f"This is an automatic hardware event.\n"
                    f"It was NOT spoken or typed by the user.\n\n"
                    f"IMPORTANT:\n"
                    f"Immediately assess the CURRENT LIVE CAMERA VIEW and determine what is causing the sensor detection.\n\n"
                    f"Use both:\n"
                    f"1. The sensor information:\n"
                    f"   - direction: {dir_clean}\n"
                    f"   - distance: {dist_int} mm\n"
                    f"2. The latest available camera view.\n\n"
                    f"Do not rely only on the sensor distance.\n"
                    f"Do not assume what the object is.\n"
                    f"Use the camera to identify it whenever possible.\n\n"
                    f"Then give ONE extremely short navigation instruction (approximately 5-12 words).\n"
                    f"Prioritize immediate navigation information.\n"
                    f"Do not explain your reasoning.\n"
                    f"Do not mention the internal sensor event."
                )
            )

            # Ensure the existing Live session context has the absolute freshest, unbuffered camera frame
            # at the exact moment this sensor event is processed:
            if camera and getattr(session, "_session", None) and session.is_connected:
                fresh_jpeg = camera.get_latest_jpeg()
                if fresh_jpeg:
                    try:
                        await session._session.send_realtime_input(
                            video=types.Blob(data=fresh_jpeg, mime_type="image/jpeg")
                        )
                    except Exception as e:
                        logger.debug(f"Fresh frame send error before sensor alert: {e}")

            gemini_is_speaking_turn = False
            # Mute microphone to prevent Gemini hearing its own alert
            audio.set_state(GatedAudioDevice.STATE_RESPONDING)
            try:
                await session.send_text(prompt)
                # Required console log: [GEMINI] Sensor event sent
                print(f"[GEMINI] Sensor event sent")
            except Exception as e:
                print(f"[-] Error injecting sensor event to Gemini: {e}")

    # Launch background BLE sensor listener in dedicated MTA worker thread (resolves Windows COM STA issue)
    ble_stop_event = threading.Event()
    ble_thread = None
    if start_ble_sensor_thread:
        ble_thread = start_ble_sensor_thread(
            on_sensor_trigger=handle_sensor_obstacle,
            on_raw_reading=sos_manager.update_sensor_reading,
            main_loop=asyncio.get_running_loop(),
            stop_event=ble_stop_event,
            trigger_distance_mm=SENSOR_TRIGGER_DISTANCE_MM,
            reset_distance_mm=SENSOR_RESET_DISTANCE_MM,
            re_alert_distance_change_mm=SENSOR_RE_ALERT_DISTANCE_CHANGE_MM,
        )

    # Background monitor loop for emergency camera black & all-sensor conditions
    async def _sos_monitor_loop():
        while session.is_connected:
            try:
                await asyncio.sleep(0.2)
                cam_black = camera.is_black_confirmed if camera else False
                if sos_manager:
                    await sos_manager.poll_check(cam_black)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"SOS monitor loop error: {e}")

    sos_monitor_task = asyncio.create_task(_sos_monitor_loop())

    # Call Active Screen
    print("=" * 70)
    print(" >>> CALL IN PROGRESS <<<")
    print(" * Say \"Hey Marvin\" to speak - mic audio is gated until triggered!")
    print(f" * Wearable BLE Obstacle Nav: Trigger < {SENSOR_TRIGGER_DISTANCE_MM}mm | Reset > {SENSOR_RESET_DISTANCE_MM}mm | Re-Alert: -{SENSOR_RE_ALERT_DISTANCE_CHANGE_MM}mm closer.")
    print(f" * Emergency SOS Layer: Active (Camera Dark -> Confirmation {SOS_CONFIRMATION_TIMEOUT_SECONDS}s).")
    print(" * Test fake sensor values anytime by typing:")
    print("     - 'sensor left 400', 'sensor center 350', or 'sensor right 500'")
    print("     - 'sensor reset' (resets hysteresis states)")
    print("     - or raw JSON: '{\"sensor\": \"LEFT\", \"distance_mm\": 400}'")
    print(" * SOS controls & simulations:")
    print("     - 'sos status' (check SOS state and sensor distances)")
    print("     - 'sos reset' (reset SOS state to NORMAL)")
    print("     - 'sos trigger' (force immediate SOS trigger)")
    print("     - 'sos black on/off' (simulate camera dark condition)")
    print("     - 'sos sensors <mm>' (e.g. 'sos sensors 200' to set all sensors below threshold)")
    print("     - 'sos test' (run automated 7-test simulation suite)")
    print(" * Action + Memory Layer: Active (Medications, Appointments, Transport).")
    print(" * Memory controls:")
    print("     - 'memory status' (view memory statistics)")
    print("     - 'memory actions' (view pending/recorded actions)")
    print("     - 'memory medications' (view active medications)")
    print("     - 'memory appointments' (view appointments)")
    print("     - 'memory clear-test-data' (reset memory storage)")
    print("     - 'test intent <query>' (test intent parsing and extraction)")
    print(" * Camera streams context continuously (~1 FPS).")
    print(" * You can also type questions below and press Enter anytime.")
    print(" * Type 'status' for stats, or 'exit' to quit.")
    loop = asyncio.get_running_loop()

    # Clean asynchronous keyboard input queue
    input_queue = asyncio.Queue()

    def keyboard_reader():
        while session.is_connected:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                loop.call_soon_threadsafe(input_queue.put_nowait, line.strip())
            except Exception:
                break

    kb_thread = threading.Thread(target=keyboard_reader, daemon=True, name="KeyboardReader")
    kb_thread.start()

    try:
        while session.is_connected:
            try:
                text = await asyncio.wait_for(input_queue.get(), timeout=0.25)
            except asyncio.TimeoutError:
                continue

            text = text.strip().strip('"').strip("'")
            if not text:
                continue

            if text.lower() in {"exit", "quit", "q"}:
                break

            if text.lower() in {"wake", "wakeup", "wake up"}:
                handle_wake_detected()
                continue

            # If SOS confirmation is currently waiting for user response, handle it
            if sos_manager and sos_manager.state == SosManager.STATE_SOS_CHECKING:
                handled = await sos_manager.handle_user_response(text)
                if handled:
                    if sos_manager.state == SosManager.STATE_NORMAL:
                        audio.set_state(GatedAudioDevice.STATE_WAITING)
                    elif sos_manager.state == SosManager.STATE_SOS_ACTIVE:
                        audio.set_state(GatedAudioDevice.STATE_RESPONDING)
                    continue

            # SOS CLI control and simulation commands
            low_text = text.lower()
            if low_text.startswith("sos"):
                sos_parts = low_text.split()
                sub = sos_parts[1] if len(sos_parts) > 1 else "status"

                if sub == "status":
                    cam_b = camera.is_black_confirmed if camera else False
                    print(f"\n--- SOS Status ---")
                    print(f"State:            {sos_manager.state if sos_manager else 'N/A'}")
                    print(f"Camera Dark:      {cam_b}")
                    if sos_manager:
                        dists = sos_manager.latest_sensor_distances
                        print(f"Sensors:          LEFT={dists['LEFT']}mm, CENTER={dists['CENTER']}mm, RIGHT={dists['RIGHT']}mm (Mode: Camera Dark Only)")
                    print("------------------\n")
                    continue

                elif sub == "reset":
                    if sos_manager:
                        sos_manager.reset()
                    if camera:
                        camera.set_simulated_black(False)
                    audio.set_state(GatedAudioDevice.STATE_WAITING)
                    print("[SOS] System reset to NORMAL. Simulated camera black cleared.")
                    continue

                elif sub == "trigger":
                    if sos_manager:
                        print("[SOS] Manual emergency trigger requested.")
                        sos_manager.state = SosManager.STATE_SOS_ACTIVE
                        audio.set_state(GatedAudioDevice.STATE_RESPONDING)
                        await sos_manager._invoke_trigger_sos(reason="CLI manual test trigger")
                    continue

                elif sub == "black":
                    turn_on = len(sos_parts) > 2 and sos_parts[2] in {"on", "true", "1"}
                    if camera:
                        camera.set_simulated_black(turn_on)
                    print(f"[SOS] Simulated camera dark: {'ON' if turn_on else 'OFF'}")
                    continue

                elif sub == "sensors":
                    s_val = float(sos_parts[2]) if (len(sos_parts) > 2 and sos_parts[2].replace(".", "", 1).isdigit()) else 200.0
                    if sos_manager:
                        sos_manager.update_sensor_reading("LEFT", s_val)
                        sos_manager.update_sensor_reading("CENTER", s_val)
                        sos_manager.update_sensor_reading("RIGHT", s_val)
                    print(f"[SOS] Updated all sensors to {s_val:.0f} mm")
                    continue

                elif sub == "test":
                    from sos_manager import run_unit_tests
                    await run_unit_tests()
                    continue

            # Action + Memory CLI commands
            if low_text.startswith("memory"):
                mem_parts = low_text.split()
                sub = mem_parts[1] if len(mem_parts) > 1 else "status"

                if not memory_manager:
                    print("[MEMORY] MemoryManager module not initialized.")
                    continue

                if sub == "status":
                    dash = memory_manager.get_frontend_dashboard_data()
                    print("\n--- Naythr Action & Memory Status ---")
                    print(f"Data Directory:   {memory_manager.data_dir}")
                    print(f"Pending Actions:  {dash['counts']['pending_actions']}")
                    print(f"Active Meds:      {dash['counts']['medications']}")
                    print(f"Appointments:     {dash['counts']['appointments']}")
                    print(f"Recent Activity:  {len(dash['recent_activity'])} entries")
                    print("-------------------------------------\n")
                    continue

                elif sub == "actions":
                    actions = memory_manager.get_actions()
                    print(f"\n--- Stored Actions ({len(actions)}) ---")
                    if not actions:
                        print(" (No actions recorded yet)")
                    for a in actions:
                        print(f" [{a['id']}] {a['type']} ({a['intent']}) - Status: {a['status']}")
                        print(f"   Request:  '{a['user_request']}'")
                        print(f"   Entities: {json.dumps(a['entities'])}")
                    print("---------------------------------------\n")
                    continue

                elif sub == "medications":
                    meds = memory_manager.get_medications()
                    print(f"\n--- Active Medications ({len(meds)}) ---")
                    if not meds:
                        print(" (No medications recorded yet)")
                    for m in meds:
                        info = m['medication']
                        print(f" [{m['id']}] {info.get('name', 'Unknown').title()} - Status: {m['status']}")
                        print(f"   Frequency: {info.get('frequency') or 'None'} | Dosage: {info.get('dosage') or 'None'} | Duration: {info.get('duration') or 'None'}")
                    print("----------------------------------------\n")
                    continue

                elif sub == "appointments":
                    appts = memory_manager.get_appointments()
                    print(f"\n--- Recorded Appointments ({len(appts)}) ---")
                    if not appts:
                        print(" (No appointments recorded yet)")
                    for ap in appts:
                        ent = ap['entities']
                        print(f" [{ap['id']}] {ent.get('place', 'Unknown')} - Status: {ap['status']}")
                        print(f"   Date: {ent.get('date') or 'None'} | Time: {ent.get('time') or 'None'}")
                    print("-------------------------------------------\n")
                    continue

                elif sub in {"clear-test-data", "reset", "clear"}:
                    memory_manager.reset_memory()
                    print("[MEMORY] All memory files and activity history cleared.")
                    continue

            # Testing Intent Extraction CLI command
            if low_text.startswith("test intent"):
                query = text[11:].strip().strip('"').strip("'")
                if not query:
                    print("[INTENT] Please provide text to parse: test intent <text>")
                    continue
                if intent_parser:
                    res = intent_parser.parse(query)
                    print("\n--- Intent Extraction Result ---")
                    print(f"Utterance:     '{query}'")
                    print(f"Should Store:  {res.should_store}")
                    print(f"Category:      {res.category}")
                    print(f"Intent:        {res.intent}")
                    print(f"Confidence:    {res.confidence}")
                    print(f"Entities:      {json.dumps(res.entities, indent=2)}")
                    print(f"Clarification: {res.requires_clarification} ({res.clarification_prompt})")
                    print("--------------------------------\n")
                    if res.should_store and memory_manager:
                        rec = memory_manager.process_intent(res, source="cli_test")
                        if rec:
                            print(f"[MEMORY] Successfully stored record -> ID: {rec['id']}\n")
                continue

            # Sensor simulation triggers for CLI & Web Dashboard testing
            if low_text.startswith("sensor ") or low_text in {"left", "center", "right", "front"}:
                parts = low_text.split()
                if len(parts) >= 2:
                    action_dir = parts[1].upper()
                    dist = float(parts[2]) if (len(parts) >= 3 and parts[2].replace(".", "", 1).isdigit()) else 400.0
                else:
                    action_dir = low_text.upper()
                    dist = 400.0

                if action_dir in {"RESET", "CLEAR"}:
                    if manual_hysteresis:
                        manual_hysteresis.reset_all()
                    print("[SENSOR] All sensor alert states reset to NORMAL.")
                    continue

                if action_dir in {"LEFT", "CENTER", "RIGHT", "FRONT"}:
                    target_dir = "CENTER" if action_dir in ("CENTER", "FRONT") else action_dir
                    if sos_manager:
                        sos_manager.update_sensor_reading(target_dir, dist)
                    if manual_hysteresis:
                        should_trigger = manual_hysteresis.process_reading(target_dir, dist)
                        if should_trigger:
                            await handle_sensor_obstacle(target_dir, dist)
                    else:
                        await handle_sensor_obstacle(target_dir, dist)
                    continue

            # Raw JSON simulation from CLI/Dashboard: e.g. {"sensor": 2, "position": "Left", "distance_mm": 123, "motor_triggered": true}
            if text.startswith("{") and text.endswith("}"):
                if parse_sensor_readings and manual_hysteresis:
                    print(f"[BLE RAW] {text}")
                    fake_readings = parse_sensor_readings(text)
                    for d, d_dist in fake_readings.items():
                        if sos_manager:
                            sos_manager.update_sensor_reading(d, d_dist)
                        if manual_hysteresis.process_reading(d, d_dist):
                            await handle_sensor_obstacle(d, d_dist)
                    continue

            if text.lower() == "status":
                stats = camera.get_stats()
                mic_db = audio.current_db if audio else 0.0
                ambient = 20.0 * np.log10(max(1.0, audio.ambient_rms)) if audio else 0.0
                mem_counts = memory_manager.get_frontend_dashboard_data()["counts"] if memory_manager else {}
                print(f"\n--- System Status ---")
                print(f"State:        {audio.state}")
                print(f"SOS State:    {sos_manager.state if sos_manager else 'N/A'}")
                print(f"Camera Dark:  {camera.is_black_confirmed if camera else False}")
                print(f"Actions/Mem:  Pending: {mem_counts.get('pending_actions', 0)} | Meds: {mem_counts.get('medications', 0)} | Appts: {mem_counts.get('appointments', 0)}")
                print(f"Camera:       {'Active [OK]' if stats['healthy'] else 'No Signal [!]'}")
                print(f"BLE Client:   {'Active' if ble_thread and ble_thread.is_alive() else 'Inactive'}")
                print(f"Frames Sent:  {session.frames_sent}")
                print(f"Mic Level:    {mic_db:.1f} dB (Ambient Noise Floor: {ambient:.1f} dB)")
                print(f"Pause Wait:   {silence_wait_sec:.1f}s")
                print(f"Gemini Model: {session.model}")
                print("---------------------\n")
                continue

            # Process intent from typed user inquiry before sending to Gemini
            if intent_parser and memory_manager:
                t_res = intent_parser.parse(text)
                if t_res.should_store:
                    t_rec = memory_manager.process_intent(t_res, source="user_typed")
                    if t_rec:
                        print(f"[ACTION/MEMORY] Recorded {t_res.category} ({t_res.intent}) -> ID: {t_rec['id']}")
                        if t_res.requires_clarification and t_res.clarification_prompt:
                            print(f"[ACTION/MEMORY] Needs info: {t_res.clarification_prompt}")

            # User typed a question directly
            gemini_is_speaking_turn = False
            print(f"\nYou (Typed) > {text}")
            audio.set_state(GatedAudioDevice.STATE_RESPONDING)
            try:
                await session.send_text(text)
            except Exception as e:
                print(f"\n[-] Error sending text: {e}\n")

    finally:
        print("\n\nEnding live call...")
        if sos_monitor_task and not sos_monitor_task.done():
            sos_monitor_task.cancel()
        if ble_stop_event:
            ble_stop_event.set()
        await session.close()
        camera.stop()
        print("Call ended. Goodbye!")


def main():
    parser = argparse.ArgumentParser(description="Gemini Live Video Call with IP Camera & 'Hey Marvin' Wake Word")
    parser.add_argument(
        "--camera",
        type=str,
        default=CAMERA_URL,
        help=f"IP Camera URL or webcam index (default: {CAMERA_URL})",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="",
        help="Gemini API Key (default: loaded from top of file or .env)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help=f"Gemini Live Model (default: {MODEL_NAME})",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=FRAME_RATE,
        help=f"Video frame rate sent to Gemini (default: {FRAME_RATE} FPS)",
    )
    parser.add_argument(
        "--threshold-db",
        type=float,
        default=MIC_THRESHOLD_DB,
        help=f"Baseline mic dB sensitivity threshold (default: {MIC_THRESHOLD_DB} dB)",
    )
    parser.add_argument(
        "--silence-wait",
        type=float,
        default=SILENCE_WAIT_SECONDS,
        help=f"Pause wait in seconds before responding (default: {SILENCE_WAIT_SECONDS}s)",
    )

    args = parser.parse_args()
    resolved_api_key = get_api_key(args.api_key)

    if not resolved_api_key:
        print("\n[!] ERROR: GEMINI_API_KEY is not set.")
        print("    Please set GEMINI_API_KEY in your .env file or at the top of main.py.\n")
        sys.exit(1)

    try:
        asyncio.run(
            run_live_call(
                camera_source=args.camera,
                api_key=resolved_api_key,
                model=args.model,
                fps=args.fps,
                threshold_db=args.threshold_db,
                silence_wait_sec=args.silence_wait,
            )
        )
    except KeyboardInterrupt:
        print("\n[!] Program interrupted by user.")
    except Exception as e:
        print(f"\n[!] Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
