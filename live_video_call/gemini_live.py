"""Gemini Multimodal Live API session manager for real-time video stream, mic audio, and Q&A."""

import asyncio
import logging
from typing import Callable, Optional, List
import numpy as np
from google import genai
from google.genai import types

try:
    from .live_config import (
        GEMINI_API_KEY,
        GEMINI_LIVE_MODEL,
        FALLBACK_LIVE_MODEL,
        FRAME_INTERVAL_SEC,
        SYSTEM_INSTRUCTION,
    )
    from .camera_stream import CameraStream
    from .audio_stream import AudioDevice
except (ImportError, ValueError):
    from live_config import (
        GEMINI_API_KEY,
        GEMINI_LIVE_MODEL,
        FALLBACK_LIVE_MODEL,
        FRAME_INTERVAL_SEC,
        SYSTEM_INSTRUCTION,
    )
    from camera_stream import CameraStream
    from audio_stream import AudioDevice

logger = logging.getLogger("GeminiLive")


class GeminiLiveSession:
    """Manages a persistent Multimodal Live connection with Gemini.
    
    Streams:
      - Camera frames at low frequency (~1 FPS)
      - Microphone audio continuously (16kHz PCM)
    Receives:
      - Spoken audio responses played in real time (24kHz PCM)
      - Real-time transcription of both user and Gemini speech
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        camera: Optional[CameraStream] = None,
        audio: Optional[AudioDevice] = None,
        frame_interval_sec: float = FRAME_INTERVAL_SEC,
        system_instruction: str = SYSTEM_INSTRUCTION,
    ):
        self.api_key = api_key or GEMINI_API_KEY
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Please provide it in .env or at the top of main.py."
            )

        self.model = model or GEMINI_LIVE_MODEL
        self.camera = camera
        self.audio = audio
        self.frame_interval_sec = frame_interval_sec
        self.system_instruction = system_instruction

        self._client: Optional[genai.Client] = None
        self._session = None
        self._session_context = None

        self._is_connected = False
        self._frames_sent = 0
        self._audio_chunks_sent = 0

        self._receive_task: Optional[asyncio.Task] = None
        self._video_task: Optional[asyncio.Task] = None
        self._audio_task: Optional[asyncio.Task] = None
        self._mic_queue: asyncio.Queue = asyncio.Queue()

        # Callbacks for UI / CLI
        self.on_user_speech: Optional[Callable[[str], None]] = None
        self.on_gemini_speech: Optional[Callable[[str], None]] = None
        self.on_turn_complete: Optional[Callable[[], None]] = None
        self.on_interrupted: Optional[Callable[[], None]] = None
        self.frame_sent_callbacks: List[Callable[[int], None]] = []

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def frames_sent(self) -> int:
        return self._frames_sent

    async def connect(self) -> bool:
        """Connect to the Gemini Multimodal Live API."""
        if self._is_connected:
            return True

        self._client = genai.Client(api_key=self.api_key)

        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            system_instruction=self.system_instruction,
        )

        models_to_try = [self.model]
        if FALLBACK_LIVE_MODEL != self.model:
            models_to_try.append(FALLBACK_LIVE_MODEL)

        last_error = None
        for m in models_to_try:
            try:
                logger.info(f"Connecting to Gemini Live API with model: {m}...")
                self._session_context = self._client.aio.live.connect(model=m, config=config)
                self._session = await self._session_context.__aenter__()
                self.model = m
                self._is_connected = True
                logger.info(f"Successfully connected to Gemini Live using {m}")
                break
            except Exception as e:
                logger.warning(f"Connection failed with model {m}: {e}")
                last_error = e

        if not self._is_connected:
            raise RuntimeError(f"Could not connect to Gemini Live: {last_error}")

        loop = asyncio.get_running_loop()

        # 1. Start Receive loop first to drain setup messages
        self._receive_task = asyncio.create_task(
            self._receive_loop(), name="GeminiReceiver"
        )

        # 2. Start Audio hardware if available
        if self.audio:
            self.audio.start(loop=loop, in_queue=self._mic_queue)
            self._audio_task = asyncio.create_task(
                self._audio_send_loop(), name="GeminiMicSender"
            )

        # 3. Start Video streaming loop
        if self.camera:
            self._video_task = asyncio.create_task(
                self._video_stream_loop(), name="GeminiVideoStream"
            )

        return True

    async def _audio_send_loop(self):
        """Continuously stream microphone audio to Gemini Live."""
        try:
            while self._is_connected:
                chunk = await self._mic_queue.get()
                if not self._is_connected or not self._session:
                    break

                try:
                    await self._session.send_realtime_input(
                        audio=types.Blob(data=chunk, mime_type="audio/pcm;rate=16000")
                    )
                    self._audio_chunks_sent += 1
                except Exception as e:
                    logger.debug(f"Audio send error: {e}")

        except asyncio.CancelledError:
            logger.debug("Audio send loop cancelled.")
        except Exception as e:
            logger.error(f"Audio send loop error: {e}")

    async def _video_stream_loop(self):
        """Send camera frames at low frequency (~1 FPS)."""
        logger.info(f"Starting video stream (~{1.0 / self.frame_interval_sec:.1f} FPS)...")
        try:
            while self._is_connected:
                if self.camera:
                    jpeg_bytes = self.camera.get_latest_jpeg()
                    if jpeg_bytes:
                        try:
                            await self._session.send_realtime_input(
                                video=types.Blob(data=jpeg_bytes, mime_type="image/jpeg")
                            )
                            self._frames_sent += 1
                            for cb in self.frame_sent_callbacks:
                                try:
                                    cb(self._frames_sent)
                                except Exception:
                                    pass
                        except Exception as e:
                            logger.warning(f"Error sending video frame: {e}")

                await asyncio.sleep(self.frame_interval_sec)
        except asyncio.CancelledError:
            logger.debug("Video stream loop cancelled.")
        except Exception as e:
            logger.error(f"Video stream loop error: {e}")

    async def _receive_loop(self):
        """Continuously process incoming server messages from Gemini Live across all turns."""
        while self._is_connected and self._session:
            try:
                async for response in self._session.receive():
                    server_content = getattr(response, "server_content", None)
                    if not server_content:
                        continue

                    # 1. Handle user interruption (barge-in)
                    if getattr(server_content, "interrupted", False):
                        if self.audio:
                            self.audio.interrupt()
                        if self.on_interrupted:
                            try:
                                self.on_interrupted()
                            except Exception:
                                pass

                    # 2. Incoming audio from Gemini -> play through speakers
                    model_turn = getattr(server_content, "model_turn", None)
                    if model_turn:
                        for part in model_turn.parts:
                            if part.inline_data and part.inline_data.data:
                                if self.audio:
                                    self.audio.play_audio(part.inline_data.data)

                    # 3. User speech transcription
                    input_trans = getattr(server_content, "input_transcription", None)
                    if input_trans and input_trans.text:
                        if self.on_user_speech:
                            try:
                                self.on_user_speech(input_trans.text)
                            except Exception:
                                pass

                    # 4. Gemini speech transcription
                    output_trans = getattr(server_content, "output_transcription", None)
                    if output_trans and output_trans.text:
                        if self.on_gemini_speech:
                            try:
                                self.on_gemini_speech(output_trans.text)
                            except Exception:
                                pass

                    # 5. Turn completion
                    if getattr(server_content, "turn_complete", False):
                        if self.on_turn_complete:
                            try:
                                self.on_turn_complete()
                            except Exception:
                                pass

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in Gemini receive loop: {e}")
                await asyncio.sleep(0.1)

    async def send_text(self, text: str):
        """Send text message into the live session using realtime input."""
        if not self._is_connected or not self._session:
            raise RuntimeError("Gemini Live session is not connected.")

        await self._session.send_realtime_input(text=text)

    async def wake_up(
        self,
        prompt: str = "Wake up call: Greet the user in one short sentence and confirm you are awake and watching the live camera feed.",
    ):
        """Send a wake-up call to prime the session and prompt Gemini to speak first."""
        if not self._is_connected or not self._session:
            raise RuntimeError("Gemini Live session is not connected.")
        await self.send_text(prompt)

    async def ask(self, question: str, on_chunk: Optional[Callable[[str], None]] = None) -> str:
        """Send a question and wait for Gemini's response turn to complete."""
        chunks: List[str] = []
        turn_done = asyncio.Event()

        prev_on_speech = self.on_gemini_speech
        prev_on_turn = self.on_turn_complete

        def _temp_speech(chunk: str):
            chunks.append(chunk)
            if on_chunk:
                try:
                    on_chunk(chunk)
                except Exception:
                    pass
            if prev_on_speech:
                try:
                    prev_on_speech(chunk)
                except Exception:
                    pass

        def _temp_turn():
            turn_done.set()
            if prev_on_turn:
                try:
                    prev_on_turn()
                except Exception:
                    pass

        self.on_gemini_speech = _temp_speech
        self.on_turn_complete = _temp_turn

        try:
            await self.send_text(question)
            await asyncio.wait_for(turn_done.wait(), timeout=15.0)
        except asyncio.TimeoutError:
            pass
        finally:
            self.on_gemini_speech = prev_on_speech
            self.on_turn_complete = prev_on_turn

        return "".join(chunks)

    async def close(self):
        """Gracefully disconnect and cleanup session tasks."""
        self._is_connected = False

        for task in [self._video_task, self._audio_task, self._receive_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        if self.audio:
            self.audio.stop()

        if self._session_context:
            try:
                await self._session_context.__aexit__(None, None, None)
            except Exception:
                pass
            self._session = None
            self._session_context = None

        logger.info("Gemini Live session closed.")
