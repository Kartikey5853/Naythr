"""Audio input (microphone) and output (speaker) streams for Gemini Live call."""

import asyncio
import logging
import queue
import threading
import time
from typing import Optional
import numpy as np
import sounddevice as sd

logger = logging.getLogger("AudioStream")


class AudioDevice:
    """Manages simultaneous PC microphone input, real-time dB calculation, and speaker output."""

    def __init__(
        self,
        input_sample_rate: int = 16000,
        output_sample_rate: int = 24000,
        chunk_size: int = 1024,
        threshold_db: float = 48.0,
    ):
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.chunk_size = chunk_size
        self.threshold_db = threshold_db

        self._in_stream: Optional[sd.RawInputStream] = None
        self._out_stream: Optional[sd.RawOutputStream] = None

        self._in_queue: asyncio.Queue = None
        self._loop: asyncio.AbstractEventLoop = None

        self._out_queue: queue.Queue = queue.Queue()
        self._out_thread: Optional[threading.Thread] = None

        self._running = False
        self._is_speaking = False

        # Real-time metrics
        self.current_db: float = 0.0
        self.current_rms: float = 0.0
        self.is_voice_detected: bool = False

    def start(self, loop: asyncio.AbstractEventLoop, in_queue: asyncio.Queue):
        """Start microphone and speaker streams."""
        if self._running:
            return

        self._loop = loop
        self._in_queue = in_queue
        self._running = True

        # 1. Start Speaker Output Stream & Worker Thread
        try:
            self._out_stream = sd.RawOutputStream(
                samplerate=self.output_sample_rate,
                channels=1,
                dtype="int16",
            )
            self._out_stream.start()
            self._out_thread = threading.Thread(
                target=self._playback_worker, daemon=True, name="AudioPlayback"
            )
            self._out_thread.start()
            logger.info("Speaker output stream started (24kHz).")
        except Exception as e:
            logger.warning(f"Failed to start speaker output stream: {e}")
            self._out_stream = None

        # 2. Start Microphone Input Stream
        try:
            self._in_stream = sd.RawInputStream(
                samplerate=self.input_sample_rate,
                channels=1,
                dtype="int16",
                blocksize=self.chunk_size,
                callback=self._mic_callback,
            )
            self._in_stream.start()
            logger.info("Microphone input stream started (16kHz).")
        except Exception as e:
            logger.warning(f"Failed to start microphone input stream: {e}")
            self._in_stream = None

    def _mic_callback(self, indata, frames, time_info, status):
        """Called by sounddevice for each microphone chunk."""
        if not self._running:
            return

        # Calculate RMS and decibel (dB) relative to full-scale int16
        samples = np.frombuffer(indata, dtype=np.int16)
        rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
        db = float(20.0 * np.log10(max(1.0, rms)))

        self.current_rms = rms
        self.current_db = db
        self.is_voice_detected = db >= self.threshold_db

        chunk_to_send = bytes(indata)

        # Echo suppression ONLY during speaker playback
        if self._is_speaking:
            # If volume is below threshold + 10, it's just speaker bleed from PC speakers
            if db < (self.threshold_db + 10.0):
                chunk_to_send = b"\x00" * len(indata)
            else:
                # User intentionally spoke loudly to interrupt Gemini
                self.interrupt()

        # Put into asyncio queue for streaming to Gemini
        if self._loop and self._in_queue and not self._loop.is_closed():
            try:
                self._loop.call_soon_threadsafe(self._in_queue.put_nowait, chunk_to_send)
            except Exception:
                pass

    def _playback_worker(self):
        """Continuously write incoming audio chunks to PC speakers."""
        while self._running:
            try:
                chunk = self._out_queue.get(timeout=0.05)
                if chunk is None:
                    break
                if self._out_stream and self._running:
                    self._is_speaking = True
                    self._out_stream.write(chunk)
                if self._out_queue.empty():
                    self._is_speaking = False
            except queue.Empty:
                self._is_speaking = False
            except Exception as e:
                self._is_speaking = False
                logger.debug(f"Playback error: {e}")

    def play_audio(self, pcm_data: bytes):
        """Enqueue PCM audio chunks to play through speaker."""
        if self._running and pcm_data:
            self._out_queue.put(pcm_data)

    def interrupt(self):
        """Clear queued audio immediately when interrupted."""
        while not self._out_queue.empty():
            try:
                self._out_queue.get_nowait()
            except queue.Empty:
                break
        self._is_speaking = False

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking

    def get_meter_display(self) -> str:
        """Return formatted ASCII decibel meter string."""
        # Map 30 dB (silence) to 90 dB (loud) into 10 characters
        normalized = max(0.0, min(1.0, (self.current_db - 30.0) / 60.0))
        bars = int(round(normalized * 10))
        bar_str = "#" * bars + "-" * (10 - bars)
        status = "VOICE" if self.is_voice_detected else "idle"
        return f"{self.current_db:4.1f} dB [{bar_str}] ({status})"

    def stop(self):
        """Stop audio hardware streams."""
        self._running = False
        if self._in_stream:
            try:
                self._in_stream.stop()
                self._in_stream.close()
            except Exception:
                pass
            self._in_stream = None

        if self._out_queue:
            self._out_queue.put(None)

        if self._out_stream:
            try:
                self._out_stream.stop()
                self._out_stream.close()
            except Exception:
                pass
            self._out_stream = None

        logger.info("Audio device stopped.")
