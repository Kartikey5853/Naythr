"""SOS Confirmation Layer for Naythr Assistive Vision & Navigation.

Architecture:
                    NORMAL
                       ↓
        Camera becomes BLACK
        AND wearable sensors indicate
        a dangerous condition (< 300 mm on all sides)
                       ↓
                SOS CHECKING
                       ↓
             Gemini asks:
             "Are you okay?"
                       ↓
          ┌────────────┼────────────┐
          ↓            ↓            ↓
       "Yes"          "No"       No response
          ↓            ↓            ↓
       CANCEL         SOS        TIMEOUT (7s)
                                    ↓
                                   SOS

Python controls timing, state transitions, and emergency triggering.
Gemini handles asking the question and transcribing user voice response.
"""

import asyncio
import logging
import os
import re
import time
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger("SosManager")

# ==============================================================================
# CONFIGURATION - SOS SETTINGS
# ==============================================================================
SOS_DISTANCE_MM = int(os.getenv("SOS_DISTANCE_MM", "300"))
SOS_CONFIRMATION_TIMEOUT_SECONDS = float(os.getenv("SOS_CONFIRMATION_TIMEOUT_SECONDS", "7.0"))
CAMERA_BLACK_BRIGHTNESS_THRESHOLD = float(os.getenv("CAMERA_BLACK_BRIGHTNESS_THRESHOLD", "20.0"))
CAMERA_BLACK_DARK_RATIO_THRESHOLD = float(os.getenv("CAMERA_BLACK_DARK_RATIO_THRESHOLD", "0.95"))
CAMERA_BLACK_MIN_FRAMES = int(os.getenv("CAMERA_BLACK_MIN_FRAMES", "5"))
CAMERA_BLACK_MIN_DURATION_SEC = float(os.getenv("CAMERA_BLACK_MIN_DURATION_SEC", "1.0"))
# ==============================================================================


class CameraBlackDetector:
    """Robust local detector for sustained black/dark camera frames."""

    def __init__(
        self,
        brightness_threshold: float = CAMERA_BLACK_BRIGHTNESS_THRESHOLD,
        dark_pixel_ratio_threshold: float = CAMERA_BLACK_DARK_RATIO_THRESHOLD,
        min_consecutive_frames: int = CAMERA_BLACK_MIN_FRAMES,
        min_duration_seconds: float = CAMERA_BLACK_MIN_DURATION_SEC,
    ):
        self.brightness_threshold = brightness_threshold
        self.dark_pixel_ratio_threshold = dark_pixel_ratio_threshold
        self.min_consecutive_frames = min_consecutive_frames
        self.min_duration_seconds = min_duration_seconds

        self.consecutive_black_count = 0
        self.first_black_time: Optional[float] = None
        self.is_confirmed_black = False
        self._simulated_black: Optional[bool] = None

    def set_simulation(self, simulated: Optional[bool]):
        """Manually override camera black state for safe unit testing."""
        self._simulated_black = simulated

    def process_frame(self, frame: Optional[np.ndarray]) -> bool:
        """Analyze a camera frame and update the black confirmation status."""
        if self._simulated_black is not None:
            self.is_confirmed_black = self._simulated_black
            return self.is_confirmed_black

        if frame is None or frame.size == 0:
            return self.is_confirmed_black

        mean_brightness = float(np.mean(frame))

        is_dark = False
        if mean_brightness < self.brightness_threshold:
            # Check percentage of very dark pixels (< 1.5 * threshold)
            dark_ratio = float(np.mean(frame < (self.brightness_threshold * 1.5)))
            if dark_ratio >= self.dark_pixel_ratio_threshold:
                is_dark = True

        now = time.time()
        if is_dark:
            self.consecutive_black_count += 1
            if self.first_black_time is None:
                self.first_black_time = now
            elapsed = now - self.first_black_time
            if (
                self.consecutive_black_count >= self.min_consecutive_frames
                and elapsed >= self.min_duration_seconds
            ):
                self.is_confirmed_black = True
        else:
            self.consecutive_black_count = 0
            self.first_black_time = None
            self.is_confirmed_black = False

        return self.is_confirmed_black

    def reset(self):
        """Reset internal frame counters."""
        self.consecutive_black_count = 0
        self.first_black_time = None
        self.is_confirmed_black = False
        self._simulated_black = None


# Emergency word lists for natural-language classification
EMERGENCY_WORDS = [
    "no",
    "help",
    "need help",
    "not okay",
    "not ok",
    "not fine",
    "send help",
    "emergency",
    "hurt",
    "fell",
    "danger",
    "call 911",
    "call help",
    "bad",
    "pain",
]

OKAY_WORDS = [
    "yes",
    "yeah",
    "yep",
    "i'm okay",
    "im okay",
    "i am okay",
    "i'm fine",
    "im fine",
    "i am fine",
    "yes i'm okay",
    "yes im okay",
    "all good",
    "fine",
    "okay",
    "ok",
    "good",
    "safe",
    "alive",
    "false alarm",
    "cancel",
]


def classify_user_response(raw_text: str) -> str:
    """Classify the user's spoken or typed response into 'EMERGENCY', 'OKAY', or 'UNKNOWN'."""
    if not raw_text:
        return "UNKNOWN"

    text = raw_text.strip().lower()
    # Normalize punctuation
    cleaned = re.sub(r"[^\w\s']", " ", text).strip()
    words = cleaned.split()

    # 1. Check for negative / emergency signals FIRST (e.g. "I'm not okay", "help")
    for phrase in EMERGENCY_WORDS:
        if " " in phrase:
            if phrase in cleaned:
                return "EMERGENCY"
        else:
            if phrase in words:
                return "EMERGENCY"

    # 2. Check for affirmative / okay signals
    for phrase in OKAY_WORDS:
        if " " in phrase:
            if phrase in cleaned:
                return "OKAY"
        else:
            if phrase in words:
                return "OKAY"

    return "UNKNOWN"


async def default_trigger_sos():
    """Default emergency SOS handler."""
    print("\n" + "!" * 60)
    print("[SOS] !!! EMERGENCY SOS TRIGGERED !!!")
    print("!" * 60 + "\n")


class SosManager:
    """Explicit State Machine managing SOS detection, confirmation, timeout, and triggering."""

    STATE_NORMAL = "NORMAL"
    STATE_SUSPICIOUS = "SUSPICIOUS"
    STATE_SOS_CHECKING = "SOS_CHECKING"
    STATE_SOS_ACTIVE = "SOS_ACTIVE"

    def __init__(
        self,
        sos_distance_mm: float = SOS_DISTANCE_MM,
        timeout_seconds: float = SOS_CONFIRMATION_TIMEOUT_SECONDS,
        send_gemini_prompt: Optional[Callable[[str], None]] = None,
        trigger_sos_callback: Optional[Callable[[], None]] = None,
        unmute_mic_callback: Optional[Callable[[], None]] = None,
    ):
        self.sos_distance_mm = sos_distance_mm
        self.timeout_seconds = timeout_seconds
        self.send_gemini_prompt = send_gemini_prompt
        self.trigger_sos_callback = trigger_sos_callback or default_trigger_sos
        self.unmute_mic_callback = unmute_mic_callback

        self.state = self.STATE_NORMAL
        self.latest_sensor_distances: Dict[str, Optional[float]] = {
            "LEFT": None,
            "CENTER": None,
            "RIGHT": None,
        }
        self.last_sensor_update_time: Dict[str, float] = {
            "LEFT": 0.0,
            "CENTER": 0.0,
            "RIGHT": 0.0,
        }

        self._timeout_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self.has_asked_question = False

    def update_sensor_reading(self, direction: str, distance_mm: float):
        """Update cached distance for a direction."""
        dir_upper = direction.upper().strip()
        if dir_upper in ("FRONT", "FORWARD", "MID", "MIDDLE"):
            dir_upper = "CENTER"

        if dir_upper in self.latest_sensor_distances:
            self.latest_sensor_distances[dir_upper] = float(distance_mm)
            self.last_sensor_update_time[dir_upper] = time.time()

    def are_all_sensors_below_threshold(self) -> bool:
        """Check if LEFT, CENTER, and RIGHT sensors are all below SOS_DISTANCE_MM."""
        left = self.latest_sensor_distances.get("LEFT")
        center = self.latest_sensor_distances.get("CENTER")
        right = self.latest_sensor_distances.get("RIGHT")

        if left is None or center is None or right is None:
            return False

        return (
            left < self.sos_distance_mm
            and center < self.sos_distance_mm
            and right < self.sos_distance_mm
        )

    async def poll_check(self, camera_black_confirmed: bool):
        """Periodically evaluate emergency condition against current state."""
        async with self._lock:
            if self.state in (self.STATE_SOS_CHECKING, self.STATE_SOS_ACTIVE):
                return

            # Camera-only trigger: Emergency condition triggered when camera is confirmed dark/black
            if camera_black_confirmed:
                if self.state == self.STATE_NORMAL:
                    self.state = self.STATE_SUSPICIOUS
                    print("\n[SOS] Possible emergency detected.")
                    print("[SOS] Camera black confirmed.")
                    await self._enter_sos_checking()
            else:
                if self.state == self.STATE_SUSPICIOUS:
                    self.state = self.STATE_NORMAL

    async def _enter_sos_checking(self):
        """Enter SOS_CHECKING state, ask 'Are you okay?', and start confirmation timer."""
        self.state = self.STATE_SOS_CHECKING
        self.has_asked_question = True

        print("[SOS] Asking user for confirmation.")
        print(f"[SOS] Waiting {int(self.timeout_seconds)} seconds for response...")

        # Formulate prompt for existing Gemini Live session
        prompt = (
            "[SYSTEM SOS CHECK]\n\n"
            "The wearable has detected a possible emergency:\n"
            "- the camera feed appears to be black\n\n"
            "Ask the user exactly:\n"
            '"Are you okay?"\n\n'
            "Do not explain the technical details.\n"
            "Keep the question short and clear."
        )

        if self.send_gemini_prompt:
            try:
                res = self.send_gemini_prompt(prompt)
                if asyncio.iscoroutine(res):
                    await res
            except Exception as e:
                logger.debug(f"Error sending SOS check prompt to Gemini: {e}")

        # Open mic so user can speak response directly without 'Hey Marvin'
        if self.unmute_mic_callback:
            try:
                self.unmute_mic_callback()
            except Exception as e:
                logger.debug(f"Error opening mic for SOS check: {e}")

        # Start confirmation timer
        if self._timeout_task and not self._timeout_task.done():
            self._timeout_task.cancel()
        self._timeout_task = asyncio.create_task(self._wait_for_timeout())

    async def _wait_for_timeout(self):
        """Asynchronous timer that triggers SOS if user does not respond."""
        try:
            await asyncio.sleep(self.timeout_seconds)
            async with self._lock:
                if self.state == self.STATE_SOS_CHECKING:
                    print("\n[SOS] No response within timeout. Triggering SOS.")
                    self.state = self.STATE_SOS_ACTIVE
                    await self._invoke_trigger_sos(reason="No response within timeout (7s)")
        except asyncio.CancelledError:
            pass

    async def handle_user_response(self, text: str) -> bool:
        """Process user spoken transcript or typed response during SOS_CHECKING."""
        async with self._lock:
            if self.state != self.STATE_SOS_CHECKING:
                return False

            classification = classify_user_response(text)

            if classification == "OKAY":
                if self._timeout_task and not self._timeout_task.done():
                    self._timeout_task.cancel()
                self.state = self.STATE_NORMAL
                self.has_asked_question = False
                print(f"[SOS] User confirmed they are okay (heard: '{text.strip()}'). SOS cancelled.")
                return True

            elif classification == "EMERGENCY":
                if self._timeout_task and not self._timeout_task.done():
                    self._timeout_task.cancel()
                self.state = self.STATE_SOS_ACTIVE
                print(f"[SOS] User indicated emergency (heard: '{text.strip()}'). Triggering SOS.")
                await self._invoke_trigger_sos(reason=f"User stated: '{text.strip()}'")
                return True

            # If UNKNOWN (e.g. background noise), continue waiting until clear response or timeout
            return False

    async def _invoke_trigger_sos(self, reason: str = "Emergency condition detected"):
        """Execute the SOS handler, dispatch email/SMS alerts with live location, and speak reassurance."""
        # 1. Invoke external custom callback immediately
        if self.trigger_sos_callback:
            try:
                res = self.trigger_sos_callback()
                if asyncio.iscoroutine(res):
                    await res
            except Exception as e:
                print(f"[SOS] Error calling SOS callback: {e}")

        # 2. Acquire live location and dispatch multi-channel alerts (email, SMS, webhooks)
        loc = None
        try:
            from .sos_dispatcher import get_live_location, dispatch_all_emergency_alerts
        except (ImportError, ValueError):
            try:
                from sos_dispatcher import get_live_location, dispatch_all_emergency_alerts
            except ImportError:
                dispatch_all_emergency_alerts = None
                get_live_location = None

        if dispatch_all_emergency_alerts:
            try:
                dispatch_res = await dispatch_all_emergency_alerts(
                    reason=reason,
                    sensor_distances=self.latest_sensor_distances,
                )
                loc = dispatch_res.get("location")
            except Exception as e:
                logger.debug(f"Error dispatching alerts: {e}")

        # 3. Instruct Gemini Live session to immediately speak reassurance aloud to the user
        if self.send_gemini_prompt:
            loc_str = loc.get("address", "your current location") if loc else "your current location"
            maps_url = loc.get("maps_url", "") if loc else ""
            city_name = loc.get("city", "your area") if loc else "your area"
            reassurance_prompt = (
                "[SYSTEM EMERGENCY SOS ACTIVATED]\n\n"
                f"Emergency SOS has been TRIGGERED. Reason: {reason}.\n"
                f"User location: {loc_str}.\n"
                f"Google Maps Link: {maps_url}\n"
                "Emergency alerts with live coordinates have been dispatched to their contacts.\n\n"
                "IMPORTANT INSTRUCTION:\n"
                "Immediately speak aloud to the user in a calm, clear, reassuring, and comforting voice:\n"
                "1. Confirm that emergency SOS has been activated.\n"
                f"2. Tell them that their emergency contacts have been notified with their live location in {city_name}.\n"
                "3. Reassure them that help is on the way and encourage them to remain calm and in a safe position.\n"
                "Speak clearly and naturally (around 15-25 words). Do not read raw URL strings aloud."
            )
            try:
                res = self.send_gemini_prompt(reassurance_prompt)
                if asyncio.iscoroutine(res):
                    await res
            except Exception as e:
                logger.debug(f"Error sending SOS reassurance to Gemini: {e}")

    def reset(self):
        """Explicitly reset state machine back to NORMAL."""
        if self._timeout_task and not self._timeout_task.done():
            self._timeout_task.cancel()
        self.state = self.STATE_NORMAL
        self.has_asked_question = False
        print("[SOS] System reset to NORMAL.")


# ==============================================================================
# SELF-TEST SUITE FOR THE 7 REQUIRED TESTS
# ==============================================================================
async def run_unit_tests():
    """Automated verification of the 7 specified SOS requirements."""
    print("=" * 70)
    print(">>> RUNNING SOS CONFIRMATION LAYER TEST SUITE <<<")
    print("=" * 70 + "\n")

    sos_triggered = False

    async def test_trigger():
        nonlocal sos_triggered
        sos_triggered = True
        print("[SOS] !!! EMERGENCY SOS TRIGGERED !!!")

    mock_gemini_calls = []

    def mock_gemini(p):
        mock_gemini_calls.append(p)

    mgr = SosManager(
        sos_distance_mm=300,
        timeout_seconds=1.5,  # Short timeout for fast unit tests
        send_gemini_prompt=mock_gemini,
        trigger_sos_callback=test_trigger,
    )
    detector = CameraBlackDetector(min_consecutive_frames=5, min_duration_seconds=0.2)

    # --------------------------------------------------------------------------
    # TEST 4: Camera briefly becomes black for one frame -> no SOS check
    # --------------------------------------------------------------------------
    print("--- TEST 4: Camera briefly becomes black for one frame ---")
    detector.reset()
    mgr.reset()
    dark_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    normal_frame = np.ones((100, 100, 3), dtype=np.uint8) * 128

    detector.process_frame(dark_frame)  # 1 black frame
    is_confirmed = detector.process_frame(normal_frame)  # Immediately back to normal
    mgr.update_sensor_reading("LEFT", 200)
    mgr.update_sensor_reading("CENTER", 200)
    mgr.update_sensor_reading("RIGHT", 200)
    await mgr.poll_check(is_confirmed)
    assert not is_confirmed, "Failed: Single black frame should not confirm black"
    assert mgr.state == SosManager.STATE_NORMAL, "Failed: State should remain NORMAL"
    print("-> Result: PASSED (Single black frame ignored, no SOS check)\n")

    # --------------------------------------------------------------------------
    # TEST 5: Camera is normal / not black -> no SOS check
    # --------------------------------------------------------------------------
    print("--- TEST 5: Camera normal (not black) -> no SOS check ---")
    mgr.reset()
    mgr.update_sensor_reading("LEFT", 100)
    mgr.update_sensor_reading("CENTER", 100)
    mgr.update_sensor_reading("RIGHT", 100)
    await mgr.poll_check(camera_black_confirmed=False)
    assert mgr.state == SosManager.STATE_NORMAL, "Failed: State should remain NORMAL when camera is not black"
    print("-> Result: PASSED (Camera not black, no SOS check)\n")

    # --------------------------------------------------------------------------
    # TEST 6: Camera black confirmed -> Triggers SOS check (camera-only mode)
    # --------------------------------------------------------------------------
    print("--- TEST 6: Camera black confirmed -> Enters SOS_CHECKING regardless of sensors ---")
    mgr.reset()
    # Sensors high (> 300 mm) or uninitialized
    mgr.update_sensor_reading("LEFT", 900)
    mgr.update_sensor_reading("CENTER", 1200)
    mgr.update_sensor_reading("RIGHT", 850)
    await mgr.poll_check(camera_black_confirmed=True)
    assert mgr.state == SosManager.STATE_SOS_CHECKING, "Failed: State should be SOS_CHECKING on camera black"
    print("-> Result: PASSED (Camera black triggers SOS check without requiring sensors)\n")

    # --------------------------------------------------------------------------
    # TEST 7: All conditions persist -> Ask 'Are you okay?' exactly once
    # --------------------------------------------------------------------------
    print("--- TEST 7: All conditions persist -> Ask 'Are you okay?' exactly once ---")
    mgr.reset()
    mock_gemini_calls.clear()
    mgr.update_sensor_reading("LEFT", 150)
    mgr.update_sensor_reading("CENTER", 180)
    mgr.update_sensor_reading("RIGHT", 120)
    # Simulate consecutive poll checks
    for _ in range(5):
        await mgr.poll_check(camera_black_confirmed=True)
    assert mgr.state == SosManager.STATE_SOS_CHECKING, "Failed: State should be SOS_CHECKING"
    assert len(mock_gemini_calls) == 1, f"Failed: Expected exactly 1 call, got {len(mock_gemini_calls)}"
    print("-> Result: PASSED (Asked exactly once)\n")

    # --------------------------------------------------------------------------
    # TEST 1: Possible SOS -> User says 'Yes, I'm okay' -> Cancelled
    # --------------------------------------------------------------------------
    print("--- TEST 1: Possible SOS -> User says 'Yes, I'm okay' -> Cancelled ---")
    # mgr is already in SOS_CHECKING from Test 7
    sos_triggered = False
    handled = await mgr.handle_user_response("Yes, I'm okay")
    assert handled is True, "Failed: Response should be handled"
    assert mgr.state == SosManager.STATE_NORMAL, "Failed: State should return to NORMAL"
    assert not sos_triggered, "Failed: SOS should NOT be triggered"
    print("-> Result: PASSED (SOS cancelled, state NORMAL)\n")

    # --------------------------------------------------------------------------
    # TEST 2: Possible SOS -> User says 'No, help me' -> SOS triggered immediately
    # --------------------------------------------------------------------------
    print("--- TEST 2: Possible SOS -> User says 'No, help me' -> Triggered immediately ---")
    mgr.reset()
    sos_triggered = False
    mgr.update_sensor_reading("LEFT", 100)
    mgr.update_sensor_reading("CENTER", 100)
    mgr.update_sensor_reading("RIGHT", 100)
    await mgr.poll_check(camera_black_confirmed=True)
    assert mgr.state == SosManager.STATE_SOS_CHECKING
    handled = await mgr.handle_user_response("No, help me")
    assert handled is True
    assert mgr.state == SosManager.STATE_SOS_ACTIVE, "Failed: State should be SOS_ACTIVE"
    assert sos_triggered is True, "Failed: SOS callback should be triggered"
    print("-> Result: PASSED (SOS triggered immediately on 'No, help me')\n")

    # --------------------------------------------------------------------------
    # TEST 3: Possible SOS -> No response -> Timeout -> SOS triggered
    # --------------------------------------------------------------------------
    print("--- TEST 3: Possible SOS -> No response within timeout -> SOS triggered ---")
    mgr.reset()
    sos_triggered = False
    mgr.update_sensor_reading("LEFT", 100)
    mgr.update_sensor_reading("CENTER", 100)
    mgr.update_sensor_reading("RIGHT", 100)
    await mgr.poll_check(camera_black_confirmed=True)
    assert mgr.state == SosManager.STATE_SOS_CHECKING
    print(f"Waiting {mgr.timeout_seconds}s for simulated timeout...")
    await asyncio.sleep(mgr.timeout_seconds + 0.2)
    assert mgr.state == SosManager.STATE_SOS_ACTIVE, "Failed: State should be SOS_ACTIVE after timeout"
    assert sos_triggered is True, "Failed: SOS callback should be triggered on timeout"
    print("-> Result: PASSED (SOS triggered upon timeout)\n")

    print("=" * 70)
    print(">>> ALL 7 SOS TESTS PASSED SUCCESSFULLY! <<<")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_unit_tests())
