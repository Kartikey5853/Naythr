"""Gemini Live Video Call - IP Camera + PC Mic/Speaker Visual Assistant.

Open this folder and run:
    python main.py
"""

# ==============================================================================
# CONFIGURATION - CHANGE YOUR SETTINGS HERE
# ==============================================================================

# IP Camera URL (e.g. "http://192.168.1.4:8080/video", "rtsp://...", or "0" for webcam)
CAMERA_URL = "http://172.31.3.232:8080/video"

# Gemini API Key (leave empty "" to automatically load from .env in the project)



# Live Model & Frame Rate
MODEL_NAME = "gemini-3.1-flash-live-preview"   # Fast Multimodal Live model (fallback: gemini-3.8-live)
FRAME_RATE = 1.0                              # Frames per second sent to Gemini (~1 FPS recommended)

# Wake-Up Call Settings
ENABLE_WAKE_UP_CALL = True       # Gemini speaks first to confirm it is awake and watching the camera
WAKE_UP_PROMPT = "Wake up call: Greet the user in one short, natural sentence and confirm you can see their live camera feed."

# Microphone & Audio Settings
ENABLE_MIC = True                # Enable PC microphone
ENABLE_SPEAKER = True            # Enable PC speaker audio output
MIC_THRESHOLD_DB = 70.0          # Sensitivity threshold in dB (typical speech is 50-70 dB; background is 30-45 dB)
SHOW_DECIBEL_METER = True        # Display real-time microphone sound level meter

# ==============================================================================

import argparse
import asyncio
import logging
import os
from pathlib import Path
import sys
import time

# Windows console encoding fix for clean terminal output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure imports work whether run as `python main.py` or `python -m live_video_call.main`
_CURRENT_DIR = Path(__file__).resolve().parent
if str(_CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(_CURRENT_DIR))
if str(_CURRENT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_CURRENT_DIR.parent))

try:
    from .camera_stream import CameraStream
    from .audio_stream import AudioDevice
    from .gemini_live import GeminiLiveSession
    from .live_config import GEMINI_API_KEY as ENV_GEMINI_KEY
except (ImportError, ValueError):
    from camera_stream import CameraStream
    from audio_stream import AudioDevice
    from gemini_live import GeminiLiveSession
    from live_config import GEMINI_API_KEY as ENV_GEMINI_KEY

# Silence verbose background logs
logging.basicConfig(level=logging.WARNING)


def get_api_key(cli_key: str = "") -> str:
    """Resolve API key: Top of file > CLI arg > .env file."""
    if GEMINI_API_KEY.strip():
        return GEMINI_API_KEY.strip()
    if cli_key and cli_key.strip():
        return cli_key.strip()
    if ENV_GEMINI_KEY and ENV_GEMINI_KEY.strip():
        return ENV_GEMINI_KEY.strip()
    return os.getenv("GEMINI_API_KEY", "").strip()


async def run_live_call(
    camera_source: str,
    api_key: str,
    model: str,
    fps: float,
    enable_mic: bool = True,
    enable_speaker: bool = True,
    threshold_db: float = MIC_THRESHOLD_DB,
    show_meter: bool = SHOW_DECIBEL_METER,
    enable_wake_up_call: bool = ENABLE_WAKE_UP_CALL,
    wake_up_prompt: str = WAKE_UP_PROMPT,
):
    """Main live call session with simultaneous camera, microphone, and speaker."""
    frame_interval = 1.0 / max(0.1, fps)

    print("=" * 70)
    print(" [LIVE VIDEO CALL] Gemini Live + IP Camera + PC Microphone/Speaker ")
    print("=" * 70)

    # 1. Initialize Camera
    print(f"\n[1/3] Connecting to Camera: {camera_source}")
    camera = CameraStream(source=camera_source)
    camera.start()

    for _ in range(6):
        if camera.is_healthy():
            break
        await asyncio.sleep(0.25)

    if camera.is_healthy():
        print("   [+] Camera connected and receiving live frames.")
    else:
        print(f"   [!] Camera at '{camera_source}' not detected yet (auto-reconnecting).")

    # 2. Initialize Audio
    audio = None
    if enable_mic or enable_speaker:
        print(f"\n[2/3] Initializing Audio (Threshold: {threshold_db:.1f} dB)...")
        try:
            audio = AudioDevice(threshold_db=threshold_db)
            print(f"   [+] Microphone & Speaker active. (Voice gate: {threshold_db:.1f} dB)")
        except Exception as e:
            print(f"   [!] Audio initialization warning: {e}")
            audio = None

    # 3. Connect to Gemini Live
    print(f"\n[3/3] Connecting to Gemini Live ({model})...")
    session = GeminiLiveSession(
        api_key=api_key,
        model=model,
        camera=camera,
        audio=audio,
        frame_interval_sec=frame_interval,
        system_instruction=(
            "You are a helpful AI assistant in a live video and voice call with the user. "
            "You continuously receive real-time video frames from the user's camera and live audio from their microphone. "
            "When receiving a wake-up call or greeting, acknowledge warmly in one concise sentence, confirm you are awake, and confirm you can see the camera feed. "
            "Respond naturally, concisely, and directly by voice to what you see and hear."
        ),
    )

    # Transcription & Speech callbacks for console
    gemini_is_speaking_turn = False

    def on_user_speech(transcript: str):
        nonlocal gemini_is_speaking_turn
        gemini_is_speaking_turn = False
        sys.stdout.write("\r" + " " * 60 + "\r")
        print(f"You (Voice) > {transcript}")

    def on_gemini_speech(chunk: str):
        nonlocal gemini_is_speaking_turn
        if not gemini_is_speaking_turn:
            sys.stdout.write("\r" + " " * 60 + "\r")
            print("Gemini > ", end="", flush=True)
            gemini_is_speaking_turn = True
        print(chunk, end="", flush=True)

    def on_turn_complete():
        nonlocal gemini_is_speaking_turn
        if gemini_is_speaking_turn:
            print("\n")
            gemini_is_speaking_turn = False

    def on_interrupted():
        nonlocal gemini_is_speaking_turn
        print("\n[Interrupted by you]")
        gemini_is_speaking_turn = False

    session.on_user_speech = on_user_speech
    session.on_gemini_speech = on_gemini_speech
    session.on_turn_complete = on_turn_complete
    session.on_interrupted = on_interrupted

    try:
        await session.connect()
        print(f"   [+] Connected to Gemini Live ({session.model})!")
    except Exception as e:
        print(f"   [-] Failed to connect: {e}")
        if audio:
            audio.stop()
        camera.stop()
        return

    # 4. Wake-Up Call before user interaction begins
    if enable_wake_up_call:
        print("\n[+] Initiating Wake-Up Call with Gemini...")

        # Wait briefly for camera to push at least 1 live frame if camera is connected
        if camera:
            for _ in range(15):
                if session.frames_sent > 0:
                    break
                await asyncio.sleep(0.1)

        wake_up_done = asyncio.Event()
        prev_turn_callback = session.on_turn_complete

        def _wake_turn_complete():
            if prev_turn_callback:
                try:
                    prev_turn_callback()
                except Exception:
                    pass
            wake_up_done.set()

        session.on_turn_complete = _wake_turn_complete

        try:
            await session.wake_up(wake_up_prompt)
            # Wait up to 7 seconds for Gemini to speak its wake-up greeting
            try:
                await asyncio.wait_for(wake_up_done.wait(), timeout=7.0)
            except asyncio.TimeoutError:
                pass
        except Exception as e:
            print(f"   [!] Wake-up call notice: {e}")
        finally:
            session.on_turn_complete = prev_turn_callback

        print("[+] Wake-up call completed. Gemini is awake and ready!\n")

    # Call Active Screen
    print("=" * 70)
    print(" >>> CALL IN PROGRESS <<<")
    print(" * Speak into your PC microphone - Gemini hears you and talks back!")
    print(f" * Voice threshold: {threshold_db:.1f} dB (Change in main.py top)")
    print(" * Camera is streaming live frames (~1 FPS).")
    print(" * You can also type questions below and press Enter anytime.")
    print(" * Type 'wake' to trigger wake-up call again, 'status' for stats, or 'exit' to quit.")
    print("=" * 70 + "\n")

    loop = asyncio.get_running_loop()

    # Optional background task to display live decibel meter
    meter_task = None
    if show_meter and audio:
        async def meter_loop():
            last_meter_time = 0
            while session.is_connected:
                await asyncio.sleep(0.4)
                # Only print meter when Gemini is not speaking
                if not gemini_is_speaking_turn and audio:
                    meter_str = audio.get_meter_display()
                    # Print in-place using carriage return
                    sys.stdout.write(f"\r[Mic: {meter_str}] ")
                    sys.stdout.flush()

        meter_task = asyncio.create_task(meter_loop())

    # Asynchronous keyboard input queue so typing never blocks the session
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

    import threading
    kb_thread = threading.Thread(target=keyboard_reader, daemon=True, name="KeyboardReader")
    kb_thread.start()

    try:
        while session.is_connected:
            try:
                # Wait for typed input or check loop every 0.2s
                text = await asyncio.wait_for(input_queue.get(), timeout=0.2)
            except asyncio.TimeoutError:
                continue

            text = text.strip().strip('"').strip("'")
            if not text:
                continue

            if text.lower() in {"exit", "quit", "q"}:
                break

            if text.lower() in {"wake", "wakeup", "wake up"}:
                gemini_is_speaking_turn = False
                print(f"\nYou (Wake-Up) > {text}")
                try:
                    await session.wake_up(wake_up_prompt)
                except Exception as e:
                    print(f"\n[-] Error sending wake-up call: {e}\n")
                continue

            if text.lower() == "status":
                stats = camera.get_stats()
                mic_db = audio.current_db if audio else 0.0
                print(f"\n--- System Status ---")
                print(f"Camera:       {'Active [OK]' if stats['healthy'] else 'No Signal [!]'}")
                print(f"Frames Sent:  {session.frames_sent}")
                print(f"Mic Level:    {mic_db:.1f} dB (Threshold: {threshold_db:.1f} dB)")
                print(f"Gemini Model: {session.model}")
                print("---------------------\n")
                continue

            # User typed a question
            gemini_is_speaking_turn = False
            print(f"\nYou (Typed) > {text}")
            try:
                await session.send_text(text)
            except Exception as e:
                print(f"\n[-] Error sending text: {e}\n")

    finally:
        if meter_task and not meter_task.done():
            meter_task.cancel()
        print("\n\nEnding live call...")
        await session.close()
        camera.stop()
        print("Call ended. Goodbye!")


def main():
    parser = argparse.ArgumentParser(description="Gemini Live Video Call with IP Camera & Mic")
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
        "--threshold",
        type=float,
        default=MIC_THRESHOLD_DB,
        help=f"Mic dB sensitivity threshold (default: {MIC_THRESHOLD_DB} dB)",
    )
    parser.add_argument(
        "--no-mic",
        action="store_true",
        help="Disable microphone input",
    )
    parser.add_argument(
        "--no-speaker",
        action="store_true",
        help="Disable speaker audio output",
    )
    parser.add_argument(
        "--no-meter",
        action="store_true",
        help="Disable live decibel meter display",
    )
    parser.add_argument(
        "--no-wake-up",
        action="store_true",
        help="Disable automatic wake-up greeting on call start",
    )
    parser.add_argument(
        "--wake-prompt",
        type=str,
        default=WAKE_UP_PROMPT,
        help="Custom wake-up prompt to send to Gemini",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Launch browser Web UI instead of CLI",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Web UI port (default: 8000)",
    )
    args = parser.parse_args()

    api_key = get_api_key(args.api_key)
    if not api_key:
        print("\n[-] Error: GEMINI_API_KEY is not set!")
        print("Please either:")
        print("  1. Set GEMINI_API_KEY at the top of main.py")
        print("  2. Or set GEMINI_API_KEY in your .env file")
        print("  3. Or pass --api-key <YOUR_KEY>\n")
        sys.exit(1)

    if args.web:
        try:
            from .web_ui import run_web_server
        except (ImportError, ValueError):
            from web_ui import run_web_server

        run_web_server(
            camera_url=args.camera,
            api_key=api_key,
            model=args.model,
            fps=args.fps,
            port=args.port,
        )
    else:
        try:
            asyncio.run(
                run_live_call(
                    camera_source=args.camera,
                    api_key=api_key,
                    model=args.model,
                    fps=args.fps,
                    enable_mic=(not args.no_mic) and ENABLE_MIC,
                    enable_speaker=(not args.no_speaker) and ENABLE_SPEAKER,
                    threshold_db=args.threshold,
                    show_meter=(not args.no_meter) and SHOW_DECIBEL_METER,
                    enable_wake_up_call=(not args.no_wake_up) and ENABLE_WAKE_UP_CALL,
                    wake_up_prompt=args.wake_prompt,
                )
            )
        except KeyboardInterrupt:
            print("\nCall ended.")


if __name__ == "__main__":
    main()
