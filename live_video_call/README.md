# Gemini Live Video Call (IP Camera + PC Microphone & Speaker)

A real-time **Gemini Multimodal Live** call integrating your **IP Camera** and your **PC Microphone & Speaker**.

- **Live Decibel (dB) Meter**: Displays your microphone sound level in real time (`[Mic: 54.2 dB [#####-----] (VOICE)]`) so you know your mic is active.
- **Configurable Voice Threshold**: Set `MIC_THRESHOLD_DB` at the top of `main.py` to match your room noise floor.
- **Continuous Multi-Turn Call**: Speaks and listens back-and-forth continuously without going silent.
- **Low-Latency Camera**: Streams live visual frames (~1 FPS) from your IP camera to Gemini.
- **Live Voice Output**: Gemini speaks back in real time through your PC speakers.

```text
       ┌───────────┐
       │ IP Camera │──(1 FPS Video)──┐
       └───────────┘                 │
                                     ▼
┌──────────────┐             ┌─────────────────────┐             ┌──────────────┐
│  PC Mic (In) │──(PCM 16k)─>│ Gemini Live Session │──(PCM 24k)─>│ Speaker (Out)│
└──────────────┘             └─────────────────────┘             └──────────────┘
                                     │
                                     ▼
                      Terminal Transcripts & dB Meter
```

---

## Configuration (Top of `main.py`)

Open `live_video_call/main.py`:

```python
# ==============================================================================
# CONFIGURATION - CHANGE YOUR SETTINGS HERE
# ==============================================================================

# IP Camera URL (e.g. "http://192.168.1.4:8080/video", "rtsp://...", or "0" for webcam)
CAMERA_URL = "http://192.168.1.4:8080/video"

# Gemini API Key (leave empty "" to automatically load from .env in the project)
GEMINI_API_KEY = ""

# Live Model & Frame Rate
MODEL_NAME = "gemini-3.1-flash-live-preview"
FRAME_RATE = 1.0

# Wake-Up Call Settings
ENABLE_WAKE_UP_CALL = True       # Gemini speaks first to confirm it is awake and watching the camera
WAKE_UP_PROMPT = "Wake up call: Greet the user in one short, natural sentence and confirm you can see their live camera feed."

# Microphone Settings & Decibel Threshold
ENABLE_MIC = True                # Enable PC microphone input
ENABLE_SPEAKER = True            # Enable PC speaker playback
MIC_THRESHOLD_DB = 70.0          # Sensitivity threshold in dB (typical speech is 50-70 dB; background is 30-45 dB)
SHOW_DECIBEL_METER = True        # Display real-time microphone sound level meter

# ==============================================================================
```

---

## How to Run

Open terminal in `live_video_call`:

```bash
cd live_video_call
python main.py
```

### What You'll See

```text
======================================================================
 [LIVE VIDEO CALL] Gemini Live + IP Camera + PC Microphone/Speaker 
======================================================================

[1/3] Connecting to Camera: http://192.168.1.4:8080/video
   [+] Camera connected and receiving live frames.

[2/3] Initializing Audio (Threshold: 50.0 dB)...
   [+] Microphone & Speaker active. (Voice gate: 50.0 dB)

[3/3] Connecting to Gemini Live (gemini-3.1-flash-live-preview)...
   [+] Connected to Gemini Live (gemini-3.1-flash-live-preview)!

[+] Initiating Wake-Up Call with Gemini...
Gemini > Hey there, I'm here and can see your feed clearly.
[+] Wake-up call completed. Gemini is awake and ready!

======================================================================
 >>> CALL IN PROGRESS <<<
 * Speak into your PC microphone - Gemini hears you and talks back!
 * Voice threshold: 70.0 dB (Change in main.py top)
 * Camera is streaming live frames (~1 FPS).
 * You can also type questions below and press Enter anytime.
 * Type 'wake' to trigger wake-up call again, 'status' for stats, or 'exit' to quit.
======================================================================

[Mic: 54.2 dB [#####-----] (VOICE)]
You (Voice) > What do you see?
Gemini > I see a laptop on a desk with a coffee mug to the right.
```

- When you speak into your microphone above the threshold (e.g. 50 dB), `(VOICE)` is detected, Gemini transcribes your voice, and answers through the speakers.
- You can also type anytime without blocking or stopping the session.
- Type `exit` or press `Ctrl+C` to quit.
