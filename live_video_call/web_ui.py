"""Lightweight Web UI server for Gemini Live Video Call."""

import asyncio
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import time
from typing import Set

import websockets
from websockets.asyncio.server import serve, ServerConnection

try:
    from .camera_stream import CameraStream
    from .gemini_live import GeminiLiveSession
except (ImportError, ValueError):
    from camera_stream import CameraStream
    from gemini_live import GeminiLiveSession

logger = logging.getLogger("WebUI")

_CURRENT_CAMERA: CameraStream = None
_CURRENT_SESSION: GeminiLiveSession = None


class VideoHttpHandler(BaseHTTPRequestHandler):
    """Serves static index.html and MJPEG video feed."""

    def log_message(self, format, *args):
        # Silence standard HTTP access logs to keep terminal clean
        pass

    def do_GET(self):
        global _CURRENT_CAMERA, _CURRENT_SESSION

        if self.path in {"/", "/index.html"}:
            html_path = Path(__file__).resolve().parent / "index.html"
            if html_path.exists():
                content = html_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_error(404, "index.html not found")
            return

        if self.path == "/status":
            stats = _CURRENT_CAMERA.get_stats() if _CURRENT_CAMERA else {}
            stats["frames_to_gemini"] = _CURRENT_SESSION.frames_sent if _CURRENT_SESSION else 0
            stats["gemini_connected"] = _CURRENT_SESSION.is_connected if _CURRENT_SESSION else False
            body = json.dumps(stats).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/video_feed":
            # MJPEG stream
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()

            try:
                while True:
                    if _CURRENT_CAMERA:
                        jpeg_bytes = _CURRENT_CAMERA.get_latest_jpeg()
                        if jpeg_bytes:
                            self.wfile.write(b"--frame\r\n")
                            self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                            self.wfile.write(jpeg_bytes)
                            self.wfile.write(b"\r\n")
                    time.sleep(0.05)  # ~20 FPS browser preview
            except (BrokenPipeError, ConnectionResetError):
                pass
            return

        self.send_error(404, "Not Found")


class WebBridge:
    """Bridges WebSocket clients with the Gemini Live Session."""

    def __init__(self, camera: CameraStream, session: GeminiLiveSession, ws_port: int):
        self.camera = camera
        self.session = session
        self.ws_port = ws_port
        self.active_clients: Set[ServerConnection] = set()

        # Connect hooks from session
        self.session.frame_sent_callbacks.append(self._on_frame_sent)

    def _on_frame_sent(self, count: int):
        self.broadcast({"type": "frame_sent", "count": count})

    def broadcast(self, message: dict):
        if not self.active_clients:
            return
        payload = json.dumps(message)
        for client in list(self.active_clients):
            try:
                asyncio.create_task(client.send(payload))
            except Exception:
                pass

    async def handle_client(self, websocket: ServerConnection):
        self.active_clients.add(websocket)
        logger.info("Web client connected to WebSocket.")
        try:
            # Initial state messages
            await websocket.send(json.dumps({
                "type": "config",
                "model": self.session.model,
            }))
            await websocket.send(json.dumps({
                "type": "camera_status",
                "healthy": self.camera.is_healthy(),
            }))
            await websocket.send(json.dumps({
                "type": "frame_sent",
                "count": self.session.frames_sent,
            }))

            async for raw_msg in websocket:
                try:
                    data = json.loads(raw_msg)
                    if data.get("type") == "ask":
                        question = data.get("question", "").strip()
                        if question:
                            logger.info(f"User asked: {question}")

                            def on_chunk(chunk: str):
                                asyncio.create_task(
                                    websocket.send(json.dumps({"type": "chunk", "text": chunk}))
                                )

                            answer = await self.session.ask(question, on_chunk=on_chunk)
                            await websocket.send(json.dumps({
                                "type": "turn_complete",
                                "text": answer,
                            }))
                except Exception as e:
                    logger.error(f"Error handling WS message: {e}")
        finally:
            self.active_clients.discard(websocket)
            logger.info("Web client disconnected.")


async def _start_async_services(camera: CameraStream, session: GeminiLiveSession, ws_port: int):
    # Connect Gemini Live
    await session.connect()
    logger.info("Gemini Live session connected for Web UI.")

    bridge = WebBridge(camera, session, ws_port)
    async with serve(bridge.handle_client, "0.0.0.0", ws_port):
        logger.info(f"WebSocket server listening on ws://localhost:{ws_port}")
        # Periodic camera health broadcaster
        while session.is_connected:
            bridge.broadcast({
                "type": "camera_status",
                "healthy": camera.is_healthy(),
            })
            await asyncio.sleep(2.0)


def run_web_server(camera_url: str, api_key: str, model: str, fps: float = 1.0, port: int = 8000):
    """Run both HTTP MJPEG server and WebSocket Q&A server."""
    global _CURRENT_CAMERA, _CURRENT_SESSION

    print("=" * 65)
    print(" 🌐  Gemini Live Video Call - Web Interface  🌐 ")
    print("=" * 65)

    _CURRENT_CAMERA = CameraStream(source=camera_url)
    _CURRENT_CAMERA.start()

    frame_interval = 1.0 / max(0.1, fps)
    _CURRENT_SESSION = GeminiLiveSession(
        api_key=api_key,
        model=model,
        camera=_CURRENT_CAMERA,
        frame_interval_sec=frame_interval,
    )

    # Start HTTP server in a background daemon thread
    http_server = ThreadingHTTPServer(("0.0.0.0", port), VideoHttpHandler)
    http_thread = threading.Thread(target=http_server.serve_forever, daemon=True, name="HttpServer")
    http_thread.start()

    ws_port = port + 1
    print(f"\n🚀 Web UI is live at: http://localhost:{port}")
    print(f"📡 WebSocket streaming on ws://localhost:{ws_port}")
    print("Open http://localhost:{port} in your browser to view the camera & chat.")
    print("Press Ctrl+C in terminal to stop.\n" + "=" * 65 + "\n")

    try:
        asyncio.run(_start_async_services(_CURRENT_CAMERA, _CURRENT_SESSION, ws_port))
    except KeyboardInterrupt:
        print("\nStopping web server...")
    finally:
        http_server.shutdown()
        if _CURRENT_SESSION:
            asyncio.run(_CURRENT_SESSION.close())
        if _CURRENT_CAMERA:
            _CURRENT_CAMERA.stop()
        print("Web server stopped.")
