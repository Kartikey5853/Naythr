"""
Android ADB Phone Dialer & Companion App Helper Module for Naythr / Marvin.
==========================================================================
Handles wireless and USB Android device detection, status checks,
multi-device targeting via ANDROID_DEVICE_ID or auto-selection of a single ready device,
phone number validation, shell injection protection, and direct cellular calling via
Marvin Call Companion app or ADB ACTION_CALL.

Key features:
1. Companion App Integration (`com.naythr.callcompanion`):
   - Directly triggers cellular phone calls without requiring the user to tap Call on screen.
   - Bypasses Wireless ADB shell UID 2000 restrictions legally via granted app permissions.
2. Direct ADB `ACTION_CALL`:
   - Supported when device has root or 'USB debugging (Security settings)' enabled.
3. Strict Sanitization:
   - Sanitizes and validates phone numbers against injection attempts.
4. Honest Status Reporting:
   - Distinguishes CALL_INITIATED, PERMISSION_BLOCKED, DIALER_OPENED, and DISCONNECTED.
   - Never simulates success or silently opens dialer keypad when direct calling is requested.
"""

import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("AndroidDialer")

DEFAULT_ADB_PATH = r"C:\Users\raj55\Downloads\platform-tools-latest-windows\platform-tools\adb.exe"

# Companion App configuration
COMPANION_PACKAGE = "com.naythr.callcompanion"
COMPANION_CALL_ACTIVITY = f"{COMPANION_PACKAGE}/.CallActivity"
COMPANION_ACTION = "com.naythr.callcompanion.ACTION_CALL"


class AndroidDialer:
    """Manages Android ADB connection and launches phone call and companion app intents."""

    def __init__(self, adb_path: Optional[str] = None, device_id: Optional[str] = None):
        self._custom_adb_path = adb_path
        self._custom_device_id = device_id

    def get_adb_command(self) -> str:
        """
        Resolve the ADB executable path.
        Priority:
        1. Explicit path passed to constructor or ADB_PATH env var
        2. Known default path if it exists on disk
        3. 'adb' from system PATH
        """
        candidates = []
        if self._custom_adb_path:
            candidates.append(self._custom_adb_path)

        env_adb = os.getenv("ADB_PATH", "").strip()
        if env_adb:
            candidates.append(env_adb)

        candidates.append(DEFAULT_ADB_PATH)

        for cand in candidates:
            if cand and os.path.isfile(cand):
                return cand

        # Fallback to system PATH
        if shutil.which("adb"):
            return "adb"

        # Return env_adb or default if set, else 'adb'
        return env_adb or DEFAULT_ADB_PATH or "adb"

    def get_configured_device_id(self) -> Optional[str]:
        """Return configured target device ID from constructor or ANDROID_DEVICE_ID env var."""
        if self._custom_device_id and self._custom_device_id.strip():
            return self._custom_device_id.strip()
        env_dev = os.getenv("ANDROID_DEVICE_ID", "").strip()
        return env_dev if env_dev else None

    def _get_subprocess_kwargs(self) -> Dict[str, Any]:
        """Return subprocess kwargs appropriate for host OS."""
        kwargs: Dict[str, Any] = {
            "capture_output": True,
            "text": True,
            "timeout": 5,
        }
        if sys.platform == "win32":
            # Suppress console window popup on Windows
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
        return kwargs

    def resolve_target_device(self) -> Tuple[bool, str, str, Optional[str]]:
        """
        Query ADB devices and resolve the appropriate target device.
        Returns:
            (is_connected, status_code, message, selected_device_id)
            status_code can be:
            - 'DEVICE_READY': Single device ready or specified ANDROID_DEVICE_ID ready
            - 'MULTIPLE_DEVICES': Multiple devices ready, but ANDROID_DEVICE_ID not set
            - 'DISCONNECTED': No devices found or specified device not connected/offline
            - 'UNAUTHORIZED': Specified or all detected devices unauthorized
            - 'ADB_NOT_FOUND': adb executable not found
            - 'ERROR': Subprocess or timeout failure
        """
        adb_cmd = self.get_adb_command()
        try:
            res = subprocess.run([adb_cmd, "devices"], **self._get_subprocess_kwargs())
            if res.returncode != 0:
                return False, "ERROR", f"ADB returned exit code {res.returncode}: {res.stderr.strip()}", None

            output = res.stdout.strip()
            lines = [line.strip() for line in output.splitlines() if line.strip()]

            # Parse device lines (Format: `<device_id>\t<state>`)
            devices_map: Dict[str, str] = {}
            for line in lines:
                if line.startswith("List of devices"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    dev_id, state = parts[0], parts[1]
                    devices_map[dev_id] = state

            configured_id = self.get_configured_device_id()

            # Case A: Specific device ID configured via constructor or ANDROID_DEVICE_ID env var
            if configured_id:
                if configured_id not in devices_map:
                    return False, "DISCONNECTED", f"Configured device '{configured_id}' is not connected via ADB.", None

                state = devices_map[configured_id]
                if state == "device":
                    return True, "DEVICE_READY", f"Device '{configured_id}' connected and ready.", configured_id
                elif state == "unauthorized":
                    return False, "UNAUTHORIZED", f"Device '{configured_id}' is unauthorized. Please accept debugging prompt on phone.", configured_id
                elif state == "offline":
                    return False, "OFFLINE", f"Device '{configured_id}' is offline.", configured_id
                else:
                    return False, "DISCONNECTED", f"Device '{configured_id}' is in state '{state}'.", configured_id

            # Case B: No specific device ID configured -> inspect all detected devices
            if not devices_map:
                return False, "DISCONNECTED", "No Android devices found via ADB.", None

            ready_devices = [dev_id for dev_id, st in devices_map.items() if st == "device"]
            unauthorized_devices = [dev_id for dev_id, st in devices_map.items() if st == "unauthorized"]

            if len(ready_devices) == 1:
                selected_id = ready_devices[0]
                return True, "DEVICE_READY", f"Device connected and ready ({selected_id}).", selected_id
            elif len(ready_devices) > 1:
                # Check if multiple devices are just mDNS + IP address duplicates of the same device
                ip_devices = [d for d in ready_devices if re.match(r"^\d+\.\d+\.\d+\.\d+:\d+$", d)]
                mdns_devices = [d for d in ready_devices if "_adb-tls-connect" in d]
                if len(ip_devices) == 1 and len(ready_devices) == (len(ip_devices) + len(mdns_devices)):
                    selected_id = ip_devices[0]
                    return True, "DEVICE_READY", f"Device connected and ready ({selected_id}).", selected_id

                dev_list = ", ".join(ready_devices)
                return False, "MULTIPLE_DEVICES", f"Multiple Android devices found ({dev_list}). Please set ANDROID_DEVICE_ID in .env to target a specific device.", None
            elif unauthorized_devices:
                unauth_list = ", ".join(unauthorized_devices)
                return False, "UNAUTHORIZED", f"Device(s) unauthorized ({unauth_list}). Please accept debugging prompt on phone.", None
            else:
                return False, "DISCONNECTED", f"No authorized device ready. Device states: {devices_map}", None

        except FileNotFoundError:
            return False, "ADB_NOT_FOUND", f"ADB executable not found at '{adb_cmd}'.", None
        except subprocess.TimeoutExpired:
            return False, "ERROR", "ADB command timed out while querying devices.", None
        except Exception as e:
            return False, "ERROR", f"Error checking ADB device: {e}", None

    def check_device_connected(self) -> Tuple[bool, str, str]:
        """
        Check if an Android device is connected and authorized via ADB.
        Backwards-compatible signature returning (is_connected, status_code, message).
        """
        is_conn, status_code, msg, _ = self.resolve_target_device()
        return is_conn, status_code, msg

    @staticmethod
    def validate_phone_number(phone_number: str) -> Tuple[bool, str]:
        """
        Clean, validate, and sanitize phone number.
        Strictly prevents command injection or arbitrary shell characters.
        Allowed formats: digits with optional leading '+' (3 to 15 digits).
        Returns: (is_valid, cleaned_number)
        """
        if not phone_number:
            return False, ""

        raw = str(phone_number).strip()
        # Reject any shell metacharacters or command separators
        if any(c in raw for c in [";", "&", "|", "`", "$", "\n", "\r", '"', "'", "\\", ">", "<", "{", "}"]):
            logger.warning(f"Rejected phone number containing unsafe shell characters: '{raw}'")
            return False, ""

        has_plus = raw.startswith("+")
        digits = re.sub(r"[^\d]", "", raw)

        if len(digits) < 3 or len(digits) > 15:
            return False, ""

        cleaned = f"+{digits}" if has_plus else digits
        return True, cleaned

    def check_companion_installed(self, target_device_id: str) -> bool:
        """Check if Marvin Call Companion app (com.naythr.callcompanion) is installed on device."""
        adb_cmd = self.get_adb_command()
        cmd = [adb_cmd, "-s", target_device_id, "shell", "pm", "path", COMPANION_PACKAGE]
        try:
            res = subprocess.run(cmd, **self._get_subprocess_kwargs())
            return res.returncode == 0 and "package:" in (res.stdout or "")
        except Exception as e:
            logger.debug(f"Error checking companion app installation: {e}")
            return False

    def check_companion_permission(self, target_device_id: str) -> bool:
        """Check if com.naythr.callcompanion has been granted CALL_PHONE permission."""
        adb_cmd = self.get_adb_command()
        cmd = [adb_cmd, "-s", target_device_id, "shell", "dumpsys", "package", COMPANION_PACKAGE]
        try:
            res = subprocess.run(cmd, **self._get_subprocess_kwargs())
            if res.returncode != 0:
                return False
            output = res.stdout or ""
            # Search for CALL_PHONE in permissions block
            for line in output.splitlines():
                if "android.permission.CALL_PHONE" in line:
                    if "granted=false" in line.lower():
                        return False
                    if "granted=true" in line.lower() or "gids=" in line:
                        return True
            return "android.permission.CALL_PHONE: granted=true" in output
        except Exception as e:
            logger.debug(f"Error checking companion app permissions: {e}")
            return False

    def call_via_companion(self, phone_number: str, target_device_id: str) -> Dict[str, Any]:
        """
        Trigger direct cellular phone call via Marvin Call Companion app.
        Bypasses UID 2000 restrictions because the companion app holds CALL_PHONE permission.
        """
        adb_cmd = self.get_adb_command()
        # Launch companion's CallActivity with extra phone_number
        cmd = [
            adb_cmd, "-s", target_device_id, "shell", "am", "start",
            "-n", COMPANION_CALL_ACTIVITY,
            "--es", "phone_number", phone_number,
        ]
        try:
            res = subprocess.run(cmd, **self._get_subprocess_kwargs())
            stdout_str = res.stdout if isinstance(res.stdout, str) else (str(res.stdout or ""))
            stderr_str = res.stderr if isinstance(res.stderr, str) else (str(res.stderr or ""))
            output = (stdout_str + " " + stderr_str).strip()

            if res.returncode == 0 and "Error:" not in output and "SecurityException" not in output:
                print(f"[COMPANION CALL] Initiated cellular call via Marvin Call Companion to {phone_number} on '{target_device_id}'")
                return {
                    "success": True,
                    "status": "CALL_INITIATED",
                    "method": "COMPANION_APP",
                    "message": f"Successfully initiated cellular call to {phone_number} via Marvin Call Companion app.",
                    "number": phone_number,
                    "device_id": target_device_id,
                }
            else:
                err_msg = output.strip() or f"Exit code {res.returncode}"
                return {
                    "success": False,
                    "status": "ERROR",
                    "method": "COMPANION_APP",
                    "message": f"Companion app call launch failed: {err_msg}",
                    "number": phone_number,
                    "device_id": target_device_id,
                }
        except Exception as e:
            return {
                "success": False,
                "status": "ERROR",
                "method": "COMPANION_APP",
                "message": f"Error launching companion app: {e}",
                "number": phone_number,
                "device_id": target_device_id,
            }

    def make_phone_call(
        self,
        phone_number: str,
        allow_dialer_fallback: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Initiate an automatic cellular phone call without requiring the user to tap Call.

        Priority strategy:
        1. Check Marvin Call Companion app (`com.naythr.callcompanion`). If installed & permitted,
           triggers hands-free call directly.
        2. If companion is installed but missing CALL_PHONE permission, reports PERMISSION_DENIED.
        3. If companion app is not installed, attempts direct ADB `ACTION_CALL`.
           - If allowed (e.g. root or USB debugging security settings enabled): triggers direct call.
           - If blocked by Android security (`SecurityException`): reports PERMISSION_BLOCKED.
             Does NOT silently open keypad dialer unless allow_dialer_fallback is explicitly True.
        """
        # Determine fallback behavior (defaults to False to avoid keypad screen popping)
        if allow_dialer_fallback is None:
            allow_dialer_fallback = os.getenv("ALLOW_DIALER_FALLBACK", "false").lower() in ("true", "1")

        valid, cleaned_num = self.validate_phone_number(phone_number)
        if not valid:
            logger.warning(f"Invalid phone number provided for call: '{phone_number}'")
            return {
                "success": False,
                "status": "INVALID_NUMBER",
                "message": f"Invalid phone number format: '{phone_number}'",
                "number": phone_number,
            }

        is_connected, status_code, status_msg, target_device_id = self.resolve_target_device()
        if not is_connected or not target_device_id:
            logger.warning(f"Cannot place call: {status_msg}")
            return {
                "success": False,
                "status": status_code,
                "message": status_msg,
                "number": cleaned_num,
            }

        adb_cmd = self.get_adb_command()

        # Step 1: Check Marvin Call Companion App
        if self.check_companion_installed(target_device_id):
            if self.check_companion_permission(target_device_id):
                companion_res = self.call_via_companion(cleaned_num, target_device_id)
                if companion_res.get("success"):
                    return companion_res
            else:
                return {
                    "success": False,
                    "status": "PERMISSION_DENIED",
                    "message": (
                        f"Marvin Call Companion app is installed on '{target_device_id}' but lacks CALL_PHONE permission.\n"
                        f"To enable automatic hands-free calling, open 'Marvin Call Companion' on your phone and tap 'Grant Phone Permission', "
                        f"or go to Settings -> Apps -> Manage Apps -> Marvin Call Companion -> Permissions -> Phone -> Allow."
                    ),
                    "number": cleaned_num,
                    "device_id": target_device_id,
                }

        # Step 2: Attempt direct ACTION_CALL (succeeds if root or USB Debugging Security settings enabled)
        call_intent = f"tel:{cleaned_num}"
        cmd_call = [
            adb_cmd, "-s", target_device_id, "shell", "am", "start",
            "-a", "android.intent.action.CALL",
            "-d", call_intent
        ]

        try:
            res = subprocess.run(cmd_call, **self._get_subprocess_kwargs())
            stdout_str = res.stdout if isinstance(res.stdout, str) else (str(res.stdout or ""))
            stderr_str = res.stderr if isinstance(res.stderr, str) else (str(res.stderr or ""))
            output = (stdout_str + " " + stderr_str).strip()

            # Check for direct ACTION_CALL success
            if res.returncode == 0 and "Error:" not in output and "SecurityException" not in output and "Permission Denial" not in output:
                print(f"[ANDROID CALL] Placed direct cellular call on device '{target_device_id}' to number {cleaned_num}")
                return {
                    "success": True,
                    "status": "CALL_INITIATED",
                    "method": "DIRECT_ADB",
                    "message": f"Successfully initiated direct cellular call to {cleaned_num} on device '{target_device_id}'",
                    "number": cleaned_num,
                    "device_id": target_device_id,
                }

            # Detect SecurityException / Permission Denial for CALL_PHONE permission
            if "SecurityException" in output or "Permission Denial" in output or "CALL_PHONE" in output:
                logger.warning(f"[ANDROID CALL] ACTION_CALL denied on '{target_device_id}' (CALL_PHONE permission required).")
                if allow_dialer_fallback:
                    dial_res = self.open_dialer(phone_number)
                    if dial_res.get("success"):
                        dial_res["fallback"] = True
                        dial_res["message"] = (
                            f"Wireless ADB shell lacks CALL_PHONE permission on '{target_device_id}'. "
                            f"Opened phone dialer pre-populated with {cleaned_num}. Please tap Call on your phone."
                        )
                        return dial_res

                # Strict honest blocker report without opening dialer
                return {
                    "success": False,
                    "status": "PERMISSION_BLOCKED",
                    "message": (
                        f"Direct automatic calling blocked by Android on '{target_device_id}': "
                        f"SecurityException: CALL_PHONE permission denied.\n"
                        f"To enable hands-free automatic calling without tapping the screen:\n"
                        f"1. Install the Marvin Call Companion app: adb install android_companion/app/build/outputs/apk/debug/app-debug.apk\n"
                        f"2. Grant Phone permission in phone settings or app UI.\n"
                        f"Alternatively, enable 'USB debugging (Security settings)' in Android Developer Options."
                    ),
                    "number": cleaned_num,
                    "device_id": target_device_id,
                }

            err_msg = output.strip() or f"Exit code {res.returncode}"
            logger.error(f"Failed to place call via ADB on device '{target_device_id}': {err_msg}")
            return {
                "success": False,
                "status": "ERROR",
                "message": f"ADB call placement failed on '{target_device_id}': {err_msg}",
                "number": cleaned_num,
                "device_id": target_device_id,
            }

        except FileNotFoundError:
            return {
                "success": False,
                "status": "ADB_NOT_FOUND",
                "message": f"ADB executable not found at '{adb_cmd}'.",
                "number": cleaned_num,
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "status": "ERROR",
                "message": "ADB command timed out while placing call.",
                "number": cleaned_num,
                "device_id": target_device_id,
            }
        except Exception as e:
            return {
                "success": False,
                "status": "ERROR",
                "message": f"Error executing ADB call command: {e}",
                "number": cleaned_num,
                "device_id": target_device_id,
            }

    def open_dialer(self, phone_number: str) -> Dict[str, Any]:
        """
        Launch the Android dialer with the phone number using ACTION_DIAL.
        Passes `-s <device_id>` to target the resolved device.
        Pre-populates keypad dialer screen.
        """
        valid, cleaned_num = self.validate_phone_number(phone_number)
        if not valid:
            logger.warning(f"Invalid phone number provided for dialer: '{phone_number}'")
            return {
                "success": False,
                "status": "INVALID_NUMBER",
                "message": f"Invalid phone number format: '{phone_number}'",
                "number": phone_number,
            }

        is_connected, status_code, status_msg, target_device_id = self.resolve_target_device()
        if not is_connected or not target_device_id:
            logger.warning(f"Cannot open dialer: {status_msg}")
            return {
                "success": False,
                "status": status_code,
                "message": status_msg,
                "number": cleaned_num,
            }

        adb_cmd = self.get_adb_command()
        dial_intent = f"tel:{cleaned_num}"
        cmd = [
            adb_cmd, "-s", target_device_id, "shell", "am", "start",
            "-a", "android.intent.action.DIAL",
            "-d", dial_intent
        ]

        try:
            res = subprocess.run(cmd, **self._get_subprocess_kwargs())
            stdout_str = res.stdout if isinstance(res.stdout, str) else (str(res.stdout or ""))
            stderr_str = res.stderr if isinstance(res.stderr, str) else (str(res.stderr or ""))
            output = (stdout_str + " " + stderr_str).strip()

            if res.returncode == 0 and "Error:" not in output:
                print(f"[ANDROID DIALER] Opened Android phone dialer on device '{target_device_id}' with number {cleaned_num}")
                return {
                    "success": True,
                    "status": "DIALER_OPENED",
                    "message": f"Successfully opened Android dialer for {cleaned_num} on device '{target_device_id}'",
                    "number": cleaned_num,
                    "device_id": target_device_id,
                }
            else:
                err_msg = output.strip() or f"Exit code {res.returncode}"
                logger.error(f"Failed to launch Android dialer via ADB on device '{target_device_id}': {err_msg}")
                return {
                    "success": False,
                    "status": "ERROR",
                    "message": f"ADB launch dialer failed on '{target_device_id}': {err_msg}",
                    "number": cleaned_num,
                    "device_id": target_device_id,
                }

        except FileNotFoundError:
            return {
                "success": False,
                "status": "ADB_NOT_FOUND",
                "message": f"ADB executable not found at '{adb_cmd}'.",
                "number": cleaned_num,
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "status": "ERROR",
                "message": "ADB command timed out while launching dialer.",
                "number": cleaned_num,
                "device_id": target_device_id,
            }
        except Exception as e:
            return {
                "success": False,
                "status": "ERROR",
                "message": f"Error executing ADB dialer command: {e}",
                "number": cleaned_num,
                "device_id": target_device_id,
            }


# Singleton helper instance
_dialer_instance = AndroidDialer()


def place_android_call(
    phone_number: str,
    adb_path: Optional[str] = None,
    device_id: Optional[str] = None,
    allow_dialer_fallback: Optional[bool] = None,
) -> Dict[str, Any]:
    """Convenience helper function to place direct Android call (Companion App or ACTION_CALL)."""
    if adb_path or device_id:
        dialer = AndroidDialer(adb_path=adb_path, device_id=device_id)
    else:
        dialer = _dialer_instance
    return dialer.make_phone_call(phone_number, allow_dialer_fallback=allow_dialer_fallback)


def open_android_dialer(
    phone_number: str,
    adb_path: Optional[str] = None,
    device_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Convenience helper function to open Android dialer (ACTION_DIAL)."""
    if adb_path or device_id:
        dialer = AndroidDialer(adb_path=adb_path, device_id=device_id)
    else:
        dialer = _dialer_instance
    return dialer.open_dialer(phone_number)
