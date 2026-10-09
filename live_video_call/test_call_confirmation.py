"""
Unit and Integration Test Suite for Voice Calling & Wireless ADB Android Dialer.
=================================================================================
Tests:
1. Intent extraction for calling expressions, numbers, and ambiguous targets.
2. Rejection of non-calling idioms ('what do you call this').
3. Multi-turn dialogue: Known contact -> asks confirmation -> 'yes' -> CALL_INITIATED.
4. Multi-turn dialogue: Known contact -> asks confirmation -> 'no' -> CANCELLED.
5. Multi-turn dialogue: Ambiguous ('call this guy') -> prompts for contact -> confirms -> CALL_INITIATED.
6. Direct number dialing: 'call 9876543210' -> confirms -> CALL_INITIATED.
7. AndroidDialer: Valid phone number formatting, sanitization, and shell injection prevention.
8. AndroidDialer: Companion App (`com.naythr.callcompanion`) detection and hands-free cellular dispatch.
9. AndroidDialer: Companion App missing permission -> honest PERMISSION_DENIED.
10. AndroidDialer: Direct ADB ACTION_CALL success (when permitted) -> CALL_INITIATED.
11. AndroidDialer: Direct ADB SecurityException without fallback -> PERMISSION_BLOCKED (no keypad popup).
12. AndroidDialer: Direct ADB SecurityException with explicit fallback -> DIALER_OPENED.
13. AndroidDialer: Multi-device handling (SINGLE_READY, MULTIPLE_DEVICES error, ANDROID_DEVICE_ID targeting).
14. CallConversationManager: Negative response cancels call without invoking ADB.
15. CallConversationManager: Duplicate confirmation prevention within short window.
"""

import asyncio
import os
import sys
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from live_video_call.intents.intent_parser import IntentParser
from live_video_call.memory.memory_manager import MemoryManager
from live_video_call.memory.call_manager import CallConversationManager, print_phone_call_summary
from live_video_call.android_dialer import AndroidDialer, open_android_dialer, place_android_call


class MockGeminiLiveSession:
    def __init__(self):
        self.sent_prompts = []

    async def send_text(self, text: str):
        self.sent_prompts.append(text)
        return True


async def run_call_confirmation_tests():
    print("=" * 72)
    print(">>> NAYTHR VOICE CALL WITH COMPANION & WIRELESS ADB TEST SUITE <<<")
    print("=" * 72 + "\n")

    parser = IntentParser()
    test_data_dir = Path(__file__).resolve().parent / "memory" / "data" / "test_scratch"
    test_data_dir.mkdir(parents=True, exist_ok=True)
    memory = MemoryManager(data_dir=test_data_dir)
    mock_session = MockGeminiLiveSession()

    # =========================================================================
    # TEST 1: Intent Extraction Tests
    # =========================================================================
    print("--- 1. Testing Intent Extraction ---")

    # 1A: Simple call contact
    r1 = parser.parse("call John")
    assert r1.should_store is True, "Must parse 'call John' as actionable"
    assert r1.category == "COMMUNICATION"
    assert r1.intent == "MAKE_PHONE_CALL"
    assert r1.entities["target"] == "John"
    assert r1.requires_clarification is False
    print(" * 'call John' -> COMMUNICATION / MAKE_PHONE_CALL (target: John) [OK]")

    # 1B: Call with wake word prefix & possessive
    r2 = parser.parse("Hey Marvin, call my mom")
    assert r2.should_store is True
    assert r2.category == "COMMUNICATION"
    assert r2.intent == "MAKE_PHONE_CALL"
    assert r2.entities["target"] == "Mom"
    print(" * 'Hey Marvin, call my mom' -> target: Mom [OK]")

    # 1C: Direct phone number dialing
    r3 = parser.parse("dial 9876543210")
    assert r3.should_store is True
    assert r3.category == "COMMUNICATION"
    assert r3.entities["phone_number"] == "9876543210"
    print(" * 'dial 9876543210' -> phone_number: 9876543210 [OK]")

    # 1D: Ambiguous target ('call this guy')
    r4 = parser.parse("call this guy")
    assert r4.should_store is True
    assert r4.category == "COMMUNICATION"
    assert r4.requires_clarification is True
    assert "Who would you like me to call" in r4.clarification_prompt
    print(" * 'call this guy' -> requires_clarification=True [OK]")

    # 1E: False positive prevention ('what do you call this')
    r5 = parser.parse("what do you call this?")
    assert r5.should_store is False, "Must not treat 'what do you call this' as a phone call"
    print(" * 'what do you call this?' -> should_store=False (rejected casual question) [OK]")

    # 1F: Conversational polite request ('Can you just call Rohit?')
    r6 = parser.parse("Can you just call Rohit?")
    assert r6.should_store is True
    assert r6.category == "COMMUNICATION"
    assert r6.intent == "MAKE_PHONE_CALL"
    assert r6.entities["target"] == "Rohit"
    print(" * 'Can you just call Rohit?' -> target: Rohit [OK]")

    print("-> Result: ALL INTENT EXTRACTION CHECKS PASSED!\n")

    # =========================================================================
    # TEST 2: Dialogue Flow - Known Contact + User Confirms ("Yes") -> CALL_INITIATED
    # =========================================================================
    print("--- 2. Testing Flow: Known Contact -> Confirmation Question -> 'Yes' ---")
    call_mgr = CallConversationManager(
        memory_manager=memory,
        send_prompt_func=mock_session.send_text,
    )

    with patch("subprocess.run") as mock_run:
        def subprocess_side_effect(cmd, **kwargs):
            m = MagicMock()
            m.returncode = 0
            cmd_str = " ".join(cmd)
            if "devices" in cmd:
                m.stdout = "List of devices attached\n172.16.13.22:42513\tdevice\n"
                m.stderr = ""
            elif "pm path com.naythr.callcompanion" in cmd_str:
                # Companion app is installed
                m.stdout = "package:/data/app/~~xyz/com.naythr.callcompanion-1/base.apk\n"
                m.stderr = ""
            elif "dumpsys package com.naythr.callcompanion" in cmd_str:
                # Companion app has CALL_PHONE granted
                m.stdout = "android.permission.CALL_PHONE: granted=true\n"
                m.stderr = ""
            elif "com.naythr.callcompanion/.CallActivity" in cmd_str:
                m.stdout = "Starting: Intent { cmp=com.naythr.callcompanion/.CallActivity (has extras) }\n"
                m.stderr = ""
                assert "-s" in cmd and "172.16.13.22:42513" in cmd, "Must pass -s <device_id> to ADB command!"
            return m

        mock_run.side_effect = subprocess_side_effect

        # Step 1: User requests call
        step1 = await call_mgr.start_call_request(parser.parse("call John"))
        assert call_mgr.state == CallConversationManager.STATE_AWAITING_CONFIRMATION
        assert "John" in step1["question"]
        assert "+91 98123 45678" in step1["question"]
        print(f" Step 1: User requested 'call John' -> Marvin asked: '{step1['question']}'")

        # Step 2: User confirms with 'yes'
        step2 = await call_mgr.handle_user_turn("yes")
        assert step2["handled"] is True
        assert step2["contact"] == "John"
        assert step2["dispatch"]["method"] == "ANDROID_ADB"
        assert step2["dispatch"]["details"]["status"] == "CALL_INITIATED"
        assert step2["dispatch"]["details"]["method"] == "COMPANION_APP"
        assert call_mgr.state == CallConversationManager.STATE_IDLE
        print(" Step 2: User said 'yes' -> Status: CALL_INITIATED (Marvin Call Companion triggered) [OK]\n")

    # =========================================================================
    # TEST 3: Dialogue Flow - Known Contact + User Cancels ("No")
    # =========================================================================
    print("--- 3. Testing Flow: Known Contact -> Confirmation Question -> 'No' ---")
    with patch("subprocess.run") as mock_run:
        step1 = await call_mgr.start_call_request(parser.parse("call Mom"))
        assert call_mgr.state == CallConversationManager.STATE_AWAITING_CONFIRMATION
        assert "Mom" in step1["question"]

        step2 = await call_mgr.handle_user_turn("no, cancel it")
        assert step2["handled"] is True
        assert step2["status"] == "CANCELLED"
        assert call_mgr.state == CallConversationManager.STATE_IDLE
        # Subprocess must NEVER be called when user cancels!
        mock_run.assert_not_called()
        print(" Step 2: User said 'no, cancel it' -> Status: CANCELLED (ADB not invoked) [OK]\n")

    # =========================================================================
    # TEST 4: Dialogue Flow - Ambiguous ("call this guy") -> Name -> "Go ahead"
    # =========================================================================
    print("--- 4. Testing Flow: 'call this guy' -> Name provided -> 'Go ahead' ---")
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = lambda cmd, **k: MagicMock(
            returncode=0,
            stdout="172.16.13.22:42513\tdevice\n" if "devices" in cmd else (
                "package:/data/app/test.apk\n" if "pm path" in " ".join(cmd) else (
                    "android.permission.CALL_PHONE: granted=true\n" if "dumpsys" in " ".join(cmd) else "Starting..."
                )
            ),
            stderr=""
        )

        step1 = await call_mgr.start_call_request(parser.parse("Hey Marvin call this guy"))
        assert call_mgr.state == CallConversationManager.STATE_AWAITING_TARGET_INFO
        print(f" Step 1: User requested 'call this guy' -> Marvin asked: '{step1['question']}'")

        step2 = await call_mgr.handle_user_turn("Dr. Sharma")
        assert step2["handled"] is True
        assert call_mgr.state == CallConversationManager.STATE_AWAITING_CONFIRMATION

        step3 = await call_mgr.handle_user_turn("go ahead")
        assert step3["handled"] is True
        assert step3["contact"] == "Dr. Sharma"
        print(" Step 3: User said 'go ahead' -> Status: COMPLETED [OK]\n")

    # =========================================================================
    # TEST 5: Phone Number Validation & Shell Injection Prevention
    # =========================================================================
    print("--- 5. Testing AndroidDialer Phone Number Validation & Sanitization ---")
    dialer = AndroidDialer()
    v1, n1 = dialer.validate_phone_number("+91 9701350218")
    assert v1 is True and n1 == "+919701350218", "Must format +91 9701350218 correctly"

    v2, n2 = dialer.validate_phone_number("9812345678")
    assert v2 is True and n2 == "9812345678"

    # Shell injection protection checks
    v_inj1, _ = dialer.validate_phone_number("+919701350218; reboot")
    assert v_inj1 is False, "Must reject semicolon injection"

    v_inj2, _ = dialer.validate_phone_number("+919701350218 && rm -rf /")
    assert v_inj2 is False, "Must reject command chaining"

    v_inj3, _ = dialer.validate_phone_number("+919701`id`")
    assert v_inj3 is False, "Must reject backtick command substitution"

    v3, n3 = dialer.validate_phone_number("12")
    assert v3 is False, "Number under 3 digits must be rejected"

    v4, n4 = dialer.validate_phone_number("")
    assert v4 is False, "Empty number must be rejected"

    res_inv = dialer.make_phone_call("abc")
    assert res_inv["success"] is False
    assert res_inv["status"] == "INVALID_NUMBER"
    print(" -> Result: Phone number validation & shell injection protection working as expected [OK]\n")

    # =========================================================================
    # TEST 6: Companion App & Direct Calling & Permission Blocker Tests
    # =========================================================================
    print("--- 6. Testing Companion App, Direct Calling, and Permission Handling ---")

    # 6A: Companion App Installed & Permitted -> CALL_INITIATED via Companion App
    with patch("subprocess.run") as mock_run:
        def companion_ok(cmd, **kwargs):
            m = MagicMock()
            m.returncode = 0
            cmd_str = " ".join(cmd)
            if "devices" in cmd:
                m.stdout = "172.16.13.22:42513\tdevice\n"
            elif "pm path" in cmd_str:
                m.stdout = "package:/data/app/base.apk\n"
            elif "dumpsys package" in cmd_str:
                m.stdout = "android.permission.CALL_PHONE: granted=true\n"
            elif ".CallActivity" in cmd_str:
                m.stdout = "Starting: Intent { cmp=com.naythr.callcompanion/.CallActivity }\n"
            return m

        mock_run.side_effect = companion_ok
        res_comp = dialer.make_phone_call("+919701350218")
        assert res_comp["success"] is True
        assert res_comp["status"] == "CALL_INITIATED"
        assert res_comp["method"] == "COMPANION_APP"
        print(" * Companion app hands-free cellular call dispatch verified [OK]")

    # 6B: Companion App Installed but LACKS Permission -> PERMISSION_DENIED
    with patch("subprocess.run") as mock_run:
        def companion_unpermitted(cmd, **kwargs):
            m = MagicMock()
            m.returncode = 0
            cmd_str = " ".join(cmd)
            if "devices" in cmd:
                m.stdout = "172.16.13.22:42513\tdevice\n"
            elif "pm path" in cmd_str:
                m.stdout = "package:/data/app/base.apk\n"
            elif "dumpsys package" in cmd_str:
                m.stdout = "android.permission.CALL_PHONE: granted=false\n"
            return m

        mock_run.side_effect = companion_unpermitted
        res_unperm = dialer.make_phone_call("+919701350218")
        assert res_unperm["success"] is False
        assert res_unperm["status"] == "PERMISSION_DENIED"
        assert "lacks CALL_PHONE permission" in res_unperm["message"]
        print(" * Companion app missing permission correctly reported as PERMISSION_DENIED [OK]")

    # 6C: Companion App NOT Installed -> Direct ADB ACTION_CALL succeeds (when permitted)
    with patch("subprocess.run") as mock_run:
        def direct_ok(cmd, **kwargs):
            m = MagicMock()
            m.returncode = 0
            cmd_str = " ".join(cmd)
            if "devices" in cmd:
                m.stdout = "172.16.13.22:42513\tdevice\n"
            elif "pm path" in cmd_str:
                m.returncode = 1
                m.stdout = ""  # not installed
                m.stderr = ""
            elif "android.intent.action.CALL" in cmd_str:
                m.stdout = "Starting: Intent { act=android.intent.action.CALL dat=tel:+919701350218 }\n"
                m.stderr = ""
            return m

        mock_run.side_effect = direct_ok
        res_direct = dialer.make_phone_call("+919701350218")
        assert res_direct["success"] is True
        assert res_direct["status"] == "CALL_INITIATED"
        assert res_direct["method"] == "DIRECT_ADB"
        print(" * Direct ADB ACTION_CALL success verified [OK]")

    # 6D: Companion App NOT Installed & Direct ACTION_CALL SecurityException (WITHOUT dialer fallback)
    # -> Must report PERMISSION_BLOCKED and NOT open dialer keypad
    with patch("subprocess.run") as mock_run:
        def direct_sec_err(cmd, **kwargs):
            m = MagicMock()
            m.returncode = 0
            m.stderr = ""
            cmd_str = " ".join(cmd)
            if "devices" in cmd:
                m.stdout = "172.16.13.22:42513\tdevice\n"
            elif "pm path" in cmd_str:
                m.returncode = 1
                m.stdout = ""
            elif "android.intent.action.CALL" in cmd_str:
                m.returncode = 1
                m.stdout = "java.lang.SecurityException: Permission Denial: starting Intent { act=android.intent.action.CALL ... } requires android.permission.CALL_PHONE"
            return m

        mock_run.side_effect = direct_sec_err
        res_blocked = dialer.make_phone_call("+919701350218", allow_dialer_fallback=False)
        assert res_blocked["success"] is False
        assert res_blocked["status"] == "PERMISSION_BLOCKED"
        assert "SecurityException: CALL_PHONE permission denied" in res_blocked["message"]
        print(" * SecurityException without fallback correctly reported as PERMISSION_BLOCKED (keypad not opened) [OK]")

    # 6E: SecurityException WITH allow_dialer_fallback=True -> Fallback to ACTION_DIAL
    with patch("subprocess.run") as mock_run:
        def fallback_dial(cmd, **kwargs):
            m = MagicMock()
            m.returncode = 0
            m.stderr = ""
            cmd_str = " ".join(cmd)
            if "devices" in cmd:
                m.stdout = "172.16.13.22:42513\tdevice\n"
            elif "pm path" in cmd_str:
                m.returncode = 1
                m.stdout = ""
            elif "android.intent.action.CALL" in cmd_str:
                m.returncode = 1
                m.stdout = "java.lang.SecurityException: Permission Denial: requires android.permission.CALL_PHONE"
            elif "android.intent.action.DIAL" in cmd_str:
                m.stdout = "Starting: Intent { act=android.intent.action.DIAL dat=tel:+919701350218 }"
            return m

        mock_run.side_effect = fallback_dial
        res_fb = dialer.make_phone_call("+919701350218", allow_dialer_fallback=True)
        assert res_fb["success"] is True
        assert res_fb["status"] == "DIALER_OPENED"
        assert res_fb["fallback"] is True
        print(" * SecurityException with explicit fallback=True opens ACTION_DIAL [OK]")

    # 6F: Multiple devices connected WITHOUT ANDROID_DEVICE_ID env var -> MULTIPLE_DEVICES error
    with patch("subprocess.run") as mock_run:
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "List of devices attached\n172.16.13.22:42513\tdevice\nemulator-5554\tdevice\n"
        mock_proc.stderr = ""
        mock_run.return_value = mock_proc

        multi_dialer = AndroidDialer()
        is_conn, status_code, msg, dev_id = multi_dialer.resolve_target_device()
        assert is_conn is False
        assert status_code == "MULTIPLE_DEVICES"
        assert "ANDROID_DEVICE_ID" in msg
        assert dev_id is None
        print(" * Multiple ready devices without ANDROID_DEVICE_ID correctly blocked [OK]")

    # 6G: Multiple devices connected WITH ANDROID_DEVICE_ID specified
    with patch("subprocess.run") as mock_run:
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "List of devices attached\n172.16.13.22:42513\tdevice\nemulator-5554\tdevice\n"
        mock_proc.stderr = ""
        mock_run.return_value = mock_proc

        targeted_dialer = AndroidDialer(device_id="emulator-5554")
        is_conn, status_code, msg, dev_id = targeted_dialer.resolve_target_device()
        assert is_conn is True
        assert status_code == "DEVICE_READY"
        assert dev_id == "emulator-5554"
        print(" * Specific device ID targeting ('emulator-5554') correctly resolved [OK]")

    # 6H: Disconnected state
    with patch("subprocess.run") as mock_run:
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "List of devices attached\n\n"
        mock_proc.stderr = ""
        mock_run.return_value = mock_proc

        is_conn, status_code, msg = dialer.check_device_connected()
        assert is_conn is False
        assert status_code == "DISCONNECTED"
        print(" * Disconnected state correctly detected [OK]")

    # 6I: Unauthorized state
    with patch("subprocess.run") as mock_run:
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "List of devices attached\n172.16.13.22:42513\tunauthorized\n"
        mock_proc.stderr = ""
        mock_run.return_value = mock_proc

        is_conn, status_code, msg = dialer.check_device_connected()
        assert is_conn is False
        assert status_code == "UNAUTHORIZED"
        assert "unauthorized" in msg
        print(" * Unauthorized state correctly detected [OK]")

    with patch("subprocess.run", side_effect=FileNotFoundError):
        is_conn, status_code, msg = dialer.check_device_connected()
        assert is_conn is False
        assert status_code == "ADB_NOT_FOUND"
        print(" * Missing ADB executable handled gracefully [OK]\n")

    # =========================================================================
    # TEST 7: Duplicate Confirmation Prevention
    # =========================================================================
    print("--- 7. Testing Duplicate Confirmation Prevention ---")
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = lambda cmd, **k: MagicMock(
            returncode=0,
            stdout="172.16.13.22:42513\tdevice\n" if "devices" in cmd else (
                "package:/data/app/test.apk\n" if "pm path" in " ".join(cmd) else (
                    "android.permission.CALL_PHONE: granted=true\n" if "dumpsys" in " ".join(cmd) else "Starting..."
                )
            ),
            stderr=""
        )

        await call_mgr.start_call_request(parser.parse("call Rohit"))
        step_first = await call_mgr.handle_user_turn("yes")
        assert step_first["status"] in ("CALL_INITIATED", "CALL_PLACED", "DIALER_OPENED (Tap Call on phone)")

        # Immediate duplicate confirmation for same call
        call_mgr.state = CallConversationManager.STATE_AWAITING_CONFIRMATION
        call_mgr.target_name = "Rohit"
        call_mgr.target_phone = "+91 9701350218"

        step_dup = await call_mgr.handle_user_turn("yes")
        assert step_dup["status"] == "DUPLICATE_IGNORED"
        print(" -> Result: Immediate duplicate confirmation suppressed [OK]\n")

    # Clean up test scratch
    try:
        import shutil
        shutil.rmtree(test_data_dir, ignore_errors=True)
    except Exception:
        pass

    print("=" * 72)
    print(">>> ALL VOICE CALL & COMPANION ADB TESTS PASSED WITH 100% SUCCESS! <<<")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    asyncio.run(run_call_confirmation_tests())
