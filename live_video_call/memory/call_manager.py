"""
Naythr Phone Call Conversation & Smart Assistant Confirmation Manager.
=======================================================================
Handles natural voice-initiated calling with multi-turn confirmation:
1. User: "Hey Marvin, call John" (or "call Mom", "call this guy", "call 9876543210")
2. Marvin checks contact book (`contacts.json`):
   - If contact found: Marvin asks: "Do you want me to call John at +91 98123 45678?"
   - If target is ambiguous ("this guy"): Marvin asks: "Who would you like me to call, or what is their phone number?"
   - If target not found: Marvin asks: "{Target} is not in your contacts. What is their phone number?"
3. User confirms:
   - "Yes" / "Call him" / "Go ahead" -> Marvin prints PHONE CALL card and places call:
     --------------------------------
     PHONE CALL
     Contact: John
     Number:  +91 98123 45678
     Status:  PLACING_CALL
     --------------------------------
   - "No" / "Cancel" -> Marvin cancels: "Call cancelled."
"""

import asyncio
import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Callable, Dict, Optional, Any, List
import time

try:
    from live_video_call.android_dialer import AndroidDialer, open_android_dialer, place_android_call
except ImportError:
    try:
        from android_dialer import AndroidDialer, open_android_dialer, place_android_call
    except ImportError:
        AndroidDialer = None
        open_android_dialer = None
        place_android_call = None

logger = logging.getLogger("CallConversationManager")




def print_phone_call_summary(contact: str, number: str, status: str = "PLACING_CALL"):
    """Print the exact standardized terminal summary card for a phone call."""
    print("--------------------------------")
    print("PHONE CALL")
    print(f"Contact: {contact}")
    print(f"Number:  {number}")
    print(f"Status:  {status}")
    print("--------------------------------")


DEFAULT_CONTACTS = {
    "mom": {
        "name": "Mom",
        "phone": "+91 98765 43210",
        "relation": "family",
        "aliases": ["mom", "mother", "mummy", "maa", "my mom"],
    },
    "rohit": {
        "name": "Rohit",
        "phone": "+91 9701350218",
        "relation": "friend",
        "aliases": ["rohit", "rohit sharma", "my friend rohit"],
    },
    "john": {
        "name": "John",
        "phone": "+91 98123 45678",
        "relation": "friend",
        "aliases": ["john", "johnny"],
    },
    "doctor": {
        "name": "Dr. Sharma",
        "phone": "+91 99887 76655",
        "relation": "physician",
        "aliases": ["doctor", "doc", "dr sharma", "clinic", "my doctor"],
    },
    "emergency": {
        "name": "Emergency Services",
        "phone": "112",
        "relation": "emergency",
        "aliases": ["emergency", "police", "ambulance", "911", "112"],
    },
}


class CallConversationManager:
    """Manages voice calling dialogues, contact resolution, and smart confirmation."""

    STATE_IDLE = "IDLE"
    STATE_AWAITING_TARGET_INFO = "AWAITING_TARGET_INFO"
    STATE_AWAITING_CONFIRMATION = "AWAITING_CONFIRMATION"
    STATE_COMPLETED = "COMPLETED"
    STATE_CANCELLED = "CANCELLED"

    def __init__(
        self,
        memory_manager: Optional[Any] = None,
        send_prompt_func: Optional[Callable[[str], Any]] = None,
        contacts_file: Optional[Path] = None,
    ):
        self.memory_manager = memory_manager
        self.send_prompt_func = send_prompt_func

        if contacts_file:
            self.contacts_path = Path(contacts_file)
        elif memory_manager and hasattr(memory_manager, "data_dir"):
            self.contacts_path = Path(memory_manager.data_dir) / "contacts.json"
        else:
            self.contacts_path = Path(__file__).resolve().parent / "data" / "contacts.json"

        self.contacts: Dict[str, Dict[str, Any]] = self._load_contacts()

        self.state = self.STATE_IDLE
        self.target_name: Optional[str] = None
        self.target_phone: Optional[str] = None
        self.user_request: str = ""
        self.last_completed_call_time: float = 0.0
        self.last_completed_number: str = ""
        self._lock = asyncio.Lock()


    @property
    def is_active(self) -> bool:
        """Returns True if currently waiting for contact info or confirmation."""
        return self.state in (self.STATE_AWAITING_TARGET_INFO, self.STATE_AWAITING_CONFIRMATION)

    def reset(self):
        """Reset conversation state to IDLE."""
        self.state = self.STATE_IDLE
        self.target_name = None
        self.target_phone = None
        self.user_request = ""

    def _load_contacts(self) -> Dict[str, Dict[str, Any]]:
        """Load contacts from JSON file or initialize with defaults."""
        if self.contacts_path.is_file():
            try:
                with open(self.contacts_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
            except Exception as e:
                logger.warning(f"Error loading contacts from {self.contacts_path}: {e}")
        return dict(DEFAULT_CONTACTS)

    def save_contact(self, key: str, name: str, phone: str, relation: str = "contact") -> None:
        """Save a new or updated contact to contacts.json."""
        clean_key = key.lower().strip()
        self.contacts[clean_key] = {
            "name": name,
            "phone": phone,
            "relation": relation,
            "aliases": [clean_key, name.lower()],
        }
        try:
            self.contacts_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.contacts_path, "w", encoding="utf-8") as f:
                json.dump(self.contacts, f, indent=2)
        except Exception as e:
            logger.error(f"Error writing contacts to {self.contacts_path}: {e}")

    def lookup_contact(self, query: str) -> Optional[Dict[str, Any]]:
        """Find a contact by name, key, or alias."""
        if not query:
            return None
        q = query.lower().strip()
        q = re.sub(r"^(?:my|the|a)\s+", "", q).strip()

        # 1. Direct key match
        if q in self.contacts:
            return self.contacts[q]

        # 2. Alias match
        for key, info in self.contacts.items():
            aliases = [a.lower() for a in info.get("aliases", [])]
            if q in aliases:
                return info

        # 3. Name substring match
        for key, info in self.contacts.items():
            c_name = info.get("name", "").lower()
            if q == c_name or q in c_name or c_name in q:
                return info

        return None

    async def _ask_through_session(self, question: str):
        """Ask the question through the existing live session."""
        print(f"\nMarvin > {question}")
        if self.send_prompt_func:
            prompt = (
                "[SYSTEM DIALOGUE PROMPT]\n\n"
                "The user requested to make a phone call.\n"
                f"Speak aloud ONLY this exact sentence to the user:\n"
                f'"{question}"\n\n'
                "Do not add extra commentary. Do not describe the camera view. Speak only this question clearly."
            )
            try:
                res = self.send_prompt_func(prompt)
                if asyncio.iscoroutine(res):
                    await res
            except Exception as e:
                logger.debug(f"Error sending dialogue question to Gemini: {e}")

    async def start_call_request(self, intent_result: Any) -> Dict[str, Any]:
        """Initiate calling dialogue from parsed intent."""
        async with self._lock:
            self.reset()
            self.user_request = getattr(intent_result, "user_request", "")
            entities = getattr(intent_result, "entities", {})

            phone_num = entities.get("phone_number")
            raw_target = (entities.get("target") or entities.get("contact_name") or entities.get("raw_target") or "").strip()

            # Case A: Direct phone number provided (e.g. "call 9876543210")
            if phone_num:
                self.target_name = "Phone Number"
                self.target_phone = phone_num
                self.state = self.STATE_AWAITING_CONFIRMATION
                q = f"Do you want me to call {phone_num}?"
                await self._ask_through_session(q)
                return {"state": self.state, "question": q, "status": "PENDING_CONFIRMATION"}

            # Case B: Contact lookup in contact book
            matched = self.lookup_contact(raw_target)
            if matched:
                self.target_name = matched.get("name", raw_target.title())
                self.target_phone = matched.get("phone", "")
                self.state = self.STATE_AWAITING_CONFIRMATION
                q = f"Do you want me to call {self.target_name} at {self.target_phone}?"
                await self._ask_through_session(q)
                return {"state": self.state, "question": q, "status": "PENDING_CONFIRMATION"}

            # Case C: Ambiguous target ("this guy", "him") or unknown contact
            vague_words = {"this guy", "that guy", "him", "her", "them", "someone", "somebody"}
            self.state = self.STATE_AWAITING_TARGET_INFO
            if raw_target.lower() in vague_words or not raw_target:
                q = "Who would you like me to call, or what is their phone number?"
            else:
                self.target_name = raw_target.title()
                q = f"{self.target_name} is not in your contacts. What is their phone number?"

            await self._ask_through_session(q)
            return {"state": self.state, "question": q, "status": "PENDING_INFORMATION"}

    async def handle_user_turn(self, utterance: str) -> Dict[str, Any]:
        """Handle user's reply during multi-turn dialogue."""
        async with self._lock:
            if not self.is_active:
                return {"handled": False}

            cleaned = utterance.strip()
            lowered = re.sub(r"[\.\?!,;]+$", "", cleaned).lower().strip()

            # ------------------------------------------------------------------
            # Phase 1: Waiting for Contact Info / Phone Number
            # ------------------------------------------------------------------
            if self.state == self.STATE_AWAITING_TARGET_INFO:
                # Check if user spoke/typed a phone number
                phone_digits = re.sub(r"[^\d\+]", "", lowered)
                if len(phone_digits) >= 5 and re.search(r"\d{5,}", phone_digits):
                    self.target_phone = phone_digits
                    self.target_name = self.target_name or "Phone Number"
                    self.state = self.STATE_AWAITING_CONFIRMATION
                    q = f"Do you want me to call {self.target_phone}?"
                    await self._ask_through_session(q)
                    return {"handled": True, "state": self.state, "question": q}

                # Check if user mentioned a known contact name
                matched = self.lookup_contact(lowered)
                if matched:
                    self.target_name = matched.get("name", lowered.title())
                    self.target_phone = matched.get("phone", "")
                    self.state = self.STATE_AWAITING_CONFIRMATION
                    q = f"Do you want me to call {self.target_name} at {self.target_phone}?"
                    await self._ask_through_session(q)
                    return {"handled": True, "state": self.state, "question": q}

                # If user said cancel
                if any(w in lowered.split() for w in ["cancel", "no", "stop", "nevermind"]):
                    return await self._cancel_call()

                # Otherwise still need phone number
                self.target_name = cleaned.title()
                q = f"What is {self.target_name}'s phone number?"
                await self._ask_through_session(q)
                return {"handled": True, "state": self.state, "question": q}

            # ------------------------------------------------------------------
            # Phase 2: Waiting for Confirmation (YES or NO)
            # ------------------------------------------------------------------
            elif self.state == self.STATE_AWAITING_CONFIRMATION:
                affirmative_words = {
                    "yes", "yeah", "yep", "sure", "ok", "okay", "call", "call him",
                    "call her", "call them", "go ahead", "confirm", "please do",
                    "dial", "do it", "please", "yes call", "yes call him", "yes please",
                }
                negative_words = {
                    "no", "nope", "cancel", "don't", "dont call", "don't call",
                    "stop", "nevermind", "abort", "wait", "no cancel", "do not",
                }

                # Match negative signals FIRST
                if any(w in lowered for w in negative_words) or lowered in negative_words:
                    return await self._cancel_call()

                # Match affirmative signals
                if any(w in lowered for w in affirmative_words) or lowered in affirmative_words:
                    return await self._complete_call()

                # Ambiguous answer: Clarify prompt
                q = f"Please confirm: say yes to call {self.target_name or self.target_phone}, or no to cancel."
                await self._ask_through_session(q)
                return {"handled": True, "state": self.state, "question": q}

            return {"handled": False}

    async def _complete_call(self) -> Dict[str, Any]:
        """Finalize confirmed phone call, print terminal card, and record action."""
        now = time.time()
        contact_display = self.target_name or "Unknown"
        number_display = self.target_phone or "Unknown Number"

        # Deduplication check: ignore duplicate confirmations within 5.0 seconds
        if (now - self.last_completed_call_time) < 5.0 and self.last_completed_number == number_display:
            logger.warning("Duplicate call confirmation ignored within deduplication window.")
            self.reset()
            return {
                "handled": True,
                "state": self.STATE_COMPLETED,
                "status": "DUPLICATE_IGNORED",
                "contact": contact_display,
                "number": number_display,
            }

        self.last_completed_call_time = now
        self.last_completed_number = number_display
        self.state = self.STATE_COMPLETED

        # Attempt real dial dispatch if Android ADB or Twilio is available
        dispatch_res = self._dispatch_dial(number_display)

        # Determine real status and spoken prompt based on dispatch result
        adb_details = dispatch_res.get("details") if isinstance(dispatch_res.get("details"), dict) else {}
        sub_status = adb_details.get("status") if isinstance(adb_details, dict) else ""

        if sub_status in ("CALL_INITIATED", "CALL_PLACED") or dispatch_res.get("method") == "TWILIO":
            card_status = "CALL_INITIATED"
            spoken_confirm = f"Calling {contact_display} now."
            gemini_instruction = f"Confirm to the user in one short sentence:\n\"{spoken_confirm}\""
        elif sub_status in ("PERMISSION_BLOCKED", "PERMISSION_DENIED"):
            card_status = "PERMISSION_BLOCKED (Companion App / CALL_PHONE required)"
            spoken_confirm = f"Automatic call failed: phone permission is required for {contact_display}."
            gemini_instruction = (
                f"Inform the user clearly: \"{spoken_confirm}\"\n"
                "CRITICAL: Do NOT say you are placing the call. Do NOT say 'Calling now'. State only that the call failed due to missing permission."
            )
            print("\n=======================================================")
            print("[SETUP REQUIRED] To enable automatic calling without screen tapping:")
            print(" 1. APK has been built and pushed to phone at: /sdcard/Download/app-debug.apk")
            print(" 2. Open File Manager on phone -> Downloads -> tap 'app-debug.apk' to install.")
            print(" 3. Open 'Marvin Call Companion' -> tap 'Grant Phone Permission' -> select 'Allow'.")
            print("=======================================================\n")
        elif sub_status == "DIALER_OPENED" or (isinstance(adb_details, dict) and adb_details.get("fallback")):
            card_status = "DIALER_OPENED (Tap Call on phone)"
            spoken_confirm = f"Opened phone dialer for {contact_display}. Please tap Call on your phone."
            gemini_instruction = f"Inform the user in one sentence:\n\"{spoken_confirm}\""
        else:
            err_msg = adb_details.get('message', 'ADB disconnected') if isinstance(adb_details, dict) else 'ADB error'
            card_status = f"CALL_FAILED ({err_msg})"
            spoken_confirm = f"Could not connect to Android phone via ADB for {contact_display}."
            gemini_instruction = (
                f"Inform the user: \"{spoken_confirm}\"\n"
                "Do NOT claim the call is being placed."
            )

        print_phone_call_summary(contact_display, number_display, status=card_status)

        print(f"\nMarvin > {spoken_confirm}")
        if self.send_prompt_func:
            try:
                res = self.send_prompt_func(
                    f"[SYSTEM DIALOGUE PROMPT]\n\n{gemini_instruction}"
                )
                if asyncio.iscoroutine(res):
                    await res
            except Exception:
                pass

        # Persist action to memory manager
        action_record = None
        if self.memory_manager:
            entities = {
                "contact": contact_display,
                "phone": number_display,
                "status": "CONFIRMED" if dispatch_res.get("success") else "FAILED",
            }
            try:
                action_record = self.memory_manager.save_action(
                    action_type="COMMUNICATION",
                    intent="MAKE_PHONE_CALL",
                    status="CONFIRMED" if dispatch_res.get("success") else "FAILED",
                    user_request=self.user_request or f"Call {contact_display}",
                    entities=entities,
                    source="voice_dialogue",
                )
                self.memory_manager._log_activity(
                    request=self.user_request or f"Call {contact_display}",
                    summary=f"Phone call action: {contact_display} ({number_display}) - Status: {card_status}",
                    category="COMMUNICATION",
                    intent="MAKE_PHONE_CALL",
                )
            except Exception as e:
                logger.error(f"Error saving phone call action to memory: {e}")

        result = {
            "handled": True,
            "state": self.state,
            "status": card_status,
            "contact": contact_display,
            "number": number_display,
            "action_record": action_record,
            "dispatch": dispatch_res,
        }
        self.reset()
        return result

    async def _cancel_call(self) -> Dict[str, Any]:
        """Cancel phone call request."""
        self.state = self.STATE_CANCELLED
        contact_display = self.target_name or "Unknown"
        number_display = self.target_phone or "N/A"

        print_phone_call_summary(contact_display, number_display, status="CANCELLED")

        spoken_cancel = "Call cancelled."
        print(f"\nMarvin > {spoken_cancel}")
        if self.send_prompt_func:
            try:
                res = self.send_prompt_func(
                    f"[SYSTEM DIALOGUE PROMPT]\n\nInform the user in one short phrase:\n\"{spoken_cancel}\""
                )
                if asyncio.iscoroutine(res):
                    await res
            except Exception:
                pass

        if self.memory_manager:
            try:
                self.memory_manager.save_action(
                    action_type="COMMUNICATION",
                    intent="MAKE_PHONE_CALL",
                    status="CANCELLED",
                    user_request=self.user_request or f"Call {contact_display}",
                    entities={"contact": contact_display, "phone": number_display, "status": "CANCELLED"},
                    source="voice_dialogue",
                )
            except Exception:
                pass

        result = {
            "handled": True,
            "state": self.state,
            "status": "CANCELLED",
            "contact": contact_display,
            "number": number_display,
        }
        self.reset()
        return result

    def _dispatch_dial(self, phone_number: str) -> Dict[str, Any]:
        """Execute real cellular dial via Android ADB (ACTION_CALL with ACTION_DIAL fallback) or Twilio."""
        # 1. Android ADB call intent via place_android_call module
        if place_android_call:
            try:
                adb_res = place_android_call(phone_number)
                if adb_res.get("success"):
                    return {"method": "ANDROID_ADB", "success": True, "details": adb_res}
                else:
                    logger.info(f"[PHONE DISPATCH NOTE] Android ADB status: {adb_res.get('status')} - {adb_res.get('message')}")
                    return {"method": "ANDROID_ADB", "success": False, "details": adb_res}
            except Exception as e:
                logger.debug(f"Android ADB call invocation note: {e}")

        # 2. Twilio Voice REST API
        twilio_sid = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
        twilio_token = os.getenv("TWILIO_AUTH_TOKEN", "").strip()
        twilio_from = os.getenv("TWILIO_FROM_NUMBER", "").strip()

        if twilio_sid and twilio_token and twilio_from:
            try:
                import urllib.request
                import urllib.parse
                url = f"https://api.twilio.com/2010-04-01/Accounts/{twilio_sid}/Calls.json"
                data = urllib.parse.urlencode({
                    "To": phone_number,
                    "From": twilio_from,
                    "Url": "http://demo.twilio.com/docs/voice.xml",
                }).encode("utf-8")
                req = urllib.request.Request(url, data=data)
                import base64
                creds = base64.b64encode(f"{twilio_sid}:{twilio_token}".encode("utf-8")).decode("ascii")
                req.add_header("Authorization", f"Basic {creds}")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    if resp.status in (200, 201):
                        print(f"[PHONE DISPATCH] Twilio Voice call initiated to {phone_number}!")
                        return {"method": "TWILIO", "success": True, "details": "Twilio call initiated"}
            except Exception as e:
                logger.debug(f"Twilio call dispatch note: {e}")

        # 3. No active dispatch backend available
        return {"method": "NONE", "success": False, "details": {"message": "No active Android ADB or Twilio calling backend connected"}}



