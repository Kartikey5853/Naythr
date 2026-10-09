"""
Naythr Multi-Turn Ride Request Conversation Manager.

Handles conversational slot-filling for ride booking requests:
- If user says: "Book a ride for me."
  Detects BOOK_RIDE, prompts through the live session:
  "Where are you going from?"
  Then:
  "Where are you going to?"
  Stores the answers and prints the completed request to terminal:
  --------------------------------
  RIDE REQUEST
  From: College
  To: KIMS Hospital
  Status: PENDING_CONFIRMATION
  --------------------------------
- Does NOT actually book anything (stays PENDING_CONFIRMATION).
- If user already provides both: "Book me a bike from College to KIMS Hospital."
  Directly extracts and displays without unnecessary questions.
- Does not guess missing information.
"""

import asyncio
import logging
import re
from typing import Callable, Dict, Optional, Any

logger = logging.getLogger("RideConversationManager")


def print_ride_request_summary(pickup: str, destination: str, status: str = "PENDING_CONFIRMATION"):
    """Print the exact standardized terminal summary for a ride request."""
    print("--------------------------------")
    print("RIDE REQUEST")
    print(f"From: {pickup}")
    print(f"To: {destination}")
    print(f"Status: {status}")
    print("--------------------------------")


class RideConversationManager:
    """Manages the interactive slot-filling dialogue for booking a ride."""

    STATE_IDLE = "IDLE"
    STATE_AWAITING_FROM = "AWAITING_FROM"
    STATE_AWAITING_TO = "AWAITING_TO"
    STATE_COMPLETED = "COMPLETED"

    def __init__(
        self,
        memory_manager: Optional[Any] = None,
        send_prompt_func: Optional[Callable[[str], Any]] = None,
    ):
        self.memory_manager = memory_manager
        self.send_prompt_func = send_prompt_func

        self.state = self.STATE_IDLE
        self.current_vehicle: str = "ride"
        self.pickup: Optional[str] = None
        self.destination: Optional[str] = None
        self.user_request: str = ""
        self._lock = asyncio.Lock()

    @property
    def is_active(self) -> bool:
        """Returns True if currently waiting for a location from the user."""
        return self.state in (self.STATE_AWAITING_FROM, self.STATE_AWAITING_TO)

    def reset(self):
        """Reset conversation state to IDLE."""
        self.state = self.STATE_IDLE
        self.current_vehicle = "ride"
        self.pickup = None
        self.destination = None
        self.user_request = ""

    def _clean_location(self, text: str) -> str:
        """Clean conversational prefixes and punctuation from a location input."""
        s = text.strip()
        # Strip trailing punctuation
        s = re.sub(r"[\.\?!,;]+$", "", s).strip()
        # Remove conversational prefixes
        prefixes = [
            r"^(?:i am at|i'm at|from|pickup from|pickup at|starting from)\s+",
            r"^(?:going to|headed to|to|drop at|drop off at)\s+",
            r"^(?:take me to|it's|its)\s+",
        ]
        for p in prefixes:
            s = re.sub(p, "", s, flags=re.IGNORECASE).strip()
        s = re.sub(r"[\.\?!,;]+$", "", s).strip()
        return s

    async def _ask_through_session(self, question: str):
        """Ask the question to the user through the existing live session."""
        print(f"\nMarvin > {question}")
        if self.send_prompt_func:
            prompt = (
                "[SYSTEM DIALOGUE PROMPT]\n\n"
                "The user is booking a ride.\n"
                f"Ask the user exactly:\n"
                f'"{question}"\n\n'
                "Do not add extra commentary. Speak only this question clearly."
            )
            try:
                res = self.send_prompt_func(prompt)
                if asyncio.iscoroutine(res):
                    await res
            except Exception as e:
                logger.debug(f"Error sending dialogue question to Gemini: {e}")

    async def _complete_ride(self) -> Dict[str, Any]:
        """Finalize ride request, persist to storage, and print terminal block."""
        self.state = self.STATE_COMPLETED
        print_ride_request_summary(self.pickup, self.destination, status="PENDING_CONFIRMATION")

        action_record = None
        if self.memory_manager:
            entities = {
                "vehicle": self.current_vehicle,
                "pickup": self.pickup,
                "destination": self.destination,
                "destination_type": "hospital" if "hospital" in (self.destination or "").lower() else "location",
                "requested_time": None,
            }
            action_record = self.memory_manager.save_action(
                action_type="TRANSPORT",
                intent="BOOK_RIDE",
                user_request=self.user_request or f"Book a ride from {self.pickup} to {self.destination}",
                entities=entities,
                status="PENDING_CONFIRMATION",
            )

        # Notify Gemini Live session to confirm
        if self.send_prompt_func:
            prompt = (
                "[SYSTEM RIDE REQUEST RECORDED]\n\n"
                f"Ride request recorded: From {self.pickup} to {self.destination}.\n"
                "Status: PENDING_CONFIRMATION (Do NOT execute external booking).\n\n"
                "In one short sentence, confirm that the ride request from "
                f"{self.pickup} to {self.destination} is recorded and pending confirmation."
            )
            try:
                res = self.send_prompt_func(prompt)
                if asyncio.iscoroutine(res):
                    await res
            except Exception as e:
                logger.debug(f"Error sending ride confirmation to Gemini: {e}")

        result = {
            "status": "COMPLETED",
            "pickup": self.pickup,
            "destination": self.destination,
            "vehicle": self.current_vehicle,
            "action_record": action_record,
            "question": None,
        }

        # Return state to IDLE for subsequent requests
        self.state = self.STATE_IDLE
        return result

    async def start_or_process_intent(self, intent_result: Any) -> Optional[Dict[str, Any]]:
        """Evaluate an IntentResult.
        
        If both FROM and TO are present: directly completes and displays without asking questions.
        If FROM is missing: asks 'Where are you going from?'.
        If TO is missing: asks 'Where are you going to?'.
        """
        async with self._lock:
            if getattr(intent_result, "intent", None) != "BOOK_RIDE":
                return None

            entities = getattr(intent_result, "entities", {}) or {}
            self.user_request = getattr(intent_result, "user_request", "")
            self.current_vehicle = entities.get("vehicle") or "ride"
            self.pickup = entities.get("pickup")
            self.destination = entities.get("destination")

            # Case 1: Both From and To already provided directly!
            if self.pickup and self.destination:
                return await self._complete_ride()

            # Case 2: Pickup is missing -> Ask "Where are you going from?"
            if not self.pickup:
                self.state = self.STATE_AWAITING_FROM
                question = "Where are you going from?"
                await self._ask_through_session(question)
                return {
                    "status": "AWAITING_FROM",
                    "question": question,
                    "action_record": None,
                }

            # Case 3: Destination is missing -> Ask "Where are you going to?"
            if not self.destination:
                self.state = self.STATE_AWAITING_TO
                question = "Where are you going to?"
                await self._ask_through_session(question)
                return {
                    "status": "AWAITING_TO",
                    "question": question,
                    "action_record": None,
                }

            return None

    async def handle_user_input(self, text: str) -> Optional[Dict[str, Any]]:
        """Handle user response while in AWAITING_FROM or AWAITING_TO state."""
        async with self._lock:
            if not self.is_active:
                return None

            low = text.strip().lower()
            if low in ("cancel", "never mind", "nevermind", "stop", "abort"):
                print("[RIDE] Ride request cancelled.")
                self.reset()
                return {"status": "CANCELLED", "question": None, "action_record": None}

            cleaned = self._clean_location(text)
            if not cleaned:
                # Still waiting for a valid answer
                q = "Where are you going from?" if self.state == self.STATE_AWAITING_FROM else "Where are you going to?"
                await self._ask_through_session(q)
                return {"status": self.state, "question": q, "action_record": None}

            if self.state == self.STATE_AWAITING_FROM:
                self.pickup = cleaned
                if not self.destination:
                    self.state = self.STATE_AWAITING_TO
                    question = "Where are you going to?"
                    await self._ask_through_session(question)
                    return {
                        "status": "AWAITING_TO",
                        "question": question,
                        "action_record": None,
                    }
                else:
                    return await self._complete_ride()

            elif self.state == self.STATE_AWAITING_TO:
                self.destination = cleaned
                if not self.pickup:
                    self.state = self.STATE_AWAITING_FROM
                    question = "Where are you going from?"
                    await self._ask_through_session(question)
                    return {
                        "status": "AWAITING_FROM",
                        "question": question,
                        "action_record": None,
                    }
                else:
                    return await self._complete_ride()

            return None
