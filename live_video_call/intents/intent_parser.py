"""
Structured Intent & Entity Parser for Naythr Action + Memory Layer
===================================================================
Converts natural user utterances into structured intent results.
Strictly respects:
- DO NOT store casual conversation (navigation, greetings, visual queries).
- NEVER invent information (missing fields remain None / null).
- Distinguishes "prescribed" from "purchase/order".
- Flags incomplete or ambiguous requests for user clarification.
- Validates data before handing to MemoryManager.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


# Casual conversation patterns that must NEVER be persisted
CASUAL_PATTERNS = [
    r"^(what('s| is) (around|in front of|behind|near) me\??)",
    r"^(tell me what('s| is) in front of me\??)",
    r"^(what do you see\??)",
    r"^(read (this|the) (note|paper|sign|book|screen|text)\??)",
    r"^(move (forward|back|left|right)|stop|slow down|step)\b",
    r"^(hello|hi|hey|good (morning|afternoon|evening)|howdy)\b",
    r"^(hey marvin|hello marvin|hi marvin)\b",
    r"^(who are you|what is your name)\??",
    r"^(thank you|thanks|bye|goodbye|see you)\b",
    r"^(yes|no|okay|ok|sure|fine|cancel|nevermind)\b",
    r"^(help me navigation|where am i)\??",
]

# Destination types keywords mapping
DESTINATION_TYPE_KEYWORDS = {
    "hospital": ["hospital", "clinic", "kims", "apollo", "care", "max", "fortis", "dispensary"],
    "home": ["home", "residence", "house", "apartment"],
    "work": ["office", "work", "workplace"],
    "pharmacy": ["pharmacy", "medical store", "chemist", "apothecary", "drugstore"],
    "transit": ["airport", "station", "metro", "bus stop", "terminal"],
}

# Vehicle keywords mapping
VEHICLE_KEYWORDS = {
    "bike": ["bike", "motorcycle", "scooter", "two wheeler", "two-wheeler"],
    "cab": ["cab", "taxi", "car", "uber", "ola", "auto", "rickshaw"],
}


@dataclass
class IntentResult:
    """Structured representation of parsed user intent."""
    should_store: bool
    category: Optional[str] = None           # MEDICATION, APPOINTMENT, TRANSPORT, PURCHASE, REMINDER, HEALTHCARE, OTHER_ACTION
    intent: Optional[str] = None             # e.g. MEDICATION_PRESCRIPTION, BOOK_RIDE, CREATE_APPOINTMENT_MEMORY
    confidence: float = 0.0
    entities: Dict[str, Any] = field(default_factory=dict)
    requires_clarification: bool = False
    clarification_prompt: Optional[str] = None
    user_request: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "should_store": self.should_store,
            "category": self.category,
            "intent": self.intent,
            "confidence": round(self.confidence, 2),
            "entities": self.entities,
            "requires_clarification": self.requires_clarification,
            "clarification_prompt": self.clarification_prompt,
            "user_request": self.user_request,
        }


class IntentParser:
    """Deterministic, robust intent and entity extraction engine."""

    def __init__(self, known_places: Optional[List[Dict[str, str]]] = None):
        self.known_places = known_places or [
            {"name": "KIMS Hospital", "type": "hospital"},
            {"name": "Apollo Hospital", "type": "hospital"},
        ]

    def parse(self, text: str) -> IntentResult:
        """Parse raw speech transcript or typed query into an IntentResult."""
        if not text or not text.strip():
            return IntentResult(should_store=False, user_request=text or "")

        cleaned = text.strip()
        # Normalize trailing punctuation for clean regex matching
        normalized = re.sub(r"[\.\?!,;]+$", "", cleaned).strip()
        lowered = normalized.lower()

        # 1. Filter out casual conversation and visual queries
        if self._is_casual_or_navigation(lowered):
            return IntentResult(
                should_store=False,
                confidence=0.99,
                user_request=cleaned,
            )

        # 2. Check for Purchase / Ordering requests (takes precedence over prescription if "buy/order" is present)
        purchase_res = self._extract_purchase_intent(cleaned, normalized, lowered)
        if purchase_res:
            return purchase_res

        # 3. Check for Medication / Prescription statements
        med_res = self._extract_medication_intent(cleaned, normalized, lowered)
        if med_res:
            return med_res

        # 4. Check for Appointment statements
        appt_res = self._extract_appointment_intent(cleaned, normalized, lowered)
        if appt_res:
            return appt_res

        # 5. Check for Transport / Ride booking requests
        trans_res = self._extract_transport_intent(cleaned, normalized, lowered)
        if trans_res:
            return trans_res

        # 6. Check for Reminders
        rem_res = self._extract_reminder_intent(cleaned, normalized, lowered)
        if rem_res:
            return rem_res

        # 7. Check for general Healthcare metrics / info
        health_res = self._extract_healthcare_intent(cleaned, normalized, lowered)
        if health_res:
            return health_res

        # Default: Unrecognized / Non-actionable statement
        return IntentResult(
            should_store=False,
            confidence=0.1,
            user_request=cleaned,
        )

    def _is_casual_or_navigation(self, text: str) -> bool:
        """Check if utterance is normal navigation, visual inquiry, or casual chat."""
        if text in {"what", "who", "where", "hi", "hello", "hey", "yes", "no", "stop", "marvin"}:
            return True

        for pat in CASUAL_PATTERNS:
            if re.search(pat, text, re.IGNORECASE):
                return True

        # Pure vision questions without memory intent
        if re.search(r"\b(what is this|what are these|look at this|can you see|describe this)\b", text):
            return True

        return False

    def _extract_medication_intent(self, raw_text: str, norm_text: str, lowered: str) -> Optional[IntentResult]:
        """Detect and parse prescription and medication statements."""
        prescribed_patterns = [
            r"\b(prescribed|prescription)\b",
            r"\bdoctor (told|advised|asked) me to take\b",
            r"\btake (this|the)?\s*medicine\b",
            r"\bmy (medicine|medication|tablets|pills) (is|are)\b",
            r"\bstarted taking\b",
        ]
        is_med = any(re.search(pat, lowered) for pat in prescribed_patterns)
        if not is_med:
            return None

        name: Optional[str] = None
        dosage: Optional[str] = None
        frequency: Optional[str] = None
        duration: Optional[str] = None

        # Split off frequency/duration clauses to isolate medicine name
        split_delims = r"\b(?:twice|once|three times|thrice|every|\d+\s*times|for\s+\d+|for\s+one|for\s+two|for\s+three|for\s+four|for\s+five)\b"
        head_lower = re.split(split_delims, lowered)[0].strip()

        # Check: prescribed [me] <name>
        presc_match = re.search(r"prescribed\s+(?:me\s+)?(.+)", head_lower)
        if presc_match:
            start, end = presc_match.span(1)
            raw_cand = norm_text[start:end].strip()
            # Clean leading/trailing particles
            raw_cand = re.sub(r"^(?:this|the|some|a|my)\s+", "", raw_cand, flags=re.IGNORECASE).strip()
            if raw_cand:
                name = raw_cand

        # Check: told me to take <name>
        if not name:
            take_match = re.search(r"(?:told|advised|asked)\s+me\s+to\s+take\s+(.+)", head_lower)
            if take_match:
                start, end = take_match.span(1)
                raw_cand = norm_text[start:end].strip()
                raw_cand = re.sub(r"^(?:this|the|some|a|my)\s+", "", raw_cand, flags=re.IGNORECASE).strip()
                if raw_cand:
                    name = raw_cand

        # Fallback check for "medicine <X>"
        if not name:
            med_x = re.search(r"\b(medicine\s+[a-zA-Z0-9]+)\b", norm_text, re.IGNORECASE)
            if med_x:
                name = med_x.group(1).strip()

        # If name is generic "medicine", keep as "unspecified medicine"
        if not name or name.lower() in {"medicine", "medication", "tablets", "pills"}:
            name = "unspecified medicine"

        # Extract dosage: e.g. "500 mg", "10ml", "1 tablet"
        dosage_match = re.search(r"\b(\d+\s*(?:mg|g|ml|mcg|tablets?|pills?|drops?))\b", lowered)
        if dosage_match:
            dosage = dosage_match.group(1).strip()

        # Extract frequency: "twice a day", "once daily", "three times a day", etc.
        if re.search(r"\b(twice\s+(?:a|per)\s+day|twice\s+daily|2\s+times\s+(?:a|per)\s+day)\b", lowered):
            frequency = "twice_daily"
        elif re.search(r"\b(once\s+(?:a|per)\s+day|once\s+daily|1\s+time\s+(?:a|per)\s+day)\b", lowered):
            frequency = "once_daily"
        elif re.search(r"\b(three\s+times\s+(?:a|per)\s+day|thrice\s+(?:a|per)\s+day|3\s+times\s+(?:a|per)\s+day)\b", lowered):
            frequency = "three_times_daily"
        elif re.search(r"\b(every\s+\d+\s+hours?)\b", lowered):
            m = re.search(r"\b(every\s+\d+\s+hours?)\b", lowered)
            frequency = m.group(1) if m else None

        # Extract duration: "for five days", "for 5 days", "for 2 weeks", etc.
        duration_match = re.search(r"\bfor\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(days?|weeks?|months?)\b", lowered)
        if duration_match:
            num_str = duration_match.group(1)
            unit_str = duration_match.group(2)
            num_map = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
                       "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
            num = num_map.get(num_str, num_str)
            duration = f"{num} {unit_str}"

        entities = {
            "name": name,
            "dosage": dosage,
            "frequency": frequency,
            "duration": duration,
        }

        needs_clarification = (name is None or name == "unspecified medicine")
        return IntentResult(
            should_store=True,
            category="MEDICATION",
            intent="MEDICATION_PRESCRIPTION",
            confidence=0.94,
            entities=entities,
            requires_clarification=needs_clarification,
            clarification_prompt="What is the name of the medicine?" if needs_clarification else None,
            user_request=raw_text,
        )

    def _extract_purchase_intent(self, raw_text: str, norm_text: str, lowered: str) -> Optional[IntentResult]:
        """Detect intent to buy or order medicine or items."""
        order_patterns = [
            r"\b(order|buy|purchase|refill|get me)\s+(?:this\s+|my\s+|the\s+)?(medicine|medication|prescription|pills|tablets?)\b",
            r"\border\s+([a-zA-Z0-9_\-\s]+?)\s+from\s+(?:the\s+)?pharmacy\b",
            r"\bbuy\s+([a-zA-Z0-9_\-\s]+?)\s+from\s+(?:the\s+)?(?:pharmacy|medical store|chemist)\b",
        ]
        is_purchase = any(re.search(pat, lowered) for pat in order_patterns)
        if not is_purchase:
            return None

        product = "medicine"
        for pat in order_patterns:
            m = re.search(pat, lowered)
            if m:
                cand = m.group(m.lastindex).strip() if m.lastindex else "medicine"
                if cand and cand not in {"this", "my", "the"}:
                    product = cand
                break

        entities = {
            "product": product,
            "quantity": None,
            "pharmacy": None,
            "items": None,
        }

        pharm_match = re.search(r"\bfrom\s+([a-zA-Z0-9_\-\s]+?)(?:\s+pharmacy|\s+medical store|$)", norm_text, re.IGNORECASE)
        if pharm_match:
            entities["pharmacy"] = pharm_match.group(1).strip().title()

        return IntentResult(
            should_store=True,
            category="PURCHASE",
            intent="PURCHASE_MEDICATION",
            confidence=0.90,
            entities=entities,
            requires_clarification=True,  # All purchases must remain pending confirmation
            clarification_prompt=f"Would you like me to prepare an order for {product}?",
            user_request=raw_text,
        )

    def _extract_appointment_intent(self, raw_text: str, norm_text: str, lowered: str) -> Optional[IntentResult]:
        """Detect and extract medical or personal appointments."""
        if not re.search(r"\b(appointment|consultation|doctor('s)? visit|checkup|meeting with doctor)\b", lowered):
            return None

        place: Optional[str] = None
        destination_type: Optional[str] = None
        date: Optional[str] = None
        appt_time: Optional[str] = None
        purpose: Optional[str] = None

        # Extract date: "tomorrow", "today", "on Monday", etc.
        date_match = re.search(r"\b(tomorrow|today|next week|on\s+[a-zA-Z]+\s+\d+|on\s+[a-zA-Z]+)\b", lowered)
        if date_match:
            date = date_match.group(1).strip()

        # Extract time: "at 10 AM", "at 10:00", "at 10", "at 4 PM"
        time_match = re.search(r"\bat\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\b", lowered)
        if time_match:
            raw_t = time_match.group(1).strip()
            if ":" not in raw_t and not any(x in raw_t for x in ["am", "pm"]):
                appt_time = f"{raw_t}:00"
            else:
                appt_time = raw_t

        # Extract place: find "at <Place>" where after is not a time/number
        for m in re.finditer(r"\bat\s+", norm_text, re.IGNORECASE):
            after = norm_text[m.end():].strip()
            if re.match(r"^\d", after):
                continue
            place_part = re.split(r"\b(?:tomorrow|today|next|on|at\s+\d)\b", after, flags=re.IGNORECASE)[0].strip()
            place_part = re.sub(r"[\.\?!,;]+$", "", place_part).strip()
            if place_part:
                raw_place = place_part
                matched_known = self._match_known_place(raw_place)
                if matched_known:
                    place = matched_known["name"]
                    destination_type = matched_known["type"]
                else:
                    place = raw_place
                    destination_type = self._infer_destination_type(raw_place)
                break

        entities = {
            "place": place,
            "destination_type": destination_type,
            "date": date,
            "time": appt_time,
            "purpose": purpose,
        }

        return IntentResult(
            should_store=True,
            category="APPOINTMENT",
            intent="CREATE_APPOINTMENT_MEMORY",
            confidence=0.93,
            entities=entities,
            requires_clarification=(place is None),
            clarification_prompt="Which hospital or clinic is the appointment at?" if place is None else None,
            user_request=raw_text,
        )

    def _extract_transport_intent(self, raw_text: str, norm_text: str, lowered: str) -> Optional[IntentResult]:
        """Detect and extract transportation and ride booking requests."""
        trans_patterns = [
            r"\b(book|need|want|get)\s+(?:me\s+)?(?:a\s+)?(bike|ride|cab|taxi|car|auto)\b",
            r"\b(book|need)\s+(?:a\s+)?ride\b",
            r"\brange\s+transportation\b",
        ]
        is_trans = any(re.search(pat, lowered) for pat in trans_patterns)
        if not is_trans:
            return None

        vehicle = None
        destination = None
        destination_type = None
        pickup = None
        requested_time = None

        # Determine vehicle type
        for v_name, v_keywords in VEHICLE_KEYWORDS.items():
            if any(k in lowered for k in v_keywords):
                vehicle = v_name
                break
        if not vehicle and "ride" in lowered:
            vehicle = "ride"

        # Determine destination: find "to <Place>" (skip infinitives like "to book")
        for m in re.finditer(r"\bto\s+", norm_text, re.IGNORECASE):
            after = norm_text[m.end():].strip()
            if not re.match(r"^(?:book|get|order|take|have|see|visit|ride|reach)\b", after, re.IGNORECASE):
                dest_part = re.split(r"\b(?:from|at\s+\d|now|tomorrow)\b", after, flags=re.IGNORECASE)[0].strip()
                dest_part = re.sub(r"[\.\?!,;]+$", "", dest_part).strip()
                if dest_part:
                    raw_dest = dest_part
                    matched_known = self._match_known_place(raw_dest)
                    if matched_known:
                        destination = matched_known["name"]
                        destination_type = matched_known["type"]
                    else:
                        destination = raw_dest
                        destination_type = self._infer_destination_type(raw_dest)
                    break

        # Check pickup location if explicitly provided: "from <Pickup>"
        from_match = re.search(r"\bfrom\s+([a-zA-Z0-9_\-\s]+?)(?:\s+(?:to|$))", norm_text, re.IGNORECASE)
        if from_match:
            pickup = from_match.group(1).strip()

        # Check requested time
        time_match = re.search(r"\bat\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\b", lowered)
        if time_match:
            requested_time = time_match.group(1).strip()

        entities = {
            "vehicle": vehicle,
            "destination": destination,
            "destination_type": destination_type,
            "pickup": pickup,
            "requested_time": requested_time,
        }

        needs_clarification = destination is None
        prompt = None
        if needs_clarification:
            prompt = "Where would you like to go?"
        elif destination and destination.lower() in {"hospital", "the hospital"}:
            prompt = "Which hospital would you like to go to?"

        return IntentResult(
            should_store=True,
            category="TRANSPORT",
            intent="BOOK_RIDE",
            confidence=0.95,
            entities=entities,
            requires_clarification=needs_clarification,
            clarification_prompt=prompt,
            user_request=raw_text,
        )

    def _extract_reminder_intent(self, raw_text: str, norm_text: str, lowered: str) -> Optional[IntentResult]:
        """Detect and extract reminder requests."""
        if not re.search(r"\b(remind me|set a reminder)\b", lowered):
            return None

        item = None
        rem_time = None

        m = re.search(r"\bremind me\s+(?:about\s+|to\s+)?(.+?)(?:\s+at\s+([a-zA-Z0-9_\-:\s]+?))?$", norm_text, re.IGNORECASE)
        if m:
            item = m.group(1).strip()
            rem_time = m.group(2).strip() if m.group(2) else None

        entities = {
            "reminder_item": item,
            "time": rem_time,
        }

        return IntentResult(
            should_store=True,
            category="REMINDER",
            intent="SET_REMINDER",
            confidence=0.91,
            entities=entities,
            requires_clarification=(rem_time is None),
            clarification_prompt="What time should I remind you?" if rem_time is None else None,
            user_request=raw_text,
        )

    def _extract_healthcare_intent(self, raw_text: str, norm_text: str, lowered: str) -> Optional[IntentResult]:
        """Detect healthcare readings or medical status updates."""
        health_patterns = [
            r"\b(blood pressure|bp reading|sugar level|glucose|heart rate|pulse)\b",
            r"\bmy weight is\b",
            r"\bfeeling (dizzy|nauseous|sick|feverish)\b",
        ]
        if not any(re.search(pat, lowered) for pat in health_patterns):
            return None

        entities = {
            "metric": "health_status",
            "details": raw_text,
        }

        return IntentResult(
            should_store=True,
            category="HEALTHCARE",
            intent="RECORD_HEALTH_METRIC",
            confidence=0.85,
            entities=entities,
            user_request=raw_text,
        )

    def _match_known_place(self, raw_query: str) -> Optional[Dict[str, str]]:
        """Match query string against known frequent user places."""
        q = raw_query.lower().strip()
        for p in self.known_places:
            name = p["name"].lower()
            if q == name or q in name or name in q:
                return p
            if "kims" in q and "kims" in name:
                return p
        return None

    def _infer_destination_type(self, raw_dest: str) -> Optional[str]:
        """Infer destination type (hospital, pharmacy, home, etc.) from name."""
        q = raw_dest.lower()
        for dtype, keywords in DESTINATION_TYPE_KEYWORDS.items():
            if any(k in q for k in keywords):
                return dtype
        return None
