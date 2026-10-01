"""
Naythr Action & Memory Manager
==============================
Manages structured JSON persistence with atomic writes, de-duplication,
action status lifecycles, and dashboard payload generation.
"""

import os
import json
import time
import uuid
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger("MemoryManager")

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"


class MemoryManager:
    """Central Memory & Action Manager using atomic JSON storage."""

    def __init__(self, data_dir: Optional[Path] = None, deduplication_window_sec: float = 60.0):
        self.data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        self.deduplication_window_sec = deduplication_window_sec
        self._ensure_data_files()
        # In-memory recent request tracking for de-duplication: key -> timestamp
        self._recent_requests: Dict[str, float] = {}

    def _ensure_data_files(self):
        """Ensure data directory and base JSON files exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        defaults = {
            "actions.json": [],
            "medications.json": [],
            "appointments.json": [],
            "activity.json": [],
            "user_context.json": {
                "frequent_places": [
                    {"name": "KIMS Hospital", "type": "hospital"}
                ],
                "preferences": {
                    "preferred_ride": None
                }
            }
        }
        for filename, initial_val in defaults.items():
            fpath = self.data_dir / filename
            if not fpath.exists():
                self._atomic_write(fpath, initial_val)

    def _atomic_write(self, filepath: Path, data: Any):
        """Write JSON data atomically to prevent corruption during crash or power loss."""
        tmp_file = filepath.with_suffix(filepath.suffix + ".tmp")
        try:
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_file, filepath)
        except Exception as e:
            logger.error(f"Failed atomic write to {filepath}: {e}")
            if tmp_file.exists():
                try:
                    tmp_file.unlink()
                except Exception:
                    pass
            raise

    def _read_json(self, filename: str) -> Any:
        """Safely read JSON data from file."""
        fpath = self.data_dir / filename
        if not fpath.exists():
            return [] if filename != "user_context.json" else {}
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading {filename}: {e}")
            return [] if filename != "user_context.json" else {}

    def _is_duplicate_request(self, user_request: str, intent_name: str) -> bool:
        """Check if identical request was received within the de-duplication window."""
        norm_text = "".join(user_request.lower().split())
        key = f"{intent_name}:{norm_text}"
        now = time.time()

        # Clean old entries
        self._recent_requests = {
            k: t for k, t in self._recent_requests.items()
            if (now - t) < self.deduplication_window_sec
        }

        if key in self._recent_requests:
            return True

        self._recent_requests[key] = now
        return False

    # --------------------------------------------------------------------------
    # PROCESSING FROM INTENT RESULT
    # --------------------------------------------------------------------------
    def process_intent(self, intent_result: Any, source: str = "user_voice") -> Optional[Dict[str, Any]]:
        """Process an IntentResult, route to appropriate storage, and log activity."""
        if not intent_result.should_store:
            return None

        # Check de-duplication
        if self._is_duplicate_request(intent_result.user_request, intent_result.intent or "UNKNOWN"):
            logger.info(f"Duplicate request ignored within {self.deduplication_window_sec}s window: '{intent_result.user_request}'")
            return None

        category = intent_result.category
        if category == "MEDICATION":
            record = self.save_medication(
                name=intent_result.entities.get("name"),
                dosage=intent_result.entities.get("dosage"),
                frequency=intent_result.entities.get("frequency"),
                duration=intent_result.entities.get("duration"),
                intent=intent_result.intent or "MEDICATION_PRESCRIPTION",
                user_request=intent_result.user_request,
                source=source,
            )
            self._log_activity(
                request=intent_result.user_request,
                summary=f"Prescription recorded: {intent_result.entities.get('name')}",
                category=category,
                intent=intent_result.intent or "MEDICATION_PRESCRIPTION",
            )
            return record

        elif category == "APPOINTMENT":
            record = self.save_appointment(
                place=intent_result.entities.get("place"),
                destination_type=intent_result.entities.get("destination_type"),
                date=intent_result.entities.get("date"),
                time_str=intent_result.entities.get("time"),
                purpose=intent_result.entities.get("purpose"),
                intent=intent_result.intent or "CREATE_APPOINTMENT_MEMORY",
                user_request=intent_result.user_request,
                source=source,
            )
            self._log_activity(
                request=intent_result.user_request,
                summary=f"Appointment recorded: {intent_result.entities.get('place')}",
                category=category,
                intent=intent_result.intent or "CREATE_APPOINTMENT_MEMORY",
            )
            return record

        else:
            # Action-type records (Transport, Purchase, Reminder, Healthcare, Other)
            status = "PENDING_CONFIRMATION"
            if intent_result.requires_clarification:
                status = "PENDING_INFORMATION"

            record = self.save_action(
                action_type=category,
                intent=intent_result.intent or "ACTION",
                status=status,
                user_request=intent_result.user_request,
                entities=intent_result.entities,
                source=source,
            )
            summary_desc = f"{category.title()} request"
            if category == "TRANSPORT":
                v = intent_result.entities.get("vehicle", "ride")
                d = intent_result.entities.get("destination", "unknown")
                summary_desc = f"Ride request: {v} -> {d}"
            elif category == "PURCHASE":
                p = intent_result.entities.get("product", "item")
                summary_desc = f"Order request: {p}"

            self._log_activity(
                request=intent_result.user_request,
                summary=summary_desc,
                category=category,
                intent=intent_result.intent or "ACTION",
            )
            return record

    # --------------------------------------------------------------------------
    # ACTION RECORDS
    # --------------------------------------------------------------------------
    def save_action(
        self,
        action_type: str,
        intent: str,
        user_request: str,
        entities: Dict[str, Any],
        status: str = "PENDING_CONFIRMATION",
        source: str = "user_voice",
    ) -> Dict[str, Any]:
        """Save an actionable task record into actions.json."""
        actions = self._read_json("actions.json")
        now_dt = datetime.now()
        record_id = f"act_{now_dt.strftime('%Y%m%d')}_{uuid.uuid4().hex[:6]}"

        record = {
            "id": record_id,
            "timestamp": now_dt.isoformat(),
            "type": action_type,
            "intent": intent,
            "status": status,
            "user_request": user_request,
            "entities": entities,
            "integration": {
                "provider": None,
                "executed": False,
            },
            "source": source,
            "verification": {
                "confirmed_by_user": False,
                "needs_clarification": (status == "PENDING_INFORMATION"),
            },
        }
        actions.append(record)
        self._atomic_write(self.data_dir / "actions.json", actions)
        return record

    def get_actions(
        self,
        status: Optional[str] = None,
        action_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve stored actions with optional filtering."""
        actions = self._read_json("actions.json")
        if status:
            actions = [a for a in actions if a.get("status") == status]
        if action_type:
            actions = [a for a in actions if a.get("type") == action_type]
        return actions

    def get_pending_actions(self) -> List[Dict[str, Any]]:
        """Get all actions waiting for user confirmation or information."""
        actions = self._read_json("actions.json")
        return [
            a for a in actions
            if a.get("status") in {"PENDING_CONFIRMATION", "PENDING_INFORMATION"}
        ]

    def update_action_status(self, action_id: str, new_status: str) -> bool:
        """Update the status of an existing action."""
        actions = self._read_json("actions.json")
        found = False
        for a in actions:
            if a.get("id") == action_id:
                a["status"] = new_status
                if new_status == "READY":
                    a.setdefault("verification", {})["confirmed_by_user"] = True
                found = True
                break
        if found:
            self._atomic_write(self.data_dir / "actions.json", actions)
        return found

    # --------------------------------------------------------------------------
    # MEDICATION RECORDS
    # --------------------------------------------------------------------------
    def save_medication(
        self,
        name: Optional[str],
        dosage: Optional[str] = None,
        frequency: Optional[str] = None,
        duration: Optional[str] = None,
        intent: str = "MEDICATION_PRESCRIPTION",
        user_request: str = "",
        status: str = "ACTIVE",
        source: str = "user_voice",
    ) -> Dict[str, Any]:
        """Save a medication record strictly without invented fields."""
        meds = self._read_json("medications.json")
        now_dt = datetime.now()
        med_id = f"med_{uuid.uuid4().hex[:6]}"

        record = {
            "id": med_id,
            "type": "MEDICATION",
            "intent": intent,
            "timestamp": now_dt.isoformat(),
            "status": status,
            "medication": {
                "name": name,
                "dosage": dosage,
                "frequency": frequency,
                "duration": duration,
            },
            "source": source,
            "verified": False,
            "user_request": user_request,
        }
        meds.append(record)
        self._atomic_write(self.data_dir / "medications.json", meds)
        return record

    def get_medications(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve stored medications."""
        meds = self._read_json("medications.json")
        if status:
            meds = [m for m in meds if m.get("status") == status]
        return meds

    # --------------------------------------------------------------------------
    # APPOINTMENT RECORDS
    # --------------------------------------------------------------------------
    def save_appointment(
        self,
        place: Optional[str],
        destination_type: Optional[str] = None,
        date: Optional[str] = None,
        time_str: Optional[str] = None,
        purpose: Optional[str] = None,
        intent: str = "CREATE_APPOINTMENT_MEMORY",
        user_request: str = "",
        status: str = "RECORDED",
        source: str = "user_voice",
    ) -> Dict[str, Any]:
        """Save a structured appointment record."""
        appts = self._read_json("appointments.json")
        now_dt = datetime.now()
        appt_id = f"appt_{uuid.uuid4().hex[:6]}"

        record = {
            "id": appt_id,
            "type": "APPOINTMENT",
            "intent": intent,
            "timestamp": now_dt.isoformat(),
            "status": status,
            "entities": {
                "place": place,
                "destination_type": destination_type,
                "date": date,
                "time": time_str,
                "purpose": purpose,
            },
            "source": source,
            "user_request": user_request,
        }
        appts.append(record)
        self._atomic_write(self.data_dir / "appointments.json", appts)
        return record

    def get_appointments(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve stored appointments."""
        appts = self._read_json("appointments.json")
        if status:
            appts = [a for a in appts if a.get("status") == status]
        return appts

    # --------------------------------------------------------------------------
    # ACTIVITY HISTORY
    # --------------------------------------------------------------------------
    def _log_activity(self, request: str, summary: str, category: str, intent: str):
        """Append an entry to lightweight activity history."""
        activity = self._read_json("activity.json")
        now_dt = datetime.now()
        entry = {
            "timestamp": now_dt.isoformat(),
            "time_str": now_dt.strftime("%H:%M"),
            "request": request,
            "summary": summary,
            "category": category,
            "intent": intent,
        }
        activity.append(entry)
        # Keep last 100 entries max
        if len(activity) > 100:
            activity = activity[-100:]
        self._atomic_write(self.data_dir / "activity.json", activity)

    def get_recent_activity(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get latest activity entries in reverse chronological order."""
        activity = self._read_json("activity.json")
        return list(reversed(activity[-limit:]))

    # --------------------------------------------------------------------------
    # USER CONTEXT
    # --------------------------------------------------------------------------
    def get_user_context(self) -> Dict[str, Any]:
        """Get stored user preferences and frequent places."""
        return self._read_json("user_context.json")

    def add_frequent_place(self, name: str, place_type: str = "place"):
        """Add an explicitly known place to user context."""
        ctx = self.get_user_context()
        places = ctx.setdefault("frequent_places", [])
        if not any(p.get("name", "").lower() == name.lower() for p in places):
            places.append({"name": name, "type": place_type})
            self._atomic_write(self.data_dir / "user_context.json", ctx)

    # --------------------------------------------------------------------------
    # RESET / TESTING
    # --------------------------------------------------------------------------
    def reset_memory(self, category: Optional[str] = None):
        """Clear memory data for testing or user request."""
        if category == "actions" or category is None:
            self._atomic_write(self.data_dir / "actions.json", [])
        if category == "medications" or category is None:
            self._atomic_write(self.data_dir / "medications.json", [])
        if category == "appointments" or category is None:
            self._atomic_write(self.data_dir / "appointments.json", [])
        if category == "activity" or category is None:
            self._atomic_write(self.data_dir / "activity.json", [])
        self._recent_requests.clear()
        logger.info(f"Memory reset complete (category: {category or 'all'}).")

    # --------------------------------------------------------------------------
    # FRONTEND DASHBOARD PAYLOAD GENERATOR
    # --------------------------------------------------------------------------
    def get_frontend_dashboard_data(self) -> Dict[str, Any]:
        """Compile a clean, formatted payload for the frontend dashboard cards."""
        pending_actions = self.get_pending_actions()
        medications = self.get_medications(status="ACTIVE")
        appointments = self.get_appointments()
        recent_activity = self.get_recent_activity(limit=10)
        user_context = self.get_user_context()

        return {
            "pending_actions": [
                {
                    "id": a["id"],
                    "icon": "🚗" if a["type"] == "TRANSPORT" else ("💊" if a["type"] == "PURCHASE" else "📌"),
                    "title": f"{a['entities'].get('vehicle', 'Ride').title()} → {a['entities'].get('destination', 'Destination')}"
                             if a["type"] == "TRANSPORT"
                             else f"Order {a['entities'].get('product', 'Item')}",
                    "status_label": "Waiting for confirmation" if a["status"] == "PENDING_CONFIRMATION" else "Waiting for information",
                    "status": a["status"],
                    "original_request": a["user_request"],
                    "timestamp": a["timestamp"],
                }
                for a in pending_actions
            ],
            "medications": [
                {
                    "id": m["id"],
                    "name": (m["medication"].get("name") or "Medicine").title(),
                    "frequency": m["medication"].get("frequency", "Unspecified").replace("_", " ").title() if m["medication"].get("frequency") else "Unspecified",
                    "dosage": m["medication"].get("dosage") or "Not specified",
                    "duration": m["medication"].get("duration") or "Ongoing",
                    "status": m["status"],
                }
                for m in medications
            ],
            "appointments": [
                {
                    "id": ap["id"],
                    "place": ap["entities"].get("place", "Medical Center"),
                    "date": (ap["entities"].get("date") or "Upcoming").title(),
                    "time": ap["entities"].get("time") or "Time unconfirmed",
                    "status": ap["status"],
                }
                for ap in appointments
            ],
            "recent_activity": [
                {
                    "time": item.get("time_str", "--:--"),
                    "summary": item.get("summary", "Activity"),
                    "category": item.get("category", "GENERAL"),
                }
                for item in recent_activity
            ],
            "user_context": user_context,
            "counts": {
                "pending_actions": len(pending_actions),
                "medications": len(medications),
                "appointments": len(appointments),
            }
        }
