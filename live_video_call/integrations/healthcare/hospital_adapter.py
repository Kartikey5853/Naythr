"""
Hospital & Healthcare Provider Adapter (Placeholder)
====================================================
Implements HospitalProvider contract for future appointment and consultation systems.
Does NOT perform real appointment bookings or clinical decisions.
"""

from typing import Dict, Any, List, Optional
from ..base import HospitalProvider


class HospitalAdapter(HospitalProvider):
    """Adapter for future hospital scheduling integrations."""

    def __init__(self, hospital_name: str = "KIMS Hospital", api_key: Optional[str] = None):
        super().__init__(name=hospital_name)
        self.api_key = api_key
        self.is_configured = bool(api_key)

    async def check_available_slots(
        self,
        hospital_name: str,
        department: Optional[str] = None,
        preferred_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return [{
            "hospital": hospital_name,
            "department": department or "General Medicine",
            "available_slots": [],
            "notice": "Hospital scheduling placeholder. No live slots connected.",
        }]

    async def schedule_appointment(
        self,
        hospital_name: str,
        slot_id: str,
        patient_name: str,
        user_confirmation_token: str,
    ) -> Dict[str, Any]:
        raise NotImplementedError("Live hospital appointment booking is disabled.")
