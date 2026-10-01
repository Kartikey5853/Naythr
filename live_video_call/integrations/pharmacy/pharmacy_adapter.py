"""
Pharmacy Adapter (Placeholder)
==============================
Implements PharmacyProvider contract for future online pharmacy integrations.
Does NOT place real medicine orders or prescribe drugs.
"""

from typing import Dict, Any, List, Optional
from ..base import PharmacyProvider


class PharmacyAdapter(PharmacyProvider):
    """Adapter for future online pharmacy order integrations."""

    def __init__(self, partner_name: str = "GenericPharmacy", api_key: Optional[str] = None):
        super().__init__(name=partner_name)
        self.api_key = api_key
        self.is_configured = bool(api_key)

    async def search_medicines(self, query: str) -> List[Dict[str, Any]]:
        return [{
            "query": query,
            "provider": self.name,
            "available": False,
            "notice": "Pharmacy integration placeholder. No live catalog connected.",
        }]

    async def check_prescription_requirement(self, medicine_name: str) -> bool:
        # Default safety: assume prescription is required
        return True

    async def order_prescription(
        self,
        medicine_name: str,
        quantity: int,
        delivery_address: str,
        user_confirmation_token: str,
    ) -> Dict[str, Any]:
        raise NotImplementedError("Live prescription ordering is disabled.")
