"""
Rapido Transport Provider Adapter (Placeholder)
===============================================
Implements TransportProvider contract for future Rapido integration.
Does NOT perform real bookings or assume undocumented APIs exist.
"""

from typing import Dict, Any, List, Optional
from ..base import TransportProvider


class RapidoProvider(TransportProvider):
    """Adapter for future Rapido ride-hailing services."""

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(name="Rapido")
        self.api_key = api_key
        self.is_configured = bool(api_key)

    async def search_rides(
        self,
        pickup_location: str,
        destination: str,
        vehicle_type: Optional[str] = "bike",
    ) -> List[Dict[str, Any]]:
        if not self.is_configured:
            # Informative structural placeholder
            return [{
                "provider": self.name,
                "vehicle": vehicle_type or "bike",
                "pickup": pickup_location,
                "destination": destination,
                "status": "API_NOT_CONFIGURED",
                "notice": "Future integration placeholder. No live booking executed.",
            }]
        raise NotImplementedError("Live Rapido API endpoint not implemented.")

    async def estimate_fare(
        self,
        pickup_location: str,
        destination: str,
        vehicle_type: Optional[str] = "bike",
    ) -> Dict[str, Any]:
        return {
            "provider": self.name,
            "estimated_fare": None,
            "currency": "INR",
            "eta_minutes": None,
            "status": "UNAVAILABLE",
        }

    async def book_ride(
        self,
        quote_id: str,
        pickup_location: str,
        destination: str,
        user_confirmation_token: str,
    ) -> Dict[str, Any]:
        raise NotImplementedError("Live Rapido ride booking is disabled.")

    async def cancel_ride(self, ride_id: str) -> bool:
        return False
