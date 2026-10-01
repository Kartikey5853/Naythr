"""
Naythr Provider Integration Architecture
========================================
Base abstract classes defining contracts for external service adapters.
Core Naythr code does NOT know provider-specific implementation details.
Adapters hide all third-party details and provide future extension points.
"""

from typing import Dict, Any, List, Optional


class BaseProvider:
    """Base class for all third-party service provider adapters."""

    def __init__(self, name: str, is_configured: bool = False):
        self.name = name
        self.is_configured = is_configured

    async def is_available(self) -> bool:
        """Check if provider API credentials and endpoints are available."""
        return self.is_configured


class TransportProvider(BaseProvider):
    """Abstract interface for ride-hailing services (e.g. Rapido, Uber, Ola)."""

    def __init__(self, name: str):
        super().__init__(name=name, is_configured=False)

    async def search_rides(
        self,
        pickup_location: str,
        destination: str,
        vehicle_type: Optional[str] = "bike",
    ) -> List[Dict[str, Any]]:
        """Search available rides and options."""
        raise NotImplementedError("Provider API integration not yet configured.")

    async def estimate_fare(
        self,
        pickup_location: str,
        destination: str,
        vehicle_type: Optional[str] = "bike",
    ) -> Dict[str, Any]:
        """Get price and ETA quote for ride."""
        raise NotImplementedError("Provider API integration not yet configured.")

    async def book_ride(
        self,
        quote_id: str,
        pickup_location: str,
        destination: str,
        user_confirmation_token: str,
    ) -> Dict[str, Any]:
        """Execute ride booking upon user confirmation."""
        raise NotImplementedError("Provider API integration not yet configured.")

    async def cancel_ride(self, ride_id: str) -> bool:
        """Cancel an existing ride."""
        raise NotImplementedError("Provider API integration not yet configured.")


class PharmacyProvider(BaseProvider):
    """Abstract interface for online pharmacies and medicine ordering."""

    def __init__(self, name: str):
        super().__init__(name=name, is_configured=False)

    async def search_medicines(self, query: str) -> List[Dict[str, Any]]:
        """Search catalog for medicine availability."""
        raise NotImplementedError("Pharmacy API integration not yet configured.")

    async def check_prescription_requirement(self, medicine_name: str) -> bool:
        """Check if doctor's prescription is required."""
        raise NotImplementedError("Pharmacy API integration not yet configured.")

    async def order_prescription(
        self,
        medicine_name: str,
        quantity: int,
        delivery_address: str,
        user_confirmation_token: str,
    ) -> Dict[str, Any]:
        """Order medicine through pharmacy adapter."""
        raise NotImplementedError("Pharmacy API integration not yet configured.")


class HospitalProvider(BaseProvider):
    """Abstract interface for hospital consultations and appointments."""

    def __init__(self, name: str):
        super().__init__(name=name, is_configured=False)

    async def check_available_slots(
        self,
        hospital_name: str,
        department: Optional[str] = None,
        preferred_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Check open consultation slots."""
        raise NotImplementedError("Hospital scheduling API integration not yet configured.")

    async def schedule_appointment(
        self,
        hospital_name: str,
        slot_id: str,
        patient_name: str,
        user_confirmation_token: str,
    ) -> Dict[str, Any]:
        """Schedule a doctor consultation appointment."""
        raise NotImplementedError("Hospital scheduling API integration not yet configured.")
