"""
Comprehensive Unit Test Suite for Naythr Action + Memory Layer
==============================================================
Tests all 10 core requirements:
- TEST 1: "I was prescribed amoxicillin." -> MEDICATION stored, dosage/frequency null
- TEST 2: "Doctor told me to take amoxicillin twice a day for five days." -> MEDICATION, twice_daily, 5 days
- TEST 3: "I want to book a bike to KIMS Hospital." -> TRANSPORT, BOOK_RIDE, PENDING_CONFIRMATION, no external booking
- TEST 4: "What's around me?" -> nothing persisted
- TEST 5: "Hello Marvin." -> nothing persisted
- TEST 6: "I have an appointment at KIMS tomorrow at 10." -> APPOINTMENT stored
- TEST 7: "Order my medicine." -> PURCHASE action, PENDING_INFORMATION/CONFIRMATION, does not execute
- TEST 8: Missing destination: "Book me a bike." -> destination is null, needs clarification
- TEST 9: Missing medication dosage: "I was prescribed medicine X." -> dosage = null
- TEST 10: Duplicate request handling within short window -> no duplicate actions created
"""

import sys
import tempfile
import shutil
from pathlib import Path

# Ensure import paths
_CURRENT_DIR = Path(__file__).resolve().parent
_LIVE_CALL_DIR = _CURRENT_DIR.parent
_ROOT_DIR = _LIVE_CALL_DIR.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))
if str(_LIVE_CALL_DIR) not in sys.path:
    sys.path.insert(0, str(_LIVE_CALL_DIR))

from intents.intent_parser import IntentParser
from memory.memory_manager import MemoryManager


def run_all_tests():
    print("=" * 70)
    print(">>> RUNNING NAYTHR ACTION + MEMORY LAYER TEST SUITE <<<")
    print("=" * 70 + "\n")

    # Use a temporary directory for safe, isolated test storage
    test_dir = Path(tempfile.mkdtemp(prefix="naythr_test_memory_"))
    try:
        parser = IntentParser()
        memory = MemoryManager(data_dir=test_dir, deduplication_window_sec=5.0)

        # ----------------------------------------------------------------------
        # TEST 1: "I was prescribed amoxicillin."
        # ----------------------------------------------------------------------
        print("--- TEST 1: 'I was prescribed amoxicillin.' ---")
        t1_text = "I was prescribed amoxicillin."
        res1 = parser.parse(t1_text)
        assert res1.should_store is True, "Test 1 failed: should_store must be True"
        assert res1.category == "MEDICATION", f"Test 1 failed: Expected category MEDICATION, got {res1.category}"
        assert res1.entities.get("name") == "amoxicillin", f"Test 1 failed: Expected name 'amoxicillin', got {res1.entities.get('name')}"
        assert res1.entities.get("dosage") is None, f"Test 1 failed: dosage must be None, got {res1.entities.get('dosage')}"
        assert res1.entities.get("frequency") is None, f"Test 1 failed: frequency must be None, got {res1.entities.get('frequency')}"
        assert res1.entities.get("duration") is None, f"Test 1 failed: duration must be None, got {res1.entities.get('duration')}"

        record1 = memory.process_intent(res1)
        assert record1 is not None, "Test 1 failed: record was not created"
        meds = memory.get_medications()
        assert len(meds) == 1, f"Test 1 failed: Expected 1 medication in storage, got {len(meds)}"
        assert meds[0]["medication"]["name"] == "amoxicillin"
        assert meds[0]["medication"]["dosage"] is None
        print("-> Result: PASSED (MEDICATION stored, dosage/frequency strictly null)\n")

        # ----------------------------------------------------------------------
        # TEST 2: "Doctor told me to take amoxicillin twice a day for five days."
        # ----------------------------------------------------------------------
        print("--- TEST 2: 'Doctor told me to take amoxicillin twice a day for five days.' ---")
        t2_text = "Doctor told me to take amoxicillin twice a day for five days."
        res2 = parser.parse(t2_text)
        assert res2.should_store is True
        assert res2.category == "MEDICATION"
        assert res2.entities.get("name") == "amoxicillin"
        assert res2.entities.get("frequency") == "twice_daily", f"Test 2 failed: Expected 'twice_daily', got {res2.entities.get('frequency')}"
        assert res2.entities.get("duration") == "5 days", f"Test 2 failed: Expected '5 days', got {res2.entities.get('duration')}"
        assert res2.entities.get("dosage") is None

        record2 = memory.process_intent(res2)
        assert record2 is not None
        meds2 = memory.get_medications()
        assert len(meds2) == 2
        assert meds2[1]["medication"]["frequency"] == "twice_daily"
        assert meds2[1]["medication"]["duration"] == "5 days"
        print("-> Result: PASSED (MEDICATION, frequency=twice_daily, duration=5 days)\n")

        # ----------------------------------------------------------------------
        # TEST 3: "I want to book a bike to KIMS Hospital."
        # ----------------------------------------------------------------------
        print("--- TEST 3: 'I want to book a bike to KIMS Hospital.' ---")
        t3_text = "I want to book a bike to KIMS Hospital."
        res3 = parser.parse(t3_text)
        assert res3.should_store is True
        assert res3.category == "TRANSPORT"
        assert res3.intent == "BOOK_RIDE"
        assert res3.entities.get("vehicle") == "bike"
        assert res3.entities.get("destination") == "KIMS Hospital"
        assert res3.entities.get("destination_type") == "hospital"

        record3 = memory.process_intent(res3)
        assert record3 is not None
        assert record3["status"] == "PENDING_CONFIRMATION", f"Test 3 failed: Status must be PENDING_CONFIRMATION, got {record3['status']}"
        assert record3["integration"]["executed"] is False, "Test 3 failed: Must NOT execute live booking"
        pending = memory.get_pending_actions()
        assert len(pending) == 1
        assert pending[0]["id"] == record3["id"]
        print("-> Result: PASSED (TRANSPORT, BOOK_RIDE, bike, KIMS Hospital, PENDING_CONFIRMATION, NO external booking)\n")

        # ----------------------------------------------------------------------
        # TEST 4: "What's around me?"
        # ----------------------------------------------------------------------
        print("--- TEST 4: 'What's around me?' ---")
        t4_text = "What's around me?"
        res4 = parser.parse(t4_text)
        assert res4.should_store is False, f"Test 4 failed: Normal navigation must have should_store=False"
        record4 = memory.process_intent(res4)
        assert record4 is None, "Test 4 failed: Must not store normal conversation"
        print("-> Result: PASSED (nothing persisted)\n")

        # ----------------------------------------------------------------------
        # TEST 5: "Hello Marvin."
        # ----------------------------------------------------------------------
        print("--- TEST 5: 'Hello Marvin.' ---")
        t5_text = "Hello Marvin."
        res5 = parser.parse(t5_text)
        assert res5.should_store is False, f"Test 5 failed: Casual greeting must have should_store=False"
        record5 = memory.process_intent(res5)
        assert record5 is None, "Test 5 failed: Must not store greeting"
        print("-> Result: PASSED (nothing persisted)\n")

        # ----------------------------------------------------------------------
        # TEST 6: "I have an appointment at KIMS tomorrow at 10."
        # ----------------------------------------------------------------------
        print("--- TEST 6: 'I have an appointment at KIMS tomorrow at 10.' ---")
        t6_text = "I have an appointment at KIMS tomorrow at 10."
        res6 = parser.parse(t6_text)
        assert res6.should_store is True
        assert res6.category == "APPOINTMENT"
        assert res6.intent == "CREATE_APPOINTMENT_MEMORY"
        assert res6.entities.get("place") == "KIMS Hospital"
        assert res6.entities.get("date") == "tomorrow"
        assert res6.entities.get("time") == "10:00"

        record6 = memory.process_intent(res6)
        assert record6 is not None
        appts = memory.get_appointments()
        assert len(appts) == 1
        assert appts[0]["entities"]["place"] == "KIMS Hospital"
        assert appts[0]["entities"]["date"] == "tomorrow"
        print("-> Result: PASSED (APPOINTMENT stored properly)\n")

        # ----------------------------------------------------------------------
        # TEST 7: "Order my medicine."
        # ----------------------------------------------------------------------
        print("--- TEST 7: 'Order my medicine.' ---")
        t7_text = "Order my medicine."
        res7 = parser.parse(t7_text)
        assert res7.should_store is True
        assert res7.category == "PURCHASE"
        assert res7.intent == "PURCHASE_MEDICATION"
        record7 = memory.process_intent(res7)
        assert record7 is not None
        # Consequential action must be PENDING_INFORMATION or PENDING_CONFIRMATION
        assert record7["status"] in {"PENDING_CONFIRMATION", "PENDING_INFORMATION"}
        assert record7["integration"]["executed"] is False
        print("-> Result: PASSED (PURCHASE action, status=PENDING_INFORMATION/CONFIRMATION, does NOT execute)\n")

        # ----------------------------------------------------------------------
        # TEST 8: Missing destination: "Book me a bike."
        # ----------------------------------------------------------------------
        print("--- TEST 8: Missing destination: 'Book me a bike.' ---")
        t8_text = "Book me a bike."
        res8 = parser.parse(t8_text)
        assert res8.should_store is True
        assert res8.category == "TRANSPORT"
        assert res8.entities.get("destination") is None, f"Test 8 failed: Destination must be None, got {res8.entities.get('destination')}"
        assert res8.requires_clarification is True
        record8 = memory.process_intent(res8)
        assert record8 is not None
        assert record8["status"] == "PENDING_INFORMATION"
        assert record8["entities"]["destination"] is None
        print("-> Result: PASSED (Destination is strictly null, status=PENDING_INFORMATION)\n")

        # ----------------------------------------------------------------------
        # TEST 9: Missing medication dosage: "I was prescribed medicine X."
        # ----------------------------------------------------------------------
        print("--- TEST 9: Missing medication dosage: 'I was prescribed medicine X.' ---")
        t9_text = "I was prescribed medicine X."
        res9 = parser.parse(t9_text)
        assert res9.should_store is True
        assert res9.category == "MEDICATION"
        assert res9.entities.get("name") == "medicine X"
        assert res9.entities.get("dosage") is None, f"Test 9 failed: Dosage must be None, got {res9.entities.get('dosage')}"
        record9 = memory.process_intent(res9)
        assert record9 is not None
        assert record9["medication"]["dosage"] is None
        print("-> Result: PASSED (Dosage strictly null when not provided)\n")

        # ----------------------------------------------------------------------
        # TEST 10: Duplicate request handling
        # ----------------------------------------------------------------------
        print("--- TEST 10: Duplicate request handling within window ---")
        dup_text = "I want to book a bike to KIMS Hospital."
        dup_res = parser.parse(dup_text)
        # First call was already stored in Test 3.
        # Second immediate call with identical request:
        dup_record = memory.process_intent(dup_res)
        assert dup_record is None, "Test 10 failed: Duplicate request must be ignored within window"

        # Verify count of transport actions did not increase
        actions = memory.get_actions(action_type="TRANSPORT")
        # In test 3 we created 1 complete ride, in test 8 we created 1 incomplete ride = 2 total
        assert len(actions) == 2, f"Test 10 failed: Expected 2 transport actions, got {len(actions)}"
        print("-> Result: PASSED (Duplicate request correctly suppressed)\n")

        # ----------------------------------------------------------------------
        # DASHBOARD PAYLOAD VERIFICATION
        # ----------------------------------------------------------------------
        print("--- DASHBOARD PAYLOAD VERIFICATION ---")
        dash_data = memory.get_frontend_dashboard_data()
        assert "pending_actions" in dash_data
        assert "medications" in dash_data
        assert "appointments" in dash_data
        assert "recent_activity" in dash_data
        assert len(dash_data["pending_actions"]) >= 1
        assert len(dash_data["medications"]) >= 1
        assert len(dash_data["appointments"]) >= 1
        assert len(dash_data["recent_activity"]) >= 1
        print("-> Result: PASSED (Dashboard payload generated cleanly)\n")

        print("=" * 70)
        print(">>> ALL 10 ACTION + MEMORY TESTS PASSED SUCCESSFULLY! <<<")
        print("=" * 70)

    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    run_all_tests()
