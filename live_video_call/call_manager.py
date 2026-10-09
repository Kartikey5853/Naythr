"""
Top-level alias / redirect for CallConversationManager.
Re-exports CallConversationManager and helpers from live_video_call.memory.call_manager.
"""

from live_video_call.memory.call_manager import (
    CallConversationManager,
    print_phone_call_summary,
    DEFAULT_CONTACTS,
)

__all__ = [
    "CallConversationManager",
    "print_phone_call_summary",
    "DEFAULT_CONTACTS",
]
