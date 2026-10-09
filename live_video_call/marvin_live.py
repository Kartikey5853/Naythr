"""Gemini Live Video Call with 'Hey Marvin' Wake Word (Alias for main.py).

Run directly:
    python marvin_live.py
or:
    python main.py
"""

import sys
from pathlib import Path

_CURRENT_DIR = Path(__file__).resolve().parent
if str(_CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(_CURRENT_DIR))

from main import main

if __name__ == "__main__":
    main()
