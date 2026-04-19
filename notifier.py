# notifier.py
# Centralized notifications (prevents double beeps / spam)
import time
import sys
import subprocess
from pathlib import Path

_last_event_time = {}

def _bell():
    # Terminal bell (often muted on macOS terminals)
    sys.stdout.write("\a")
    sys.stdout.flush()

def _mac_beep():
    """
    macOS guaranteed beep using built-in system sound.
    Uses afplay (built-in).
    """
    # This file exists on macOS
    sound = "/System/Library/Sounds/Glass.aiff"
    try:
        subprocess.Popen(
            ["afplay", sound],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return True
    except Exception:
        return False

def notify(event: str, cooldown: float = 1.0, beep: bool = True):
    """
    event: "flip" | "warning" | "done"
    cooldown: seconds to prevent spam
    """
    now = time.monotonic()
    last = _last_event_time.get(event, 0.0)
    if now - last < cooldown:
        return
    _last_event_time[event] = now

    if beep:
        # Prefer real mac sound; fallback to terminal bell
        if not _mac_beep():
            _bell()

    # Logs
    if event == "flip":
        print("🔄 FLIP NOW")
    elif event == "warning":
        print("⚠️ WARNING: CENTER ZONE")
    elif event == "done":
        print("✅ FOOD READY")
