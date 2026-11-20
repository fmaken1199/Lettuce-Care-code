"""Small agent protocol. Payloads are JSON-compatible dictionaries."""
from __future__ import annotations
from datetime import datetime, timezone
import time
from typing import Any

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

class BaseAgent:
    name = "base"
    goal = ""

    def __init__(self):
        self.state: dict[str, Any] = {"runs": 0, "last_status": None}
        self._started = time.perf_counter()

    def begin(self):
        self._started = time.perf_counter()
        self.state["runs"] += 1

    def result(self, data=None, status="ok", decisions=None, messages=None):
        if status not in {"ok", "partial", "blocked", "error"}:
            raise ValueError("Invalid agent status")
        self.state["last_status"] = status
        return {"agent": self.name, "goal": self.goal, "status": status,
                "data": data or {}, "decisions": decisions or [],
                "messages": messages or [], "timestamp": utc_now(),
                "elapsed_ms": round((time.perf_counter()-self._started)*1000, 2),
                "state": dict(self.state)}

def decision(action: str, reason: str, **details) -> dict:
    return {"action": action, "reason": reason, **details}

CLASS_NAMES = ["Anthracnose", "Bacterial", "Bottom_Rot", "Downy_Mildew",
               "Healthy", "Lettuce_Mosaic_Virus", "Powdery_Mildew",
               "Septoria_Blight", "Viral", "Wilt_and_Leaf_Blight"]

# Use None for missing readings; preserve valid zeroes.
