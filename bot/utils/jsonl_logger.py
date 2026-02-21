"""Simple JSONL file logger utility."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Optional


class JsonlLogger:
    """Thread-safe JSONL logger for append-only records."""

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    def log(self, payload: Dict[str, Any], ts: Optional[datetime] = None) -> None:
        record = {
            "timestamp": (ts or datetime.now(timezone.utc)).isoformat(),
            **payload,
        }
        line = json.dumps(record, default=str)
        with self._lock:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    @property
    def path(self) -> Path:
        return self._path
