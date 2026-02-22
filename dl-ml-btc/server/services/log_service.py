"""
Log service - Captures application logs into an in-memory circular buffer
"""
import logging
from collections import deque
from datetime import datetime
from threading import Lock


class InMemoryLogHandler(logging.Handler):
    """Custom logging handler that stores log records in a deque."""

    def __init__(self, max_entries=500):
        super().__init__()
        self._buffer = deque(maxlen=max_entries)
        self._lock = Lock()

    def emit(self, record):
        entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": self.format(record),
        }
        with self._lock:
            self._buffer.append(entry)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def get_logs(self, level=None, search=None, limit=200):
        with self._lock:
            logs = list(self._buffer)

        if level:
            level_upper = level.upper()
            logs = [l for l in logs if l["level"] == level_upper]

        if search:
            search_lower = search.lower()
            logs = [l for l in logs if search_lower in l["message"].lower()]

        return logs[-limit:]

    def clear(self):
        with self._lock:
            self._buffer.clear()

    @property
    def count(self):
        return len(self._buffer)


# --------------- Singleton Setup ---------------
log_handler = InMemoryLogHandler(max_entries=500)
log_handler.setFormatter(logging.Formatter("%(message)s"))

# Attach to root logger so every library / module log is captured
_root = logging.getLogger()
_root.addHandler(log_handler)
