"""Background thread that watches the replay folders (Demos / DemosEpic)."""
from __future__ import annotations

import threading
import time
import traceback

from . import replays, store


class Scanner:
    def __init__(self):
        self.state = {"running": False, "last_scan": None, "last_new": None}
        self._wake = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="scanner")

    def start(self):
        self._thread.start()

    def poke(self):
        self._wake.set()

    def _loop(self):
        while True:
            try:
                store.backup()
                self.state["running"] = True
                rep = replays.scan(self.state)
                self.state["replays"] = rep
                if rep.get("read"):
                    self.state["last_new"] = time.time()
            except Exception:
                traceback.print_exc()
            finally:
                self.state.update(running=False, last_scan=time.time())
            interval = max(2, int(store.load_config().get("poll_seconds") or 5))
            self._wake.wait(interval)
            self._wake.clear()
