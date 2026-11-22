"""Export every match: CSV (one row per player per game) or JSON (everything)."""
from __future__ import annotations

import csv
import io
import json

from . import stats, store
from .tracked import TRACKED_STATS


def _matches():
    cfg = store.load_config()
    ms = store.all_matches()
    me = stats.detect_me([m for m in ms if not m["hidden"]], cfg)
    stats.mark_duplicates(ms)
    aliases = cfg.get("aliases") or []
    for m in ms:
        stats.enrich(m, me)
        stats.apply_aliases(m, aliases, me)
    return ms


def to_csv() -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["match_id", "played_at", "mode", "ranked", "finished", "your_result", "hidden", "duplicate",
                "side", "team", "team_color", "team_score", "team_won", "player", "alias", "is_you",
                "mvp", *TRACKED_STATS])
    for m in _matches():
        for ti, t in enumerate(m["teams"]):
            side = "" if m.get("my_team") is None else ("us" if ti == m["my_team"] else "them")
            for p in t["players"]:
                w.writerow([m["id"], m["played_at"], m.get("playlist"), int(bool(m.get("ranked"))),
                            int(bool(m.get("is_final"))), m.get("result") or "", int(bool(m.get("hidden"))),
                            int(bool(m.get("duplicate_of"))), side, t.get("name"), t.get("color"), t.get("score"),
                            int(bool(t.get("winner"))), p["name"], p.get("alias") or "", int(p.get("key") == "me"),
                            int(m.get("mvp") == p["name"]), *[p["stats"].get(k, "") for k in TRACKED_STATS]])
    return buf.getvalue()


def to_json() -> str:
    out = []
    for m in _matches():
        out.append({
            "id": m["id"], "played_at": m["played_at"], "mode": m.get("playlist"), "ranked": bool(m.get("ranked")),
            "finished": bool(m.get("is_final")), "your_result": m.get("result"), "hidden": bool(m.get("hidden")),
            "duplicate_of": m.get("duplicate_of"), "mvp": m.get("mvp"),
            "teams": [{"name": t.get("name"), "color": t.get("color"), "score": t.get("score"), "won": bool(t.get("winner")),
                       "players": [{"name": p["name"], "alias": p.get("alias"), "is_you": p.get("key") == "me",
                                    "stats": {k: p["stats"].get(k) for k in TRACKED_STATS}} for p in t["players"]]}
                      for t in m["teams"]],
        })
    return json.dumps({"app": "RL Tracker", "stats": TRACKED_STATS, "matches": out}, indent=2)
