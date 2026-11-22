"""Read Rocket League .replay files with the bundled rrrocket parser.

rrrocket (github.com/nickbabcock/rrrocket, v0.11.9) lives in tools/rrrocket/ and
only ever *reads* replay files. Everything else in the app goes through
`header()` (fast, ~10 ms) or `load()` (full network data, ~0.5 s), so the parser
could be swapped later without touching the rest of the code.
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess

from . import store

AUTO_NAME = re.compile(r"^([0-9A-Fa-f]{32}|[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12})\.replay$")


def exe_path() -> str | None:
    hits = glob.glob(os.path.join(store.ROOT, "tools", "rrrocket", "**", "rrrocket.exe"), recursive=True)
    return hits[0] if hits else None


def _run(path: str, network: bool) -> dict:
    exe = exe_path()
    if not exe:
        raise RuntimeError("rrrocket.exe not found in tools/rrrocket")
    args = [exe, "-n", path] if network else [exe, path]
    out = subprocess.run(args, capture_output=True, check=True, timeout=120,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
    return json.loads(out)


def header(path: str) -> dict:
    """Match info only: players + scoreboard stats, score, date, custom name."""
    d = _run(path, network=False)
    p = d.get("properties", {})
    return _summary(path, p)


def load(path: str) -> dict:
    """Header summary plus the full decoded network stream:
    {"header": {...}, "frames": [...], "objects": [...], "names": [...]}"""
    d = _run(path, network=True)
    return {
        "header": _summary(path, d.get("properties", {})),
        "frames": (d.get("network_frames") or {}).get("frames", []),
        "objects": d.get("objects", []),
        "names": d.get("names", []),
    }


def _platform(p: dict) -> str:
    v = (p.get("Platform") or {}).get("value", "")
    return v.replace("OnlinePlatform_", "") or "Unknown"


def _summary(path: str, p: dict) -> dict:
    players = []
    for s in p.get("PlayerStats") or []:
        oid = str(s.get("OnlineID") or "")
        plat = _platform(s)
        # Epic accounts report OnlineID 0 in the header; fall back to name so it's still stable-ish.
        pid = f"{plat}:{oid}" if oid not in ("", "0") else f"{plat}:name:{s.get('Name')}"
        players.append({
            "name": s.get("Name"), "team": s.get("Team"), "platform": plat, "id": pid, "bot": bool(s.get("bBot")),
            "stats": {"score": s.get("Score"), "goals": s.get("Goals"), "assists": s.get("Assists"),
                      "saves": s.get("Saves"), "shots": s.get("Shots")},
        })
    name = os.path.basename(path)
    custom_title = (p.get("ReplayName") or "").strip()
    return {
        "file": name, "guid": p.get("Id"), "match_guid": p.get("MatchGUID"),
        "date": p.get("Date"),                       # local time "YYYY-MM-DD HH-MM-SS"
        "match_type": p.get("MatchType"), "team_size": p.get("TeamSize"), "map": p.get("MapName"),
        "score": [p.get("Team0Score") or 0, p.get("Team1Score") or 0], "winner": p.get("WinningTeam"),
        "seconds": p.get("TotalSecondsPlayed"), "frames": p.get("NumFrames"), "fps": p.get("RecordFPS"),
        "recorded_by": p.get("PlayerName"),
        "goals": [{"frame": g.get("frame"), "player": g.get("PlayerName"), "team": g.get("PlayerTeam")} for g in p.get("Goals") or []],
        "players": players,
        # Retention rule: custom if the file name isn't RL's auto name OR a name was typed in game.
        "custom_name": custom_title or None,
        "is_custom": bool(custom_title) or not AUTO_NAME.match(name),
    }
