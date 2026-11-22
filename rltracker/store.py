"""SQLite storage + JSON settings."""
from __future__ import annotations

import json
import os
import sqlite3
import threading

from .tracked import TRACKED_STATS, keep_tracked, REPLAY_ONLY

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _data_dir() -> str:
    """The project's data folder, or RLTRACKER_DATA if set (used for tests)."""
    return os.environ.get("RLTRACKER_DATA") or os.path.join(ROOT, "data")


DATA_DIR = _data_dir()
DB_PATH = os.path.join(DATA_DIR, "rltracker.db")
CONFIG_PATH = os.path.join(DATA_DIR, "config.json")

os.makedirs(DATA_DIR, exist_ok=True)
_lock = threading.RLock()


def conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH, timeout=10)
    c.row_factory = sqlite3.Row
    return c


def init():
    with _lock, conn() as c:
        c.execute("PRAGMA journal_mode=WAL")
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT UNIQUE,
                played_at TEXT,
                data TEXT,            -- parsed JSON (possibly user-edited)
                edited INTEGER DEFAULT 0,
                hidden INTEGER DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS ix_matches_played ON matches(played_at);
            CREATE TABLE IF NOT EXISTS match_notes (   -- your own notes/tags, kept apart from the match data
                match_id INTEGER PRIMARY KEY,
                note TEXT,
                tags TEXT              -- JSON list
            );
            """
        )
        cols = {r["name"] for r in c.execute("PRAGMA table_info(matches)")}
        if "deleted_at" not in cols:
            c.execute("ALTER TABLE matches ADD COLUMN deleted_at TEXT")  # legacy column, kept for old databases
        if "freed_bytes" not in cols:
            c.execute("ALTER TABLE matches ADD COLUMN freed_bytes INTEGER")


# ---------------------------------------------------------------- size
def storage_stats() -> dict:
    with conn() as c:
        r = c.execute("SELECT COUNT(*) n, COALESCE(SUM(LENGTH(data)),0) data FROM matches").fetchone()
    db_bytes = sum(os.path.getsize(p) for p in (DB_PATH, DB_PATH + "-wal") if os.path.exists(p))
    return {"db_bytes": db_bytes, "matches": r["n"], "data_bytes": r["data"], "backup_dir": BACKUP_DIR}


BACKUP_DIR = os.path.join(DATA_DIR, "backups")


def backup(keep: int = 14):
    """Daily copy of the database: once replays are recycled it's the only copy."""
    import datetime as _dt
    import sqlite3 as _sq

    os.makedirs(BACKUP_DIR, exist_ok=True)
    target = os.path.join(BACKUP_DIR, f"rltracker-{_dt.date.today():%Y%m%d}.db")
    if os.path.exists(target):
        return
    with _lock, conn() as src, _sq.connect(target) as dst:
        src.backup(dst)
    import re as _re
    # only the dated daily copies are rotated; named copies ("rltracker-before-…") are always kept
    olds = sorted(f for f in os.listdir(BACKUP_DIR) if _re.fullmatch(r"rltracker-\d{8}\.db", f))
    for f in olds[:-keep]:
        os.remove(os.path.join(BACKUP_DIR, f))


def backup_named(label: str) -> str:
    """A one-off copy of the database before a big change (never rotated away)."""
    import datetime as _dt
    import sqlite3 as _sq
    os.makedirs(BACKUP_DIR, exist_ok=True)
    safe = "".join(ch if ch.isalnum() or ch in "-._" else "-" for ch in label)[:60]
    target = os.path.join(BACKUP_DIR, f"rltracker-{safe}-{_dt.datetime.now():%Y%m%d-%H%M}.db")
    with _lock, conn() as src, _sq.connect(target) as dst:
        src.backup(dst)
    return target


# ---------------------------------------------------------------- settings
DEFAULT_CONFIG = {
    "player_names": [],     # your in-game name(s); empty = auto-detect
    "aliases": [],          # [{"name": in-game name, "alias": what you call them}]; alias "me" = you
    "poll_seconds": 5,
    "session_cutoff_min": 45,     # a gap longer than this between games starts a new session
}


def load_config() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, encoding="utf-8-sig") as f:  # tolerate a BOM (Notepad / PowerShell)
                cfg.update(json.load(f))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            # Settings exist but can't be read: never recycle replays on a guess.
            cfg["replay_retention"] = False
            cfg["config_error"] = True
    return cfg


def save_config(cfg: dict):
    with _lock, open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


# ---------------------------------------------------------------- matches
def _dump(d: dict) -> str:
    """Compact JSON of just the numbers and names."""
    d = json.loads(json.dumps(d))
    d.pop("note", None)
    d.pop("tags", None)
    for t in d.get("teams", []):
        for pl in t.get("players", []):
            pl.pop("raw", None)
            pl["stats"] = {k: v for k, v in pl.get("stats", {}).items() if k in TRACKED_STATS}
            pl.pop("mmr", None)          # MMR isn't tracked (user's choice)
            pl.pop("mmr_delta", None)
            if not d.get("replay"):      # ball hits / demos only exist when a replay provided them
                for k in REPLAY_ONLY:
                    pl["stats"].pop(k, None)
    if not d.get("replay"):
        d["columns"] = [c for c in d.get("columns", []) if c.get("key") not in REPLAY_ONLY]
    d["columns"] = keep_tracked(d.get("columns", []))
    for t in d.get("teams", []):
        t.pop("cols", None)
    return json.dumps(d, separators=(",", ":"))


def _strip_ignored(d: dict) -> dict:
    """Keep only what's tracked (see tracked.py), whatever was stored:
    no MMR, and ball hits only when they came from a replay."""
    if not d.get("replay"):
        d["columns"] = [c for c in d.get("columns", []) if c.get("key") not in REPLAY_ONLY]
    d["columns"] = keep_tracked(d.get("columns", []))
    for t in d.get("teams", []):
        for p in t.get("players", []):
            p["stats"] = {k: v for k, v in p.get("stats", {}).items() if k in TRACKED_STATS}
            if not d.get("replay"):
                for k in REPLAY_ONLY:
                    p["stats"].pop(k, None)
            p.pop("raw", None)
            p.pop("mmr", None)
            p.pop("mmr_delta", None)
    return d


MIN_GAME_S = 90  # games that lasted 1:30 or less (early forfeits) are left out of everything


def short_match_ids(c) -> set[int]:
    """Matches whose replay says the game lasted MIN_GAME_S or less."""
    try:
        rows = c.execute("SELECT match_id, MAX(json_extract(header, '$.seconds')) s FROM replay_files "
                         "WHERE match_id IS NOT NULL GROUP BY match_id").fetchall()
    except sqlite3.OperationalError:
        return set()
    return {r["match_id"] for r in rows if r["s"] is not None and r["s"] <= MIN_GAME_S}


def all_matches(include_hidden=True) -> list[dict]:
    with conn() as c:
        short = short_match_ids(c)
        q = "SELECT * FROM matches" + ("" if include_hidden else " WHERE hidden=0") + " ORDER BY played_at"
        notes = _notes(c)
        out = []
        for r in c.execute(q):
            if r["id"] in short:
                continue
            d = _strip_ignored(json.loads(r["data"]))
            d.update(id=r["id"], edited=bool(r["edited"]), hidden=bool(r["hidden"]), deleted_at=r["deleted_at"])
            d.update(notes.get(r["id"], {"note": "", "tags": []}))
            out.append(d)
        return out


def matches_signature() -> tuple:
    """Changes whenever matches are added, removed, hidden or rewritten — without loading them."""
    with conn() as c:
        r = c.execute("SELECT count(*), max(id), total(hidden), total(length(data)), group_concat(length(data) * id % 9973) "
                      "FROM matches").fetchone()
        return tuple(r[:4]) + (hash(r[4]),)


def _notes(c, mid: int | None = None) -> dict[int, dict]:
    try:
        q = "SELECT * FROM match_notes" + (" WHERE match_id=?" if mid is not None else "")
        rows = c.execute(q, (mid,) if mid is not None else ()).fetchall()
    except sqlite3.OperationalError:
        return {}
    return {r["match_id"]: {"note": r["note"] or "", "tags": json.loads(r["tags"] or "[]")} for r in rows}


def clean_tag(t: str) -> str:
    return " ".join(str(t).strip().lower().split())[:30]


def set_notes(mid: int, note: str | None = None, tags: list[str] | None = None, add: list[str] | None = None,
              remove: list[str] | None = None):
    """Your note and tags for a match (None = leave as is)."""
    with _lock, conn() as c:
        cur = _notes(c, mid).get(mid, {"note": "", "tags": []})
        n = cur["note"] if note is None else str(note)[:2000]
        ts = cur["tags"] if tags is None else tags
        ts = list(ts) + list(add or [])
        rm = {clean_tag(x) for x in remove or []}
        out = []
        for t in ts:
            t = clean_tag(t)
            if t and t not in out and t not in rm:
                out.append(t)
        c.execute("INSERT INTO match_notes(match_id, note, tags) VALUES (?,?,?) "
                  "ON CONFLICT(match_id) DO UPDATE SET note=excluded.note, tags=excluded.tags",
                  (mid, n, json.dumps(out[:12])))
        return {"note": n, "tags": out[:12]}


def all_tags() -> list[dict]:
    with conn() as c:
        cnt: dict[str, int] = {}
        for v in _notes(c).values():
            for t in v["tags"]:
                cnt[t] = cnt.get(t, 0) + 1
    return sorted(({"tag": t, "games": n} for t, n in cnt.items()), key=lambda e: (-e["games"], e["tag"]))


def get_match(mid: int) -> dict | None:
    with conn() as c:
        r = c.execute("SELECT * FROM matches WHERE id=?", (mid,)).fetchone()
        if not r:
            return None
        d = _strip_ignored(json.loads(r["data"]))
        d.update(id=r["id"], edited=bool(r["edited"]), hidden=bool(r["hidden"]), deleted_at=r["deleted_at"])
        d.update(_notes(c, mid).get(mid, {"note": "", "tags": []}))
        return d


def set_hidden(mid: int, hidden: bool):
    with _lock, conn() as c:
        c.execute("UPDATE matches SET hidden=? WHERE id=?", (1 if hidden else 0, mid))


