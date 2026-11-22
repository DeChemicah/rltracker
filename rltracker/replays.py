"""Rocket League replays: watch, extract, store as matches, retain.

Anti-cheat safety (hard rules, see docs/REPLAY_PLAN.md):
- Only *reads* files the game has already saved. Never touches the game process
  (no process checks of any kind), its install folder, or its settings.
- Files in the Demos folder are only read once >=10 s old, and only ever
  *recycled* (never edited/written), and only when the newest file there is
  >=30 min old, so nothing changes while a play session may be going on.

Retention (user's rule, 2026-11-21):
- Never recycle the replays listed in docs/protected_replays.txt (existed on
  2026-11-21), custom-named replays (file name not RL's auto name, or a name
  typed in game), or replays marked "Keep replay".
- Otherwise keep the newest 10. Older ones go to the Recycle Bin only after
  their data is extracted and the saved match data reads back.
"""
from __future__ import annotations

import json
import os
import traceback
from datetime import datetime, timezone

from . import replay_extract, replay_io, replay_stats, store

KEEP_NEWEST = 10
MIN_AGE_S = 10                  # file must be this old before it's read
QUIET_MIN = 30                  # Demos must be untouched this long before recycling
PROTECTED_FILE = os.path.join(store.DATA_DIR, "protected_replays.txt")  # optional list of replay names never to touch
REPLAY_STATS = ["score", "goals", "assists", "saves", "shots"]  # exact from the replay header
TEAM_COLOR = {0: "blue", 1: "orange"}


# ---------------------------------------------------------------- setup
def init():
    with store._lock, store.conn() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS replay_files (
                path TEXT PRIMARY KEY, mtime REAL, status TEXT, error TEXT, guid TEXT,
                replay_date TEXT, is_custom INTEGER, custom_name TEXT, protected INTEGER,
                match_id INTEGER, header TEXT, track TEXT, recycled_at TEXT,
                processed_at TEXT DEFAULT CURRENT_TIMESTAMP);
            CREATE INDEX IF NOT EXISTS ix_replay_files_match ON replay_files(match_id);
            CREATE TABLE IF NOT EXISTS replay_keep (guid TEXT PRIMARY KEY);
            """
        )
    # First run on this PC: replays of games played before now are never imported or deleted
    # (the app only works with games played after it was installed).
    cfg = store.load_config()
    if not cfg.get("protect_before") and not cfg.get("config_error"):
        cfg["protect_before"] = datetime.now().replace(microsecond=0).isoformat()
        store.save_config(cfg)


def demos_folders() -> list[str]:
    """Rocket League's replay folders in your (possibly OneDrive-redirected) Documents:
    Demos (Steam) and DemosEpic (Epic Games). Each is only included if it exists."""
    cands = []
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                            r"Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders") as k:
            cands.append(os.path.expandvars(winreg.QueryValueEx(k, "Personal")[0]))
    except OSError:
        pass
    home = os.path.expanduser("~")
    cands += [os.path.join(home, "OneDrive", "Documents"), os.path.join(home, "Documents")]
    for docs in cands:
        base = os.path.join(docs, "My Games", "Rocket League", "TAGame")
        found = [os.path.normpath(os.path.join(base, n)) for n in ("Demos", "DemosEpic")
                 if os.path.isdir(os.path.join(base, n))]
        if found:
            return found
    return []


def demos_folder() -> str | None:
    f = demos_folders()
    return f[0] if f else None


def replay_folders(cfg: dict | None = None) -> list[str]:
    cfg = cfg or store.load_config()
    out = []
    if cfg.get("replays_auto", True):
        out += demos_folders()
    for f in cfg.get("replay_folders") or []:
        if f and os.path.normpath(f) not in out:
            out.append(os.path.normpath(f))
    return out


# Replays of games played before the app was installed (config "protect_before", set on first run)
# are never imported or deleted, even if the file only shows up later (e.g. Steam Cloud).
def protect_before() -> datetime:
    try:
        return datetime.fromisoformat(store.load_config().get("protect_before") or "")
    except ValueError:
        return datetime.now()  # unreadable: protect everything up to now


def is_old_game(replay_date: str | None) -> bool:
    d = _parse_date(replay_date)
    return d is not None and d < protect_before()


def protected_names() -> set[str]:
    names = set()
    for src in (PROTECTED_FILE,):
        try:
            with open(src, encoding="utf-8") as f:
                for line in f:
                    if line.strip() and not line.startswith("#"):
                        names.add(line.split("\t")[0].strip().lower())
        except OSError:
            pass
    return names


# ---------------------------------------------------------------- pairing / overriding
def _parse_date(s: str | None) -> datetime | None:
    try:
        return datetime.strptime(s, "%Y-%m-%d %H-%M-%S")
    except (TypeError, ValueError):
        return None


def fix_date(hdr: dict, file_mtime: float | None) -> dict:
    """Replays saved by the game carry local time. Some (e.g. downloaded from Match
    History, no "recorded by") carry UTC instead. A game can't end after its replay
    file was written, so a date later than the file's own time is read as UTC and
    converted to local time (or, if that still doesn't fit, the file time is used)."""
    d = _parse_date(hdr.get("date"))
    if d is None or not file_mtime or d.timestamp() <= file_mtime + 300:
        return hdr
    local = d.replace(tzinfo=timezone.utc).astimezone().replace(tzinfo=None)
    if not (file_mtime - 6 * 3600 <= local.timestamp() <= file_mtime + 300):
        local = datetime.fromtimestamp(file_mtime)
    hdr["date_raw"] = hdr["date"]
    hdr["date"] = local.strftime("%Y-%m-%d %H-%M-%S")
    return hdr


def match_from_replay(hdr: dict, file_info: dict) -> dict | None:
    """A match built from a replay: exact stats."""
    if not hdr.get("players"):
        return None  # very old replays have no scoreboard in the header
    when = _parse_date(hdr.get("date"))
    teams = []
    for idx in (0, 1):
        ps = [{"name": rp["name"], "platform_id": rp["id"], "replay_name": rp["name"],
               "stats": {k: rp["stats"].get(k) for k in REPLAY_STATS}, **({"bot": True} if rp.get("bot") else {})}
              for rp in hdr["players"] if rp.get("team") == idx]
        teams.append({"name": TEAM_COLOR[idx].upper(), "color": TEAM_COLOR[idx], "score": hdr["score"][idx],
                      "winner": hdr.get("winner") == idx, "players": ps})
    return {
        "source": "replay", "path": "replay:" + (hdr.get("guid") or file_info["file"]),
        "played_at": (when or datetime.now()).isoformat(), "is_scoreboard": True, "is_final": True,
        "columns": [{"key": k, "label": k.upper()} for k in REPLAY_STATS], "teams": teams,
        "team_size": hdr.get("team_size") or max(len(t["players"]) for t in teams),
        "ranked": None, "local_player": hdr.get("recorded_by"), "replay": file_info,
    }


def _file_info(path: str, hdr: dict) -> dict:
    return {"file": os.path.basename(path), "guid": hdr.get("guid"), "custom_name": hdr.get("custom_name"),
            "date": hdr.get("date"), "map": hdr.get("map"), "match_type": hdr.get("match_type")}


def _save_data(mid: int, data: dict):
    keep = {k: v for k, v in data.items() if k not in ("id", "edited", "hidden", "deleted_at")}
    with store._lock, store.conn() as c:
        c.execute("UPDATE matches SET data=? WHERE id=?", (store._dump(keep), mid))


def _insert_match(data: dict) -> int:
    with store._lock, store.conn() as c:
        cur = c.execute("INSERT INTO matches(path, played_at, data) VALUES (?,?,?)",
                        (data["path"], data["played_at"], store._dump(data)))
        return cur.lastrowid


def _remember_my_id(data: dict):
    """The replay's recorder is you: remember that account id (finds you under any name)."""
    rec = data.get("local_player") or ""
    cfg = store.load_config()
    ids = set(cfg.get("my_ids") or [])
    for t in data.get("teams", []):
        for p in t.get("players", []):
            if p.get("platform_id") and p.get("replay_name") == rec and p["platform_id"] not in ids:
                ids.add(p["platform_id"])
                cfg["my_ids"] = sorted(ids)
                store.save_config(cfg)
                return


def _track_path(mid: int) -> str:
    return os.path.join(replay_extract.TRACKS_DIR, f"{mid}.npz")


# ---------------------------------------------------------------- processing
def process_file(path: str, mtime: float, cfg: dict) -> str:
    """Read one replay: header -> pair or create match -> full track file."""
    hdr = fix_date(replay_io.header(path), mtime)
    name = os.path.basename(path)
    prot = name.lower() in protected_names() or is_old_game(hdr.get("date"))
    info = _file_info(path, hdr)
    row = {"path": path, "mtime": mtime, "guid": hdr.get("guid"), "replay_date": hdr.get("date"),
           "is_custom": int(hdr["is_custom"]), "custom_name": hdr.get("custom_name"), "protected": int(prot),
           "header": json.dumps(hdr)}
    if prot:
        _record(row, status="protected_skipped")  # user's decision: replays from before 21 Nov 2026 are never imported
        return "protected_skipped"
    # same replay already linked (e.g. re-read)?
    with store.conn() as c:
        prev = c.execute("SELECT match_id FROM replay_files WHERE guid=? AND match_id IS NOT NULL", (hdr.get("guid"),)).fetchone()
    mid = prev["match_id"] if prev else None
    if mid is None:
        data = match_from_replay(hdr, info)
        if not data:
            _record(row, status="no_scoreboard")
            return "no_scoreboard"
        mid = _insert_match(data)
        _remember_my_id(store.get_match(mid) or {})
    meta, arrays, events = replay_extract.extract(path)
    track = replay_extract.save(str(mid), meta, arrays, events)
    row.update(match_id=mid, track=track)
    _record(row, status="extracted")
    _attach_stats(mid, track)
    return "extracted"


def _attach_stats(mid: int, track: str):
    """Phase-2 stats (boost/movement/demos) stored on the match's players."""
    try:
        m = store.get_match(mid)
        if m and track and os.path.exists(track):
            _save_data(mid, replay_stats.attach(m, track))
    except Exception:
        traceback.print_exc()


def backfill_assists():
    """Games stored before assists were tracked: take them from the saved replay header (exact)."""
    with store.conn() as c:
        rows = [dict(r) for r in c.execute(
            "SELECT f.match_id, f.header FROM replay_files f JOIN matches m ON m.id = f.match_id "
            "WHERE f.header IS NOT NULL AND m.data NOT LIKE '%\"assists\"%'")]
    for r in rows:
        try:
            hdr = json.loads(r["header"] or "{}")
        except ValueError:
            continue
        m = store.get_match(r["match_id"])
        if not m or not hdr.get("players"):
            continue
        by_id = {p.get("id"): p for p in hdr["players"] if p.get("id")}
        by_name = {p.get("name"): p for p in hdr["players"]}
        changed = False
        for t in m.get("teams", []):
            for p in t.get("players", []):
                hp = by_id.get(p.get("platform_id")) or by_name.get(p.get("replay_name") or p.get("name"))
                a = ((hp or {}).get("stats") or {}).get("assists")
                if a is not None and p.setdefault("stats", {}).get("assists") is None:
                    p["stats"]["assists"] = a
                    changed = True
        if changed:
            if not any(c_.get("key") == "assists" for c_ in m.get("columns", [])):
                m.setdefault("columns", []).append({"key": "assists", "label": "ASSISTS"})
            _save_data(r["match_id"], m)


def refresh_stats():
    """Recompute stats for matches whose stored version is out of date."""
    backfill_assists()
    with store.conn() as c:
        rows = [dict(r) for r in c.execute("SELECT match_id, track FROM replay_files WHERE status='extracted' AND match_id IS NOT NULL")]
    for r in rows:
        m = store.get_match(r["match_id"])
        if m and m.get("rstats_version") != replay_stats.STATS_VERSION:
            _attach_stats(r["match_id"], r["track"])


def _record(row: dict, status: str, error: str | None = None):
    with store._lock, store.conn() as c:
        c.execute("INSERT OR REPLACE INTO replay_files(path, mtime, status, error, guid, replay_date, is_custom, custom_name, "
                  "protected, match_id, header, track, processed_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)",
                  (row["path"], row.get("mtime"), status, error, row.get("guid"), row.get("replay_date"), row.get("is_custom"),
                   row.get("custom_name"), row.get("protected"), row.get("match_id"), row.get("header"), row.get("track")))


def scan(state: dict | None = None) -> dict:
    """Read new replays, then apply retention. Returns a small report."""
    cfg = store.load_config()
    report = {"read": 0, "skipped": 0, "errors": 0, "recycled": 0, "kept": []}
    if not replay_io.exe_path():
        return report
    with store.conn() as c:
        known = {r["path"]: r["mtime"] for r in c.execute("SELECT path, mtime FROM replay_files")}
    now = datetime.now().timestamp()
    for folder in replay_folders(cfg):
        try:
            files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".replay")]
        except OSError:
            continue
        for p in sorted(files, key=lambda x: os.path.getmtime(x)):
            try:
                mt = os.path.getmtime(p)
            except OSError:
                continue
            if now - mt < MIN_AGE_S or known.get(p) == mt:
                continue
            if state is not None:
                state["current"] = os.path.basename(p)
            try:
                res = process_file(p, mt, cfg)
                report["read" if res == "extracted" else "skipped"] += 1
                if res == "extracted" and state is not None:
                    state["last_new"] = now
            except Exception as e:
                traceback.print_exc()
                _record({"path": p, "mtime": mt}, status="error", error=f"{type(e).__name__}: {e}")
                report["errors"] += 1
    refresh_stats()
    report.update(retention(cfg))
    return report


# ---------------------------------------------------------------- retention
def retention(cfg: dict | None = None, dry_run: bool = False) -> dict:
    cfg = cfg or store.load_config()
    out = {"recycled": 0, "kept": [], "would_recycle": []}
    if not cfg.get("replay_retention", False):  # off until turned on in Settings
        return out
    prot = protected_names()
    with store.conn() as c:
        rows = [dict(r) for r in c.execute("SELECT * FROM replay_files")]
        keep_guids = {r["guid"] for r in c.execute("SELECT guid FROM replay_keep")}
    files = []
    for folder in replay_folders(cfg):  # Steam + Epic folders share one "newest 10"
        try:
            files += [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".replay")]
        except OSError:
            continue
    for _once in (1,):
        if not files:
            break
        newest = max(os.path.getmtime(f) for f in files)
        if datetime.now().timestamp() - newest < QUIET_MIN * 60:
            out["kept"].append({"file": "*", "reason": f"a replay was saved in the last {QUIET_MIN} min — waiting"})
            break
        by_path = {r["path"]: r for r in rows}
        pool = []
        for f in files:
            name = os.path.basename(f)
            r = by_path.get(f)
            if name.lower() in prot or (r and (r.get("protected") or is_old_game(r.get("replay_date")))):
                continue  # existed on 2026-11-21, or a game from before then: never touched
            if not replay_io.AUTO_NAME.match(name) or (r and r.get("is_custom")):
                continue  # custom-named: never touched
            if r and r.get("guid") in keep_guids:
                continue
            pool.append((f, r))
        pool.sort(key=lambda x: ((x[1] or {}).get("replay_date") or "", os.path.getmtime(x[0])), reverse=True)
        for f, r in pool[KEEP_NEWEST:]:
            reason = None
            if not r or r.get("status") != "extracted":
                reason = "not read yet"
            elif not r.get("track") or not os.path.exists(r["track"]):
                reason = "track file missing"
            else:
                try:
                    replay_extract.load_track(r["track"])
                except Exception:
                    reason = "track file unreadable"
            if not reason and r.get("match_id") and not store.get_match(r["match_id"]):
                reason = "its match isn't in the database"
            if reason:
                out["kept"].append({"file": os.path.basename(f), "reason": reason})
                continue
            if dry_run:
                out["would_recycle"].append(os.path.basename(f))
                continue
            try:
                from send2trash import send2trash
                send2trash(f)
                with store._lock, store.conn() as c:
                    c.execute("UPDATE replay_files SET recycled_at=CURRENT_TIMESTAMP WHERE path=?", (f,))
                out["recycled"] += 1
            except Exception as e:
                out["kept"].append({"file": os.path.basename(f), "reason": f"couldn't recycle: {e}"})
    return out


def set_keep(mid: int, keep: bool) -> bool:
    with store._lock, store.conn() as c:
        r = c.execute("SELECT guid FROM replay_files WHERE match_id=?", (mid,)).fetchone()
        if not r or not r["guid"]:
            return False
        if keep:
            c.execute("INSERT OR IGNORE INTO replay_keep(guid) VALUES (?)", (r["guid"],))
        else:
            c.execute("DELETE FROM replay_keep WHERE guid=?", (r["guid"],))
    return True


def match_replay_info(mid: int) -> dict | None:
    with store.conn() as c:
        r = c.execute("SELECT path, guid, custom_name, is_custom, protected, recycled_at, track FROM replay_files WHERE match_id=?", (mid,)).fetchone()
        if not r:
            return None
        keep = bool(c.execute("SELECT 1 FROM replay_keep WHERE guid=?", (r["guid"],)).fetchone())
    return {"file": os.path.basename(r["path"]), "custom_name": r["custom_name"], "is_custom": bool(r["is_custom"]),
            "protected": bool(r["protected"]), "recycled": bool(r["recycled_at"]), "on_disk": os.path.exists(r["path"]),
            "keep": keep, "track": bool(r["track"] and os.path.exists(r["track"])),
            "track_kb": round(os.path.getsize(r["track"]) / 1024) if r["track"] and os.path.exists(r["track"]) else None}


def summary() -> dict:
    cfg = store.load_config()
    with store.conn() as c:
        st = {r["status"]: r["n"] for r in c.execute("SELECT status, COUNT(*) n FROM replay_files GROUP BY status")}
        errs = [dict(r) for r in c.execute("SELECT path, error FROM replay_files WHERE status='error' LIMIT 10")]
        kept = c.execute("SELECT COUNT(*) FROM replay_keep").fetchone()[0]
    tracks = 0
    if os.path.isdir(replay_extract.TRACKS_DIR):
        tracks = sum(os.path.getsize(os.path.join(replay_extract.TRACKS_DIR, f)) for f in os.listdir(replay_extract.TRACKS_DIR))
    files = []
    for folder in replay_folders(cfg):
        try:
            files += [f for f in os.listdir(folder) if f.lower().endswith(".replay")]
        except OSError:
            pass
    prot = protected_names()
    return {"folders": replay_folders(cfg), "parser": bool(replay_io.exe_path()), "status": st, "errors": errs,
            "on_disk": len(files), "protected_on_disk": sum(f.lower() in prot for f in files),
            "kept": kept, "tracks_mb": round(tracks / 1e6, 1), "keep_newest": KEEP_NEWEST,
            "retention_on": bool(cfg.get("replay_retention", False)),
            "preview": retention(cfg, dry_run=True)}
