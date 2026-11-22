"""In-app updates from GitHub releases.

Only ever started by the person clicking "Update" in Settings. It downloads the latest release
of the configured repository, backs up the database and the current code, replaces the app's
code files and restarts the app. Never touches data/ (database, settings, backups) or .venv/,
and never touches Rocket League.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import zipfile

from . import store
from .version import UPDATE_REPO, VERSION

UA = "RL-Tracker-updater"
API = "https://api.github.com/repos/{repo}/releases/latest"
MAX_ZIP = 60 * 1024 * 1024
# what an update may replace (relative to the app folder); everything else is left alone
ALLOWED = ("rltracker/", "tools/", "docs/", "start.bat", "install.ps1", "requirements.txt", "README.md", "INSTALL.md",
           "CLAUDE.md", "LICENSE", ".gitignore")
NEVER = ("data/", ".venv/", ".git/")
UPDATES_DIR = os.path.join(store.DATA_DIR, "updates")
_cache: dict = {}


def repo() -> str:
    return (store.load_config().get("update_repo") or UPDATE_REPO or "").strip().strip("/")


def parse(v: str) -> tuple:
    v = (v or "").strip().lstrip("vV")
    out = []
    for part in v.split("."):
        num = "".join(ch for ch in part if ch.isdigit())
        out.append(int(num) if num else 0)
    return tuple(out + [0] * (3 - len(out)))


def _get(url: str, limit: int = 2_000_000, timeout: int = 20) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read(limit + 1)
    if len(data) > limit:
        raise ValueError("download is larger than expected")
    return data


def check(force: bool = False) -> dict:
    """{'configured', 'current', 'latest', 'available', 'notes', 'url', 'published'} (cached for an hour)."""
    rp = repo()
    base = {"configured": bool(rp), "repo": rp, "current": VERSION}
    if not rp:
        return base
    if not force and _cache.get("repo") == rp and time.time() - _cache.get("at", 0) < 3600:
        return {**base, **_cache["info"]}
    try:
        rel = json.loads(_get(API.format(repo=rp)))
        tag = rel.get("tag_name") or ""
        info = {"latest": tag.lstrip("vV"), "available": parse(tag) > parse(VERSION), "notes": rel.get("body") or "",
                "url": rel.get("html_url"), "published": rel.get("published_at"), "zip": rel.get("zipball_url")}
    except Exception as e:  # offline, repo missing, rate limit…
        return {**base, "error": f"Couldn't check for updates ({type(e).__name__}). Check your internet connection and the update source."}
    _cache.update(repo=rp, at=time.time(), info=info)
    return {**base, **info}


def _members(zf: zipfile.ZipFile) -> tuple[str, list[tuple[zipfile.ZipInfo, str]]]:
    """The release's top folder and its (member, path inside the app) pairs, safe paths only."""
    root = next((n[: -len("rltracker/version.py")] for n in zf.namelist() if n.endswith("rltracker/version.py")), None)
    if root is None:
        raise ValueError("this release doesn't look like RL Tracker")
    out = []
    for info in zf.infolist():
        if info.is_dir() or not info.filename.startswith(root):
            continue
        rel = info.filename[len(root):].replace("\\", "/")
        if not rel or rel.startswith("/") or ".." in rel.split("/") or ":" in rel:
            continue
        if rel.startswith(NEVER) or "__pycache__/" in rel:
            continue
        if any(rel == a or (a.endswith("/") and rel.startswith(a)) for a in ALLOWED):
            out.append((info, rel))
    return root, out


def _release_version(zf: zipfile.ZipFile, root: str) -> str:
    txt = zf.read(root + "rltracker/version.py").decode("utf-8", "replace")
    for line in txt.splitlines():
        head, eq, val = line.partition("=")
        if eq and head.strip() == "VERSION":
            return val.strip().strip("\"'")
    return ""


def apply(zip_bytes: bytes | None = None, app_root: str | None = None) -> dict:
    """Install the latest release. zip_bytes/app_root are for testing on a throwaway copy."""
    app_root = app_root or store.ROOT
    if zip_bytes is None:
        info = check(force=True)
        if info.get("error"):
            return {"ok": False, "error": info["error"]}
        if not info.get("available"):
            return {"ok": False, "error": "You already have the latest version."}
        zip_bytes = _get(info["zip"], limit=MAX_ZIP, timeout=120)
    zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    root, members = _members(zf)
    new_v = _release_version(zf, root)
    if not new_v:
        return {"ok": False, "error": "The release has no version number."}
    os.makedirs(UPDATES_DIR, exist_ok=True)
    # 1. back up the database and the current code
    try:
        store.backup_named(f"before-update-to-{new_v}")
    except Exception:
        pass
    prev = os.path.join(UPDATES_DIR, f"previous-{VERSION}")
    if os.path.exists(prev):
        shutil.rmtree(prev, ignore_errors=True)
    os.makedirs(prev, exist_ok=True)
    touched = sorted({rel for _i, rel in members})
    for rel in touched:
        src = os.path.join(app_root, rel)
        if os.path.isfile(src):
            dst = os.path.join(prev, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
    old_req = _read(os.path.join(app_root, "requirements.txt"))
    # 2. write the new files; on any error put the old ones back
    try:
        for info, rel in members:
            dst = os.path.join(app_root, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with zf.open(info) as s, open(dst + ".new", "wb") as d:
                shutil.copyfileobj(s, d)
            os.replace(dst + ".new", dst)
    except Exception as e:
        for rel in touched:
            b = os.path.join(prev, rel)
            if os.path.isfile(b):
                shutil.copy2(b, os.path.join(app_root, rel))
        return {"ok": False, "error": f"Update failed and was rolled back ({type(e).__name__}: {e})."}
    if _read(os.path.join(app_root, "requirements.txt")) != old_req:
        open(os.path.join(UPDATES_DIR, "pip_needed"), "w").close()  # start.bat installs new packages
    return {"ok": True, "version": new_v, "files": len(members), "backup": prev}


def _read(p: str) -> str:
    try:
        with open(p, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


def restart_soon(delay: float = 1.0):
    """Start a fresh copy of the app (after this one has stopped) and exit."""
    bat = os.path.join(store.ROOT, "start.bat")

    def go():
        flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
        subprocess.Popen(f'cmd /c "timeout /t 2 /nobreak >nul & call "{bat}" --no-browser"', cwd=store.ROOT, creationflags=flags)
        os._exit(0)
    threading.Timer(delay, go).start()
