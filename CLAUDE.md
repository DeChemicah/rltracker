# RL Tracker

Local browser app that turns saved Rocket League replays (Steam `Demos` and Epic `DemosEpic` folders) into stats.
Run: `start.bat` → http://127.0.0.1:5177. Browser-only, Windows, Python 3.10+. Install steps: `INSTALL.md`.

## Rules (for any AI assistant working on this copy)

- **Anti-cheat: zero risk.** Never touch the Rocket League process in any way: no process access, memory reading,
  injection, overlays, input simulation, or even checking whether it's running. Only read files the game already saved
  (the Demos / DemosEpic folders). **Never read the game's log files** (`TAGame\Logs`).
- **Never hard-delete user files.** Replays only go to the Recycle Bin through the app's own retention setting, after
  their data is saved, and only when the user turned it on.
- Replays of games played before the app was installed (`data/config.json` › `protect_before`) are never imported or
  deleted.
- **The `data` folder is the user's**: database, settings, backups. Back up the database before changing data; never
  delete it. Test changes against a throwaway copy (`RLTRACKER_DATA=<temp dir>`), never the real `data`.
- Ask the user before downloading or installing anything.
- The server listens on 127.0.0.1 only. Keep it that way.

## Layout

- `rltracker/` the app (Flask + SQLite). `static/` the pages. `debrief.py` session reports, `insights.py` analysis,
  `replays.py` reading and retention, `updater.py` in-app updates, `version.py` version and update source.
- `tools/rrrocket/` the bundled replay parser (rrrocket, MIT licence, github.com/nickbabcock/rrrocket).
- `data/` (not in the repository) the user's database, settings and backups.
