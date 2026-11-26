# RL Tracker

Turns the replays Rocket League saves into stats, report cards and session reports, shown in your browser.
Everything runs and is stored on your own PC.

## Install (Windows, about 1 minute)

Nothing else to install: the download includes its own Python.

1. **Download RL Tracker:** https://github.com/DeChemicah/rltracker/releases/latest/download/RLTracker.zip (about 27 MB)
2. Right-click the zip › **Extract All…**, and put the `RLTracker` folder somewhere you'll keep it (e.g. your Documents).
3. Open the folder and double-click **`start.bat`**. Your browser opens the app at http://127.0.0.1:5177.
   If Windows shows "Windows protected your PC", click **More info › Run anyway**.

Next time, just double-click `start.bat` again. Updates install from inside the app (Settings › Updates).

More detail, troubleshooting, or installing with an AI assistant such as Claude Code: see [INSTALL.md](INSTALL.md).

## What you get

| Page | What's there |
|---|---|
| Home | Your current or last session, the last game's grade, key numbers, form, and stats you pinned |
| Stats | Every stat in one table (you, your average teammate, the lobby, percentile, trend, best game), plus topic tabs: performance, boost & movement, positioning, wins & clutch, trends |
| Games | Every game; open one for the scoreboard, every stat, timeline, heatmaps, boost pad map and shot map |
| Sessions | Games grouped into sessions with report cards, a shot map and a **Session report** (what to work on, for the team and each player) |
| People | Teammates and lineups, and compare up to three players |
| Records | Single-game bests and streaks, all time or per session |
| Settings | Your name, aliases, who appears in lists, replays, updates, export |

Press **Ctrl+K** anywhere to search for a stat, page or player.

## Safety

RL Tracker never touches the game: it only reads replay files Rocket League already saved, and never its logs.
Replays of games played before you installed it are never read or touched. Replay clean-up (keep only the newest 10,
moved to the Recycle Bin after their data is saved) is off until you turn it on.

## Updates

When a new version is out, a badge appears at the top of the app. **Settings › Updates › Update now** downloads it,
backs up your data and the current version, and restarts the app. Your data is never touched.

## Credits

Replays are read with [rrrocket](https://github.com/nickbabcock/rrrocket) (MIT licence), bundled in `tools/rrrocket`.
