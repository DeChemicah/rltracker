# Publishing RL Tracker for friends

Your working copy is `D:\RLTRACKER` (private: it has your data, notes and history). Friends get a clean copy from a
GitHub repository, built from your working copy by `tools/publish.py` into `D:\RLTRACKER-public`.

## One-time setup

1. Create a **public** repository on GitHub, e.g. `yourname/rltracker` (empty: no README).
2. Put its name in `rltracker/version.py` › `UPDATE_REPO = "yourname/rltracker"` (in `D:\RLTRACKER`).
3. Build the public copy and connect it to GitHub:

   ```powershell
   D:\RLTRACKER\.venv\Scripts\python D:\RLTRACKER\tools\publish.py
   cd D:\RLTRACKER-public
   git remote add origin https://github.com/yourname/rltracker.git
   git push -u origin main
   ```

4. On GitHub: **Releases › Draft a new release**, tag `v1.0.0` (must match `VERSION`), add a few lines of notes,
   **Publish release**.
5. Send friends the repository link; its front page has the download link and steps.

## Every update

1. In `D:\RLTRACKER`, bump `VERSION` in `rltracker/version.py` (e.g. `1.0.1`).
2. `D:\RLTRACKER\.venv\Scripts\python D:\RLTRACKER\tools\publish.py` (copies the app and commits it in the public copy).
3. `cd D:\RLTRACKER-public` then `git push`.
4. On GitHub, publish a release with tag `v1.0.1` and notes on what changed. The notes are shown in the app.
   Attach `D:\RLTRACKER-release\RLTracker.zip` (built by publish.py) to the release: friends' download link
   `https://github.com/<you>/rltracker/releases/latest/download/RLTracker.zip` always gets the newest one.

Friends then see "Update to v1.0.1" at the top of their app.

## What is never published

Your `data` folder, `.venv`, `CLAUDE.md`/`docs/REPLAY_PLAN.md` (your project notes), `docs/protected_replays.txt`,
`.claude`, `backups`. `tools/publish.py` only copies the allow-list at its top.
