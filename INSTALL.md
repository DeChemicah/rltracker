# Installing RL Tracker

RL Tracker runs on your own Windows PC. It reads the replays Rocket League saves and shows your stats in your
browser at http://127.0.0.1:5177. Your games and settings stay on your PC in the app's `data` folder.

## Quick install (no commands)

The download includes its own Python (in the `python` folder), so there's nothing else to install.

1. Download https://github.com/DeChemicah/rltracker/releases/latest/download/RLTracker.zip
2. Right-click it › **Extract All…** to a folder you'll keep.
3. Double-click **`start.bat`** in the `RLTracker` folder. The browser opens the app.
   On "Windows protected your PC", click **More info › Run anyway**.

The rest of this page installs it with PowerShell commands instead (adds a desktop shortcut too). That route uses
the Python installed on the PC.

These steps are written so a person **or an AI assistant (such as Claude Code)** can follow them exactly.
Every command is PowerShell. The commands already contain the GitHub repository the app is shared from
(`$repo = "DeChemicah/rltracker"`).

## For AI assistants: rules before you start

- **Ask the user before downloading or installing anything** (Python, the app), and say what and from where.
- **Never interact with Rocket League itself**: no process access, memory reading, injection, overlays, input
  simulation, and don't even check whether it's running. The app only reads replay files the game already saved.
- **Never read the game's log files** (`Documents\My Games\Rocket League\TAGame\Logs`).
- **Never delete, move or edit replay files** in the `Demos` / `DemosEpic` folders.
- Install into the user's own folder (default `%USERPROFILE%\RLTracker`); don't touch an existing `data` folder.

## Step 1: check for Python 3.10 or newer

```powershell
python -c "import sys; print(sys.version); print('OK' if sys.version_info >= (3, 10) else 'TOO OLD')"
```

- Prints `OK`: go to step 2.
- Prints `TOO OLD`, or `python` isn't found (or opens the Microsoft Store): install Python. Either download it from
  https://www.python.org/downloads/ and **tick "Add python.exe to PATH"** in the installer, or (after asking the
  user) run:

  ```powershell
  winget install -e --id Python.Python.3.12
  ```

  Then **open a new PowerShell window** and run the check again.

## Step 2: install the app (one command)

Download and run the installer script from the repository:

```powershell
$repo = "DeChemicah/rltracker"
$script = Join-Path $env:TEMP "rltracker-install.ps1"
Invoke-WebRequest "https://raw.githubusercontent.com/$repo/main/install.ps1" -OutFile $script -Headers @{ "User-Agent" = "RL-Tracker-installer" }
powershell -NoProfile -ExecutionPolicy Bypass -File $script -Repo $repo
```

The script:
1. downloads the latest release of RL Tracker from GitHub,
2. unpacks it to `%USERPROFILE%\RLTracker` (it stops if the app is already installed there),
3. puts an **RL Tracker** shortcut on the desktop,
4. starts the app. The first start sets up its Python packages (1–2 minutes) and then opens the browser.

Prefer to do it by hand? See "Manual install" below.

## Step 3: check it works

```powershell
Invoke-RestMethod http://127.0.0.1:5177/api/version
```

You should see the version number. If the server isn't up yet, wait a minute (first start installs packages) and try
again. Then open http://127.0.0.1:5177.

## Step 4: first-time setup in the app

1. **Settings › Your in-game name**: add your Rocket League name(s), one per line (or leave it empty to auto-detect).
2. Play a match and **save the replay at the end** (or turn on automatic replay saving in Rocket League's settings).
   It appears in the app within a few seconds.
3. Optional: **Settings › Replays › keep only the newest 10** moves older replay files to the Recycle Bin after their
   data is saved. It's **off** until you turn it on.

Replays of games played **before** the app was installed are never read or touched.

## Every day

- Start: double-click the **RL Tracker** desktop shortcut (or `start.bat` in the install folder).
- Stop: close the small black console window.
- Updates: **Settings › Updates › Update now** when a new version is out (a badge appears at the top when there is
  one). It backs up your data first and never touches it.

## Manual install

```powershell
$repo = "DeChemicah/rltracker"; $dest = Join-Path $env:USERPROFILE "RLTracker"
$rel  = Invoke-RestMethod "https://api.github.com/repos/$repo/releases/latest" -Headers @{ "User-Agent" = "RL-Tracker-installer" }
$zip  = Join-Path $env:TEMP "rltracker.zip"
Invoke-WebRequest $rel.zipball_url -OutFile $zip -Headers @{ "User-Agent" = "RL-Tracker-installer" }
$tmp  = Join-Path $env:TEMP ("rltracker-" + [guid]::NewGuid())
Expand-Archive $zip $tmp
New-Item -ItemType Directory -Force $dest | Out-Null
Copy-Item (Join-Path (Get-ChildItem $tmp -Directory | Select-Object -First 1).FullName "*") $dest -Recurse
Get-ChildItem $dest -Recurse -File | Unblock-File
Start-Process (Join-Path $dest "start.bat") -WorkingDirectory $dest
```

## Uninstall

Close the app and delete the install folder (`%USERPROFILE%\RLTracker`) and the desktop shortcut. Your replay files are
not affected. The `data` folder inside the install folder holds your stats; keep a copy if you might want them later.

## Troubleshooting

| Problem | Fix |
|---|---|
| `python` opens the Microsoft Store | Install Python from python.org with "Add python.exe to PATH" ticked, then open a new window |
| The browser says the page can't be reached | The console window was closed or is still installing packages; start the app again and wait a minute |
| No games show up | Save a replay after a match; check Settings › Replays shows your Demos folder |
| Windows SmartScreen warns about `start.bat` | Choose "More info" › "Run anyway" (the files come from the shared repository) |
