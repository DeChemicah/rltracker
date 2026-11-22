# RL Tracker installer: downloads the latest release from GitHub, unpacks it, adds a desktop shortcut, starts it.
# Usage:  powershell -NoProfile -ExecutionPolicy Bypass -File install.ps1 -Repo owner/name
# Safe to read: it only writes to the install folder, the temp folder and one desktop shortcut.
# It never touches Rocket League, its replays or its logs.
param(
    [string]$Repo = "DeChemicah/rltracker",
    [string]$Dest = (Join-Path $env:USERPROFILE "RLTracker")
)
$ErrorActionPreference = "Stop"
$ua = @{ "User-Agent" = "RL-Tracker-installer" }

$placeholder = "__" + "REPO" + "__"   # written in two parts so publishing doesn't replace it
if ($Repo -eq $placeholder -or $Repo -notmatch "^[\w.-]+/[\w.-]+$") {
    throw "Tell the installer which GitHub repository to use, e.g.  -Repo yourname/rltracker"
}

# 1. Python 3.10+
$ok = $false
try { $ok = (& python -c "import sys; print(sys.version_info >= (3, 10))" 2>$null) -eq "True" } catch {}
if (-not $ok) {
    Write-Host "Python 3.10 or newer is needed." -ForegroundColor Yellow
    Write-Host "Install it from https://www.python.org/downloads/ (tick 'Add python.exe to PATH'),"
    Write-Host "or run:  winget install -e --id Python.Python.3.12"
    Write-Host "Then open a new PowerShell window and run this installer again."
    exit 1
}

# 2. Already installed?
if (Test-Path (Join-Path $Dest "rltracker\version.py")) {
    Write-Host "RL Tracker is already installed in $Dest." -ForegroundColor Yellow
    Write-Host "To update it, open the app and use Settings > Updates."
    exit 0
}

# 3. Download the latest release
Write-Host "Finding the latest release of $Repo ..."
$rel = Invoke-RestMethod "https://api.github.com/repos/$Repo/releases/latest" -Headers $ua
$zip = Join-Path $env:TEMP "rltracker-$($rel.tag_name).zip"
Write-Host "Downloading $($rel.tag_name) ..."
Invoke-WebRequest $rel.zipball_url -OutFile $zip -Headers $ua

# 4. Unpack into the install folder
$tmp = Join-Path $env:TEMP ("rltracker-" + [guid]::NewGuid())
Expand-Archive $zip $tmp
$inner = Get-ChildItem $tmp -Directory | Select-Object -First 1
if (-not (Test-Path (Join-Path $inner.FullName "rltracker\version.py"))) { throw "The download doesn't look like RL Tracker." }
New-Item -ItemType Directory -Force $Dest | Out-Null
Copy-Item (Join-Path $inner.FullName "*") $Dest -Recurse -Force
Get-ChildItem $Dest -Recurse -File | Unblock-File
Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item $zip -Force -ErrorAction SilentlyContinue

# 5. Desktop shortcut
$desktop = [Environment]::GetFolderPath("Desktop")
$lnk = (New-Object -ComObject WScript.Shell).CreateShortcut((Join-Path $desktop "RL Tracker.lnk"))
$lnk.TargetPath = Join-Path $Dest "start.bat"
$lnk.WorkingDirectory = $Dest
$lnk.Description = "Start RL Tracker"
$lnk.Save()

# 6. Start it
Write-Host "Installed RL Tracker $($rel.tag_name) in $Dest" -ForegroundColor Green
Write-Host "Starting it now. The first start sets up its Python packages (1-2 minutes), then opens http://127.0.0.1:5177"
Start-Process (Join-Path $Dest "start.bat") -WorkingDirectory $Dest
