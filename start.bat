@echo off
cd /d "%~dp0"
rem The download from GitHub ships its own Python in the "python" folder, so nothing needs installing.
if exist python\python.exe (
  if exist data\updates\pip_needed (
    rem after an update that changed the packages list, install the new packages
    python\python.exe -m pip install --disable-pip-version-check -r requirements.txt && del data\updates\pip_needed
  )
  python\python.exe -m rltracker %*
  exit /b
)
rem Otherwise (installed with git or install.ps1) use the Python installed on this PC.
if not exist .venv\Scripts\python.exe (
  echo First run: setting up Python environment...
  python -m venv .venv || (echo Python 3.10+ is required: https://www.python.org/downloads/ & pause & exit /b 1)
  .venv\Scripts\python -m pip install --upgrade pip
  .venv\Scripts\pip install -r requirements.txt || (pause & exit /b 1)
)
rem after an update that changed the packages list, install the new packages
if exist data\updates\pip_needed (
  .venv\Scripts\pip install -r requirements.txt && del data\updates\pip_needed
)
.venv\Scripts\python -m rltracker %*
