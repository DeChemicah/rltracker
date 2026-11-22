@echo off
cd /d "%~dp0"
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
