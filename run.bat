@echo off
setlocal
cd /d "%~dp0"
title Argument Mining and Socratic Feedback

where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install Python 3.10, 3.11 or 3.12 from https://www.python.org/downloads/
  echo IMPORTANT: tick "Add python.exe to PATH" in the installer, then run this file again.
  start https://www.python.org/downloads/
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo [1/3] Creating virtual environment...
  python -m venv .venv
  if errorlevel 1 goto :error
)
call ".venv\Scripts\activate.bat"

if not exist ".venv\deps_installed.flag" (
  echo [2/3] Installing packages. This can take 5-15 minutes the first time...
  python -m pip install --upgrade pip
  if errorlevel 1 goto :error
  python -m pip install -r requirements.txt
  if errorlevel 1 goto :error
  echo Trying optional package bitsandbytes ^(4-bit loading, NVIDIA GPU only^)...
  python -m pip install bitsandbytes
  echo installed> ".venv\deps_installed.flag"
)

echo [3/3] Starting the app. Your browser will open. Close this window to stop it.
python -c "import utils; utils.ensure_dirs()"
python -m streamlit run app.py
goto :eof

:error
echo.
echo Something went wrong during setup. Please read the messages above.
pause
exit /b 1
