@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Please run run.bat once first to install everything.
  pause
  exit /b 1
)
call ".venv\Scripts\activate.bat"
echo Training the BERT baseline in small-sample mode...
python train.py --target bert --sample-mode %*
echo.
pause
