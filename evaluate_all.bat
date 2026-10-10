@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Please run run.bat once first to install everything.
  pause
  exit /b 1
)
call ".venv\Scripts\activate.bat"
echo Evaluating the heuristic demo engine on the built-in sample...
python evaluate.py --engine heuristic --dataset sample --split all
echo.
pause
