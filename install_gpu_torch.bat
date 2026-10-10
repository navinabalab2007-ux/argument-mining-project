@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Please run run.bat once first.
  pause
  exit /b 1
)
call ".venv\Scripts\activate.bat"
echo Installing the NVIDIA CUDA build of PyTorch (only for computers with an NVIDIA GPU)...
python -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu121
echo.
pause
