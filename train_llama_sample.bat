@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Please run run.bat once first to install everything.
  pause
  exit /b 1
)
call ".venv\Scripts\activate.bat"
echo Small test fine-tuning of Llama 3.1 8B with LoRA (needs Hugging Face access + GPU or lots of RAM)...
python train.py --sample-mode %*
echo.
pause
