@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set HF_HUB_OFFLINE=1
set PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

if not exist ".venv\Scripts\anime-trivia.exe" (
  echo Anime Trivia is not installed. Run scripts\install_windows.ps1 first.
  pause
  exit /b 1
)

if not exist "config.json" (
  echo config.json is missing. Calibration has not been completed.
  pause
  exit /b 1
)

echo DRY RUN: answers will be logged but no keys will be sent.
echo Press F12 to stop.
echo.
".venv\Scripts\anime-trivia.exe" --config "config.json" --dry-run
echo.
echo Dry run stopped with exit code %errorlevel%.
pause
