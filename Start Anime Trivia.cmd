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

set DIRTY=
where git >nul 2>&1 && (
  for /f "delims=" %%L in ('git status --porcelain src tests 2^>nul') do set DIRTY=1
)
if defined DIRTY (
  echo.
  echo WARNING: src/ or tests/ has UNCOMMITTED changes. The code about to run is not the tested,
  echo committed code. Commit or revert before a quiz. Press any key to launch anyway.
  git status --short src tests
  pause >nul
)
for /f "delims=" %%C in ('git log -1 --format^=%%h 2^>nul') do echo Running commit %%C

echo Startup checks: OCR warm-up, every solver must answer a real clue, and the writer types and
echo erases "ok" in the live #anime-chat composer (pause typing for 2 seconds when asked).
echo The app refuses to arm if any of those fail. Capture starts after ARMED.
echo Any red or green card is answered, even mid-quiz. The answer is typed as real input at green;
echo if Chrome is in front, Discord is raised once you pause typing and focus is returned after.
echo Afterwards: ".venv\Scripts\anime-trivia.exe --report" prints what happened in every round.
echo Press F12 to stop.
echo.
".venv\Scripts\anime-trivia.exe" --config "config.json"
echo.
echo Anime Trivia stopped with exit code %errorlevel%.
pause
