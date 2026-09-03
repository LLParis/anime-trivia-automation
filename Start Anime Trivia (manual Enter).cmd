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

echo MANUAL-ENTER MODE: the app does everything except press Enter.
echo Use this only if the automatic Enter is not posting in a real quiz.
echo At green the answer is typed into #anime-chat and verified, then left in the box for YOU to
echo press Enter. If you do not send it, the app erases its own text when the next round starts.
echo Same startup checks as the normal launcher (OCR warm-up, solver check, "ok" probe).
echo Afterwards: "Show Quiz Report.cmd" lists every round; rounds show as rehearsal, not sent.
echo Press F12 to stop.
echo.
".venv\Scripts\anime-trivia.exe" --config "config.json" --rehearse
echo.
echo Anime Trivia stopped with exit code %errorlevel%.
pause
