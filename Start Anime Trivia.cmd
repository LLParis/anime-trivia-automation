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

echo Loading local OCR, reviewed history, account Gemini 3.7 (Antigravity), and local Qwen3.8. Capture starts after ARMED.
echo Any red or green card is answered, even if you start mid-quiz. First answer types at green;
echo different answers from other solvers follow as extra guesses while the card stays green.
echo If Chrome is in front when the card turns green, Discord is raised once you pause typing,
echo the answer is sent, and focus returns to Chrome. Log + ledger: runtime\logs, runtime
ound_ledger.jsonl.
echo Press F12 to stop.
echo.
".venv\Scripts\anime-trivia.exe" --config "config.json"
echo.
echo Anime Trivia stopped with exit code %errorlevel%.
pause
