@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
".venv\Scripts\anime-trivia.exe" --config "config.json" --report 1
echo.
echo Columns: Q, clue, which solver answered, answer(s), seconds from red to answer,
echo what was sent, what the bot revealed, and the outcome (CORRECT / WRONG / HAD IT but not sent / not resolved).
echo Use --report 3 for the last three launches. The same table is saved under runtime\logs.
pause
