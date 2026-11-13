@echo off
setlocal
cd /d "%~dp0"
title ChatCarry

rem Find Python: the "py" launcher first, then "python".
set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY where python >nul 2>nul && set "PY=python"
if not defined PY goto :nopython

rem First run only: make a private Python environment next to the app.
if not exist ".venv\Scripts\python.exe" (
  echo Setting up ChatCarry for the first time. This takes about a minute...
  %PY% -m venv .venv
  if errorlevel 1 goto :setupfailed
)
".venv\Scripts\python.exe" -c "import flask" >nul 2>nul
if errorlevel 1 (
  echo Installing Flask...
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 goto :setupfailed
)

rem Make a .env file the first time, so there is a place to keep the Gemini key.
if not exist ".env" if exist ".env.example" (
  copy ".env.example" ".env" >nul
  echo Created a .env file. Open it in Notepad and paste your Gemini key after GEMINI_API_KEY= to skip pasting it each time.
  echo.
)

set "CHATCARRY_OPEN=1"
".venv\Scripts\python.exe" app.py
echo.
echo ChatCarry has stopped.
pause
exit /b 0

:nopython
echo Python was not found on this computer.
echo Install Python 3 from https://www.python.org/downloads/ and tick "Add python.exe to PATH".
echo Then double-click this file again.
pause
exit /b 1

:setupfailed
echo.
echo Setup failed. Check your internet connection and try again.
pause
exit /b 1
