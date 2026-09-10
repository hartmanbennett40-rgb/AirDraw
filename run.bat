@echo off
REM Starts AirDraw: creates/uses a venv, installs deps if needed, runs the app.
cd /d "%~dp0"

if not exist ".venv" (
    py -3 -m venv .venv
)

call .venv\Scripts\activate.bat
pip install -q -r requirements.txt

python -m airdraw.app
