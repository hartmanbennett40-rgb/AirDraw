@echo off
REM Starts AirDraw: creates/uses a venv, installs deps if needed, runs the app.
cd /d "%~dp0"

REM mediapipe has no wheels for very new Python versions, so prefer 3.11 if
REM the py launcher has it installed; otherwise fall back to whatever is on PATH.
where py >nul 2>nul
if %errorlevel%==0 (
    py -3.11 --version >nul 2>nul
    if %errorlevel%==0 (
        set PYLAUNCHER=py -3.11
    ) else (
        set PYLAUNCHER=py -3
    )
) else (
    where python >nul 2>nul
    if %errorlevel%==0 (
        set PYLAUNCHER=python
    ) else (
        echo ERROR: Python was not found on PATH.
        echo Install Python 3.11 from https://www.python.org/downloads/
        echo and make sure "Add python.exe to PATH" is checked during setup.
        exit /b 1
    )
)

if exist ".venv" if not exist ".venv\Scripts\activate.bat" (
    echo Removing incomplete .venv from a previous failed setup...
    rmdir /s /q .venv
)

if not exist ".venv" (
    %PYLAUNCHER% -m venv .venv
    if not exist ".venv\Scripts\activate.bat" (
        echo ERROR: Failed to create the virtual environment.
        exit /b 1
    )
)

call .venv\Scripts\activate.bat
pip install -q -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies. See the pip error above.
    echo If the error mentions mediapipe and "no matching distribution",
    echo your Python version is too new - install Python 3.11 and re-run.
    exit /b 1
)

python -m airdraw.app
