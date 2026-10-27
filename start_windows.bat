@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"
chcp 65001 >nul

echo TAISIA 0.4 - Windows 11
set "PYTHON_COMMAND="
where python >nul 2>&1
if not errorlevel 1 (
    python -c "import sys,venv,ensurepip; assert (3,11) <= sys.version_info[:2] <= (3,13)" >nul 2>&1
    if not errorlevel 1 set "PYTHON_COMMAND=python"
)
if not defined PYTHON_COMMAND (
    where py >nul 2>&1
    if not errorlevel 1 (
        py -3.12 -c "import venv,ensurepip" >nul 2>&1
        if not errorlevel 1 set "PYTHON_COMMAND=py -3.12"
    )
)
if not defined PYTHON_COMMAND (
    echo Python 3.11-3.13 not found. Install Python 3.12 from python.org and enable Add to PATH.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating local Python environment...
    %PYTHON_COMMAND% -m venv ".venv"
    if errorlevel 1 (
        echo Failed to create .venv. Ensure Python is installed with venv and pip.
        pause
        exit /b 1
    )
)

".venv\Scripts\python.exe" -c "import numpy,scipy,matplotlib,sounddevice" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies from requirements.txt. Internet may be required on first launch.
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Package installation failed. See error messages above.
        pause
        exit /b 1
    )
)

echo Running quick startup smoke tests...
".venv\Scripts\python.exe" -m unittest discover -s tests -p test_smoke.py -q
if errorlevel 1 (
    echo Mathematical tests failed. TAISIA will NOT launch.
    pause
    exit /b 1
)
echo Starting TAISIA...
".venv\Scripts\python.exe" app.py
if errorlevel 1 (
    echo TAISIA exited with an error. See details above.
    pause
    exit /b 1
)
exit /b 0
