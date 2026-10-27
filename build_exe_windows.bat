@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo First start_windows.bat to create the local environment.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m pip install "pyinstaller>=6,<7"
if errorlevel 1 (pause & exit /b 1)
".venv\Scripts\python.exe" -m PyInstaller --noconfirm --clean --onedir --windowed --name TAISIA --collect-all sounddevice app.py
if errorlevel 1 (pause & exit /b 1)
echo EXE folder: dist\TAISIA\
pause
