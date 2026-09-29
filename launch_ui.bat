@echo off
setlocal enabledelayedexpansion

REM Navigate to script directory dynamically
cd /d "%~dp0"

echo ========================================================
echo            Photo Organizer Web Studio Launcher
echo ========================================================
echo.

REM Activate local virtual environment if present
if exist "%~dp0.venv\Scripts\activate.bat" (
    echo Activating virtual environment: .venv
    call "%~dp0.venv\Scripts\activate.bat"
)

REM Detect Python executable (python, py, or python3)
set "PY_CMD="
python --version >nul 2>&1
if !errorlevel! equ 0 (
    set "PY_CMD=python"
) else (
    py -0 >nul 2>&1
    if !errorlevel! equ 0 (
        set "PY_CMD=py"
    ) else (
        python3 --version >nul 2>&1
        if !errorlevel! equ 0 (
            set "PY_CMD=python3"
        )
    )
)

if not defined PY_CMD (
    echo [ERROR] Python was not detected on your system.
    echo Please install Python 3.10+ or ensure it is added to your PATH.
    echo.
    pause
    exit /b 1
)

REM Optional argument: drag-and-drop folder or command line path
set "TARGET_DIR=%~1"

REM Automatically open default web browser after server starts
start "" cmd /c "timeout /t 2 /nobreak >nul && start http://127.0.0.1:8080"

echo Starting Photo Organizer Web Studio at http://127.0.0.1:8080
echo Close this window or press Ctrl+C to terminate the server.
echo.

if defined TARGET_DIR (
    !PY_CMD! photo_organizer.py --ui "%TARGET_DIR%"
) else (
    !PY_CMD! photo_organizer.py --ui
)

if !errorlevel! neq 0 (
    echo.
    echo Server stopped with error code !errorlevel!.
    pause
)
