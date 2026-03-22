@echo off
REM Driver Fatigue Monitor - Setup Script for Windows
REM This script sets up the virtual environment, installs dependencies, and downloads the model

setlocal enabledelayedexpansion
cd /d "%~dp0"

echo.
echo ============================================================
echo Driver Fatigue Monitor - Setup Script
echo ============================================================
echo.

REM Run the Python setup script
python setup.py

if %errorlevel% neq 0 (
    echo.
    echo ============================================================
    echo Setup failed. Please check the errors above.
    echo ============================================================
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Setup complete! You can now run the app with:
echo   .venv\Scripts\python.exe -m streamlit run fatigue_app.py
echo ============================================================
echo.
pause
