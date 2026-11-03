@echo off
title Driver Drowsiness Detection

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Khong tim thay moi truong Python .venv
    echo Hay tao moi truong ao va cai requirements.txt
    pause
    exit /b 1
)

echo ===================================
echo DRIVER DROWSINESS DETECTION
echo ===================================
echo.
echo Dang khoi dong he thong...
echo.

".venv\Scripts\python.exe" app.py

echo.
echo He thong da dung.
pause