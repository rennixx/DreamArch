@echo off
REM Dream Architect - Start Script (Windows)

echo.
echo ============================================================
echo   Dream Architect - Starting Application
echo ============================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.9+
    pause
    exit /b 1
)

REM Check if Node.js is available
cd /d "%~dp0frontend"
if not exist "node_modules" (
    echo [INFO] Installing frontend dependencies...
    call npm install
)

echo.
echo [1/2] Starting Backend (FastAPI)...
echo.
start "Dream Architect Backend" cmd /k "cd /d %~dp0backend && venv\Scripts\activate && uvicorn main:app --reload --host 0.0.0.0 --port 8000"

timeout /t 3 /nobreak >nul

echo.
echo [2/2] Starting Frontend (Next.js)...
echo.
start "Dream Architect Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo ============================================================
echo   Dream Architect is starting...
echo ============================================================
echo.
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:8000
echo API Docs: http://localhost:8000/docs
echo.
echo Press Ctrl+C in each window to stop.
echo.

timeout /t 2 /nobreak >nul
start http://localhost:3000
