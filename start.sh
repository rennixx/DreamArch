#!/bin/bash
# Dream Architect - Start Script (Linux/Mac)

echo ""
echo "============================================================"
echo "  Dream Architect - Starting Application"
echo "============================================================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python not found. Please install Python 3.9+"
    exit 1
fi

# Check if Node.js is available
if ! command -v npm &> /dev/null; then
    echo "[ERROR] Node.js not found. Please install Node.js 18+"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Install frontend dependencies if needed
if [ ! -d "frontend/node_modules" ]; then
    echo "[INFO] Installing frontend dependencies..."
    cd frontend
    npm install
    cd ..
fi

echo ""
echo "[1/2] Starting Backend (FastAPI)..."
echo ""

# Start backend in background
cd backend
if [ ! -d "venv" ]; then
    echo "[INFO] Creating Python virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
cd ..

echo "[INFO] Backend started (PID: $BACKEND_PID)"
sleep 3

echo ""
echo "[2/2] Starting Frontend (Next.js)..."
echo ""

# Start frontend in background
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo "[INFO] Frontend started (PID: $FRONTEND_PID)"

echo ""
echo "============================================================"
echo "  Dream Architect is running!"
echo "============================================================"
echo ""
echo "  Backend:   http://localhost:8000"
echo "  Frontend:  http://localhost:3000"
echo "  API Docs:  http://localhost:8000/docs"
echo ""
echo "  Press Ctrl+C to stop both servers"
echo ""

# Wait for interrupt
trap "echo ''; echo 'Stopping...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT

wait
