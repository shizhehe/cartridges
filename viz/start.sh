#!/bin/bash

# Start the dataset visualization application

echo "Starting Dataset Visualization App..."

# Check if Python backend dependencies are installed
if ! python3 -c "import fastapi, uvicorn" 2>/dev/null; then
    echo "Installing Python dependencies..."
    pip3 install --upgrade pip setuptools wheel
    pip3 install -r requirements.txt
fi

# Start the Python backend server
echo "Starting FastAPI server on port 8000..."
python3 src/server.py &
BACKEND_PID=$!

# Wait a moment for the backend to start
sleep 2

# Start the Vite development server
echo "Starting Vite development server..."
npm run dev &
FRONTEND_PID=$!

# Function to cleanup processes on exit
cleanup() {
    echo "Stopping servers..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

echo "
Dataset Visualization App is running!
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

Press Ctrl+C to stop both servers.
"

# Wait for both processes
wait $BACKEND_PID $FRONTEND_PID