#!/bin/bash
# start.sh — Launch both FastAPI backend and Streamlit frontend

echo "💧 Starting Water Tracker AI..."

# Check .env file
if [ ! -f .env ]; then
    cp .env.example .env
    echo "⚠️  .env file created from .env.example — add your ANTHROPIC_API_KEY!"
fi

# Install dependencies
pip install -r requirements.txt -q

# Start FastAPI backend in background
echo "🚀 Starting FastAPI backend on port 8000..."
cd backend && uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
cd ..

# Wait for backend to start
sleep 3

# Start Streamlit frontend
echo "🌊 Starting Streamlit frontend on port 8501..."
cd frontend && streamlit run app.py --server.port 8501 --server.headless true &
FRONTEND_PID=$!
cd ..

echo ""
echo "✅ Water Tracker AI is running!"
echo "   Backend API:  http://localhost:8000"
echo "   API Docs:     http://localhost:8000/docs"
echo "   Dashboard:    http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop all services."

# Wait and cleanup
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo 'Stopped.'" EXIT
wait
