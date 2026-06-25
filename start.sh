#!/bin/bash

# Define colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=======================================${NC}"
echo -e "${BLUE}   Citation Cartography & Graph RAG    ${NC}"
echo -e "${BLUE}=======================================${NC}"

# Function to handle graceful shutdown
cleanup() {
    echo -e "\n${YELLOW}Shutting down services...${NC}"
    if [ -n "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null
    fi
    if [ -n "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null
    fi
    echo -e "${GREEN}Shutdown complete.${NC}"
    exit 0
}

# Register the cleanup function for Ctrl+C
trap cleanup SIGINT SIGTERM

# 1. Check for Ollama
echo -e "\n${YELLOW}[1/4] Checking Ollama service...${NC}"
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo -e "${RED}Error: Ollama is not running. Please start Ollama first (e.g., 'ollama serve' in another terminal).${NC}"
    exit 1
else
    echo -e "${GREEN}Ollama is running.${NC}"
fi

# 2. Start Python Backend
echo -e "\n${YELLOW}[2/4] Starting FastAPI Backend...${NC}"
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Start uvicorn in the background
uvicorn api:app --host 0.0.0.0 --port 8000 > /dev/null 2>&1 &
BACKEND_PID=$!

# Wait for backend to be healthy
sleep 3
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo -e "${RED}Failed to start backend server.${NC}"
    exit 1
fi
echo -e "${GREEN}Backend running on http://localhost:8000${NC}"

# 3. Start Next.js Frontend
echo -e "\n${YELLOW}[3/4] Starting Next.js Frontend...${NC}"
cd frontend

# Install dependencies if node_modules is missing
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}Installing frontend dependencies...${NC}"
    npm install
fi

# Start next.js in the background
npm run dev > /dev/null 2>&1 &
FRONTEND_PID=$!

# Wait a moment for frontend to initialize
sleep 4
if ! kill -0 $FRONTEND_PID 2>/dev/null; then
    echo -e "${RED}Failed to start frontend server.${NC}"
    cleanup
fi
echo -e "${GREEN}Frontend running on http://localhost:3000${NC}"

# 4. Open Browser
echo -e "\n${YELLOW}[4/4] Opening Browser...${NC}"
cd ..
URL="http://localhost:3000"

if command -v xdg-open > /dev/null; then
    xdg-open "$URL"
elif command -v open > /dev/null; then
    open "$URL"
elif command -v start > /dev/null; then
    start "$URL"
elif command -v python3 > /dev/null; then
    python3 -m webbrowser "$URL"
else
    echo -e "${YELLOW}Could not open browser automatically. Please navigate to $URL${NC}"
fi

echo -e "\n${GREEN}=======================================${NC}"
echo -e "${GREEN}All systems operational. Press Ctrl+C to stop.${NC}"
echo -e "${GREEN}=======================================${NC}"

# Wait indefinitely for user to press Ctrl+C
wait $BACKEND_PID $FRONTEND_PID
