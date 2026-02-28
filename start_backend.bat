@echo off
echo Starting AI Code Reviewer Backend...
cd backend
if exist venv\Scripts\activate (
    call venv\Scripts\activate
) else (
    echo ERROR: Virtual environment not found. Run setup.bat first!
    pause
    exit /b 1
)
echo.
echo  Backend starting at: http://localhost:8000
echo  API Docs at:         http://localhost:8000/docs
echo  Press Ctrl+C to stop
echo.
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
