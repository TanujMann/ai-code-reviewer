@echo off
REM ============================================================
REM  AI Code Reviewer - Windows Setup Script
REM  Run this once to set up the entire project
REM ============================================================

echo.
echo  ====================================================
echo   AI Code Reviewer - Setup Script (Windows)
echo  ====================================================
echo.

REM ── Step 1: Check Python ────────────────────────────────────
echo [1/6] Checking Python installation...
python --version 2>NUL
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python not found! 
    echo Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)
echo OK

REM ── Step 2: Check Node.js ────────────────────────────────────
echo.
echo [2/6] Checking Node.js installation...
node --version 2>NUL
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Node.js not found!
    echo Please install Node.js from https://nodejs.org
    pause
    exit /b 1
)
echo OK

REM ── Step 3: Backend Setup ────────────────────────────────────
echo.
echo [3/6] Setting up Python backend...
cd backend

python -m venv venv
call venv\Scripts\activate

pip install -r requirements.txt --quiet

REM Create .env from example
if not exist ".env" (
    copy .env.example .env
    echo Created .env from template
)

cd ..
echo OK

REM ── Step 4: Fine-tuning Setup ────────────────────────────────
echo.
echo [4/6] Setting up fine-tuning environment...
cd fine-tuning
pip install -r requirements.txt --quiet 2>NUL
cd ..
echo OK

REM ── Step 5: VS Code Extension Setup ─────────────────────────
echo.
echo [5/6] Setting up VS Code Extension...
cd extension
npm install --silent
npm run compile
cd ..
echo OK

REM ── Step 6: Verify Installation ──────────────────────────────
echo.
echo [6/6] Verifying setup...
cd backend
call venv\Scripts\activate
python -c "import fastapi, uvicorn, pydantic; print('Backend deps OK')"
cd ..
echo.
echo  ====================================================
echo   Setup Complete!
echo  ====================================================
echo.
echo  Next steps:
echo.
echo  1. Start the backend:
echo     cd backend
echo     venv\Scripts\activate
echo     uvicorn app.main:app --reload
echo.
echo  2. Open VS Code Extension:
echo     cd extension
echo     code .
echo     Press F5 to launch Extension Development Host
echo.
echo  3. (Optional) Fine-tune your own model:
echo     cd fine-tuning
echo     python scripts/prepare_dataset.py
echo     python scripts/train.py
echo.
echo  API Docs: http://localhost:8000/docs
echo.
pause
