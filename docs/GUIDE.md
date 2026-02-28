# 📖 Step-by-Step Guide: AI Code Review Assistant

## What You're Building

```
┌──────────────────────────────────────────────┐
│            VS Code Editor                     │
│  ┌─────────────────────────────────────────┐ │
│  │  def get_user(id):                       │ │
│  │      query = f"SELECT * WHERE id={id}" ←─┼─┼── 🔴 SQL Injection warning
│  │      return db.execute(query)            │ │
│  └─────────────────────────────────────────┘ │
│                                              │
│  ┌────────────────┐  Sidebar Panel:          │
│  │ Right-click →  │  📊 Score: 15/100 F      │
│  │ AI: Review     │  🐛 2 bugs found         │
│  │ This File      │  🔒 1 SQL injection       │
│  └────────────────┘  💡 3 improvements       │
└──────────────────────────────────────────────┘
         │ HTTP POST
         ▼
┌──────────────────┐
│  FastAPI Backend │  ← Your Python server
│  localhost:8000  │
└──────────────────┘
         │
         ▼
┌──────────────────┐
│  AI Model        │  ← Fine-tuned CodeLlama
│  (or demo mode)  │    or demo rules engine
└──────────────────┘
```

---

## Prerequisites

Install these before starting:

| Tool | Download | Check |
|------|----------|-------|
| Python 3.10+ | python.org | `python --version` |
| Node.js 18+ | nodejs.org | `node --version` |
| VS Code | code.visualstudio.com | `code --version` |
| Git | git-scm.com | `git --version` |

---

## PHASE 1: Run the Backend (Demo Mode — No GPU Needed)

### Step 1.1: Install dependencies
```batch
cd ai-code-reviewer\backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Step 1.2: Configure environment
```batch
copy .env.example .env
```

Open `.env` and set:
```
MODEL_BACKEND=demo
```

> **Demo mode** uses intelligent rule-based analysis.
> It detects real security patterns, bugs, and code issues
> without needing any GPU or API key. Perfect for development!

### Step 1.3: Start the server
```batch
uvicorn app.main:app --reload
```

You should see:
```
INFO:     AI Code Reviewer v1.0.0 starting up
INFO:     Model backend: demo
INFO:     ✅ API ready! Docs at: http://localhost:8000/docs
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 1.4: Test the API
Open your browser: http://localhost:8000/docs

Click **POST /api/v1/review** → **Try it out** → paste this code:
```json
{
  "code": "def get_user(id):\n    query = f\"SELECT * FROM users WHERE id = {id}\"\n    return db.execute(query)",
  "language": "python"
}
```

Click Execute. You should get back a JSON response with SQL injection detected! ✅

---

## PHASE 2: Install the VS Code Extension

### Step 2.1: Install dependencies
```batch
cd ai-code-reviewer\extension
npm install
npm run compile
```

### Step 2.2: Launch extension in VS Code
```batch
code .
```

In VS Code, press **F5** (or Run → Start Debugging).

A new VS Code window opens — this is the **Extension Development Host**.
This is where your extension runs!

### Step 2.3: Test the extension

1. In the Extension Development Host, create a new file `test.py`
2. Paste this code:
```python
import sqlite3

def get_user(username):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    query = f"SELECT * FROM users WHERE username = '{username}'"
    cursor.execute(query)
    return cursor.fetchone()
```

3. Right-click in the editor → **"AI: Review This File"**
   (or press `Ctrl+Shift+R`)

4. You should see:
   - 🔴 Red squiggles on the SQL injection line
   - The sidebar panel showing the full review
   - Status bar showing the score

---

## PHASE 3: Fine-tune Your Own Model (Optional — Needs GPU)

> Skip this phase if you don't have a GPU.
> The demo mode already shows the full end-to-end system for your resume!

### Step 3.1: Prepare the dataset
```batch
cd ai-code-reviewer\fine-tuning
pip install -r requirements.txt
python scripts\prepare_dataset.py
```

This creates `data\train.json` and `data\val.json`.

### Step 3.2: Fine-tune with LoRA

**No GPU (CPU only — slow but works):**
```batch
python scripts\train.py --model microsoft/phi-2 --epochs 1 --use_4bit false
```

**With GPU (16GB VRAM — recommended):**
```batch
python scripts\train.py --model codellama/CodeLlama-7b-Instruct-hf --epochs 3
```

**Free GPU option — Google Colab:**
1. Upload `fine-tuning/` to Google Drive
2. Open Colab with T4 GPU (free)
3. Run the training script
4. Download the `models/` folder

### Step 3.3: Switch backend to local model
Edit `backend/.env`:
```
MODEL_BACKEND=local
MODEL_PATH=../fine-tuning/models/code-reviewer-lora/final
```

---

## PHASE 4: Switch to OpenAI (Best Quality, No GPU)

Get an API key from platform.openai.com

Edit `backend/.env`:
```
MODEL_BACKEND=openai
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o-mini
```

Restart the backend. Now your extension uses GPT-4o-mini!

---

## Troubleshooting

### "Cannot connect to backend"
- Make sure backend is running: `uvicorn app.main:app --reload`
- Check port 8000 is not blocked
- Try: http://localhost:8000/health

### "Module not found" errors
```batch
cd backend
venv\Scripts\activate
pip install -r requirements.txt
```

### Extension not showing up
- Make sure you pressed F5 in the extension's VS Code window
- Check the Debug Console for errors
- Try: `npm run compile` before F5

### TypeScript compile errors
```batch
cd extension
npm install
npx tsc --version   # Should be 5.x
npm run compile
```

---

## Resume Talking Points

When explaining this project in interviews:

**"What did you build?"**
> "An AI-powered VS Code extension that reviews code in real-time.
> It detects bugs, security vulnerabilities, and suggests improvements
> with line-level precision using a fine-tuned LLM."

**"What tech did you use?"**
> "Python FastAPI for the backend, TypeScript for the VS Code extension,
> and I fine-tuned CodeLlama-7B using LoRA/PEFT — which let me train
> a 7B parameter model in 8GB of VRAM instead of 80GB."

**"What was the hardest part?"**
> "Parsing unstructured LLM output into structured diagnostic data
> that VS Code could display as inline squiggles with the exact
> line numbers. I ended up using a regex-based JSON extractor with
> graceful fallbacks."

**"What would you improve?"**
> "I'd add a tree-sitter based AST parser for more precise line detection,
> add streaming responses so suggestions appear progressively, and
> publish it to the VS Code marketplace."
