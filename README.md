# 🤖 AI Code Review Assistant

A full-stack AI-powered code review tool built with:
- **Fine-tuned LLM** (CodeBERT / CodeLlama via Hugging Face)
- **FastAPI** backend with REST API
- **VS Code Extension** for seamless developer experience

---

## 📁 Project Structure

```
ai-code-reviewer/
├── backend/                  # FastAPI Python backend
│   ├── app/
│   │   ├── api/              # Route handlers
│   │   ├── core/             # Config, settings
│   │   ├── models/           # Pydantic models
│   │   └── services/         # LLM service logic
│   ├── tests/
│   └── requirements.txt
├── extension/                # VS Code Extension (Node.js)
│   ├── src/
│   │   ├── extension.ts      # Main extension entry
│   │   ├── reviewer.ts       # API communication
│   │   └── webview.ts        # Sidebar UI
│   └── package.json
├── fine-tuning/              # LLM fine-tuning scripts
│   ├── data/                 # Training datasets
│   ├── scripts/              # Training code
│   └── models/               # Saved model weights
└── docs/                     # Architecture & setup docs
```

---

## 🚀 Quick Start

### 1. Backend Setup
```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### 2. Fine-tune Model (Optional - uses pretrained by default)
```bash
cd fine-tuning
pip install -r requirements.txt
python scripts/train.py
```

### 3. VS Code Extension
```bash
cd extension
npm install
npm run compile
# Press F5 in VS Code to launch Extension Development Host
```

---

## ✨ Features

| Feature | Description |
|---|---|
| 🐛 Bug Detection | Identifies potential bugs, null refs, off-by-one errors |
| 📊 Code Quality Score | Rates code 0-100 with detailed breakdown |
| 💡 Improvement Suggestions | Line-by-line refactoring suggestions |
| 🔒 Security Analysis | Detects SQL injection, XSS, hardcoded secrets |
| 📝 Docstring Generator | Auto-generates missing documentation |
| ⚡ Real-time Review | Review on save or via right-click menu |

---

## 🧠 Architecture

```
VS Code Extension (TypeScript)
        │
        │ HTTP POST /review
        ▼
FastAPI Backend (Python)
        │
        │ Inference
        ▼
Fine-tuned LLM
(CodeBERT + CodeLlama 7B)
        │
        ▼
Structured JSON Response
{ bugs, suggestions, score, security }
```

---

## 📊 Tech Stack

| Layer | Technology |
|---|---|
| LLM | CodeLlama-7B + CodeBERT |
| Fine-tuning | Hugging Face Transformers, PEFT/LoRA |
| Backend | FastAPI, Python 3.10+ |
| Extension | TypeScript, VS Code API |
| Deployment | Docker, Hugging Face Spaces |

---

## 🎓 Resume Points

- Fine-tuned **CodeLlama-7B** using **LoRA/PEFT** on code review datasets
- Built **REST API** with FastAPI handling concurrent LLM inference
- Developed **VS Code Extension** with 1000+ lines of TypeScript
- Achieved **X% improvement** in bug detection over baseline GPT-3.5
