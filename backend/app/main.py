import os
import time
import re

from fastapi import FastAPI, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger


# ===============================
# 🔐 API KEY CONFIG
# ===============================

API_KEY = os.getenv("API_KEY")


def verify_api_key(x_api_key: str = Header(None)):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API key not configured on server")

    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    return x_api_key


# ===============================
# 🚀 APP SETUP
# ===============================

app = FastAPI(
    title="AI Code Reviewer",
    description="AI-powered code review API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    logger.info("Starting AI Code Reviewer v1.0.0")


# ===============================
# 🌍 PUBLIC ROUTES
# ===============================

@app.get("/")
async def root():
    return {
        "message": "AI Code Reviewer API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/api/v1/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_backend": "demo",
        "version": "1.0.0"
    }


# ===============================
# 🧠 PROTECTED ROUTES
# ===============================

@app.post("/api/v1/review")
async def review_code(
    request: dict,
    api_key: str = Depends(verify_api_key)
):
    start = time.time()
    code = request.get("code", "")
    language = request.get("language", "python")

    bugs = []
    security_issues = []
    improvements = []
    score = 100

    # -----------------------------
    # Security Checks
    # -----------------------------

    if re.search(r'f["\'].*SELECT.*\{', code, re.IGNORECASE):
        security_issues.append({
            "line": None,
            "issue_type": "SQL Injection",
            "description": "f-string used in SQL query — vulnerable to SQL injection",
            "severity": "critical",
            "fix": "Use parameterized queries"
        })
        score -= 30

    if "eval(" in code:
        security_issues.append({
            "line": None,
            "issue_type": "Code Injection",
            "description": "eval() allows arbitrary code execution",
            "severity": "critical",
            "fix": "Remove eval()"
        })
        score -= 25

    if "md5" in code.lower():
        security_issues.append({
            "line": None,
            "issue_type": "Weak Cryptography",
            "description": "MD5 is insecure for passwords",
            "severity": "high",
            "fix": "Use bcrypt or argon2"
        })
        score -= 20

    if re.search(r'password\s*=\s*["\']', code, re.IGNORECASE):
        security_issues.append({
            "line": None,
            "issue_type": "Hardcoded Credentials",
            "description": "Hardcoded password found",
            "severity": "critical",
            "fix": "Use environment variables"
        })
        score -= 15

    # -----------------------------
    # Bug Checks
    # -----------------------------

    if re.search(r'/\s*\w+', code) and 'def divide' in code:
        bugs.append({
            "line": None,
            "description": "Possible division by zero",
            "severity": "medium",
            "suggestion": "Check denominator before division"
        })
        score -= 10

    # -----------------------------
    # Improvements
    # -----------------------------

    if 'def ' in code and '->' not in code:
        improvements.append({
            "line": None,
            "category": "Type Hints",
            "description": "Add type hints to functions",
            "code_example": "def login(username: str, password: str) -> dict:"
        })

    score = max(0, score)

    # -----------------------------
    # Grade Calculation
    # -----------------------------

    if score >= 90:
        grade = "A"
    elif score >= 80:
        grade = "B"
    elif score >= 70:
        grade = "C"
    elif score >= 60:
        grade = "D"
    else:
        grade = "F"

    elapsed = (time.time() - start) * 1000

    return {
        "score": score,
        "grade": grade,
        "breakdown": {
            "correctness": min(100, score + 10),
            "security": max(0, 100 - len(security_issues) * 25),
            "performance": 80,
            "readability": 75,
            "maintainability": 75
        },
        "bugs": bugs,
        "security_issues": security_issues,
        "improvements": improvements,
        "summary": f"Found {len(bugs)} bugs and {len(security_issues)} security issues.",
        "language_detected": language,
        "review_time_ms": round(elapsed, 2),
        "model_used": "demo"
    }


@app.post("/api/v1/quick-scan")
async def quick_scan(
    request: dict,
    api_key: str = Depends(verify_api_key)
):
    return await review_code(request, api_key)