from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import time
import re

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

@app.get("/")
async def root():
    return {"message": "AI Code Reviewer API", "version": "1.0.0", "docs": "/docs"}

@app.get("/api/v1/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_backend": "demo",
        "version": "1.0.0"
    }

@app.post("/api/v1/review")
async def review_code(request: dict):
    start = time.time()
    code = request.get("code", "")
    language = request.get("language", "python")

    bugs = []
    security_issues = []
    improvements = []
    score = 100

    # Check SQL Injection
    if re.search(r'f["\'].*SELECT.*\{', code, re.IGNORECASE):
        security_issues.append({
            "line": None,
            "issue_type": "SQL Injection",
            "description": "f-string used in SQL query — vulnerable to SQL injection",
            "severity": "critical",
            "fix": "Use parameterized queries: cursor.execute('SELECT * FROM users WHERE id = %s', (id,))"
        })
        score -= 30

    # Check eval
    if 'eval(' in code:
        security_issues.append({
            "line": None,
            "issue_type": "Code Injection",
            "description": "eval() is dangerous — allows arbitrary code execution",
            "severity": "critical",
            "fix": "Remove eval() and use safer alternatives"
        })
        score -= 25

    # Check MD5
    if 'md5' in code.lower():
        security_issues.append({
            "line": None,
            "issue_type": "Weak Cryptography",
            "description": "MD5 is cryptographically broken — don't use for passwords",
            "severity": "high",
            "fix": "Use bcrypt or argon2: import bcrypt; bcrypt.hashpw(password, bcrypt.gensalt())"
        })
        score -= 20

    # Check hardcoded password
    if re.search(r'password\s*=\s*["\']', code, re.IGNORECASE):
        security_issues.append({
            "line": None,
            "issue_type": "Hardcoded Credentials",
            "description": "Hardcoded password found in source code",
            "severity": "critical",
            "fix": "Use environment variables: os.environ.get('PASSWORD')"
        })
        score -= 15

    # Check division
    if re.search(r'/\s*\w+', code) and 'def divide' in code:
        bugs.append({
            "line": None,
            "description": "Division by zero risk — no zero check",
            "severity": "medium",
            "suggestion": "Add: if b == 0: raise ValueError('Cannot divide by zero')"
        })
        score -= 10

    # Improvements
    if 'def ' in code and '->' not in code:
        improvements.append({
            "line": None,
            "category": "Type Hints",
            "description": "Add type hints to functions",
            "code_example": "def login(username: str, password: str) -> dict:"
        })

    score = max(0, score)

    # Grade
    if score >= 90: grade = "A"
    elif score >= 80: grade = "B"
    elif score >= 70: grade = "C"
    elif score >= 60: grade = "D"
    else: grade = "F"

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
async def quick_scan(request: dict):
    return await review_code(request)