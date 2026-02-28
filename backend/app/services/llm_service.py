"""
llm_service.py
==============
The brain of the application.
Handles code review using 3 different backends:

1. LOCAL   → Your fine-tuned CodeLlama model (best quality, needs GPU)
2. OPENAI  → GPT-4o-mini fallback (great quality, needs API key)
3. DEMO    → Rule-based analysis (works everywhere, good for testing)

The service auto-selects based on config.MODEL_BACKEND setting.
"""

import re
import time
import json
import ast
import keyword
from typing import Optional, Dict, Any
from loguru import logger

from app.core.config import settings
from app.models.schemas import (
    ReviewResponse, Bug, SecurityIssue, Improvement,
    QualityBreakdown, Language, Severity
)


# ─────────────────────────────────────────────────────────────────────────────
# BACKEND 1: LOCAL FINE-TUNED MODEL
# ─────────────────────────────────────────────────────────────────────────────

class LocalModelBackend:
    """Runs inference on your fine-tuned CodeLlama model."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
            
            logger.info(f"Loading fine-tuned model from: {settings.MODEL_PATH}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_PATH)
            base_model = AutoModelForCausalLM.from_pretrained(
                settings.BASE_MODEL_NAME,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.model = PeftModel.from_pretrained(base_model, settings.MODEL_PATH)
            self.model.eval()
            
            logger.info("✅ Local model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            raise
    
    def generate(self, code: str, language: str) -> str:
        import torch
        
        prompt = f"""### Instruction:
You are an expert code reviewer. Analyze this {language} code and provide:
1. Quality score (0-100)
2. Bugs found with line numbers
3. Security issues
4. Improvements
5. Summary

Respond in JSON format.

### Code:
```{language}
{code}
```

### Review JSON:
"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=settings.MAX_NEW_TOKENS,
                temperature=settings.TEMPERATURE,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        return response


# ─────────────────────────────────────────────────────────────────────────────
# BACKEND 2: OPENAI (Fallback)
# ─────────────────────────────────────────────────────────────────────────────

class OpenAIBackend:
    """Uses OpenAI GPT as a high-quality fallback."""
    
    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        logger.info("✅ OpenAI backend initialized")
    
    def generate(self, code: str, language: str) -> str:
        system_prompt = """You are an expert code reviewer. Analyze the given code and respond ONLY with valid JSON in this exact format:
{
  "score": <0-100>,
  "bugs": [{"line": <int or null>, "description": "<str>", "severity": "<critical|high|medium|low>", "suggestion": "<str>"}],
  "security_issues": [{"line": <int or null>, "issue_type": "<str>", "description": "<str>", "severity": "<critical|high|medium|low>", "fix": "<str>"}],
  "improvements": [{"line": <int or null>, "category": "<str>", "description": "<str>", "code_example": "<str or null>"}],
  "summary": "<one paragraph summary>"
}"""
        
        response = self.client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Review this {language} code:\n```{language}\n{code}\n```"}
            ],
            temperature=0.1,
            max_tokens=1000,
        )
        return response.choices[0].message.content


# ─────────────────────────────────────────────────────────────────────────────
# BACKEND 3: DEMO (Rule-Based — No GPU/API needed!)
# ─────────────────────────────────────────────────────────────────────────────

class DemoBackend:
    """
    Rule-based code analysis.
    Works without any AI model — great for development/demo.
    Detects real patterns using regex + AST analysis.
    """
    
    # Security patterns to check
    SECURITY_PATTERNS = {
        "SQL Injection": [
            (r'f["\'].*SELECT.*\{', "f-string used in SQL query — SQL injection risk"),
            (r'%.*SELECT.*%s', "String formatting used in SQL query"),
            (r'\.format\(.*SELECT', "str.format() used in SQL query"),
            (r'execute\(["\'].*\+', "String concatenation in SQL execute()"),
        ],
        "Hardcoded Secret": [
            (r'password\s*=\s*["\'][^"\']+["\']', "Hardcoded password detected"),
            (r'api_key\s*=\s*["\'][^"\']+["\']', "Hardcoded API key detected"),
            (r'secret\s*=\s*["\'][^"\']+["\']', "Hardcoded secret detected"),
            (r'AWS_SECRET|sk-[a-zA-Z0-9]{20}', "Hardcoded cloud credentials"),
        ],
        "Unsafe Deserialization": [
            (r'pickle\.loads?\(', "Unsafe pickle deserialization"),
            (r'yaml\.load\([^,)]+\)', "yaml.load() without Loader is unsafe — use yaml.safe_load()"),
            (r'eval\(', "eval() is dangerous with untrusted input"),
            (r'exec\(', "exec() is dangerous with untrusted input"),
        ],
        "Weak Cryptography": [
            (r'hashlib\.md5\(', "MD5 is cryptographically broken — use SHA-256 or bcrypt for passwords"),
            (r'hashlib\.sha1\(', "SHA-1 is deprecated for security use"),
            (r'DES\.|RC4\.|ECB', "Weak encryption algorithm detected"),
        ],
        "Path Traversal": [
            (r'open\(.*\+.*\)', "Concatenated path in open() — path traversal risk"),
        ],
        "XSS": [
            (r'innerHTML\s*=\s*.*\+', "innerHTML with concatenation — XSS risk"),
            (r'document\.write\(', "document.write() can lead to XSS"),
        ]
    }
    
    BUG_PATTERNS = {
        "Division by zero risk": (r'\/\s*\w+\b(?!\s*!=?\s*0)', None),
        "Empty except block": (r'except\s*:\s*\n\s*pass', "Silently swallowing exceptions"),
        "Missing await": (r'(?<!await\s)fetch\(|(?<!await\s)\.json\(\)', "Possible missing await"),
        "File not closed": (r'open\([^)]+\)(?!.*with)', "File opened without context manager"),
        "Mutable default argument": (r'def\s+\w+\([^)]*=\s*[\[\{]', "Mutable default argument — shared across calls"),
        "== None comparison": (r'==\s*None|!=\s*None', "Use 'is None' instead of '== None'"),
        "Bare except": (r'except\s*:', "Bare except catches all exceptions including KeyboardInterrupt"),
    }
    
    def analyze(self, code: str, language: str) -> Dict[str, Any]:
        lines = code.split('\n')
        bugs = []
        security_issues = []
        improvements = []
        
        # ── Security Checks ──────────────────────────────────
        for issue_type, patterns in self.SECURITY_PATTERNS.items():
            for pattern, description in patterns:
                for i, line in enumerate(lines, 1):
                    if re.search(pattern, line, re.IGNORECASE):
                        severity = "critical" if issue_type in ["SQL Injection", "Hardcoded Secret", "Weak Cryptography"] else "high"
                        security_issues.append({
                            "line": i,
                            "issue_type": issue_type,
                            "description": description,
                            "severity": severity,
                            "fix": self._get_fix_suggestion(issue_type)
                        })
        
        # ── Bug Checks ───────────────────────────────────────
        for bug_name, (pattern, desc) in self.BUG_PATTERNS.items():
            for i, line in enumerate(lines, 1):
                if re.search(pattern, line):
                    bugs.append({
                        "line": i,
                        "description": desc or bug_name,
                        "severity": "high" if "bare except" in bug_name.lower() or "await" in bug_name.lower() else "medium",
                        "suggestion": self._get_bug_suggestion(bug_name)
                    })
        
        # ── Style/Quality Checks ─────────────────────────────
        
        # Check for missing type hints (Python)
        if language == "python":
            fn_pattern = re.compile(r'def\s+(\w+)\s*\([^)]*\)(?!\s*->)')
            for i, line in enumerate(lines, 1):
                if fn_pattern.search(line) and '__' not in line:
                    improvements.append({
                        "line": i,
                        "category": "Type Hints",
                        "description": f"Function missing return type annotation",
                        "code_example": None
                    })
        
        # Check for long lines
        for i, line in enumerate(lines, 1):
            if len(line) > 120:
                improvements.append({
                    "line": i,
                    "category": "Readability",
                    "description": f"Line too long ({len(line)} chars) — consider breaking it up",
                    "code_example": None
                })
        
        # Check for TODO/FIXME comments
        for i, line in enumerate(lines, 1):
            if re.search(r'#\s*(TODO|FIXME|HACK|XXX)', line):
                improvements.append({
                    "line": i,
                    "category": "Code Debt",
                    "description": f"Unresolved TODO/FIXME comment",
                    "code_example": None
                })
        
        # Check for print statements in non-script code
        if language == "python":
            for i, line in enumerate(lines, 1):
                if re.search(r'^\s*print\(', line) and i > 1:
                    improvements.append({
                        "line": i,
                        "category": "Best Practice",
                        "description": "Use logging instead of print() for production code",
                        "code_example": "import logging\nlogging.info('your message')"
                    })
        
        # ── Score Calculation ────────────────────────────────
        score = 100
        critical_count = sum(1 for s in security_issues if s["severity"] == "critical")
        high_count = sum(1 for b in bugs if b["severity"] == "high")
        
        score -= critical_count * 25  # -25 per critical security issue
        score -= len(security_issues) * 10  # -10 per security issue
        score -= high_count * 15            # -15 per high severity bug
        score -= len(bugs) * 8             # -8 per bug
        score -= len(improvements) * 2     # -2 per improvement
        score = max(0, min(100, score))
        
        # Breakdown
        security_score = max(0, 100 - len(security_issues) * 20 - critical_count * 30)
        correctness_score = max(0, 100 - len(bugs) * 15)
        
        return {
            "score": score,
            "bugs": bugs[:10],              # Cap at 10
            "security_issues": security_issues[:5],
            "improvements": improvements[:8],
            "breakdown": {
                "correctness": correctness_score,
                "security": security_score,
                "performance": max(40, 90 - len(improvements) * 5),
                "readability": max(40, 85 - len(improvements) * 3),
                "maintainability": max(40, 80 - len(bugs) * 5),
            },
            "summary": self._generate_summary(score, bugs, security_issues)
        }
    
    def _get_fix_suggestion(self, issue_type: str) -> str:
        fixes = {
            "SQL Injection": "Use parameterized queries: cursor.execute('SELECT * WHERE id = ?', (user_id,))",
            "Hardcoded Secret": "Use environment variables: os.environ.get('API_KEY')",
            "Unsafe Deserialization": "Use safe alternatives: yaml.safe_load() or validate pickle sources",
            "Weak Cryptography": "Use bcrypt for passwords: bcrypt.hashpw(password, bcrypt.gensalt())",
            "Path Traversal": "Validate and sanitize file paths: os.path.realpath() and check prefix",
            "XSS": "Use textContent instead of innerHTML, or sanitize with DOMPurify",
        }
        return fixes.get(issue_type, "Review and fix the security issue")
    
    def _get_bug_suggestion(self, bug_name: str) -> str:
        suggestions = {
            "Mutable default argument": "Use None as default: def f(lst=None): if lst is None: lst = []",
            "== None comparison": "Use: if value is None: (identity check, not equality)",
            "Bare except": "Catch specific exceptions: except ValueError: or except Exception as e:",
            "File not closed": "Use context manager: with open(filename) as f: content = f.read()",
            "Missing await": "Add await keyword before async operations",
        }
        return suggestions.get(bug_name, "Review and fix the bug")
    
    def _generate_summary(self, score: int, bugs, security_issues) -> str:
        if score >= 90:
            return "Excellent code quality! Minor improvements suggested but no critical issues found."
        elif score >= 70:
            grade_word = "Good"
            main_issue = "a few improvements" if not bugs else f"{len(bugs)} bug(s)"
            return f"{grade_word} code quality with {main_issue} to address."
        elif score >= 50:
            sec_note = f" {len(security_issues)} security issue(s) need immediate attention." if security_issues else ""
            return f"Moderate quality with {len(bugs)} bug(s) found.{sec_note} Review and address issues before production."
        else:
            sec_note = f" CRITICAL: {len(security_issues)} security vulnerabilities detected!" if security_issues else ""
            return f"Significant issues found: {len(bugs)} bugs.{sec_note} This code needs substantial revision."
    
    def generate(self, code: str, language: str) -> str:
        result = self.analyze(code, language)
        return json.dumps(result)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SERVICE CLASS
# ─────────────────────────────────────────────────────────────────────────────

class LLMService:
    """
    Unified service that selects the right backend and
    parses responses into structured ReviewResponse objects.
    """
    
    def __init__(self):
        self.backend = None
        self.backend_name = settings.MODEL_BACKEND
        self._initialize_backend()
    
    def _initialize_backend(self):
        logger.info(f"Initializing backend: {self.backend_name}")
        
        if self.backend_name == "local":
            self.backend = LocalModelBackend()
        elif self.backend_name == "openai":
            if not settings.OPENAI_API_KEY:
                logger.warning("No OPENAI_API_KEY set, falling back to demo")
                self.backend_name = "demo"
                self.backend = DemoBackend()
            else:
                self.backend = OpenAIBackend()
        else:
            logger.info("Using demo (rule-based) backend")
            self.backend = DemoBackend()
        
        logger.info(f"✅ Backend ready: {self.backend_name}")
    
    @property
    def is_ready(self) -> bool:
        return self.backend is not None
    
    def detect_language(self, code: str, filename: Optional[str] = None, 
                        hint: Language = Language.AUTO) -> str:
        """Auto-detect language from filename or code patterns."""
        if hint != Language.AUTO:
            return hint.value
        
        if filename:
            ext_map = {
                ".py": "python", ".js": "javascript", ".ts": "typescript",
                ".java": "java", ".cpp": "cpp", ".c": "c", ".go": "go",
                ".rs": "rust", ".rb": "ruby", ".php": "php"
            }
            for ext, lang in ext_map.items():
                if filename.endswith(ext):
                    return lang
        
        # Pattern-based detection
        if "def " in code and "import " in code:
            return "python"
        if "function " in code or "const " in code or "let " in code:
            return "javascript"
        if "public class" in code or "System.out" in code:
            return "java"
        if "#include" in code:
            return "cpp"
        
        return "python"  # Default
    
    def calculate_grade(self, score: int) -> str:
        if score >= 90: return "A"
        if score >= 80: return "B"
        if score >= 70: return "C"
        if score >= 60: return "D"
        return "F"
    
    def parse_response(self, raw: str, language: str) -> Dict[str, Any]:
        """Parse the model's raw JSON output into structured data."""
        # Try to extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Fallback: return minimal structure
        return {
            "score": 50,
            "bugs": [],
            "security_issues": [],
            "improvements": [{"category": "General", "description": "Review code manually", "code_example": None}],
            "breakdown": {"correctness": 50, "security": 50, "performance": 50, "readability": 50, "maintainability": 50},
            "summary": "Analysis completed. Please review the code manually for detailed feedback."
        }
    
    async def review_code(self, code: str, language: str, filename: Optional[str] = None,
                          focus_areas: Optional[list] = None) -> ReviewResponse:
        """
        Main entry point: takes code and returns structured review.
        """
        start_time = time.time()
        
        # Detect language
        detected_lang = self.detect_language(code, filename, Language(language))
        
        # Generate review
        logger.info(f"Reviewing {detected_lang} code ({len(code)} chars) with {self.backend_name}")
        raw_response = self.backend.generate(code, detected_lang)
        
        # Parse response
        parsed = self.parse_response(raw_response, detected_lang)
        
        # Build structured response
        review_time = int((time.time() - start_time) * 1000)
        
        bugs = [Bug(**b) for b in parsed.get("bugs", [])]
        security_issues = [SecurityIssue(**s) for s in parsed.get("security_issues", [])]
        improvements = [Improvement(**i) for i in parsed.get("improvements", [])]
        
        breakdown_data = parsed.get("breakdown", {})
        breakdown = QualityBreakdown(
            correctness=breakdown_data.get("correctness", 70),
            security=breakdown_data.get("security", 70),
            performance=breakdown_data.get("performance", 70),
            readability=breakdown_data.get("readability", 70),
            maintainability=breakdown_data.get("maintainability", 70),
        )
        
        score = parsed.get("score", 50)
        
        return ReviewResponse(
            score=score,
            grade=self.calculate_grade(score),
            breakdown=breakdown,
            bugs=bugs,
            security_issues=security_issues,
            improvements=improvements,
            summary=parsed.get("summary", "Review complete."),
            language_detected=detected_lang,
            review_time_ms=review_time,
            model_used=self.backend_name,
        )


# Singleton instance — loaded once on startup
_service: Optional[LLMService] = None

def get_llm_service() -> LLMService:
    global _service
    if _service is None:
        _service = LLMService()
    return _service
