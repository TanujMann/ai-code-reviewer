"""
schemas.py
==========
Pydantic models for request/response validation.
These define the exact shape of data going in and out of the API.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional
from enum import Enum


class Language(str, Enum):
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CPP = "cpp"
    C = "c"
    GO = "go"
    RUST = "rust"
    AUTO = "auto"   # Auto-detect


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


# ─── Request Models ────────────────────────────────────────────────────────────

class ReviewRequest(BaseModel):
    """What the VS Code extension sends to the API."""
    code: str = Field(
        ...,
        description="The source code to review",
        min_length=1,
        max_length=10000
    )
    language: Language = Field(
        default=Language.AUTO,
        description="Programming language of the code"
    )
    filename: Optional[str] = Field(
        default=None,
        description="Optional filename for context (e.g., 'app.py')"
    )
    focus_areas: Optional[List[str]] = Field(
        default=None,
        description="Specific areas to focus on: ['security', 'performance', 'style']"
    )
    
    @validator("code")
    def code_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Code cannot be empty or whitespace only")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "code": "def divide(a, b):\n    return a / b",
                "language": "python",
                "filename": "math_utils.py"
            }
        }


# ─── Response Models ───────────────────────────────────────────────────────────

class Bug(BaseModel):
    """A single bug or issue found in the code."""
    line: Optional[int] = Field(None, description="Line number of the bug")
    description: str = Field(..., description="Description of the bug")
    severity: Severity = Field(default=Severity.MEDIUM)
    suggestion: Optional[str] = Field(None, description="How to fix it")


class SecurityIssue(BaseModel):
    """A security vulnerability found in the code."""
    line: Optional[int] = None
    issue_type: str = Field(..., description="e.g., 'SQL Injection', 'XSS', 'Hardcoded Secret'")
    description: str
    severity: Severity = Field(default=Severity.HIGH)
    fix: Optional[str] = None


class Improvement(BaseModel):
    """A code improvement suggestion."""
    line: Optional[int] = None
    category: str = Field(..., description="e.g., 'Performance', 'Readability', 'Best Practice'")
    description: str
    code_example: Optional[str] = Field(None, description="Example of the improved code")


class QualityBreakdown(BaseModel):
    """Detailed quality score breakdown."""
    correctness: int = Field(..., ge=0, le=100, description="Code correctness (0-100)")
    security: int = Field(..., ge=0, le=100)
    performance: int = Field(..., ge=0, le=100)
    readability: int = Field(..., ge=0, le=100)
    maintainability: int = Field(..., ge=0, le=100)


class ReviewResponse(BaseModel):
    """Full review response sent back to the VS Code extension."""
    
    # Overall score
    score: int = Field(..., ge=0, le=100, description="Overall code quality score")
    grade: str = Field(..., description="Letter grade: A, B, C, D, F")
    
    # Detailed breakdown
    breakdown: QualityBreakdown
    
    # Issues found
    bugs: List[Bug] = Field(default_factory=list)
    security_issues: List[SecurityIssue] = Field(default_factory=list)
    improvements: List[Improvement] = Field(default_factory=list)
    
    # Summary
    summary: str = Field(..., description="One-paragraph summary of the review")
    
    # Metadata
    language_detected: str
    review_time_ms: int = Field(..., description="Time taken for review in milliseconds")
    model_used: str = Field(..., description="Which model performed the review")
    
    class Config:
        json_schema_extra = {
            "example": {
                "score": 35,
                "grade": "D",
                "breakdown": {
                    "correctness": 40,
                    "security": 10,
                    "performance": 60,
                    "readability": 50,
                    "maintainability": 45
                },
                "bugs": [
                    {
                        "line": 2,
                        "description": "Division by zero not handled",
                        "severity": "critical",
                        "suggestion": "Add: if b == 0: raise ValueError('Division by zero')"
                    }
                ],
                "security_issues": [],
                "improvements": [
                    {
                        "category": "Best Practice",
                        "description": "Add type hints",
                        "code_example": "def divide(a: float, b: float) -> float:"
                    }
                ],
                "summary": "Critical bug found: unhandled division by zero.",
                "language_detected": "python",
                "review_time_ms": 1250,
                "model_used": "demo"
            }
        }


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_backend: str
    version: str
