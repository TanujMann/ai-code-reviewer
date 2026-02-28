from pydantic import BaseModel, Field
from typing import List, Optional

class CodeReviewRequest(BaseModel):
    code: str = Field(..., min_length=1, max_length=10000)
    language: str = Field(default="python")
    filename: Optional[str] = None

class Bug(BaseModel):
    line: Optional[int] = None
    description: str
    severity: str = "medium"
    suggestion: Optional[str] = None

class SecurityIssue(BaseModel):
    line: Optional[int] = None
    issue_type: str
    description: str
    severity: str = "high"
    fix: Optional[str] = None

class Improvement(BaseModel):
    line: Optional[int] = None
    category: str
    description: str
    code_example: Optional[str] = None

class QualityBreakdown(BaseModel):
    correctness: float = 0
    security: float = 0
    performance: float = 0
    readability: float = 0
    maintainability: float = 0

class CodeReviewResponse(BaseModel):
    score: float
    grade: str
    breakdown: QualityBreakdown
    bugs: List[Bug] = []
    security_issues: List[SecurityIssue] = []
    improvements: List[Improvement] = []
    summary: str
    language_detected: str
    review_time_ms: float
    model_used: str

class QuickScanRequest(BaseModel):
    code: str
    language: str = "python"

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_backend: str
    version: str