"""
test_api.py
===========
Tests for the FastAPI backend.
Run with: pytest tests/ -v
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

client = TestClient(app)


# ─── Health Tests ─────────────────────────────────────────────────────────────

def test_health_check():
    """Backend should always return healthy status."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "version" in data


# ─── Review Tests ─────────────────────────────────────────────────────────────

UNSAFE_SQL_CODE = '''
def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return db.execute(query)
'''

SAFE_CODE = '''
from typing import Optional
import logging

logger = logging.getLogger(__name__)

def calculate_average(numbers: list[float]) -> Optional[float]:
    """Calculate arithmetic mean."""
    if not numbers:
        return None
    return sum(numbers) / len(numbers)
'''

DIVISION_BUG_CODE = '''
def divide(a, b):
    return a / b

result = divide(10, 0)
'''


def test_review_basic():
    """Review endpoint should return valid response."""
    response = client.post("/api/v1/review", json={
        "code": DIVISION_BUG_CODE,
        "language": "python"
    })
    assert response.status_code == 200
    data = response.json()
    
    # Check required fields
    assert "score" in data
    assert "grade" in data
    assert "bugs" in data
    assert "security_issues" in data
    assert "improvements" in data
    assert "summary" in data
    assert "breakdown" in data
    
    # Score should be 0-100
    assert 0 <= data["score"] <= 100
    
    # Grade should be a letter
    assert data["grade"] in ["A", "B", "C", "D", "F"]


def test_review_detects_sql_injection():
    """Should detect SQL injection vulnerability."""
    response = client.post("/api/v1/review", json={
        "code": UNSAFE_SQL_CODE,
        "language": "python"
    })
    assert response.status_code == 200
    data = response.json()
    
    # Should have security issues
    assert len(data["security_issues"]) > 0
    
    # SQL injection should be detected
    issue_types = [issue["issue_type"] for issue in data["security_issues"]]
    assert "SQL Injection" in issue_types
    
    # Score should be low due to critical security issue
    assert data["score"] < 70


def test_review_scores_good_code_high():
    """Clean code should get a high score."""
    response = client.post("/api/v1/review", json={
        "code": SAFE_CODE,
        "language": "python"
    })
    assert response.status_code == 200
    data = response.json()
    
    # Good code should score well
    assert data["score"] >= 70
    assert data["grade"] in ["A", "B", "C"]
    
    # Should have no critical security issues
    critical = [i for i in data["security_issues"] if i["severity"] == "critical"]
    assert len(critical) == 0


def test_review_with_filename():
    """Filename should be accepted and used for language detection."""
    response = client.post("/api/v1/review", json={
        "code": "def hello(): print('world')",
        "language": "auto",
        "filename": "hello.py"
    })
    assert response.status_code == 200
    assert response.json()["language_detected"] == "python"


def test_review_empty_code():
    """Empty code should return 422 validation error."""
    response = client.post("/api/v1/review", json={
        "code": "",
        "language": "python"
    })
    assert response.status_code == 422


def test_review_code_too_long():
    """Code exceeding limit should return 413."""
    long_code = "x = 1\n" * 10000
    response = client.post("/api/v1/review", json={
        "code": long_code,
        "language": "python"
    })
    assert response.status_code == 413


def test_quick_scan():
    """Quick scan should return subset of full review."""
    response = client.post("/api/v1/quick-scan", json={
        "code": UNSAFE_SQL_CODE,
        "language": "python"
    })
    assert response.status_code == 200
    data = response.json()
    assert "score" in data
    assert "security_issues" in data


def test_supported_languages():
    """Languages endpoint should return list of languages."""
    response = client.get("/api/v1/languages")
    assert response.status_code == 200
    data = response.json()
    assert "languages" in data
    lang_ids = [l["id"] for l in data["languages"]]
    assert "python" in lang_ids
    assert "javascript" in lang_ids


# ─── Breakdown Tests ──────────────────────────────────────────────────────────

def test_review_breakdown_all_fields():
    """Breakdown should contain all 5 quality dimensions."""
    response = client.post("/api/v1/review", json={
        "code": "def foo(): pass",
        "language": "python"
    })
    data = response.json()
    breakdown = data["breakdown"]
    
    required_keys = ["correctness", "security", "performance", "readability", "maintainability"]
    for key in required_keys:
        assert key in breakdown
        assert 0 <= breakdown[key] <= 100


def test_review_metadata():
    """Response should include useful metadata."""
    response = client.post("/api/v1/review", json={
        "code": "print('hello')",
        "language": "python"
    })
    data = response.json()
    
    assert "review_time_ms" in data
    assert data["review_time_ms"] >= 0
    assert "model_used" in data
    assert "language_detected" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
